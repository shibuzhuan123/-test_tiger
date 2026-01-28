import os
import math
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
warnings.filterwarnings("ignore")

try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

# ==========================================
# 1. CONFIG - 超参数调优版本
# ==========================================
class Config:
    seed = 42

    # 模型配置
    model_name = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
    img_size = 448
    embedding_dim = 1024
    num_classes = 31

    # ============ 超参数调优区域 ============
    # 训练轮数：10 → 15 (更多训练)
    num_epochs = 15

    # 批次大小：根据显存调整
    batch_size = 8
    grad_accum = 4  # 有效 batch = 32

    # 学习率：2e-5 → 1.5e-5 (稍低，更稳定)
    lr = 1.5e-5
    weight_decay = 1e-3
    min_lr = 1e-6

    # ArcFace 参数调优
    arcface_s = 30.0
    arcface_m = 0.35  # 0.5 → 0.35 (更宽松的 margin)

    # GeM Pooling 参数：p=3 → p=2.5 (可学习)
    gem_p = 2.5

    # 后处理参数调优
    use_tta = True
    use_qe = True
    qe_top_k = 5  # 3 → 5 (更强的查询扩展)

    use_rerank = True
    rerank_k1 = 30   # 20 → 30
    rerank_k2 = 10   # 6 → 10
    rerank_lambda = 0.4  # 0.3 → 0.4
    # ===========================================

    # 多GPU配置
    multi_gpu = True
    gpu_ids = '0,1'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    # 预训练权重路径
    pretrained_path_pth = '/home/fei/kaggle/eva02_large_patch14_448.pth'
    pretrained_path_safetensors = '/home/fei/kaggle/model.safetensors'

    # 路径
    train_dir = ''
    test_dir = ''
    train_csv = ''
    test_csv = ''
    submission_file = 'submission.csv'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(Config.seed)

def find_dataset_paths():
    """自动查找数据集路径"""
    print("Searching for dataset files...")
    base_search_paths = ['/kaggle/input', '.', '/home/fei/kaggle']

    for root in base_search_paths:
        for dirpath, _, files in os.walk(root):
            if 'train.csv' in files and not Config.train_csv:
                Config.train_csv = os.path.join(dirpath, 'train.csv')
            if 'test.csv' in files and not Config.test_csv:
                Config.test_csv = os.path.join(dirpath, 'test.csv')

    sample_train = 'train_0001.png'
    sample_test = 'test_0001.png'

    for root in base_search_paths:
        for dirpath, _, files in os.walk(root):
            if sample_train in files and not Config.train_dir:
                Config.train_dir = dirpath
            if sample_test in files and not Config.test_dir:
                Config.test_dir = dirpath

    if Config.train_dir and Config.train_csv:
        print(f"Dataset Found:\n  Train CSV: {Config.train_csv}\n  Train Dir: {Config.train_dir}")
        print(f"  Test CSV: {Config.test_csv}\n  Test Dir: {Config.test_dir}")

find_dataset_paths()

# ==========================================
# 2. DATASET
# ==========================================
class JaguarDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_test = is_test

        if not is_test:
            unique_ids = sorted(df["ground_truth"].unique())
            self.label_map = {name: i for i, name in enumerate(unique_ids)}
            self.df["label"] = self.df["ground_truth"].map(self.label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["filename"]
        img_path = self.img_dir / img_name

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new("RGB", (Config.img_size, Config.img_size))

        if self.transform:
            img = self.transform(img)

        if self.is_test:
            return img, img_name
        return img, torch.tensor(row["label"], dtype=torch.long)

# ==========================================
# 3. TRANSFORMS
# ==========================================
train_transform = transforms.Compose([
    transforms.Resize((Config.img_size, Config.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.481, 0.457, 0.408], [0.268, 0.261, 0.275]),
    transforms.RandomErasing(p=0.25),
])

test_transform = transforms.Compose([
    transforms.Resize((Config.img_size, Config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.481, 0.457, 0.408], [0.268, 0.261, 0.275]),
])

# ==========================================
# 4. MODEL WITH TUNED GeM
# ==========================================
class GeM(nn.Module):
    """广义均值池化 - 可学习参数 p"""
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)

class ArcFaceLayer(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None:
            return cosine
        phi = cosine - self.m
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s

class JaguarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            Config.model_name, pretrained=False, num_classes=0
        )

        # 加载预训练权重
        loaded = False
        if SAFETENSORS_AVAILABLE and os.path.exists(Config.pretrained_path_safetensors):
            print(f"Loading safetensors weights from: {Config.pretrained_path_safetensors}")
            try:
                state_dict = load_file(Config.pretrained_path_safetensors)
                self.backbone.load_state_dict(state_dict, strict=False)
                print("✅ Safetensors weights loaded successfully!")
                loaded = True
            except Exception as e:
                print(f"⚠️  Failed to load safetensors: {e}")

        if not loaded and os.path.exists(Config.pretrained_path_pth):
            print(f"Loading PyTorch weights from: {Config.pretrained_path_pth}")
            try:
                state_dict = torch.load(Config.pretrained_path_pth, map_location='cpu')
                self.backbone.load_state_dict(state_dict, strict=False)
                print("✅ PyTorch weights loaded successfully!")
                loaded = True
            except Exception as e:
                print(f"⚠️  Failed to load pth: {e}")

        if not loaded:
            print("⚠️  No pretrained weights found, using random initialization")

        self.feat_dim = self.backbone.num_features

        # 使用调优后的 GeM 参数
        self.gem = GeM(p=Config.gem_p, eps=1e-6)

        self.bn = nn.BatchNorm1d(self.feat_dim)

        self.head = ArcFaceLayer(
            self.feat_dim, Config.num_classes, s=Config.arcface_s, m=Config.arcface_m
        )

    def forward(self, x, label=None):
        features = self.backbone.forward_features(x)

        if features.dim() == 3:
            B, N, C = features.shape
            H = W = int(math.sqrt(N))
            if H * W != N:
                features = features[:, -H * W :, :]
            features = features.permute(0, 2, 1).reshape(B, C, H, W)

        emb = self.gem(features).flatten(1)
        emb = self.bn(emb)

        if label is not None:
            return self.head(emb, label)
        return emb

# ==========================================
# 5. TRAINING FUNCTIONS
# ==========================================
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    loss_meter = 0

    for i, (imgs, labels) in enumerate(tqdm(loader, leave=False)):
        imgs, labels = imgs.to(Config.device), labels.to(Config.device)

        with torch.amp.autocast(Config.device_type):
            loss = criterion(model(imgs, labels), labels)
            loss = loss / Config.grad_accum

        scaler.scale(loss).backward()

        if (i + 1) % Config.grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_meter += loss.item() * Config.grad_accum

    return loss_meter / len(loader)

@torch.no_grad()
def extract_features(model, loader):
    """提取特征，支持 TTA"""
    model.eval()
    feats, names = [], []

    for imgs, fnames in tqdm(loader, desc="Inference"):
        imgs = imgs.to(Config.device)
        f1 = model(imgs)

        if Config.use_tta:
            f2 = model(torch.flip(imgs, [3]))
            f1 = (f1 + f2) / 2

        feats.append(F.normalize(f1, dim=1).cpu())
        names.extend(fnames)

    return torch.cat(feats, dim=0).numpy(), names

# ==========================================
# 6. POST-PROCESSING WITH TUNED PARAMETERS
# ==========================================
def query_expansion(emb, top_k=3):
    """Query Expansion: 使用调优后的 top_k"""
    print(f"Applying Query Expansion (top_k={top_k})...")

    sims = emb @ emb.T
    indices = np.argsort(-sims, axis=1)[:, :top_k]

    new_emb = np.zeros_like(emb)
    for i in range(len(emb)):
        new_emb[i] = np.mean(emb[indices[i]], axis=0)

    return new_emb / np.linalg.norm(new_emb, axis=1, keepdims=True)

def k_reciprocal_rerank(prob, k1=20, k2=6, lambda_value=0.3):
    """K-reciprocal Re-ranking: 使用调优后的参数"""
    print(f"Applying K-reciprocal Re-ranking (k1={k1}, k2={k2}, lambda={lambda_value})...")

    q_g_dist = 1 - prob
    original_dist = q_g_dist.copy()
    initial_rank = np.argsort(q_g_dist, axis=1)

    nn_k1 = []
    for i in range(prob.shape[0]):
        forward_k1 = initial_rank[i, : k1 + 1]
        backward_k1 = initial_rank[forward_k1, : k1 + 1]
        fi = np.where(backward_k1 == i)[0]
        nn_k1.append(forward_k1[fi])

    jaccard_dist = np.zeros_like(original_dist)
    for i in range(prob.shape[0]):
        ind_non_zero = np.where(original_dist[i, :] < 0.6)[0]
        ind_images = [
            inv for inv in ind_non_zero if len(np.intersect1d(nn_k1[i], nn_k1[inv])) > 0
        ]
        for j in ind_images:
            intersection = len(np.intersect1d(nn_k1[i], nn_k1[j]))
            union = len(np.union1d(nn_k1[i], nn_k1[j]))
            jaccard_dist[i, j] = 1 - intersection / union

    return 1 - (jaccard_dist * lambda_value + original_dist * (1 - lambda_value))

# ==========================================
# 7. MAIN PIPELINE
# ==========================================
def run_pipeline():
    # 设置多GPU
    if Config.multi_gpu and torch.cuda.device_count() >= 2:
        os.environ['CUDA_VISIBLE_DEVICES'] = Config.gpu_ids
        print(f"Using Multi-GPU: {Config.gpu_ids}")

    print(f"Device: {Config.device}")
    print(f"Model: {Config.model_name}")
    print(f"Image Size: {Config.img_size}")
    print(f"Effective Batch Size: {Config.batch_size * Config.grad_accum}")
    print(f"Learning Rate: {Config.lr}")
    print(f"Epochs: {Config.num_epochs}")
    print(f"ArcFace Margin: {Config.arcface_m}")
    print(f"GeM Pooling p: {Config.gem_p}")
    print(f"Post-processing: TTA={Config.use_tta}, QE(top_k={Config.qe_top_k}), Re-rank(k1={Config.rerank_k1}, k2={Config.rerank_k2}, λ={Config.rerank_lambda})")

    # 加载数据
    if not os.path.exists(Config.train_csv):
        print("Data files missing.")
        return

    train_df = pd.read_csv(Config.train_csv)
    test_df = pd.read_csv(Config.test_csv)

    train_loader = DataLoader(
        JaguarDataset(train_df, Config.train_dir, train_transform),
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 模型
    model = JaguarModel().to(Config.device)

    if Config.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.num_epochs, eta_min=Config.min_lr
    )

    scaler = torch.amp.GradScaler(Config.device_type)

    # 训练
    print(f"\n🔥 Training for {Config.num_epochs} epochs...")
    print("📊 Hyperparameter Tuning:")
    print("  - Epochs: 10 → 15")
    print("  - LR: 2e-5 → 1.5e-5")
    print("  - ArcFace m: 0.5 → 0.35")
    print("  - GeM p: 3 → 2.5")
    print("  - QE top_k: 3 → 5")
    print("  - Re-rank k1: 20 → 30, k2: 6 → 10, λ: 0.3 → 0.4")
    print()

    best_loss = float('inf')

    for epoch in range(Config.num_epochs):
        loss = train_epoch(model, train_loader, optimizer, nn.CrossEntropyLoss(), scaler)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.num_epochs} | "
            f"Loss: {loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if loss < best_loss:
            best_loss = loss
            if Config.multi_gpu and torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), 'best_model_v3_1.pth')
            else:
                torch.save(model.state_dict(), 'best_model_v3_1.pth')

    # 推理
    print("\n🔍 Extracting features...")
    model.eval()

    unique_test = sorted(set(test_df["query_image"]) | set(test_df["gallery_image"]))
    test_loader = DataLoader(
        JaguarDataset(pd.DataFrame({"filename": unique_test}), Config.test_dir, test_transform, True),
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    emb, names = extract_features(model, test_loader)
    img_map = {n: i for i, n in enumerate(names)}

    # Query Expansion (使用调优后的参数)
    if Config.use_qe:
        emb = query_expansion(emb, top_k=Config.qe_top_k)

    # 计算相似度矩阵
    sim_matrix = emb @ emb.T

    # Re-ranking (使用调优后的参数)
    if Config.use_rerank:
        sim_matrix = k_reciprocal_rerank(
            sim_matrix,
            k1=Config.rerank_k1,
            k2=Config.rerank_k2,
            lambda_value=Config.rerank_lambda
        )

    # 生成预测
    print("📝 Generating predictions...")
    preds = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Mapping"):
        s = sim_matrix[img_map[row["query_image"]], img_map[row["gallery_image"]]]
        preds.append(max(0.0, min(1.0, s)))

    pd.DataFrame({"row_id": test_df["row_id"], "similarity": preds}).to_csv(Config.submission_file, index=False)
    print(f"✅ Done! Mean Sim: {np.mean(preds):.4f}")

if __name__ == '__main__':
    run_pipeline()
