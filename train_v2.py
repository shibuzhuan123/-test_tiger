import os
import math
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import timm

# 设置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    seed = 927
    model_name = 'resnet152'
    img_size = 256  # 增大输入尺寸
    batch_size = 128
    embedding_size = 512
    num_classes = 31
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 多GPU配置
    multi_gpu = True
    gpu_ids = '0,1'

    # 预训练权重配置
    use_pretrained = True
    pretrained_path = '/home/fei/kaggle/resnet152-394f9c45.pth'

    # 新增：背景处理配置
    use_alpha_mask = True  # 使用 alpha mask 去除背景
    # 背景策略在数据集初始化时指定：
    # - 训练时：'random' 或 'random_noise'（防止模型学习背景特征）
    # - 推理时：'gray' 或 'mean'（固定背景用于评估）

    # 训练配置
    num_epochs = 20  # 增加训练轮数
    warmup_epochs = 3
    use_mixup = True  # Mixup 数据增强
    mixup_alpha = 0.2

    # 学习率配置
    lr = 0.001
    min_lr = 1e-6
    weight_decay = 1e-4

    # 路径
    train_dir = ''
    test_dir = ''
    train_csv = ''
    test_csv = ''
    submission_file = 'submission.csv'

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

    if not Config.train_dir or not Config.train_csv:
        print("WARNING: Dataset not found.")
        Config.train_csv = 'dummy_train.csv'
    else:
        print(f"Dataset Found:\n  Train CSV: {Config.train_csv}\n  Train Dir: {Config.train_dir}")
        print(f"  Test CSV: {Config.test_csv}\n  Test Dir: {Config.test_dir}")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(Config.seed)
find_dataset_paths()

# ==========================================
# 2. ENHANCED DATASET WITH ALPHA MASK
# ==========================================
def apply_alpha_mask(image, background='white'):
    """
    使用 alpha mask 去除背景

    Args:
        image: PIL Image (RGBA mode)
        background: 'white', 'black', 'gray', 'mean', 'random', or 'random_noise'
                  'random' - 随机选择 white/black/gray/mean
                  'random_noise' - 使用随机噪声作为背景
    """
    if image.mode != 'RGBA':
        return image.convert('RGB')

    # 分离通道
    r, g, b, a = image.split()

    # 随机选择背景（训练时使用）
    if background == 'random':
        background = random.choice(['white', 'black', 'gray', 'mean'])
    elif background == 'random_noise':
        # 生成随机噪声背景
        noise = np.random.randint(0, 256, (*image.size, 3), dtype=np.uint8)
        bg = Image.fromarray(noise, 'RGB')
        bg.paste(image.convert('RGB'), (0, 0), mask=a)
        return bg

    # 创建背景
    if background == 'white':
        bg = Image.new('RGB', image.size, (255, 255, 255))
    elif background == 'black':
        bg = Image.new('RGB', image.size, (0, 0, 0))
    elif background == 'gray':
        bg = Image.new('RGB', image.size, (128, 128, 128))
    elif background == 'mean':
        # 使用 ImageNet 均值作为背景
        bg = Image.new('RGB', image.size, (122, 116, 103))
    else:
        bg = Image.new('RGB', image.size, (128, 128, 128))

    # 合成：alpha 区域保留原图，其他区域用背景
    bg.paste(image.convert('RGB'), (0, 0), mask=a)
    return bg

class JaguarDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False, use_alpha=True,
                 train_background='random', test_background='gray'):
        """
        Args:
            train_background: 训练时的背景策略，推荐 'random' 或 'random_noise'
            test_background: 推理时的背景，推荐 'gray' 或 'mean'
        """
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.use_alpha = use_alpha
        self.train_background = train_background
        self.test_background = test_background

        if not self.is_test:
            self.label_map = {name: idx for idx, name in enumerate(df['ground_truth'].unique())}
            self.df['label'] = self.df['ground_truth'].map(self.label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename']
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path)

            # 使用 alpha mask 去除背景
            if self.use_alpha and image.mode == 'RGBA':
                # 训练时用随机背景，推理时用固定背景
                bg_color = self.train_background if not self.is_test else self.test_background
                image = apply_alpha_mask(image, bg_color)
            elif image.mode != 'RGB':
                image = image.convert('RGB')

        except Exception as e:
            if idx == 0: print(f"Warning: Could not load {img_path}. ({e})")
            image = Image.new('RGB', (Config.img_size, Config.img_size), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, img_name
        else:
            label = torch.tensor(row['label'], dtype=torch.long)
            return image, label

# ==========================================
# 3. MIXUP AUGMENTATION
# ==========================================
def mixup_data(x, y, alpha=0.2):
    """Mixup 数据增强"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup 损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==========================================
# 4. MODEL WITH ARCFACE
# ==========================================
class ArcFaceLayer(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None: return cosine
        phi = cosine - self.m
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class JaguarReIDModel(nn.Module):
    def __init__(self, model_name, embedding_size, num_classes, use_pretrained=True, pretrained_path=None):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)

        if use_pretrained and pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict, strict=False)
            print("Pretrained weights loaded successfully!")
        elif use_pretrained:
            print("Warning: pretrained_path not found, using random initialization")

        in_features = self.backbone.num_features
        self.neck = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.3),
            nn.Linear(in_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )
        self.head = ArcFaceLayer(embedding_size, num_classes)

    def forward(self, x, label=None):
        features = self.backbone(x)
        embeddings = self.neck(features)
        if label is not None: return self.head(embeddings, label)
        return embeddings

# ==========================================
# 5. COSINE ANNEALING LR SCHEDULER
# ==========================================
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    """余弦退火学习率调度"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(cosine_decay, min_lr / optimizer.param_groups[0]['lr'])

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ==========================================
# 6. TRAINING PIPELINE
# ==========================================
def run_pipeline():
    if Config.multi_gpu and torch.cuda.device_count() >= 2:
        os.environ['CUDA_VISIBLE_DEVICES'] = Config.gpu_ids
        print(f"Using Multi-GPU: {Config.gpu_ids}")

    print(f"Running on device: {Config.device}")
    print(f"Using Alpha Mask: {Config.use_alpha_mask}")
    print(f"Background Strategy:")
    print(f"  - Training: Random (white/black/gray/mean) - prevents background bias")
    print(f"  - Inference: Fixed gray - consistent evaluation")

    # 增强的数据增强
    transforms_train = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.RandomResizedCrop(Config.img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(Config.train_csv):
        print("Data files missing.")
        return

    train_df = pd.read_csv(Config.train_csv)
    test_df_pairs = pd.read_csv(Config.test_csv)

    train_dataset = JaguarDataset(
        train_df,
        Config.train_dir,
        transform=transforms_train,
        use_alpha=Config.use_alpha_mask,
        train_background='random',  # 训练时使用随机背景
        test_background='gray'      # 推理时使用固定背景
    )
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)

    model = JaguarReIDModel(
        Config.model_name,
        Config.embedding_size,
        Config.num_classes,
        use_pretrained=Config.use_pretrained,
        pretrained_path=Config.pretrained_path
    ).to(Config.device)

    if Config.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 学习率调度器
    num_training_steps = Config.num_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=Config.warmup_epochs * len(train_loader),
        num_training_steps=num_training_steps,
        min_lr=Config.min_lr
    )

    # Training
    print(f"Starting Training for {Config.num_epochs} epochs...")
    best_loss = float('inf')

    for epoch in range(Config.num_epochs):
        model.train()
        epoch_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.num_epochs}")
        for images, labels in loop:
            images, labels = images.to(Config.device), labels.to(Config.device)

            # Mixup
            if Config.use_mixup and random.random() < 0.5:
                images, labels_a, labels_b, lam = mixup_data(images, labels, Config.mixup_alpha)
                optimizer.zero_grad()
                outputs = model(images, labels)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                optimizer.zero_grad()
                outputs = model(images, labels)
                loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item(), lr=f"{optimizer.param_groups[0]['lr']:.6f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            # 保存最佳模型
            if Config.multi_gpu and torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), 'best_model.pth')
            else:
                torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with loss: {best_loss:.4f}")

    # Inference
    print("Starting Inference...")
    model.eval()

    unique_images = sorted(list(set(test_df_pairs['query_image']) | set(test_df_pairs['gallery_image'])))
    unique_df = pd.DataFrame({'filename': unique_images})
    test_dataset = JaguarDataset(
        unique_df,
        Config.test_dir,
        transform=transforms_test,
        is_test=True,
        use_alpha=Config.use_alpha_mask,
        train_background='random',  # 训练时用随机（这里不使用）
        test_background='gray'      # 推理时用灰色背景
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)

    embeddings_map = {}
    with torch.no_grad():
        for images, names in tqdm(test_loader, desc="Extracting"):
            images = images.to(Config.device)
            emb = F.normalize(model(images), p=2, dim=1).cpu().numpy()
            for name, e in zip(names, emb):
                embeddings_map[name] = e

    # Scoring
    similarities = []
    for _, row in tqdm(test_df_pairs.iterrows(), total=len(test_df_pairs), desc="Pairing"):
        if row['query_image'] in embeddings_map and row['gallery_image'] in embeddings_map:
            sim = np.dot(embeddings_map[row['query_image']], embeddings_map[row['gallery_image']])
            similarities.append((sim + 1) / 2)
        else:
            similarities.append(0.5)

    pd.DataFrame({'row_id': test_df_pairs['row_id'], 'similarity': similarities}).to_csv(Config.submission_file, index=False)
    print("Success! Submission saved.")

if __name__ == '__main__':
    run_pipeline()
