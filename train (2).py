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

# Suppress Pydantic/System warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. ROBUST CONFIGURATION & PATH FIX
# ==========================================
class Config:
    seed = 927
    model_name = 'resnet152' 
    img_size = 224
    batch_size = 32
    embedding_size = 512
    num_classes = 31
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Placeholder Paths (Will be auto-corrected below)
    train_dir = ''
    test_dir = ''
    train_csv = ''
    test_csv = ''
    submission_file = 'submission.csv'

def find_dataset_paths():
    """Auto-detects the correct paths for CSVs and Image directories."""
    print("Searching for dataset files...")
    base_search_paths = ['/kaggle/input', '.']
    
    # 1. Find CSVs
    for root in base_search_paths:
        for dirpath, _, files in os.walk(root):
            if 'train.csv' in files:
                Config.train_csv = os.path.join(dirpath, 'train.csv')
            if 'test.csv' in files:
                Config.test_csv = os.path.join(dirpath, 'test.csv')
                
    # 2. Find Image Directories (Look for specific sample images)
    # We look for 'train_0001.png' and 'test_0001.png' to identify the folders
    sample_train = 'train_0001.png'
    sample_test = 'test_0001.png'
    
    for root in base_search_paths:
        for dirpath, _, files in os.walk(root):
            if sample_train in files:
                Config.train_dir = dirpath
            if sample_test in files:
                Config.test_dir = dirpath
                
    # 3. Fallbacks / Verification
    if not Config.train_dir or not Config.train_csv:
        print("WARNING: Dataset not found. Using Dummy Paths (Code will generate random submission).")
        # Set dummy paths to prevent crashes before the dummy-data check
        Config.train_csv = 'dummy_train.csv' 
    else:
        print(f"Dataset Found:\n  Train CSV: {Config.train_csv}\n  Train Dir: {Config.train_dir}")

# Set Seeds
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(Config.seed)
find_dataset_paths() # Run path fixer immediately

# ==========================================
# 2. DATASET (Robust)
# ==========================================
class JaguarDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        
        if not self.is_test:
            # Map labels
            self.label_map = {name: idx for idx, name in enumerate(df['ground_truth'].unique())}
            self.df['label'] = self.df['ground_truth'].map(self.label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename']
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Robustness: If a file is missing/corrupt, return a black image instead of crashing
            # logging only once to avoid spamming
            if idx == 0: print(f"Warning: Could not load {img_path}. Returning black image. ({e})")
            image = Image.new('RGB', (Config.img_size, Config.img_size), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_test:
            return image, img_name
        else:
            label = torch.tensor(row['label'], dtype=torch.long)
            return image, label

# ==========================================
# 3. ARCFACE & MODEL
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
    def __init__(self, model_name, embedding_size, num_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        self.neck = nn.Sequential(
            nn.BatchNorm1d(in_features),
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
# 4. PIPELINE
# ==========================================
def run_pipeline():
    print(f"Running on device: {Config.device}")
    
    # Transforms
    transforms_train = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Check for Data Presence
    if not os.path.exists(Config.train_csv):
        print("Data files missing. Generating DUMMY submission for verification.")
        # Generate dummy
        if os.path.exists('sample_submission.csv'):
            sample = pd.read_csv('sample_submission.csv')
            pd.DataFrame({'row_id': sample['row_id'], 'similarity': np.random.rand(len(sample))}).to_csv(Config.submission_file, index=False)
        else:
            pd.DataFrame({'row_id': range(10), 'similarity': np.random.rand(10)}).to_csv(Config.submission_file, index=False)
        return

    # Load Data
    train_df = pd.read_csv(Config.train_csv)
    test_df_pairs = pd.read_csv(Config.test_csv)
    
    train_dataset = JaguarDataset(train_df, Config.train_dir, transform=transforms_train)
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=2)

    # Initialize Model
    model = JaguarReIDModel(Config.model_name, Config.embedding_size, Config.num_classes).to(Config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training
    print("Starting Training...")
    model.train()
    for epoch in range(10): # Increase epochs for real training
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for images, labels in loop:
            images, labels = images.to(Config.device), labels.to(Config.device)
            optimizer.zero_grad()
            loss = criterion(model(images, labels), labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

    # Inference
    print("Starting Inference...")
    model.eval()
    
    unique_images = sorted(list(set(test_df_pairs['query_image']) | set(test_df_pairs['gallery_image'])))
    unique_df = pd.DataFrame({'filename': unique_images})
    test_dataset = JaguarDataset(unique_df, Config.test_dir, transform=transforms_test, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=2)
    
    embeddings_map = {}
    with torch.no_grad():
        for images, names in tqdm(test_loader, desc="Extracting"):
            images = images.to(Config.device)
            emb = F.normalize(model(images), p=2, dim=1).cpu().numpy()
            for name, e in zip(names, emb):
                embeddings_map[name] = e
                
    # Scoring
    similarities = []
    # Optimization: Use list comprehension or vectorization if possible, loop is fine for 130k
    for _, row in tqdm(test_df_pairs.iterrows(), total=len(test_df_pairs), desc="Pairing"):
        if row['query_image'] in embeddings_map and row['gallery_image'] in embeddings_map:
            sim = np.dot(embeddings_map[row['query_image']], embeddings_map[row['gallery_image']])
            similarities.append((sim + 1) / 2)
        else:
            similarities.append(0.5) # Fallback if image load failed

    # Save
    pd.DataFrame({'row_id': test_df_pairs['row_id'], 'similarity': similarities}).to_csv(Config.submission_file, index=False)
    print("Success! Submission saved.")

if __name__ == '__main__':
    run_pipeline()