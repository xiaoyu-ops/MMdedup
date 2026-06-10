import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
import librosa
import numpy as np
from PIL import Image
from tqdm import tqdm

# ================= Configuration =================
# 1. Point directly to the raw noisy data directory.
# Set this to the absolute path that contains the original WAV files.
RAW_DATA_PATH = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\digital_swamp_audio"

# 2. Training parameters. Keep them consistent with previous runs for comparison.
BATCH_SIZE = 32
EPOCHS = 10          # Keep 10 epochs.
LEARNING_RATE = 0.001
# ===============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AudioSpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = []
        
        # Scan the raw directory.
        files = [f for f in os.listdir(root_dir) if f.endswith('.wav')]
        
        # Parse class ids while supporting prefixes such as r01_.
        class_ids = set()
        temp_samples = []
        
        for f in files:
            try:
                # Logic: strip extension, split on '-', and use the last segment.
                # Example: "r01_1-137-A-32.wav" -> "32"
                # Example: "1-137-A-32.wav" -> "32"
                name_no_ext = os.path.splitext(f)[0]
                parts = name_no_ext.split('-')
                label_str = parts[-1] 
                
                if label_str.isdigit():
                    label = int(label_str)
                    class_ids.add(label)
                    temp_samples.append((os.path.join(root_dir, f), label))
            except:
                pass
        
        self.classes = sorted(list(class_ids))
        real_label_map = {lbl: i for i, lbl in enumerate(self.classes)}
        self.samples = [(p, real_label_map[l]) for p, l in temp_samples]

        print(f"[INFO] No Dedup Baseline: Loaded {len(self.samples)} samples, {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            # Truncate or pad to 4 seconds.
            y, sr = librosa.load(path, sr=16000, duration=4)
            if len(y) < 16000*4:
                y = np.pad(y, (0, 16000*4 - len(y)))
            else:
                y = y[:16000*4]
                
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            log_S = librosa.power_to_db(S, ref=np.max)
            
            min_v, max_v = log_S.min(), log_S.max()
            if max_v - min_v > 0:
                img_arr = (255 * (log_S - min_v) / (max_v - min_v)).astype(np.uint8)
            else:
                img_arr = np.zeros((128, int(16000*4/512)+1), dtype=np.uint8)

            img = Image.fromarray(img_arr).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            return img, label
        except:
            return torch.zeros(3, 224, 224), label

def run_baseline():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"[ERROR] Raw data path not found: {RAW_DATA_PATH}")
        return

    print(f"\n[START] Training Baseline on Raw Data (No Dedup) ...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = AudioSpectrogramDataset(RAW_DATA_PATH, transform=transform)
    
    if len(dataset) == 0:
        print("[ERROR] No valid files found in raw directory.")
        return

    # Split the dataset.
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model.
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train.
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    # Test.
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing Baseline"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print("\n" + "="*50)
    print(f"[No Dedup] Final Test Acc: {acc:.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_baseline()
