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
# Dataset root directory.
DATASET_ROOT = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_deduped_datasets"

# Dataset folder names to evaluate.
TARGET_DIRS = [
    "audio_md5_deduped",
    "audio_ours_deduped", 
    "audio_mfcc_deduped"
]

# Training parameters.
BATCH_SIZE = 32
EPOCHS = 10           # 5 epochs is enough to see the trend
LEARNING_RATE = 0.001
# ===============================================

# Detect the device automatically.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AudioSpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = []
        
        # 1. Try to detect subfolders, as in a standard ImageFolder layout.
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        if len(subdirs) > 0:
            # Subdirectory names are class names.
            self.classes = sorted(subdirs)
            class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            
            for cls_name in subdirs:
                cls_folder = os.path.join(root_dir, cls_name)
                for f in os.listdir(cls_folder):
                    if f.endswith('.wav'):
                        self.samples.append((os.path.join(cls_folder, f), class_to_idx[cls_name]))
        else:
            # 2. Flat layout: parse classes from ESC-50-style filenames.
            # Format: fs-id-src-class.wav, e.g. 1-100032-A-0.wav where the final 0 is the class.
            # Deduped files may look like copy_0_1-137-A-32.wav.
            files = [f for f in os.listdir(root_dir) if f.endswith('.wav')]
            
            class_ids = set()
            temp_samples = []
            
            for f in files:
                try:
                    # Remove the .wav extension.
                    name_no_ext = os.path.splitext(f)[0]
                    # Split by '-' and use the last part as the class id.
                    parts = name_no_ext.split('-')
                    label_str = parts[-1] 
                    
                    # Ensure the class id is numeric.
                    if label_str.isdigit():
                        label = int(label_str)
                        class_ids.add(label)
                        temp_samples.append((os.path.join(root_dir, f), label))
                except:
                    pass
            
            self.classes = sorted(list(class_ids))
            # Map true label ids to 0..N indices.
            real_label_map = {lbl: i for i, lbl in enumerate(self.classes)}
            self.samples = [(p, real_label_map[l]) for p, l in temp_samples]

        print(f"[INFO] Loaded: {len(self.samples)} samples, {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            # Load audio and convert it to a spectrogram.
            # Limit duration to 4 seconds.
            y, sr = librosa.load(path, sr=16000, duration=4)
            if len(y) < 16000*4: # Pad
                y = np.pad(y, (0, 16000*4 - len(y)))
            else:
                y = y[:16000*4]
                
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            log_S = librosa.power_to_db(S, ref=np.max)
            
            # Normalize to 0-255 and convert to RGB.
            min_v, max_v = log_S.min(), log_S.max()
            if max_v - min_v > 0:
                img_arr = (255 * (log_S - min_v) / (max_v - min_v)).astype(np.uint8)
            else:
                img_arr = np.zeros((128, int(16000*4/512)+1), dtype=np.uint8)

            img = Image.fromarray(img_arr).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            return img, label
        except Exception as e:
            # Return a black image for bad data to keep training running.
            return torch.zeros(3, 224, 224), label

def train_one_dataset(folder_name):
    full_path = os.path.join(DATASET_ROOT, folder_name)
    if not os.path.exists(full_path):
        print(f"[ERROR] Folder not found: {folder_name}, skipping.")
        return "N/A"

    print(f"\n[START] Training on [{folder_name}] ...")
    
    # Preprocessing.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Build the dataset.
    dataset = AudioSpectrogramDataset(full_path, transform=transform)
    
    # Check dataset size.
    if len(dataset) < 50:
        print("[WARNING] Dataset too small (<50), skipping training.")
        return "0.00%"
    
    # Check class count.
    if len(dataset.classes) < 2:
        print("[WARNING] Less than 2 classes found, cannot perform classification.")
        return "0.00%"

    # Split into 80% train and 20% test.
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Load the model (ResNet18).
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Adapt the final fully connected layer to the actual class count.
    num_ftrs = model.fc.in_features
    actual_num_classes = len(dataset.classes)
    model.fc = nn.Linear(num_ftrs, actual_num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop.
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        # Progress bar.
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
    # Test loop.
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total > 0:
        acc = 100 * correct / total
    else:
        acc = 0.0
        
    print(f"[DONE] [{folder_name}] Final Test Acc: {acc:.2f}%")
    return f"{acc:.2f}%"

if __name__ == "__main__":
    results = {}
    print(f"[Init] Starting Downstream Training (Device: {device})")
    print("="*60)
    
    for folder in TARGET_DIRS:
        acc = train_one_dataset(folder)
        results[folder] = acc
        
    print("\n" + "="*60)
    print("=== FINAL RESULTS ===")
    print("="*60)
    print(f"{'Dataset':<25} | {'Test Acc'}")
    print("-" * 40)
    for name, acc in results.items():
        print(f"{name:<25} | {acc}")
    print("="*60)
