import os
import json
import re
import time
import csv
import math
import glob
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights 
from PIL import Image
from tqdm import tqdm

try:
    import webdataset as wds
except Exception:
    wds = None

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)

# ================= 1. Experiment configuration =================
RAW_DATA_ROOT = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_bloated\train"
JSON_RESULT_DIR = r"D:\Deduplication_framework\2026_new_experiment\result" 

METHODS_TO_COMPARE = [
    # ("No Dedup",       None),
    # ("MD5",            "md5_keep_list.json"),
    # ("pHash",          "phash_keep_list.json"),
    # ("SimCLR",         "simclr_keep_list.json"),
    # ("SemDeDup",       "semdedup_keep_list.json"),
    ("Ours (Pipeline)", "our_pipeline_keep_list.json") 
]

# Change 1: lower BATCH_SIZE to avoid OOM; use 256/512 again if VRAM is above 24 GB.
BATCH_SIZE = 256
EPOCHS = 4

NUM_CLASSES = 1000 
LR = 0.0001         
VAL_SPLIT = 0.2
SPLIT_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_FILENAME = "cached_no_dedup_list.json"

# Change 2: choose num_workers dynamically to avoid Windows process deadlocks.
NUM_WORKERS = min(8, os.cpu_count() // 2) if os.name == 'nt' else 8

# Acceleration settings.
torch.backends.cudnn.benchmark = True

# WebDataset settings; currently forced off so JSON keep-list logic works normally.
USE_WDS = False
WDS_ROOT = r"D:\Deduplication_framework_wds_shards"
# ===============================================

# ================= 2. Core helper functions =================
def is_validation_sample(filepath: str) -> bool:
    """
    Change 3: deterministically assign validation samples.
    Compute an MD5 hash from the 'parent-folder/file-name' pair so the same
    image is always assigned to the same split across machines and root paths.
    """
    if not isinstance(filepath, str):
        return False
    unique_name = f"{os.path.basename(os.path.dirname(filepath))}/{os.path.basename(filepath)}"
    h = hashlib.md5((unique_name + str(SPLIT_SEED)).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 100 < int(VAL_SPLIT * 100)

def load_image_paths(root_dir, json_file=None):
    """Load image paths from a JSON keep-list or by scanning a directory."""
    paths = []
    if json_file:
        full_json_path = os.path.join(JSON_RESULT_DIR, json_file)
        print(f"   [Load] Reading Keep-List from: {os.path.basename(full_json_path)}...")
        if os.path.exists(full_json_path):
            with open(full_json_path, 'r') as f:
                paths = json.load(f)
        else:
            print(f"   [ERROR] File not found: {full_json_path}")
    else:
        cache_path = os.path.join(JSON_RESULT_DIR, CACHE_FILENAME)
        if os.path.exists(cache_path):
            print(f"   [Cache] Loading raw paths from {CACHE_FILENAME}...")
            with open(cache_path, 'r') as f: 
                paths = json.load(f)
        elif os.path.exists(root_dir):
            print(f"   [Scan] Scanning raw directory recursively...")
            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        paths.append(os.path.join(root, file))
            with open(cache_path, 'w') as f: 
                json.dump(paths, f)
    return paths

# ===============================================

class DedupJsonDataset(Dataset):
    """Dataset only reads the given path list; split logic is handled outside."""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def _extract_label(self, filepath):
        # The correct label comes from the parent directory name (0-999).
        try:
            parent = os.path.basename(os.path.dirname(filepath))
            if parent.isdigit():
                return int(parent)
        except:
            pass
        # Fallback: support the old filename convention.
        try:
            filename = os.path.basename(filepath)
            match = re.search(r'_(\d+)(?:_aug|\.)', filename)
            if match:
                return int(match.group(1)) % NUM_CLASSES
        except:
            pass
        return 0

    def __len__(self): 
        return len(self.image_paths)

    def __getitem__(self, index):
        try:
            path = self.image_paths[index]
            img = Image.open(path).convert('RGB')
            if self.transform: 
                img = self.transform(img)
            return img, self._extract_label(path)
        except:
            return None

def run_training(method_name, json_file, raw_all_paths):
    print(f"\n" + "="*50)
    print(f"START TRAINING: {method_name} (UNFROZEN / FINE-TUNING)")
    print("="*50)

    # Change 5: use separate data augmentation for train and validation splits.
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),       # Train: random crop.
        transforms.RandomHorizontalFlip(),       # Train: random horizontal flip.
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),                  # Validation: resize to 256 first.
        transforms.CenterCrop(224),              # Validation: center crop to 224.
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 1. Get the image list for the current method.
    current_method_paths = load_image_paths(RAW_DATA_ROOT, json_file)
    if not current_method_paths:
        return None

    dataset_size = len(current_method_paths)

    # 2. Strict data split (core fix).
    # Train set: use the current method's keep-list after removing validation samples.
    train_paths = [p for p in current_method_paths if not is_validation_sample(p)]
    # Validation set: draw from the raw full data so all methods use the same exam set.
    val_paths = [p for p in raw_all_paths if is_validation_sample(p)]

    print(f"   [Data] Total Keep-List: {dataset_size}")
    print(f"   [Data] Train Split:     {len(train_paths)} images")
    print(f"   [Data] Fixed Val Split: {len(val_paths)} images")

    train_dataset = DedupJsonDataset(train_paths, transform=train_transform)
    val_dataset = DedupJsonDataset(val_paths, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True, 
        persistent_workers=(NUM_WORKERS > 0), prefetch_factor=4 if NUM_WORKERS > 0 else None,
        collate_fn=safe_collate
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True, 
        persistent_workers=(NUM_WORKERS > 0), prefetch_factor=4 if NUM_WORKERS > 0 else None,
        collate_fn=safe_collate
    )

    print("   [Model] Loading Pretrained Weights (Unfrozen)...")
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE, memory_format=torch.channels_last)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    use_amp = (DEVICE.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    model.train()
    start_time = time.time()
    best_val = 0.0

    for epoch in range(EPOCHS):
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
        for batch in pbar:
            if batch is None:
                continue
            inputs, labels = batch
            inputs = inputs.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            acc = 100 * correct / total
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{acc:.2f}%"})

        # Validate.
        if val_loader is not None and len(val_dataset) > 0:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating", leave=False):
                    if batch is None:
                        continue
                    v_inputs, v_labels = batch
                    v_inputs = v_inputs.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
                    v_labels = v_labels.to(DEVICE, non_blocking=True)
                    v_outputs = model(v_inputs)
                    _, v_pred = torch.max(v_outputs.data, 1)
                    val_total += v_labels.size(0)
                    val_correct += (v_pred == v_labels).sum().item()
            
            if val_total > 0:
                val_acc = 100 * val_correct / val_total
                best_val = max(best_val, val_acc)
                print(f"   [Val] Epoch {epoch + 1}: {val_acc:.2f}% (Best: {best_val:.2f}%)")
            model.train()

    real_time = time.time() - start_time
    if total == 0:
        print(f"[ERROR] {method_name}: No samples were read.")
        return None
    final_acc = 100 * correct / total
    
    print(f"DONE: {method_name}")
    print(f"   -> Time Cost:  {real_time:.1f}s ({(real_time/3600):.2f} hours)")
    print(f"   -> Train Acc:  {final_acc:.2f}%")
    print(f"   -> Best Val:   {best_val:.2f}%")
    
    return final_acc, real_time, dataset_size, best_val

if __name__ == "__main__":
    results = []
    print(f"Device: {DEVICE} | Batch: {BATCH_SIZE} | Workers: {NUM_WORKERS} | LR: {LR} | Mode: Unfrozen")
    
    # Change 6: load the raw full data pool once before the main loop for a fixed validation set.
    print("\n--- Initializing Fixed Validation Pool ---")
    raw_all_paths = load_image_paths(RAW_DATA_ROOT, json_file=None)
    print(f"Total raw dataset size: {len(raw_all_paths)}")
    print("------------------------------------------\n")

    for name, json_file in METHODS_TO_COMPARE:
        if json_file and not os.path.exists(os.path.join(JSON_RESULT_DIR, json_file)):
             print(f"[SKIP] {name}: File not found ({json_file})")
             continue
        
        res = run_training(name, json_file, raw_all_paths)
        if res:
            acc, time_s, size, best_val = res
            results.append({
                "Method": name,
                "Size": size,
                "Time(s)": f"{time_s:.1f}",
                "Train Acc": f"{acc:.2f}%",
                "Best Val": f"{best_val:.2f}%" if best_val else ""
            })

            # Save incrementally to tolerate interruptions.
            try:
                with open("final_ours_full_run.csv", 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=["Method", "Size", "Time(s)", "Train Acc", "Best Val"])
                    writer.writeheader()
                    writer.writerows(results)
            except Exception as e:
                print(f"[WARN] Failed to write intermediate CSV: {e}")

    print("\n" + "="*60)
    print(f"{'Method':<15} | {'Size':<10} | {'Time(s)':<15} | {'Train Acc':<10} | {'Best Val':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['Method']:<15} | {r['Size']:<10} | {r['Time(s)']:<15} | {r['Train Acc']:<10} | {r['Best Val']:<10}")
    print("="*60)
