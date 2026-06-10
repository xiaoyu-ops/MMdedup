import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np
import json
import multiprocessing

# ================= Configuration =================
IMAGE_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_bloated"
OUTPUT_JSON = r"D:\Deduplication_framework\2026_new_experiment\result\semdedup_keep_list.json"
# SemDeDup parameters.
EPSILON = 0.07  
THRESHOLD = 1 - EPSILON  # 0.93
BATCH_SIZE = 512       # Increase batch size to use RTX 3090 VRAM.
NUM_WORKERS = 8        # Increase CPU workers to speed up image loading and decoding.
# ===========================================

class ImageListDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, idx
        except Exception:
            return torch.zeros((3, 224, 224)), idx

def main():
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SemDeDup] 使用设备: {device}")

    # 1. Scan file structure.
    print("扫描文件结构...")
    class_groups = {}
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Directory NOT FOUND: {IMAGE_DIR}")
        return

    for r, d, f in os.walk(IMAGE_DIR):
        imgs = [os.path.join(r, x) for x in f if x.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if imgs: 
            class_groups[r] = imgs
            
    # 2. Prepare inference list.
    final_keep = []
    files_to_infer = []
    files_to_infer_map = {} # folder -> indices range
    
    sorted_folders = sorted(list(class_groups.keys()))
    current_idx = 0
    
    for folder in sorted_folders:
        files = class_groups[folder]
        if len(files) < 2:
            final_keep.extend(files)
        else:
            start = current_idx
            files_to_infer.extend(files)
            end = len(files_to_infer)
            files_to_infer_map[folder] = list(range(start, end))
            current_idx = end
            
    total_infer = len(files_to_infer)
    print(f"需要计算特征的图片数量: {total_infer} (直接保留单张文件夹: {len(final_keep)})")
    
    if total_infer == 0:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(final_keep, f)
        return

    # 3. Load model.
    print(f"每批处理: {BATCH_SIZE} 张, CPU 线程数: {NUM_WORKERS}")
    # Enable cuDNN auto-tuning.
    torch.backends.cudnn.benchmark = True
    print("加载 ResNet50 模型...")
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
    except:
        model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 4. Batch inference.
    print(f"开始批量特征提取 (Batch={BATCH_SIZE}, Workers={NUM_WORKERS})...")
    dataset = ImageListDataset(files_to_infer, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    # Allocate a large array for features.
    all_features = np.zeros((total_infer, 2048), dtype=np.float32)
    valid_mask = np.zeros(total_infer, dtype=bool)

    with torch.no_grad():
        for imgs, idxs in tqdm(dataloader, desc="Inference"):
            imgs = imgs.to(device)
            feats = model(imgs).squeeze() # [B, 2048]
            
            # Normalize features because SemDeDup relies on cosine similarity.
            if len(feats.shape) == 1: feats = feats.unsqueeze(0)
            norms = torch.norm(feats, p=2, dim=1, keepdim=True)
            feats = feats / (norms + 1e-8)
            
            feats_np = feats.cpu().numpy()
            idxs_np = idxs.numpy()
            norms_np = norms.cpu().numpy()
            
            all_features[idxs_np] = feats_np
            valid_batch = (norms_np.squeeze() > 0.001)
            valid_mask[idxs_np] = valid_batch

    # 5. Core SemDeDup processing.
    print("开始执行 SemDeDup 算法...")
    
    for folder, indices in tqdm(files_to_infer_map.items(), desc="SemDeDup Processing"):
        # Get valid features for the current folder.
        folder_indices = [i for i in indices if valid_mask[i]]
        if not folder_indices: continue
            
        feats = all_features[folder_indices] # [N, 2048]
        current_valid_files = [files_to_infer[i] for i in folder_indices]
        N = len(current_valid_files)
        
        if N < 2:
            final_keep.extend(current_valid_files)
            continue
            
        # === SemDeDup Logic ===
        # B. Compute the cluster centroid.
        centroid = np.mean(feats, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        
        # C. Compute similarity and sort.
        sim_to_center = np.dot(feats, centroid)
        sort_indices = np.argsort(sim_to_center)[::-1] # Descending order.
        
        # D. Dynamic deduplication.
        kept_local_indices = [] # Store local indices from 0 to N-1.
        kept_feats = []
        
        for idx in sort_indices:
            current_feat = feats[idx]
            
            if not kept_local_indices:
                kept_local_indices.append(idx)
                kept_feats.append(current_feat)
            else:
                # Compare against already retained images.
                kept_feats_arr = np.array(kept_feats)
                # [K, D] @ [D] -> [K]
                sims = np.dot(kept_feats_arr, current_feat)
                max_sim = np.max(sims)
                
                if max_sim < THRESHOLD:
                    kept_local_indices.append(idx)
                    kept_feats.append(current_feat)
                    
        # E. Collect results.
        for idx in kept_local_indices:
            final_keep.append(current_valid_files[idx])

    # 6. Save results.
    print(f"SemDeDup 完成! 总保留: {len(final_keep)}")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(final_keep, f)
    print(f"结果已保存至: {OUTPUT_JSON}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
