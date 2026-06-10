import os
import sys
import random
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import numpy as np
import csv
from collections import Counter, defaultdict
from sklearn.cluster import KMeans

# ================= Configuration =================
# Path to the imagenet_bloated folder.
IMAGE_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_bloated"

SAMPLE_SIZE = 10000 
BATCH_SIZE = 256
# SemDeDup recommends epsilon=0.07, corresponding to threshold 0.93.
SIMILARITY_THRESHOLD = 0.93 
RESULT_FILE = r"D:\Deduplication_framework\2026_new_experiment\result\image_benchmark_results.csv"
# ===========================================

def get_all_images(root_dir, limit=None):
    image_files = []
    # topdown=True lets us shuffle dirs for random sampling.
    count = 0
    for root, dirs, files in os.walk(root_dir, topdown=True):
        random.shuffle(dirs)  # Shuffle subfolder traversal order.
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
                count += 1
                if count % 1000 == 0:
                    print(f"      [Scan] Found {count} images...", end='\r')
                if limit and len(image_files) >= limit:
                    print(f"      [Scan] Reached limit of {limit} images.")
                    return image_files
    return image_files

def parse_id(filename):
    name = os.path.splitext(filename)[0]
    if "_aug" in name:
        return name.split("_aug")[0]
    return name

def log_result(method, throughput, precision, recall, gpu_mem):
    file_exists = os.path.isfile(RESULT_FILE)
    with open(RESULT_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Method", "Throughput (imgs/s)", "Precision", "Recall", "GPU Mem (GB)"])
        writer.writerow([method, f"{throughput:.1f}", f"{precision*100:.2f}%", f"{recall*100:.2f}%", f"{gpu_mem:.2f}"])
    print(f"[成功] {method} 结果已写入文件")

class BenchDataset(Dataset):
    def __init__(self, files):
        self.files = files
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert('RGB')
            return self.transform(img), self.files[idx]
        except:
            return torch.zeros(3,224,224), self.files[idx]

def run_semdedup_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Benchmark] SemDeDup (ResNet50) 评测...")
    
    # 1. Prepare data.
    print(f"   正在快速扫描前 {SAMPLE_SIZE} 个文件...")
    all_files = get_all_images(IMAGE_DIR, limit=SAMPLE_SIZE)
    if not all_files: return
    # Keep the first SAMPLE_SIZE samples; this is redundant but keeps the logic explicit.
    test_files = all_files[:SAMPLE_SIZE]
    
    # 2. Load model (ResNet50 feature extractor).
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(model.children())[:-1]) 
    model.to(device).eval()

    dataset = BenchDataset(test_files)
    # Increase num_workers for faster loading. On Windows, 4 or 8 is recommended.
    # Batch size and worker count are tuned for a 24GB GPU.
    # Avoid 16 Windows workers because startup can look stalled.
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    feats_list = []
    paths_list = []
    
    # Start timing, including feature extraction and SemDeDup computation.
    start_time = time.time()
    
    # A. Feature extraction.
    print("   正在提取特征 (Initializing DataLoader for batch processing)...")
    with torch.no_grad():
        for imgs, paths in tqdm(loader, desc="Extracting", file=sys.stdout):
            imgs = imgs.to(device)
            feats = model(imgs).squeeze()
            # Key requirement: SemDeDup needs L2-normalized features.
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            feats_list.append(feats.cpu().numpy())
            paths_list.extend(paths)
            
    embeddings = np.concatenate(feats_list)
    
    # B. Run the core SemDeDup algorithm.
    print("   正在执行 SemDeDup K-Means 聚类筛选...")
    
    # Use K-Means clustering.
    # Heuristic: assume roughly 50 images per cluster; tune for the data distribution.
    # For 10,000 images this gives about 200 clusters, closer to unsupervised use than folder labels.
    n_clusters = max(1, len(embeddings) // 50)
    print(f"      Running KMeans with k={n_clusters} ...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Group embeddings and paths by cluster label.
    cluster_groups = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_groups[label].append(i) # Store index.
        
    removed_indices = set()
    
    for label, indices in tqdm(cluster_groups.items(), desc="Filtering Clusters", file=sys.stdout):
        if len(indices) < 2: continue
        
        # Get features for the current cluster: [K, 2048].
        cluster_feats = embeddings[indices]
        
        # 1. Compute centroid.
        centroid = np.mean(cluster_feats, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        
        # 2. Compute similarity to the centroid and sort descending.
        sim_to_center = np.dot(cluster_feats, centroid)
        # argsort is ascending; [::-1] reverses to descending.
        sorted_local_indices = np.argsort(sim_to_center)[::-1]
        
        # 3. Dynamic deduplication filtering.
        kept_local_feats = []
        
        for local_idx in sorted_local_indices:
            global_idx = indices[local_idx]
            current_feat = cluster_feats[local_idx]
            
            if not kept_local_feats:
                # Always keep the item closest to the centroid.
                kept_local_feats.append(current_feat)
            else:
                # Compare with already retained items.
                # [M, 2048] @ [2048, 1] -> [M]
                sims = np.dot(np.array(kept_local_feats), current_feat)
                max_sim = np.max(sims)
                
                if max_sim < SIMILARITY_THRESHOLD:
                    # Not similar: keep it.
                    kept_local_feats.append(current_feat)
                else:
                    # Too similar: remove it.
                    removed_indices.add(global_idx)

    # Stop timing after the algorithm completes.
    total_time = time.time() - start_time
    throughput = len(test_files) / total_time
    
    # C. Compute precision and recall.
    print("   正在计算准确率指标...")
    
    # 1. Build ground truth.
    # Count how often each id appears in the sample.
    id_list = [parse_id(os.path.basename(p)) for p in paths_list]
    id_counts = Counter(id_list)
    
    # Total number of files that should be removed.
    # If an id has N copies, N-1 can be removed.
    total_gt_duplicates = sum([count - 1 for count in id_counts.values()])
    
    # 2. Count whether removals are correct.
    removed_counts = Counter() # Count removals per id.
    for idx in removed_indices:
        fid = id_list[idx]
        removed_counts[fid] += 1
        
    tp = 0
    fp = 0 # Includes unique-image removals and over-removal.
    
    # Iterate over all observed ids.
    for fid, total_count in id_counts.items():
        # Actual removed count for this id.
        rem_count = removed_counts[fid]
        
        # Maximum removable count for this id, while keeping one copy.
        # If total_count=1, max_removable=0.
        max_removable = max(0, total_count - 1)
        
        if rem_count <= max_removable:
            # Removed count is within the duplicate budget, so all are TP.
            tp += rem_count
        else:
            # Too many removals: only max_removable can be counted as TP.
            tp += max_removable
            # Extra removals are FP.
            fp += (rem_count - max_removable)
            
    # Compute precision.
    precision = tp / len(removed_indices) if len(removed_indices) > 0 else 0
    
    # Compute recall.
    recall = tp / total_gt_duplicates if total_gt_duplicates > 0 else 0
    
    # D. Get GPU memory usage.
    gpu_mem = 0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / (1024**3)

    # E. Write results.
    log_result("SemDeDup", throughput, precision, recall, gpu_mem)

if __name__ == "__main__":
    run_semdedup_benchmark()
