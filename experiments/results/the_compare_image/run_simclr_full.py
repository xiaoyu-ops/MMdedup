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
OUTPUT_JSON = r"D:\Deduplication_framework\2026_new_experiment\result\simclr_keep_list.json"
THRESHOLD = 0.93 # Keep consistent with the paper.
BATCH_SIZE = 128
NUM_WORKERS = 4 # On Windows, usually set this to 0-8 depending on CPU capacity.
# ============================================

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
            # Return a black image placeholder; near-zero-norm features are filtered later.
            return torch.zeros((3, 224, 224)), idx

def main():
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # Load model.
    print("加载 ResNet50 模型...")
    # Support both older weights arguments and newer weights enums.
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
    except:
        model = models.resnet50(pretrained=True)
        
    # Remove the final fully connected layer and keep the feature extractor.
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. Scan folders and group files.
    class_groups = {}
    print("正在扫描文件结构...")
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Directory NOT FOUND: {IMAGE_DIR}")
        return

    for r, d, f in os.walk(IMAGE_DIR):
        imgs = [os.path.join(r, x) for x in f if x.lower().endswith(('.jpg', '.png'))]
        if imgs: 
            class_groups[r] = imgs

    print(f"找到 {len(class_groups)} 个文件夹。")

    # 2. Prepare the image list that needs feature computation.
    # Folders with a single image are kept directly.
    final_keep = []
    files_to_infer = []      # List[path]
    files_to_infer_map = {}  # folder -> List[path indices in files_to_infer]
    
    sorted_folders = sorted(list(class_groups.keys()))
    
    current_idx = 0
    for folder in sorted_folders:
        files = class_groups[folder]
        if len(files) < 2:
            final_keep.extend(files)
        else:
            # Record the files_to_infer index range for this folder.
            start = current_idx
            files_to_infer.extend(files)
            end = len(files_to_infer)
            files_to_infer_map[folder] = list(range(start, end))
            current_idx = end
            
    total_infer = len(files_to_infer)
    print(f"需要计算特征的图片数量: {total_infer} (直接保留单张文件夹图片: {len(final_keep)})")

    if total_infer == 0:
        print("没有需要去重的文件夹。")
        with open(OUTPUT_JSON, "w") as f:
            json.dump(final_keep, f)
        return

    # 3. Batch inference.
    # DataLoader provides standard multiprocessing data loading for deep learning.
    print(f"开始批量特征提取 (Batch Size={BATCH_SIZE}, Workers={NUM_WORKERS})...")
    dataset = ImageListDataset(files_to_infer, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    # Store all features in a dense array. This is large but manageable for the target dataset.
    # If memory is insufficient, replace this with mmap or chunked storage.
    
    all_features = np.zeros((total_infer, 2048), dtype=np.float32)
    valid_mask = np.zeros(total_infer, dtype=bool) # True when loading succeeded and image is not black.

    with torch.no_grad():
        for imgs, idxs in tqdm(dataloader, desc="Inference"):
            imgs = imgs.to(device)
            feats = model(imgs).squeeze() # [B, 2048]
            
            # Normalize.
            if len(feats.shape) == 1:
                feats = feats.unsqueeze(0)
            
            norms = torch.norm(feats, p=2, dim=1, keepdim=True)
            feats = feats / (norms + 1e-8)
            feats_np = feats.cpu().numpy()
            
            idxs_np = idxs.numpy()
            norms_np = norms.cpu().numpy()
            
            # Store into the dense feature array.
            all_features[idxs_np] = feats_np
            
            # Simple validity check: tiny norm indicates a black image or failed load.
            # Normal ResNet features should have a larger norm before normalization.
            valid_batch = (norms_np.squeeze() > 0.001)
            valid_mask[idxs_np] = valid_batch

    # 4. Deduplicate by folder.
    print("开始按文件夹计算相似度矩阵并去重...")
    
    # Iterate over folders with recorded index mappings.
    for folder, indices in tqdm(files_to_infer_map.items(), desc="Deduplicating Groups"):
        # Get the features and paths for this folder.
        # indices are offsets in files_to_infer.
        folder_indices = [i for i in indices if valid_mask[i]]
        
        if not folder_indices:
            continue
            
        folder_feats = all_features[folder_indices] # (N, 2048)
        folder_paths = [files_to_infer[i] for i in folder_indices]
        
        N = len(folder_paths)
        if N < 2:
            final_keep.extend(folder_paths)
            continue
            
        # Compute the similarity matrix.
        # (N, 2048) @ (2048, N) -> (N, N)
        sim_mat = np.dot(folder_feats, folder_feats.T)
        np.fill_diagonal(sim_mat, 0)
        
        to_remove_local = set()
        
        for i in range(N):
            if i in to_remove_local: continue
            # Find high-similarity items.
            dups = np.where(sim_mat[i] > THRESHOLD)[0]
            for j in dups:
                if j > i: # Remove later items only.
                    to_remove_local.add(j)
        
        for i in range(N):
            if i not in to_remove_local:
                final_keep.append(folder_paths[i])

    print(f"总计保留文件数: {len(final_keep)}")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(final_keep, f)
    print(f"完成！已保存到 {OUTPUT_JSON}")

if __name__ == "__main__":
    # Windows requires freeze_support when DataLoader num_workers > 0.
    multiprocessing.freeze_support()
    main()
