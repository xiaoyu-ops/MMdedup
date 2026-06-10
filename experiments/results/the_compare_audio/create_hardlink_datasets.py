import os
import json
import shutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathlib import Path

# ================= Configuration =================
# 1. Root directory for the original data, used to compute relative paths.
# Hard links require the source and target directories to be on the same disk partition.
BASE_DIR = r"D:\Deduplication_framework\2026_new_experiment"

# 2. Output directory. Keep this as the single canonical definition.
TARGET_ROOT = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_deduped_datasets"

# 3. Task list, including Ours, MFCC, and MD5.
TASKS = [
    # === Task 1: Ours (Spectrogram + pHash) ===
    {
        "json": r"D:\Deduplication_framework\2026_new_experiment\result\audio_ours_keep_list.json",
        "src_root": r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\digital_swamp_audio",
        "target_name": "audio_ours_deduped"
    },
    
    # === Task 2: MFCC (baseline) ===
    {
        "json": r"D:\Deduplication_framework\2026_new_experiment\result\audio_mfcc_keep_list.json",
        "src_root": r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\digital_swamp_audio",
        "target_name": "audio_mfcc_deduped"
    },

    # === Task 3: MD5 (baseline) ===
    {
        "json": r"D:\Deduplication_framework\2026_new_experiment\result\audio_md5_keep_list.json",
        "src_root": r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\digital_swamp_audio",
        "target_name": "audio_md5_deduped"
    }
]
# ===========================================

# Global variables used to pass arguments into worker processes.
GLOBAL_SRC_ROOT = None
GLOBAL_TARGET_DIR = None

def init_worker(src_root, target_dir):
    """Initialize a worker process."""
    global GLOBAL_SRC_ROOT, GLOBAL_TARGET_DIR
    GLOBAL_SRC_ROOT = Path(src_root)
    GLOBAL_TARGET_DIR = Path(target_dir)

def worker_link(file_path):
    """Create a hard link in a worker process."""
    try:
        # 1. Compute the relative path.
        try:
            p = Path(file_path)
            rel_path = p.relative_to(GLOBAL_SRC_ROOT)
        except ValueError:
            p = Path(file_path)
            rel_path = p.name

        # 2. Target path.
        dest_path = GLOBAL_TARGET_DIR / rel_path
        
        # 3. Skip if the target already exists.
        if dest_path.exists():
            return True

        # 4. Create parent directories.
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # 5. Create the hard link, or copy as a fallback.
        try:
            os.link(file_path, dest_path)
        except OSError:
            shutil.copy2(file_path, dest_path)
        
        return True
    except Exception:
        return False

def run_multicore():
    # Enable multicore processing.
    num_cores = max(1, cpu_count() - 2)
    print(f"[Info] 启用 {num_cores} 核心并行构建数据集...")

    for task in TASKS:
        json_path = task['json']
        target_name = task['target_name']
        src_root = task['src_root']
        
        if not os.path.exists(json_path):
            print(f"[Warning] 跳过: 找不到 JSON {json_path}")
            continue

        target_dir = os.path.join(TARGET_ROOT, target_name)
        
        # Clean the old directory.
        if os.path.exists(target_dir):
            print(f"[Info] 正在重建目录: {target_name} ...")
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)

        # Read the keep list.
        with open(json_path, 'r') as f:
            keep_list = json.load(f)

        print(f"[Info] 开始构建: {target_name} ({len(keep_list)} 个文件)...")
        
        # Run linking in parallel.
        with Pool(processes=num_cores, initializer=init_worker, initargs=(src_root, target_dir)) as pool:
            results = list(tqdm(pool.imap_unordered(worker_link, keep_list, chunksize=100), 
                              total=len(keep_list), 
                              desc=f"Linking"))
            
        print(f"[Success] 完成! 成功: {sum(results)}/{len(keep_list)}\n")

if __name__ == "__main__":
    run_multicore()
