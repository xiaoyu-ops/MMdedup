import os
import hashlib
import json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ================= Configuration =================
IMAGE_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_bloated"
OUTPUT_JSON = r"D:\Deduplication_framework\2026_new_experiment\result\md5_keep_list.json"
# ============================================

def get_md5_worker(file_path):
    """Worker function that computes MD5 for one file."""
    try:
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash, file_path
    except Exception:
        # Return None on read errors such as corrupted files.
        return None

def main():
    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Directory NOT FOUND: {IMAGE_DIR}")
        return

    files = []
    print(f"正在扫描文件结构: {IMAGE_DIR} ...")
    for r, d, f in os.walk(IMAGE_DIR):
        for file in f:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                files.append(os.path.join(r, file))

    total_files = len(files)
    print(f"找到 {total_files} 个图片文件。")

    seen = set()
    keep = []
    
    # Set process count from CPU cores, leaving two cores for the system.
    num_processes = max(1, cpu_count() - 2)
    print(f"启动 {num_processes} 个进程进行 MD5 计算...")

    # Use Pool for multiprocessing.
    # A larger chunksize reduces inter-process communication overhead.
    with Pool(processes=num_processes) as pool:
        # imap_unordered returns out of order, which is fine and enables live progress.
        results = list(tqdm(pool.imap_unordered(get_md5_worker, files, chunksize=100), total=total_files, desc="Calculating MD5"))

    print("正在进行去重筛选...")
    for res in results:
        if res is None:
            continue
        h, f_path = res
        if h not in seen:
            seen.add(h)
            keep.append(f_path)

    print(f"原始数量: {total_files} -> 保留数量: {len(keep)}")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(keep, f)
    print(f"完成！已保存到 {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
