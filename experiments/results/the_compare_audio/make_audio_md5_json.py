import os
import json
import hashlib
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ================= Configuration =================
# Audio source data path.
AUDIO_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\digital_swamp_audio"

# Result output path.
RESULT_JSON = r"D:\Deduplication_framework\2026_new_experiment\result\audio_md5_keep_list.json"
# ===========================================

def get_files():
    files = []
    print("正在扫描音频文件...")
    for r, d, f in os.walk(AUDIO_DIR):
        for file in f:
            if file.endswith('.wav'):
                files.append(os.path.join(r, file))
    return files

def worker_md5(file_path):
    """Compute the MD5 hash for one file in a worker process."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            # Read in chunks to avoid excessive memory use.
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return (file_path, hash_md5.hexdigest())
    except Exception:
        # Return None on read failure; the main process will handle it.
        return None

if __name__ == "__main__":
    # 1. Scan files.
    all_files = get_files()
    if not all_files:
        print("错误：未找到音频文件！")
        exit()

    # 2. Prepare multiprocessing.
    num_cores = max(1, cpu_count() - 16)
    print(f"启用 {num_cores} 核心并行计算 MD5...")

    # 3. Compute hashes in parallel.
    results = []
    with Pool(processes=num_cores) as pool:
        # Use imap_unordered so the progress bar updates as results arrive.
        for res in tqdm(pool.imap_unordered(worker_md5, all_files, chunksize=50), 
                       total=len(all_files), 
                       desc="Calculating MD5"):
            if res is not None:
                results.append(res)

    # 4. Deduplication logic, aggregated in a single process.
    print("正在进行去重筛选...")
    seen_hashes = set()
    keep_list = []
    
    # Results contain (path, md5). Keep the first path seen for each hash.
    for file_path, md5_val in results:
        if md5_val not in seen_hashes:
            seen_hashes.add(md5_val)
            keep_list.append(file_path)

    # 5. Save JSON.
    os.makedirs(os.path.dirname(RESULT_JSON), exist_ok=True)
    with open(RESULT_JSON, 'w') as f:
        json.dump(keep_list, f)
        
    print(f"MD5 列表生成完毕！")
    print(f"   总文件: {len(all_files)}")
    print(f"   保留文件: {len(keep_list)}")
    print(f"   保存路径: {RESULT_JSON}")
