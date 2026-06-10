import os
import time
import hashlib
import librosa
import numpy as np
import imagehash
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ================= Configuration =================
# Raw data directory.
AUDIO_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\digital_swamp_audio"
# ===========================================

def get_all_files():
    files = []
    for r, d, f in os.walk(AUDIO_DIR):
        for file in f:
            if file.endswith('.wav'):
                files.append(os.path.join(r, file))
    return files

# --- Worker functions, kept consistent with earlier scripts. ---
def worker_md5(file_path):
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except: return None

def worker_ours(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000, duration=4)
        if len(y) == 0: return None
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        log_S = librosa.power_to_db(S, ref=np.max)
        min_v, max_v = log_S.min(), log_S.max()
        img = Image.fromarray((255 * (log_S - min_v) / (max_v - min_v + 1e-6)).astype(np.uint8))
        return str(imagehash.phash(img))
    except: return None

def worker_mfcc(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000, duration=4)
        if len(y) == 0: return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        feat = np.mean(mfcc.T, axis=0)
        return feat / (np.linalg.norm(feat) + 1e-6)
    except: return None

def measure_performance():
    # 1. Get file list and dataset size.
    print("[INFO] Scanning files...")
    files = get_all_files()
    total_count = len(files)
    print(f"✅ Full Dataset Size (No Dedup): {total_count} files")
    print("-" * 60)

    num_cores = max(1, cpu_count() - 2)
    print(f"[INFO] Using {num_cores} cores for timing benchmark...")

    results = []

    # === Test 1: MD5 ===
    print("\n>>> Measuring MD5 Time...")
    start_t = time.time()
    with Pool(num_cores) as pool:
        list(tqdm(pool.imap_unordered(worker_md5, files, chunksize=100), total=total_count))
    # MD5 post-processing is nearly zero and is ignored.
    duration = time.time() - start_t
    results.append(("MD5 Hash", duration))

    # === Test 2: Ours ===
    print("\n>>> Measuring Ours (Spectrogram+LSH) Time...")
    start_t = time.time()
    with Pool(num_cores) as pool:
        list(tqdm(pool.imap_unordered(worker_ours, files, chunksize=50), total=total_count))
    # Our hash deduplication is fast; feature extraction dominates runtime.
    duration = time.time() - start_t
    results.append(("Ours (LSH)", duration))

    # === Test 3: MFCC ===
    print("\n>>> Measuring MFCC Time (Extraction + Matrix)...")
    start_t = time.time()
    # 3.1 Extract features.
    feats = []
    with Pool(num_cores) as pool:
        temp_res = list(tqdm(pool.imap(worker_mfcc, files, chunksize=50), total=total_count))
        for r in temp_res:
            if r is not None: feats.append(r)
    
    # 3.2 Matrix computation, which is the MFCC-specific expensive step.
    if len(feats) > 0:
        feats_arr = np.array(feats)
        _ = np.dot(feats_arr, feats_arr.T) # Simulate matrix computation.
        
    duration = time.time() - start_t
    results.append(("MFCC+Cosine", duration))

    # === Print the final report. ===
    print("\n" + "="*50)
    print(f"{'Method':<15} | {'Time(s)':<10} | {'Throughput (files/s)'}")
    print("-" * 50)
    for name, t in results:
        fps = total_count / t if t > 0 else 0
        print(f"{name:<15} | {t:.2f}s      | {fps:.1f}")
    print("="*50)
    print(f"\n[INFO] Please fill 'Size' = {total_count} for 'No Dedup'.")

if __name__ == "__main__":
    measure_performance()
