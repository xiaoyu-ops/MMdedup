import os
import json
import librosa
import numpy as np
import imagehash
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ================= Configuration =================
AUDIO_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\digital_swamp_audio"
RESULT_DIR = r"D:\Deduplication_framework\2026_new_experiment\result"
# ===========================================

def get_files():
    files = []
    print("扫描音频文件...")
    for r, d, f in os.walk(AUDIO_DIR):
        for file in f:
            if file.endswith('.wav'):
                files.append(os.path.join(r, file))
    return files

# --- Worker 1: Ours (Spectrogram + Hash), single-file logic. ---
def worker_ours(file_path):
    try:
        # Read only 4 seconds for fast mode.
        y, sr = librosa.load(file_path, sr=16000, duration=4)
        if len(y) == 0: return file_path # Keep bad files.
        
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        log_S = librosa.power_to_db(S, ref=np.max)
        min_v, max_v = log_S.min(), log_S.max()
        
        if max_v - min_v > 0:
            img = Image.fromarray((255 * (log_S - min_v) / (max_v - min_v)).astype(np.uint8))
            h = str(imagehash.phash(img))
            return (file_path, h) # Return (path, hash value).
        else:
            return file_path # Keep exceptional cases.
    except:
        return file_path # Keep errored files.

# --- Worker 2: MFCC, single-file logic. ---
def worker_mfcc(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000, duration=4)
        if len(y) == 0: return None
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        feat = np.mean(mfcc.T, axis=0)
        feat = feat / (np.linalg.norm(feat) + 1e-6)
        return (file_path, feat) # Return (path, feature vector).
    except:
        return None

# ================= Main program =================
if __name__ == "__main__":
    # On Windows, multiprocessing must be guarded by if __name__ == "__main__".
    os.makedirs(RESULT_DIR, exist_ok=True)
    all_files = get_files()
    
    if not all_files:
        print("错误：没找到文件！")
        exit()

    # Use available CPU cores while leaving some for the system.
    num_cores = max(1, cpu_count() - 16)
    print(f"火力全开！正在使用 {num_cores} 个 CPU 核心并行处理...")

    # ----------------------------------------------------
    # Task 1: Generate the Ours keep list.
    # ----------------------------------------------------
    ours_json_path = os.path.join(RESULT_DIR, "audio_ours_keep_list.json")
    if not os.path.exists(ours_json_path):
        print(f"\n>>> [1/2] 正在生成 Ours 列表 ({len(all_files)} files)...")
        
        keep_ours = []
        seen_hashes = set()
        
        # Start the worker pool.
        with Pool(processes=num_cores) as pool:
            # imap_unordered returns results as they arrive so tqdm can update live.
            results = list(tqdm(pool.imap(worker_ours, all_files), total=len(all_files), desc="Ours Multiprocessing"))
            
        # Aggregate results. This step is fast enough in one process.
        for res in results:
            if isinstance(res, tuple):
                f_path, h = res
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    keep_ours.append(f_path)
            else:
                # Single path means an exceptional file; keep it directly.
                keep_ours.append(res)
                
        with open(ours_json_path, 'w') as f: json.dump(keep_ours, f)
        print(f"Ours 完成！保留: {len(keep_ours)}")
    else:
        print("Ours 列表已存在，跳过。")

    # ----------------------------------------------------
    # Task 2: Generate the MFCC keep list.
    # ----------------------------------------------------
    mfcc_json_path = os.path.join(RESULT_DIR, "audio_mfcc_keep_list.json")
    if not os.path.exists(mfcc_json_path):
        print(f"\n>>> [2/2] 正在生成 MFCC 列表...")
        
        feats = []
        valid_files = []
        
        with Pool(processes=num_cores) as pool:
            # MFCC only extracts features here; hash decisions are not needed.
            results = list(tqdm(pool.imap(worker_mfcc, all_files), total=len(all_files), desc="MFCC Extracting"))
            
        # Organize features.
        for res in results:
            if res is not None:
                valid_files.append(res[0])
                feats.append(res[1])
                
        # Compute similarity. NumPy matrix operations are already optimized.
        if feats:
            print("   正在计算矩阵 (Matrix Calculation)...")
            feats_arr = np.array(feats)
            # Simple full matrix computation.
            sim_mat = np.dot(feats_arr, feats_arr.T)
            np.fill_diagonal(sim_mat, 0)
            
            to_remove = set()
            n = len(valid_files)
            for i in tqdm(range(n), desc="Filtering"):
                if i in to_remove: continue
                dups = np.where(sim_mat[i] > 0.95)[0]
                for j in dups:
                    if j > i: to_remove.add(j)
                    
            keep_mfcc = [valid_files[i] for i in range(n) if i not in to_remove]
            
            # Add bad files back to the keep list.
            processed_set = set(valid_files)
            for f in all_files:
                if f not in processed_set: keep_mfcc.append(f)
                
            with open(mfcc_json_path, 'w') as f: json.dump(keep_mfcc, f)
            print(f"MFCC 完成！保留: {len(keep_mfcc)}")
    else:
        print("MFCC 列表已存在，跳过。")
