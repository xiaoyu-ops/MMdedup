
import os
import sys
import time
import csv
import json
from pathlib import Path
from tqdm import tqdm

# --- Add the project root to call the real Pipeline API. ---
PROJECT_ROOT = r"D:\Deduplication_framework"
sys.path.insert(0, PROJECT_ROOT)

# Dynamically add image module search paths to avoid relative import errors.
sys.path.insert(0, os.path.join(PROJECT_ROOT, "image"))

import torch
import open_clip

# Patch the Pipeline API manually to avoid import failures.
# Relative imports or environment path issues can make pipeline_api try-import fail.
# Inject dependencies here explicitly.
import image.method.pipeline_api as pipeline_api
pipeline_api.torch = torch
pipeline_api.open_clip = open_clip

from image.method.pipeline_api import (
    ImagePipelineConfig,
    EmbeddingConfig,
    DedupConfig,
    _compute_embeddings_open_clip,
    _run_deduplication
)

# ================= Configuration =================
# Point to the same dataset as benchmark_simclr_updated.
IMAGE_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_bloated"

SAMPLE_SIZE = 10000 
BATCH_SIZE = 64
EPS = 0.07 # Matches the pipeline default config.
RESULT_FILE = r"D:\Deduplication_framework\2026_new_experiment\result\image_benchmark_results.csv"
MODEL_NAME = "hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
# ===========================================


from image.method.pipeline_api import (
    ImagePipelineConfig,
    EmbeddingConfig,
    DedupConfig,
    _compute_embeddings_open_clip,
    _run_deduplication
)

# ================= Configuration =================
# Point to the same dataset as benchmark_simclr_updated.
IMAGE_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_bloated"

SAMPLE_SIZE = 10000 
BATCH_SIZE = 64
EPS = 0.07 # Matches the pipeline default config.
RESULT_FILE = r"D:\Deduplication_framework\2026_new_experiment\result\image_benchmark_results.csv"
MODEL_NAME = "hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K" # Must match configs/image_config.yaml.
# ===========================================

def get_all_images(root_dir):
    image_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    return image_files

# Ground-truth logic: filenames containing _aug are treated as the same group.
def parse_id(filename):
    name = os.path.splitext(filename)[0]
    if "_aug" in name:
        return name.split("_aug")[0]
    return name

def log_result(method, throughput, precision, recall, gpu_mem):
    file_exists = os.path.isfile(RESULT_FILE)
    try:
        with open(RESULT_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Method", "Throughput (imgs/s)", "Precision", "Recall", "GPU Mem (GB)"])
            writer.writerow([method, f"{throughput:.1f}", f"{precision*100:.2f}%", f"{recall*100:.2f}%", f"{gpu_mem:.2f}"])
        print(f"[Success] '{method}' results written to {RESULT_FILE}")
    except Exception as e:
        print(f"[Error] Failed to write CSV: {e}")

def run_real_pipeline_benchmark():
    print(f"[Benchmark] Ours (Real Pipeline Logic: SemDeDup) on imagenet_bloated subset...")
    
    # 1. Prepare data.
    all_files = get_all_images(IMAGE_DIR)
    if not all_files:
        print("No images found.")
        return

    # Simulate pipeline input using Path objects.
    test_paths = [Path(f) for f in all_files[:SAMPLE_SIZE]]
    print(f"Total files to test: {len(test_paths)}")
    
    # 2. Configure the pipeline.
    # Construct objects equivalent to the YAML config.
    emb_config = EmbeddingConfig(
        backend="open_clip",
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        device="auto"
    )
    
    # Use SemDeDup here rather than pairwise.
    # SemDeDup requires clustering; small datasets may fall back or produce few clusters.
    dedup_config = DedupConfig(
        method="semdedup", 
        eps=EPS,
        legacy_cluster_dir=None, # Force cluster recomputation.
        legacy_keep_indices_file=None
    )
    
    start_time = time.time()
    gpu_mem = 0
    
    try:
        # 3. Call the real API to extract features.
        # _compute_embeddings_open_clip returns (embeddings, valid_paths, failed_paths, backend_name).
        embeddings, valid_paths, failed_paths, _ = _compute_embeddings_open_clip(test_paths, emb_config)
        
        extract_time = time.time()
        print(f"Embedding extraction done. Shape: {embeddings.shape}")
        
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / (1024**3)

        # 4. Call the real API for SemDeDup deduplication.
        # This triggers K-Means clustering and intra-cluster deduplication.
        dedup_result = _run_deduplication(
            valid_paths,
            embeddings,
            dedup_config,
            indices=None # Do not pass indices; cluster dynamically from embeddings.
        )
        
    except Exception as e:
        print(f"Pipeline Execution Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    total_time = time.time() - start_time
    throughput = len(test_paths) / total_time if total_time > 0 else 0
    print(f"Pipeline finished. Total Throughput (Extract+SemDeDup): {throughput:.2f} imgs/s")
    
    # 5. Evaluate results against ground truth.
    
    kept_set = set([str(p) for p in dedup_result['keepers']])
    # Removed file set.
    
    # For simplicity, removed files are valid_paths minus kept_set.
    valid_paths_str = [str(p) for p in valid_paths]
    removed_files = [p for p in valid_paths_str if p not in kept_set]
    
    print(f"Kept: {len(kept_set)}, Removed: {len(removed_files)}")
    
    # Compute action-based precision and recall.
    print("Evaluating Precision & Recall...")
    
    # Parse IDs
    id_map = {str(p): parse_id(os.path.basename(str(p))) for p in valid_paths}
    all_labels = [id_map[p] for p in valid_paths_str]
    
    # A. Compute total ground-truth pairs.
    from collections import Counter
    cnt = Counter(all_labels)
    # total_gt_pairs = sum([(c*(c-1))//2 for c in cnt.values() if c > 1])
    # print(f"Total Ground Truth Duplicate Pairs: {total_gt_pairs}")
    
    # --- Use action-based evaluation, which is better suited to SemDeDup. ---
    # Precision = correctly removed redundant files / all removed files.
    # Recall = correctly removed redundant files / all ground-truth removable files.
    
    unique_ids_count = len(cnt)
    total_should_remove = len(valid_paths) - unique_ids_count
    
    tp_action = 0 # Correct removals.
    
    # Build the set of keeper ids.
    keeper_ids = set()
    for k in kept_set:
        keeper_ids.add(id_map[k])
        
    for removed in removed_files:
        rid = id_map[removed]
        if rid in keeper_ids:
            # Removed while another copy of the same id remains: correct redundant removal.
            tp_action += 1
        else:
            # Removed the only remaining copy for this id: incorrect removal.
            pass
            
    actual_precision = tp_action / len(removed_files) if len(removed_files) > 0 else 0
    actual_recall = tp_action / total_should_remove if total_should_remove > 0 else 0
    
    print(f"Precision (Reduction): {actual_precision*100:.2f}%")
    print(f"Recall (Reduction): {actual_recall*100:.2f}%")
    print(f"Peak GPU Mem: {gpu_mem:.2f} GB")

    log_result("Ours (Real SemDeDup 10k)", throughput, actual_precision, actual_recall, gpu_mem)

if __name__ == "__main__":
    run_real_pipeline_benchmark()
