import os
import time
import csv
import json
import torch
from pathlib import Path
from collections import defaultdict

# -------------------------------------------------------------
# 0. Configuration
# -------------------------------------------------------------

# Pipeline output directory. Adjust according to the big_run config.
# summary.json is read from here to locate duplicate outputs.
PIPELINE_SUMMARY_FILE = Path(r"D:\Deduplication_Framework\outputs\image_experiment\summary.json")

# Ground-truth raw image directory.
IMAGE_SOURCE_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_bloated"

# Result CSV file to append to.
RESULT_FILE = r"D:\Deduplication_framework\2026_new_experiment\result\image_benchmark_results.csv"

# -------------------------------------------------------------
# 1. Helper functions
# -------------------------------------------------------------

def parse_id(filename):
    """
    Parse filenames consistently with the baseline.
    Example: train-0_0_aug_noise.jpg -> train-0_0.
    """
    name = os.path.splitext(filename)[0]
    if "_aug" in name:
        return name.split("_aug")[0]
    return name

def log_result_csv(method, throughput, precision, recall, gpu_mem):
    """Append one result row to the CSV file."""
    file_exists = os.path.isfile(RESULT_FILE)
    try:
        with open(RESULT_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Method", "Throughput (imgs/s)", "Precision", "Recall", "GPU Mem (GB)"])
            
            # Format invalid values consistently.
            tp_str = f"{throughput:.1f}" if throughput >= 0 else "N/A"
            mem_str = f"{gpu_mem:.2f}" if gpu_mem >= 0 else "0.00"
            
            writer.writerow([method, tp_str, f"{precision*100:.2f}%", f"{recall*100:.2f}%", mem_str])
        print(f"[成功] 结果已追加到 {RESULT_FILE}")
    except Exception as e:
        print(f"[错误] 写入CSV失败: {e}")

# -------------------------------------------------------------
# 2. Main logic
# -------------------------------------------------------------

def main():
    print(f"[My Pipeline Evaluation] 开始评估...")

    # A. Load duplicate outputs from the pipeline.
    if not PIPELINE_SUMMARY_FILE.exists():
        print(f"[错误] 找不到 summary 文件: {PIPELINE_SUMMARY_FILE}")
        print("请确认 pipeline 是否已运行完毕 (check outputs/big_run)。")
        return
    
    duplicates_map = {} # keeper -> [dup1, dup2...]
    duration_total = 0.0
    processed_count = 0
    
    try:
        with open(PIPELINE_SUMMARY_FILE, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        # 1. Locate the image stage.
        image_stage = None
        if "stages" in summary:
            for stage in summary["stages"]:
                if stage.get("stage_name") == "stage2_image":
                    image_stage = stage
                    break
        
        if not image_stage:
            print("[错误] summary.json 中未找到 'stage2_image' 阶段的信息。")
            return

        # 2. Get duplicate output files from the stage metadata.
        output_paths = image_stage.get("output_paths", {})
        dup_files = output_paths.get("duplicates", [])
        
        if not dup_files:
            print("[警告] stage2_image 报告中没有 duplicates 文件输出。可能没有发现重复？")
        
        print(f"[Info] 找到 {len(dup_files)} 个分片结果文件，开始合并...")

        # 3. Read each shard file.
        for dup_file_path in dup_files:
            if not os.path.exists(dup_file_path):
                print(f"[警告] 找不到结果文件: {dup_file_path}，跳过。")
                continue
                
            with open(dup_file_path, 'r', encoding='utf-8') as df:
                items = json.load(df)
                # Item shape: [{"original": "pathA", "duplicates": [{"path": "pathB", "similarity": 0.99}, ...]}, ...]
                
                for item in items:
                    keeper = item.get("original")
                    if not keeper: continue
                    
                    dups_list = item.get("duplicates", [])
                    if not dups_list: continue

                    if keeper not in duplicates_map:
                        duplicates_map[keeper] = []
                    
                    for dup_info in dups_list:
                        d_path = dup_info.get("path")
                        if d_path:
                            duplicates_map[keeper].append(d_path)

        # 4. Read statistics such as elapsed time.
        runner_stats = image_stage.get("metadata", {}).get("runner_summary", {}).get("stats", {})
        processed_count = runner_stats.get("processed", 0)
        # Prefer elapsed_seconds from metadata because it is usually more accurate.
        duration_total = runner_stats.get("elapsed_seconds", 0)
        if duration_total == 0:
             duration_total = image_stage.get("elapsed_seconds", 0)
        
    except Exception as e:
        print(f"[严重错误] 解析过程发生异常: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Pipeline 报告耗时: {duration_total:.2f}s, 处理文件: {processed_count}")
    print(f"发现重复组 (Clusters): {len(duplicates_map)} (含 Keeper)")

    # B. Build ground truth from the actual files processed by the pipeline.
    print(f"正在读取 Pipeline 输入 Manifest 以建立 Ground Truth...")
    
    all_files_gt = []
    
    # Get input manifests from image_stage output_paths.
    # output_paths['manifests'] contains the input manifest shards processed by stage2_image.
    input_manifests = image_stage.get("output_paths", {}).get("manifests", [])
    
    if not input_manifests:
        # If no manifest list is found, fail instead of guessing the GT scope.
        print("[错误] 无法在 summary 中找到 input manifests 列表，无法确定 GT 范围。")
        return

    for mani_path in input_manifests:
        if not os.path.exists(mani_path):
            print(f"[警告] 找不到 Manifest 文件: {mani_path}")
            continue
            
        with open(mani_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Get the filename.
                    fname = os.path.basename(line)
                    all_files_gt.append(fname)
    
    if not all_files_gt:
        print("[错误] 未能从 Manifest 中读取到任何文件。")
        return

    # Count ground-truth pairs.
    id_counts = defaultdict(int)
    for name in all_files_gt:
        fid = parse_id(name)
        id_counts[fid] += 1
        
    total_gt_pairs = 0
    for count in id_counts.values():
        if count > 1:
            total_gt_pairs += (count * (count - 1)) // 2
            
    print(f"Ground Truth: 总文件 {len(all_files_gt)}, 真实重复对 {total_gt_pairs}")

    # C. Compute TP and FP.
    tp = 0
    fp = 0
    
    # Iterate over each cluster.
    for keeper_path, dup_paths in duplicates_map.items():
        # A cluster contains [keeper, dup1, dup2...].
        # Filenames are enough for id parsing.
        cluster_files = [os.path.basename(keeper_path)] + [os.path.basename(d) for d in dup_paths]
        
        # Parse ids.
        ids = [parse_id(f) for f in cluster_files]
        n = len(ids)
        if n < 2: continue
        
        # Pairwise comparison within each predicted cluster.
        # The algorithm claims every pair in the cluster is duplicate, so verify ids pair by pair.
        
        # For a cluster of k items, the algorithm claims k(k-1)/2 duplicate pairs.
        for i in range(n):
            for j in range(i+1, n):
                if ids[i] == ids[j]:
                    tp += 1
                else:
                    fp += 1
    
    # D. Compute final metrics.
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / total_gt_pairs) if total_gt_pairs > 0 else 0.0
    
    # Throughput.
    throughput = (processed_count / duration_total) if duration_total > 0 else 0
    
    # GPU memory. Offline runs cannot recover the peak, so use 0.
    gpu_mem = 0.0
    
    print("-" * 40)
    print(f"结果: Precision={precision*100:.2f}%, Recall={recall*100:.2f}%")
    print(f"TP={tp}, FP={fp}, Total GT Pairs={total_gt_pairs}")
    print("-" * 40)

    # Write CSV.
    log_result_csv("My Pipeline", throughput, precision, recall, gpu_mem)

if __name__ == "__main__":
    main()
