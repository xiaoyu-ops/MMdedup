"""Build a human annotation sheet from mined Stage 4 candidate pairs."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--target-total", type=int, default=1000)
    parser.add_argument("--audit-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260511)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = args.output_dir / "stdout.log"
    stderr_path = args.output_dir / "stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rows = _read_csv(args.candidates_csv)
            sampled = _stratified_sample(rows, target_total=args.target_total, seed=args.seed)
            audit_count = max(0, round(len(sampled) * args.audit_fraction))
            audit_ids = set(row["candidate_id"] for row in random.Random(args.seed + 1).sample(sampled, k=audit_count)) if sampled else set()
            for row in sampled:
                row["needs_audit"] = "1" if row["candidate_id"] in audit_ids else "0"
                row["label_options"] = "duplicate|near-duplicate|not-duplicate"
                row["annotation_guideline"] = (
                    "duplicate: same image and same caption meaning; "
                    "near-duplicate: visually/semantically same with small edits; "
                    "not-duplicate: meaningfully different pair. "
                    "Score-assisted rule: if image_similarity and text_similarity are both "
                    ">0.85 and <0.95, near-duplicate is acceptable; if both are >0.95, "
                    "duplicate is acceptable."
                )

            annotation_path = args.output_dir / "annotation_sheet.csv"
            _write_csv(sampled, annotation_path)
            config = {
                "experiment_id": args.experiment_id,
                "candidates_csv": str(args.candidates_csv),
                "target_total": args.target_total,
                "audit_fraction": args.audit_fraction,
                "seed": args.seed,
            }
            (args.output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            metrics = {
                "num_candidates_input": len(rows),
                "num_annotation_rows": len(sampled),
                "num_audit_rows": audit_count,
                "bucket_counts": _bucket_counts(sampled),
                "elapsed_seconds": time.time() - started,
            }
            (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            manifest = {
                "experiment_id": args.experiment_id,
                "command": " ".join(sys.argv),
                "git_commit": _git_commit(),
                "hardware": _hardware_summary(),
                "wall_clock_seconds": time.time() - started,
                "outputs": {
                    "config": str(args.output_dir / "config.yaml"),
                    "metrics": str(args.output_dir / "metrics.json"),
                    "annotation_sheet": str(annotation_path),
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                },
            }
            (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(json.dumps(metrics, indent=2), flush=True)
    return 0


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stratified_sample(rows: List[Dict[str, str]], target_total: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    by_bucket: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_bucket[row.get("bucket", "unknown")].append(row)

    buckets = ["very_high", "high", "medium", "low", "unknown"]
    active = [bucket for bucket in buckets if by_bucket.get(bucket)]
    if not active:
        return []

    per_bucket = max(1, target_total // len(active))
    sampled: List[Dict[str, str]] = []
    leftovers: List[Dict[str, str]] = []
    for bucket in active:
        bucket_rows = list(by_bucket[bucket])
        rng.shuffle(bucket_rows)
        sampled.extend(bucket_rows[:per_bucket])
        leftovers.extend(bucket_rows[per_bucket:])

    remaining = max(0, target_total - len(sampled))
    rng.shuffle(leftovers)
    sampled.extend(leftovers[:remaining])
    sampled = sampled[:target_total]
    sampled.sort(key=lambda row: row.get("candidate_id", ""))
    return sampled


def _bucket_counts(rows: List[Dict[str, str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        bucket = row.get("bucket", "unknown")
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _hardware_summary() -> str:
    return f"{platform.system()} {platform.machine()} | Python {platform.python_version()}"


if __name__ == "__main__":
    raise SystemExit(main())
