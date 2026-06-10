"""Analyze residual high-similarity candidate-edge leakage across Stage 4 splits."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SPLIT_DIRS = {
    "A_raw": "A_raw",
    "B_image_only": "B_image_only",
    "C_text_only": "C_text_only",
    "D_naive_union": "D_naive_union",
    "E_train_stage4_conservative": "E_stage4_score_assisted",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--splits-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--image-threshold", type=float, default=0.85)
    parser.add_argument("--text-threshold", type=float, default=0.95)
    parser.add_argument("--joint-threshold", type=float, default=0.85)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    keepers_by_split = {
        split: read_pair_ids(args.splits_root / split_dir / "training_manifest.csv")
        for split, split_dir in SPLIT_DIRS.items()
    }
    rows = read_candidate_rows(args.candidates_csv)

    baseline = summarize_edges(
        rows,
        kept_ids=keepers_by_split["A_raw"],
        image_threshold=args.image_threshold,
        text_threshold=args.text_threshold,
        joint_threshold=args.joint_threshold,
    )
    split_rows: list[dict[str, Any]] = []
    for split, kept_ids in keepers_by_split.items():
        summary = summarize_edges(
            rows,
            kept_ids=kept_ids,
            image_threshold=args.image_threshold,
            text_threshold=args.text_threshold,
            joint_threshold=args.joint_threshold,
        )
        summary["split"] = split
        summary["kept_pairs"] = len(kept_ids)
        for key in (
            "image_high_edges",
            "text_high_edges",
            "joint_high_edges",
            "conservative_stage4_edges",
            "strict_duplicate_edges",
        ):
            base_value = baseline[key]
            summary[f"{key}_reduction_vs_raw"] = 1.0 - (summary[key] / base_value) if base_value else 0.0
        split_rows.append(summary)

    write_csv(args.output_dir / "split_leakage_summary.csv", split_rows)
    metrics = {
        "experiment_id": args.experiment_id,
        "analysis_type": "split_candidate_edge_leakage",
        "candidates_csv": str(args.candidates_csv),
        "splits_root": str(args.splits_root),
        "num_candidate_edges": len(rows),
        "thresholds": {
            "image": args.image_threshold,
            "text": args.text_threshold,
            "joint": args.joint_threshold,
            "conservative_stage4": "image>=0.85 AND text>=0.85",
            "strict_duplicate": "image>=0.95 AND text>=0.95",
        },
        "results_by_split": {row["split"]: row for row in split_rows},
        "elapsed_seconds": time.time() - started,
        "notes": (
            "Counts residual mined candidate edges whose two endpoints both remain in each training split. "
            "This is a data-leakage / residual-near-duplicate analysis, not a VQA downstream metric."
        ),
        "outputs": {
            "split_leakage_summary": str(args.output_dir / "split_leakage_summary.csv"),
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "experiment_id": args.experiment_id,
        "command": " ".join(sys.argv),
        "git_commit": git_commit(),
        "hardware": hardware_summary(),
        "wall_clock_seconds": time.time() - started,
        "outputs": {
            "metrics": str(args.output_dir / "metrics.json"),
            "split_leakage_summary": str(args.output_dir / "split_leakage_summary.csv"),
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


def read_pair_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pair_id = (row.get("pair_id") or "").strip()
            if pair_id:
                ids.add(pair_id)
    return ids


def read_candidate_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "pair_id_a": row["pair_id_a"],
                    "pair_id_b": row["pair_id_b"],
                    "image_similarity": float(row["image_similarity"]),
                    "text_similarity": float(row["text_similarity"]),
                    "joint_similarity": float(row["joint_similarity"]),
                    "signals": row.get("signals", ""),
                }
            )
    return rows


def summarize_edges(
    rows: list[dict[str, Any]],
    kept_ids: set[str],
    image_threshold: float,
    text_threshold: float,
    joint_threshold: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "residual_candidate_edges": 0,
        "removed_candidate_edges": 0,
        "image_high_edges": 0,
        "text_high_edges": 0,
        "joint_high_edges": 0,
        "naive_union_edges": 0,
        "conservative_stage4_edges": 0,
        "strict_duplicate_edges": 0,
        "mean_image_similarity_residual": 0.0,
        "mean_text_similarity_residual": 0.0,
        "mean_joint_similarity_residual": 0.0,
    }
    image_sum = text_sum = joint_sum = 0.0
    for row in rows:
        residual = row["pair_id_a"] in kept_ids and row["pair_id_b"] in kept_ids
        if residual:
            summary["residual_candidate_edges"] += 1
            image_sum += row["image_similarity"]
            text_sum += row["text_similarity"]
            joint_sum += row["joint_similarity"]
            if row["image_similarity"] >= image_threshold:
                summary["image_high_edges"] += 1
            if row["text_similarity"] >= text_threshold:
                summary["text_high_edges"] += 1
            if row["joint_similarity"] >= joint_threshold:
                summary["joint_high_edges"] += 1
            if row["image_similarity"] >= image_threshold or row["text_similarity"] >= text_threshold:
                summary["naive_union_edges"] += 1
            if row["image_similarity"] >= image_threshold and row["text_similarity"] >= image_threshold:
                summary["conservative_stage4_edges"] += 1
            if row["image_similarity"] >= 0.95 and row["text_similarity"] >= 0.95:
                summary["strict_duplicate_edges"] += 1
        else:
            summary["removed_candidate_edges"] += 1
    residual_count = summary["residual_candidate_edges"]
    if residual_count:
        summary["mean_image_similarity_residual"] = image_sum / residual_count
        summary["mean_text_similarity_residual"] = text_sum / residual_count
        summary["mean_joint_similarity_residual"] = joint_sum / residual_count
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def hardware_summary() -> str:
    return f"{platform.system()} {platform.machine()} | Python {platform.python_version()}"


if __name__ == "__main__":
    raise SystemExit(main())
