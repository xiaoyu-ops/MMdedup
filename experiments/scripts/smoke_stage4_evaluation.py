"""Smoke test Stage 4 annotation evaluation metrics."""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path("experiments/results/plan_b_stage4/smoke_stage4_evaluation")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    labeled = root / "labeled_annotation_sheet.csv"
    _write_labeled_sheet(labeled)
    subprocess.check_call(
        [
            sys.executable,
            "experiments/scripts/evaluate_stage4_groundtruth.py",
            "--annotations-csv",
            str(labeled),
            "--output-dir",
            str(root / "eval_joint"),
            "--score",
            "joint",
            "--thresholds",
            "0.50,0.70,0.90",
            "--experiment-id",
            "smoke_stage4_eval_joint",
        ]
    )
    assert (root / "eval_joint" / "metrics.json").exists()
    assert (root / "eval_joint" / "per_threshold_metrics.csv").exists()
    print("Stage 4 evaluation smoke passed")
    return 0


def _write_labeled_sheet(target: Path) -> None:
    rows = [
        _row("cand_000000", "red_car_01", "red_car_02", 0.85, 1.0, 0.92, "duplicate"),
        _row("cand_000001", "blue_ocean_01", "red_car_01", 0.36, 0.16, 0.26, "not-duplicate"),
        _row("cand_000002", "green_forest_01", "red_car_01", 0.34, 0.28, 0.31, "not-duplicate"),
        _row("cand_000003", "green_forest_01", "red_car_02", 0.15, 0.28, 0.22, "not-duplicate"),
        _row("cand_000004", "blue_ocean_01", "green_forest_01", 0.27, 0.08, 0.18, "not-duplicate"),
        _row("cand_000005", "blue_ocean_01", "red_car_02", 0.14, 0.16, 0.15, "not-duplicate"),
    ]
    with target.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _row(
    candidate_id: str,
    pair_id_a: str,
    pair_id_b: str,
    image_similarity: float,
    text_similarity: float,
    joint_similarity: float,
    label: str,
) -> dict[str, object]:
    max_similarity = max(image_similarity, text_similarity, joint_similarity)
    return {
        "candidate_id": candidate_id,
        "pair_id_a": pair_id_a,
        "pair_id_b": pair_id_b,
        "image_path_a": f"{pair_id_a}.jpg",
        "image_path_b": f"{pair_id_b}.jpg",
        "caption_a": pair_id_a.replace("_", " "),
        "caption_b": pair_id_b.replace("_", " "),
        "image_similarity": image_similarity,
        "text_similarity": text_similarity,
        "joint_similarity": joint_similarity,
        "max_similarity": max_similarity,
        "signals": "image|text|joint",
        "bucket": "very_high" if max_similarity >= 0.95 else "low",
        "label": label,
        "annotator": "smoke",
        "audit_label": "",
        "notes": "",
    }


if __name__ == "__main__":
    raise SystemExit(main())
