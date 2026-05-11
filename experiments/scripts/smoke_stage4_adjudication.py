"""Smoke test Stage 4 annotation adjudication."""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path("experiments/results/plan_b_stage4/smoke_stage4_adjudication")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    labeled = root / "audited_annotation_sheet.csv"
    _write_audited_sheet(labeled)
    subprocess.check_call(
        [
            sys.executable,
            "experiments/scripts/adjudicate_stage4_annotations.py",
            "--annotations-csv",
            str(labeled),
            "--output-dir",
            str(root / "adjudicated"),
            "--conflict-policy",
            "mark",
            "--experiment-id",
            "smoke_stage4_adjudication",
        ]
    )
    assert (root / "adjudicated" / "adjudicated_annotations.csv").exists()
    assert (root / "adjudicated" / "metrics.json").exists()
    subprocess.check_call(
        [
            sys.executable,
            "experiments/scripts/evaluate_stage4_groundtruth.py",
            "--annotations-csv",
            str(root / "adjudicated" / "adjudicated_annotations.csv"),
            "--output-dir",
            str(root / "eval_adjudicated"),
            "--score",
            "all",
            "--thresholds",
            "0.50,0.70,0.90",
            "--experiment-id",
            "smoke_stage4_eval_adjudicated",
        ]
    )
    assert (root / "eval_adjudicated" / "metrics.json").exists()
    print("Stage 4 adjudication smoke passed")
    return 0


def _write_audited_sheet(target: Path) -> None:
    rows = [
        _row("cand_000000", "red_car_01", "red_car_02", 0.85, 1.0, 0.92, "duplicate", "duplicate", "1"),
        _row("cand_000001", "blue_ocean_01", "red_car_01", 0.36, 0.16, 0.26, "not-duplicate", "not-duplicate", "1"),
        _row("cand_000002", "green_forest_01", "red_car_01", 0.34, 0.28, 0.31, "not-duplicate", "near-duplicate", "1"),
        _row("cand_000003", "green_forest_01", "red_car_02", 0.15, 0.28, 0.22, "not-duplicate", "", "0"),
        _row("cand_000004", "blue_ocean_01", "green_forest_01", 0.27, 0.08, 0.18, "not-duplicate", "", "0"),
        _row("cand_000005", "blue_ocean_01", "red_car_02", 0.14, 0.16, 0.15, "not-duplicate", "", "0"),
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
    audit_label: str,
    needs_audit: str,
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
        "annotator": "primary_smoke",
        "audit_label": audit_label,
        "notes": "",
        "needs_audit": needs_audit,
    }


if __name__ == "__main__":
    raise SystemExit(main())
