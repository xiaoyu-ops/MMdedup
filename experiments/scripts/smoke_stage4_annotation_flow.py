"""Smoke test candidate mining and annotation sheet generation."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image


def main() -> int:
    root = Path("experiments/results/plan_b_stage4/smoke_stage4_annotation_flow")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    _write_pair(dataset, "red_car_01", (220, 40, 40), "red car parked on street")
    _write_pair(dataset, "red_car_02", (218, 42, 42), "red car parked on street")
    _write_pair(dataset, "blue_ocean_01", (40, 90, 220), "blue ocean wave")
    _write_pair(dataset, "green_forest_01", (40, 180, 90), "green forest trail")

    candidates_dir = root / "candidates"
    annotation_dir = root / "annotation"
    subprocess.check_call(
        [
            sys.executable,
            "experiments/scripts/mine_stage4_candidates.py",
            "--input-dir",
            str(dataset),
            "--output-dir",
            str(candidates_dir),
            "--backend",
            "simple",
            "--top-k",
            "2",
            "--min-similarity",
            "0.0",
            "--max-candidates",
            "10",
            "--experiment-id",
            "smoke_stage4_candidates",
        ]
    )
    subprocess.check_call(
        [
            sys.executable,
            "experiments/scripts/build_annotation_sheet.py",
            "--candidates-csv",
            str(candidates_dir / "stage4_candidate_pairs.csv"),
            "--output-dir",
            str(annotation_dir),
            "--target-total",
            "6",
            "--audit-fraction",
            "0.5",
            "--experiment-id",
            "smoke_stage4_annotation",
        ]
    )
    assert (candidates_dir / "stage4_candidate_pairs.csv").exists()
    assert (annotation_dir / "annotation_sheet.csv").exists()
    print("Stage 4 annotation flow smoke passed")
    return 0


def _write_pair(dataset: Path, stem: str, color: tuple[int, int, int], caption: str) -> None:
    Image.new("RGB", (64, 64), color).save(dataset / f"{stem}.jpg")
    (dataset / f"{stem}.txt").write_text(caption, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
