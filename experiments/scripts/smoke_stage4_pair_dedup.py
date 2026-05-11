"""Smoke test for Plan B Stage 4 pair-level deduplication."""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

from pipelines.stage4_pair_dedup import (
    Stage4Config,
    load_pairs_from_sidecar_dir,
    run_stage4_pair_dedup,
    write_stage4_outputs,
)


def main() -> int:
    root = Path("experiments/results/plan_b_stage4/smoke_stage4_pair_dedup")
    dataset = root / "dataset"
    output = root / "run"
    if root.exists():
        shutil.rmtree(root)
    dataset.mkdir(parents=True)
    output.mkdir(parents=True)

    _write_pair(dataset, "red_car_01", (220, 40, 40), "red car parked on street")
    _write_pair(dataset, "red_car_02", (218, 42, 42), "red car parked on street")
    _write_pair(dataset, "blue_ocean_01", (40, 90, 220), "blue ocean wave")
    _write_pair(dataset, "green_forest_01", (40, 180, 90), "green forest trail")

    pairs = load_pairs_from_sidecar_dir(dataset)
    result = run_stage4_pair_dedup(
        pairs,
        Stage4Config(embedding_backend="simple", joint_method="concat", tau_cross=0.65),
    )
    write_stage4_outputs(result, output)

    assert len(result.keepers) == 3, result.summary
    assert len(result.drops) == 1, result.summary
    assert result.duplicate_groups, result.summary
    print(f"Stage 4 smoke passed: {result.summary}")
    return 0


def _write_pair(dataset: Path, stem: str, color: tuple[int, int, int], caption: str) -> None:
    Image.new("RGB", (64, 64), color).save(dataset / f"{stem}.jpg")
    (dataset / f"{stem}.txt").write_text(caption, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

