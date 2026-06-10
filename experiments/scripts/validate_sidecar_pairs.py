"""Validate a sidecar image-caption dataset directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--expected-pairs", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir
    images = sorted(dataset_dir.glob("*.jpg"))
    captions = sorted(dataset_dir.glob("*.txt"))
    image_stems = {path.stem for path in images}
    caption_stems = {path.stem for path in captions}
    missing_captions = sorted(image_stems - caption_stems)
    missing_images = sorted(caption_stems - image_stems)

    manifest_path = dataset_dir / "manifest.csv"
    metrics_path = dataset_dir / "prepare_metrics.json"
    manifest_rows = []
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            manifest_rows = list(csv.DictReader(handle))

    metrics: Dict[str, Any] = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    sample_images = _sample_edges(images, args.sample_size)
    sample_captions = _sample_edges(captions, args.sample_size)
    sampled_image_bytes = []
    for path in sample_images:
        with Image.open(path) as image:
            image.verify()
        sampled_image_bytes.append(path.stat().st_size)

    empty_captions = [str(path) for path in sample_captions if not path.read_text(encoding="utf-8").strip()]
    result = {
        "dataset_dir": str(dataset_dir),
        "jpg_count": len(images),
        "txt_count": len(captions),
        "manifest_rows": len(manifest_rows),
        "metrics_saved_pairs": metrics.get("saved_pairs"),
        "sampled_images_verified": len(sample_images),
        "empty_sampled_captions": len(empty_captions),
        "missing_caption_files": len(missing_captions),
        "missing_image_files": len(missing_images),
        "min_sampled_image_bytes": min(sampled_image_bytes) if sampled_image_bytes else None,
        "max_sampled_image_bytes": max(sampled_image_bytes) if sampled_image_bytes else None,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.expected_pairs is not None:
        _assert_equal("jpg_count", len(images), args.expected_pairs)
        _assert_equal("txt_count", len(captions), args.expected_pairs)
        _assert_equal("manifest_rows", len(manifest_rows), args.expected_pairs)
        if metrics:
            _assert_equal("metrics_saved_pairs", metrics.get("saved_pairs"), args.expected_pairs)
    _assert_equal("missing_caption_files", len(missing_captions), 0)
    _assert_equal("missing_image_files", len(missing_images), 0)
    _assert_equal("empty_sampled_captions", len(empty_captions), 0)
    return 0


def _sample_edges(paths: list[Path], sample_size: int) -> list[Path]:
    if len(paths) <= sample_size:
        return paths
    head = sample_size // 2
    tail = sample_size - head
    return paths[:head] + paths[-tail:]


def _assert_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected!r}, got {actual!r}")


if __name__ == "__main__":
    raise SystemExit(main())
