"""Prepare a sidecar image-caption subset from a Hugging Face CC3M dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="WeiChow/cc3m")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-pairs", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--caption-column", default="caption")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise SystemExit("Install datasets first: uv pip install datasets") from exc

    dataset = load_dataset(args.dataset, split=args.split, streaming=args.streaming)
    manifest_path = args.output_dir / "manifest.csv"
    metrics_path = args.output_dir / "prepare_metrics.json"
    failures_path = args.output_dir / "prepare_failures.jsonl"

    saved = 0
    seen = 0
    failed = 0
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_handle, failures_path.open(
        "w", encoding="utf-8"
    ) as failures_handle:
        writer = csv.DictWriter(
            manifest_handle,
            fieldnames=["pair_id", "image_path", "caption_path", "caption", "source_index", "source_id"],
        )
        writer.writeheader()

        iterable = dataset if args.streaming else iter(dataset)
        for sample in iterable:
            if seen < args.start_index:
                seen += 1
                continue
            if saved >= args.num_pairs:
                break
            source_index = seen
            seen += 1
            try:
                caption = str(sample[args.caption_column]).strip()
                if not caption:
                    raise ValueError("empty caption")
                image = _get_image(sample, args.image_column)
                pair_id = _pair_id(sample, args.id_column, source_index)
                image_path = args.output_dir / f"{pair_id}.jpg"
                caption_path = args.output_dir / f"{pair_id}.txt"
                if not args.overwrite and image_path.exists() and caption_path.exists():
                    saved += 1
                    continue
                image.convert("RGB").save(image_path, format="JPEG", quality=95)
                caption_path.write_text(caption + "\n", encoding="utf-8")
                writer.writerow(
                    {
                        "pair_id": pair_id,
                        "image_path": str(image_path),
                        "caption_path": str(caption_path),
                        "caption": caption,
                        "source_index": source_index,
                        "source_id": str(sample.get(args.id_column, "")),
                    }
                )
                saved += 1
                if saved % 100 == 0:
                    print(f"saved {saved}/{args.num_pairs}", flush=True)
            except Exception as exc:
                failed += 1
                failures_handle.write(
                    json.dumps(
                        {
                            "source_index": source_index,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    metrics: Dict[str, Any] = {
        "dataset": args.dataset,
        "split": args.split,
        "output_dir": str(args.output_dir),
        "requested_pairs": args.num_pairs,
        "saved_pairs": saved,
        "failed_samples": failed,
        "seen_samples": seen,
        "elapsed_seconds": time.time() - started,
        "manifest": str(manifest_path),
        "failures": str(failures_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    if saved < args.num_pairs:
        raise SystemExit(f"Only saved {saved}/{args.num_pairs} pairs")
    return 0


def _get_image(sample: Dict[str, Any], image_column: str):
    image = sample[image_column]
    if hasattr(image, "convert"):
        return image
    if isinstance(image, dict) and "bytes" in image:
        from io import BytesIO
        from PIL import Image

        return Image.open(BytesIO(image["bytes"]))
    raise TypeError(f"Unsupported image object: {type(image)!r}")


def _pair_id(sample: Dict[str, Any], id_column: str, source_index: int) -> str:
    raw = str(sample.get(id_column, "")).strip()
    if not raw:
        raw = f"{source_index:09d}"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)
    return f"cc3m_{safe}"


if __name__ == "__main__":
    try:
        exit_code = main()
    except SystemExit as exc:
        if isinstance(exc.code, int):
            exit_code = exc.code
        else:
            if exc.code:
                print(exc.code, file=sys.stderr)
            exit_code = 1
    except Exception:
        traceback.print_exc()
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # Hugging Face streaming can leave shutdown cleanup hanging after outputs are
    # safely written. Keep the data-prep command deterministic for remote runs.
    os._exit(exit_code)
