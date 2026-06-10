"""Prepare a deterministic COCO caption-generation eval JSON for LLaVA.

The output intentionally matches the generic input shape consumed by
``run_llava_stage4_vqa_eval.py``: each record has an image path, a prompt-like
question, and multiple reference answers. COCO-specific caption metrics can be
computed from the same file plus the generated predictions.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = "Describe the image in one sentence."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--sample-policy",
        default="shuffle",
        choices=["shuffle", "first"],
        help="shuffle is deterministic with --seed; first uses COCO image order.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(args.annotations_json.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError(f"Unsupported COCO caption annotation shape: {args.annotations_json}")

    captions_by_image: dict[int, list[str]] = defaultdict(list)
    for row in annotations:
        try:
            image_id = int(row["image_id"])
        except (KeyError, TypeError, ValueError):
            continue
        caption = str(row.get("caption") or "").strip()
        if caption:
            captions_by_image[image_id].append(caption)

    candidates: list[dict[str, Any]] = []
    missing_images = 0
    no_caption = 0
    for image in images:
        try:
            image_id = int(image["id"])
        except (KeyError, TypeError, ValueError):
            continue
        refs = captions_by_image.get(image_id, [])
        if not refs:
            no_caption += 1
            continue
        file_name = str(image.get("file_name") or f"COCO_val2014_{image_id:012d}.jpg")
        image_path = args.image_root / file_name
        if not image_path.exists():
            missing_images += 1
            continue
        candidates.append(
            {
                "id": f"coco_val2014_{image_id}",
                "question_id": image_id,
                "image_id": image_id,
                "file_name": file_name,
                "image": str(image_path),
                "question": args.prompt,
                "answers": refs,
            }
        )

    if args.sample_policy == "shuffle":
        random.Random(args.seed).shuffle(candidates)
    else:
        candidates.sort(key=lambda row: int(row["image_id"]))
    records = candidates[: args.max_samples]

    args.output_json.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics = {
        "annotations_json": str(args.annotations_json),
        "image_root": str(args.image_root),
        "output_json": str(args.output_json),
        "sample_policy": args.sample_policy,
        "seed": args.seed,
        "prompt": args.prompt,
        "requested_samples": args.max_samples,
        "available_records": len(candidates),
        "written_records": len(records),
        "missing_images": missing_images,
        "images_without_captions": no_caption,
        "runtime_seconds": time.time() - started,
    }
    (args.output_json.parent / "prepare_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
