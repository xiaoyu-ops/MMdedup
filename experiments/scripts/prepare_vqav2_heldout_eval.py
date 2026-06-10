"""Prepare a VQAv2 held-out subset excluding an existing quick-eval subset."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from prepare_vqav2_quick_eval import (
    ANNOTATION_JSON,
    ANNOTATION_ZIP_URL,
    IMAGE_URL_TEMPLATE,
    QUESTION_JSON,
    QUESTION_ZIP_URL,
    download_file,
    read_zip_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("experiments/data/vqav2/cache"))
    parser.add_argument("--exclude-questions-json", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--verify-ssl", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.image_root.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    question_zip = args.cache_dir / "v2_Questions_Val_mscoco.zip"
    annotation_zip = args.cache_dir / "v2_Annotations_Val_mscoco.zip"
    download_file(QUESTION_ZIP_URL, question_zip, args.timeout, args.retries, verify_ssl=args.verify_ssl)
    download_file(ANNOTATION_ZIP_URL, annotation_zip, args.timeout, args.retries, verify_ssl=args.verify_ssl)

    questions_payload = read_zip_json(question_zip, QUESTION_JSON)
    annotations_payload = read_zip_json(annotation_zip, ANNOTATION_JSON)
    questions = questions_payload["questions"]
    annotations = annotations_payload["annotations"]
    annotation_by_qid = {int(row["question_id"]): row for row in annotations}
    excluded_qids = load_question_ids(args.exclude_questions_json)

    rng = random.Random(args.seed)
    candidates = [
        row
        for row in questions
        if int(row["question_id"]) in annotation_by_qid and int(row["question_id"]) not in excluded_qids
    ]
    rng.shuffle(candidates)

    selected_questions: list[dict[str, Any]] = []
    selected_annotations: list[dict[str, Any]] = []
    failed_images: list[dict[str, Any]] = []
    existing_images = 0
    downloaded_images = 0

    for question in candidates:
        if len(selected_questions) >= args.max_samples:
            break
        image_id = int(question["image_id"])
        image_path = args.image_root / f"COCO_val2014_{image_id:012d}.jpg"
        if image_path.exists() and image_path.stat().st_size > 0:
            existing_images += 1
        else:
            image_url = IMAGE_URL_TEMPLATE.format(image_id=image_id)
            ok = download_file(
                image_url,
                image_path,
                args.timeout,
                args.retries,
                raise_on_error=False,
                verify_ssl=args.verify_ssl,
            )
            if not ok:
                failed_images.append({"question_id": question["question_id"], "image_id": image_id, "url": image_url})
                continue
            downloaded_images += 1
        selected_questions.append(question)
        selected_annotations.append(annotation_by_qid[int(question["question_id"])])

    questions_out = args.output_dir / f"vqav2_val_questions_heldout_{args.max_samples}.json"
    annotations_out = args.output_dir / f"vqav2_val_annotations_heldout_{args.max_samples}.json"
    metrics_out = args.output_dir / f"vqav2_heldout_prep_{args.max_samples}_metrics.json"
    questions_out.write_text(json.dumps({"questions": selected_questions}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    annotations_out.write_text(
        json.dumps({"annotations": selected_annotations}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metrics = {
        "status": "prepared" if len(selected_questions) == args.max_samples else "partial",
        "requested_samples": args.max_samples,
        "prepared_samples": len(selected_questions),
        "excluded_question_ids": len(excluded_qids),
        "existing_images": existing_images,
        "downloaded_images": downloaded_images,
        "failed_images": failed_images[:20],
        "num_failed_images": len(failed_images),
        "questions_json": str(questions_out),
        "annotations_json": str(annotations_out),
        "image_root": str(args.image_root),
        "seed": args.seed,
        "runtime_seconds": time.time() - started_at,
    }
    metrics_out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0 if metrics["status"] == "prepared" else 2


def load_question_ids(path: Path) -> set[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("questions", payload if isinstance(payload, list) else [])
    return {int(row["question_id"]) for row in rows if isinstance(row, dict) and row.get("question_id") is not None}


if __name__ == "__main__":
    raise SystemExit(main())
