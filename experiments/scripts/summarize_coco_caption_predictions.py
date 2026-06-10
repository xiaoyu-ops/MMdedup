"""Summarize COCO caption predictions and optionally run COCO caption metrics."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-coco-metrics", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.predictions_jsonl)

    references: dict[int, list[str]] = {}
    results: list[dict[str, Any]] = []
    lengths: list[int] = []
    char_lengths: list[int] = []
    empty = 0
    for row in rows:
        image_id = int(row.get("question_id") or row.get("image_id"))
        prediction = str(row.get("prediction") or "").strip()
        refs = [str(item).strip() for item in row.get("answers", []) if str(item).strip()]
        references[image_id] = refs
        results.append({"image_id": image_id, "caption": prediction})
        token_count = len(simple_tokens(prediction))
        lengths.append(token_count)
        char_lengths.append(len(prediction))
        if not prediction:
            empty += 1

    refs_path = args.output_dir / "coco_references_min.json"
    results_path = args.output_dir / "coco_results.json"
    refs_payload = {
        "images": [{"id": image_id} for image_id in references],
        "annotations": [
            {"id": idx, "image_id": image_id, "caption": caption}
            for idx, (image_id, caption) in enumerate(
                (item for image_id, refs in references.items() for item in [(image_id, caption) for caption in refs])
            )
        ],
        "type": "captions",
        "info": {},
        "licenses": [],
    }
    refs_path.write_text(json.dumps(refs_payload, ensure_ascii=False), encoding="utf-8")
    results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    metrics: dict[str, Any] = {
        "predictions_jsonl": str(args.predictions_jsonl),
        "num_predictions": len(rows),
        "num_empty_predictions": empty,
        "mean_tokens": mean(lengths),
        "median_tokens": median(lengths),
        "mean_chars": mean(char_lengths),
        "median_chars": median(char_lengths),
        "coco_references_json": str(refs_path),
        "coco_results_json": str(results_path),
        "runtime_seconds": time.time() - started,
    }
    if args.run_coco_metrics:
        metrics["coco_metrics"] = run_coco_metrics(refs_path, results_path)
    (args.output_dir / "caption_summary_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def simple_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def mean(values: list[int]) -> float | None:
    return float(statistics.mean(values)) if values else None


def median(values: list[int]) -> float | None:
    return float(statistics.median(values)) if values else None


def run_coco_metrics(refs_path: Path, results_path: Path) -> dict[str, Any]:
    try:
        from pycocoevalcap.bleu.bleu import Bleu  # type: ignore
        from pycocoevalcap.cider.cider import Cider  # type: ignore
        from pycocoevalcap.meteor.meteor import Meteor  # type: ignore
        from pycocoevalcap.rouge.rouge import Rouge  # type: ignore
        from pycocoevalcap.spice.spice import Spice  # type: ignore
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer  # type: ignore
        from pycocotools.coco import COCO  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on remote env
        return {"status": "missing_dependency", "error": repr(exc)}

    metrics: dict[str, Any] = {"status": "ok", "metric_errors": {}}
    try:
        coco = COCO(str(refs_path))
        coco_result = coco.loadRes(str(results_path))
    except Exception as exc:  # pragma: no cover - depends on Java/METEOR/SPICE
        return {"status": "failed", "error": repr(exc)}

    image_ids = coco_result.getImgIds()
    gts = {image_id: coco.imgToAnns[image_id] for image_id in image_ids}
    res = {image_id: coco_result.imgToAnns[image_id] for image_id in image_ids}
    try:
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        metrics["tokenizer"] = "ptb"
    except Exception as exc:
        metrics["tokenizer"] = "simple_fallback"
        metrics["tokenizer_error"] = repr(exc)
        gts = simple_coco_tokenize(gts)
        res = simple_coco_tokenize(res)

    scorers: list[tuple[type[Any], tuple[Any, ...], str | list[str]]] = [
        (Bleu, (4,), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge, (), "ROUGE_L"),
        (Cider, (), "CIDEr"),
    ]
    if shutil.which("java"):
        scorers.extend([(Meteor, (), "METEOR"), (Spice, (), "SPICE")])
    else:
        metrics["metric_errors"]["METEOR"] = "skipped_missing_java"
        metrics["metric_errors"]["SPICE"] = "skipped_missing_java"

    for scorer_cls, scorer_args, method in scorers:
        try:
            scorer = scorer_cls(*scorer_args)
            score, _ = scorer.compute_score(gts, res)
        except Exception as exc:  # pragma: no cover - optional Java resources
            metrics["metric_errors"][str(method)] = repr(exc)
            continue
        if isinstance(method, list):
            for name, value in zip(method, score):
                metrics[name] = float(value)
        else:
            metrics[method] = float(score)
    if not any(name in metrics for name in ["CIDEr", "Bleu_4", "METEOR", "SPICE"]):
        metrics["status"] = "no_metric_succeeded"
    return metrics


def simple_coco_tokenize(rows_by_image: dict[int, list[dict[str, Any]]]) -> dict[int, list[str]]:
    tokenized: dict[int, list[str]] = {}
    for image_id, rows in rows_by_image.items():
        captions = []
        for row in rows:
            caption = str(row.get("caption") or "")
            captions.append(" ".join(simple_tokens(caption)))
        tokenized[image_id] = captions
    return tokenized


if __name__ == "__main__":
    raise SystemExit(main())
