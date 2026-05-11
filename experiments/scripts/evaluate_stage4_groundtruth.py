"""Evaluate Stage 4 candidate scores against annotated pair-pair labels."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import yaml


POSITIVE_LABELS = {"duplicate", "near-duplicate", "near_duplicate"}
NEGATIVE_LABELS = {"not-duplicate", "not_duplicate", "negative"}
SCORE_COLUMNS = {
    "image": "image_similarity",
    "text": "text_similarity",
    "joint": "joint_similarity",
    "max": "max_similarity",
    "naive_union": "naive_union_similarity",
}
EVAL_SCORES = ["image", "text", "naive_union", "joint", "max"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--score", default="joint", choices=sorted([*SCORE_COLUMNS, "all"]))
    parser.add_argument("--thresholds", default="0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--positive-labels", default="duplicate,near-duplicate")
    parser.add_argument("--label-column", default="auto", help="Use `auto` to prefer final_label when present, else label.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = args.output_dir / "stdout.log"
    stderr_path = args.output_dir / "stderr.log"

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rows = _read_labeled_rows(args.annotations_csv, label_column=args.label_column)
            thresholds = _parse_thresholds(args.thresholds)
            positive_labels = {label.strip() for label in args.positive_labels.split(",") if label.strip()}
            scores = EVAL_SCORES if args.score == "all" else [args.score]
            per_threshold = []
            for score in scores:
                per_threshold.extend(evaluate_thresholds(rows, score, thresholds, positive_labels))
            best = max(per_threshold, key=lambda item: (item["f1"], item["precision"], item["recall"])) if per_threshold else {}
            best_by_score = {}
            for score in scores:
                score_rows = [item for item in per_threshold if item["score"] == score]
                if score_rows:
                    best_by_score[score] = max(score_rows, key=lambda item: (item["f1"], item["precision"], item["recall"]))

            _write_per_threshold(per_threshold, args.output_dir / "per_threshold_metrics.csv")
            config = {
                "experiment_id": args.experiment_id,
                "annotations_csv": str(args.annotations_csv),
                "score": args.score,
                "scores": scores,
                "thresholds": thresholds,
                "positive_labels": sorted(positive_labels),
                "label_column": args.label_column,
            }
            (args.output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            metrics = {
                "num_labeled_rows": len(rows),
                "num_positive": sum(1 for row in rows if row["is_positive"]),
                "num_negative": sum(1 for row in rows if not row["is_positive"]),
                "score": args.score,
                "best": best,
                "best_by_score": best_by_score,
                "elapsed_seconds": time.time() - started,
            }
            (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            manifest = {
                "experiment_id": args.experiment_id,
                "command": " ".join(sys.argv),
                "git_commit": _git_commit(),
                "hardware": _hardware_summary(),
                "wall_clock_seconds": time.time() - started,
                "outputs": {
                    "config": str(args.output_dir / "config.yaml"),
                    "metrics": str(args.output_dir / "metrics.json"),
                    "per_threshold": str(args.output_dir / "per_threshold_metrics.csv"),
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                },
            }
            (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(json.dumps(metrics, indent=2), flush=True)
    return 0


def evaluate_thresholds(
    rows: Sequence[Dict[str, object]],
    score_name: str,
    thresholds: Sequence[float],
    positive_labels: set[str],
) -> List[Dict[str, object]]:
    metrics: List[Dict[str, object]] = []
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for row in rows:
            pred_positive = _predict_positive(row, score_name, threshold)
            true_positive = bool(row["is_positive"])
            if pred_positive and true_positive:
                tp += 1
            elif pred_positive and not true_positive:
                fp += 1
            elif not pred_positive and true_positive:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics.append(
            {
                "threshold": threshold,
                "score": score_name,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "positive_labels": "|".join(sorted(positive_labels)),
            }
        )
    return metrics


def _read_labeled_rows(path: Path, label_column: str = "auto") -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_similarity", "text_similarity", "joint_similarity", "max_similarity"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Annotation CSV must contain columns: {sorted(required)}")
        resolved_label_column = _resolve_label_column(reader.fieldnames, label_column)
        for raw in reader:
            label = (raw.get(resolved_label_column) or "").strip()
            if not label:
                continue
            normalized = label.lower().replace("_", "-")
            if normalized not in POSITIVE_LABELS and normalized not in NEGATIVE_LABELS:
                raise ValueError(f"Unsupported label {label!r}; use duplicate, near-duplicate, or not-duplicate")
            row: Dict[str, object] = dict(raw)
            for column in required - {"label"}:
                row[column] = float(raw[column])
            row["naive_union_similarity"] = max(float(row["image_similarity"]), float(row["text_similarity"]))
            row["label"] = normalized
            row["is_positive"] = normalized in POSITIVE_LABELS
            rows.append(row)
    if not rows:
        raise ValueError("No labeled rows found in annotation CSV")
    return rows


def _resolve_label_column(fieldnames: Sequence[str], label_column: str) -> str:
    if label_column != "auto":
        if label_column not in fieldnames:
            raise ValueError(f"Label column {label_column!r} not found in annotation CSV")
        return label_column
    if "final_label" in fieldnames:
        return "final_label"
    if "label" in fieldnames:
        return "label"
    raise ValueError("Annotation CSV must contain `label` or `final_label`")


def _write_per_threshold(rows: Sequence[Dict[str, object]], path: Path) -> None:
    fieldnames = [
        "threshold",
        "score",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1",
        "positive_labels",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_thresholds(raw: str) -> List[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one threshold is required")
    return values


def _predict_positive(row: Dict[str, object], score_name: str, threshold: float) -> bool:
    if score_name == "naive_union":
        return float(row["image_similarity"]) >= threshold or float(row["text_similarity"]) >= threshold
    return float(row[SCORE_COLUMNS[score_name]]) >= threshold


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _hardware_summary() -> str:
    return f"{platform.system()} {platform.machine()} | Python {platform.python_version()}"


if __name__ == "__main__":
    raise SystemExit(main())
