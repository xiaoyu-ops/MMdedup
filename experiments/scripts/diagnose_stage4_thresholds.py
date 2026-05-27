"""Run Stage 4 threshold diagnostics on a labeled annotation set.

This script is intentionally separate from the original evaluation script
because the ICDM revision needs a stricter diagnostic pass:

- image/text/joint are swept independently;
- naive-union is swept as an image-threshold x text-threshold grid;
- the selected thresholds can later be frozen before held-out fair evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import yaml


POSITIVE_LABELS = {"duplicate", "near-duplicate", "near_duplicate"}
NEGATIVE_LABELS = {"not-duplicate", "not_duplicate", "negative"}
SCORE_COLUMNS = {
    "image": "image_similarity",
    "text": "text_similarity",
    "joint": "joint_similarity",
    "max": "max_similarity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--thresholds", default="0.00:1.00:0.01")
    parser.add_argument("--positive-labels", default="duplicate,near-duplicate")
    parser.add_argument("--label-column", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_labeled_rows(args.annotations_csv, args.label_column)
    thresholds = parse_thresholds(args.thresholds)
    positive_labels = {item.strip() for item in args.positive_labels.split(",") if item.strip()}

    per_threshold: list[dict[str, Any]] = []
    for score_name in ("image", "text", "joint", "max"):
        for threshold in thresholds:
            per_threshold.append(score_metrics(rows, score_name, threshold, positive_labels))

    naive_grid: list[dict[str, Any]] = []
    for image_threshold in thresholds:
        for text_threshold in thresholds:
            naive_grid.append(naive_union_metrics(rows, image_threshold, text_threshold, positive_labels))

    best_by_method: dict[str, dict[str, Any]] = {}
    for score_name in ("image", "text", "joint", "max"):
        candidates = [row for row in per_threshold if row["method"] == score_name]
        best_by_method[score_name] = best_row(candidates)
    best_by_method["naive_union"] = best_row(naive_grid)

    write_csv(args.output_dir / "per_threshold_metrics.csv", per_threshold)
    write_csv(args.output_dir / "naive_union_grid_metrics.csv", naive_grid)

    config = {
        "experiment_id": args.experiment_id,
        "annotations_csv": str(args.annotations_csv),
        "thresholds": thresholds,
        "positive_labels": sorted(positive_labels),
        "label_column": args.label_column,
        "note": "Existing labeled set threshold diagnostic; do not use as final held-out evaluation.",
    }
    (args.output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    metrics = {
        "experiment_id": args.experiment_id,
        "num_labeled_rows": len(rows),
        "num_positive": sum(1 for row in rows if row["is_positive"]),
        "num_negative": sum(1 for row in rows if not row["is_positive"]),
        "threshold_count": len(thresholds),
        "best_by_method": best_by_method,
        "elapsed_seconds": time.time() - started,
        "outputs": {
            "per_threshold_metrics": str(args.output_dir / "per_threshold_metrics.csv"),
            "naive_union_grid_metrics": str(args.output_dir / "naive_union_grid_metrics.csv"),
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "command": " ".join(sys.argv),
                "git_commit": git_commit(),
                "hardware": hardware_summary(),
                "wall_clock_seconds": time.time() - started,
                "outputs": {
                    "config": str(args.output_dir / "config.yaml"),
                    "metrics": str(args.output_dir / "metrics.json"),
                    "per_threshold_metrics": str(args.output_dir / "per_threshold_metrics.csv"),
                    "naive_union_grid_metrics": str(args.output_dir / "naive_union_grid_metrics.csv"),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


def read_labeled_rows(path: Path, label_column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"image_similarity", "text_similarity", "joint_similarity", "max_similarity"}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        resolved_label_column = resolve_label_column(fieldnames, label_column)
        for raw in reader:
            label = (raw.get(resolved_label_column) or "").strip()
            if not label:
                continue
            normalized = label.lower().replace("_", "-")
            if normalized not in POSITIVE_LABELS and normalized not in NEGATIVE_LABELS:
                raise ValueError(f"Unsupported label {label!r}")
            row: dict[str, Any] = dict(raw)
            for column in required:
                row[column] = float(raw[column])
            row["label"] = normalized
            row["is_positive"] = normalized in POSITIVE_LABELS
            rows.append(row)
    if not rows:
        raise ValueError("No labeled rows found")
    return rows


def resolve_label_column(fieldnames: Iterable[str], label_column: str) -> str:
    names = list(fieldnames)
    if label_column != "auto":
        if label_column not in names:
            raise ValueError(f"Label column {label_column!r} not found")
        return label_column
    if "final_label" in names:
        return "final_label"
    if "label" in names:
        return "label"
    raise ValueError("Annotation CSV must contain label or final_label")


def parse_thresholds(raw: str) -> list[float]:
    raw = raw.strip()
    if ":" in raw:
        start_s, stop_s, step_s = raw.split(":")
        start = float(start_s)
        stop = float(stop_s)
        step = float(step_s)
        if step <= 0:
            raise ValueError("threshold step must be positive")
        values = []
        current = start
        while current <= stop + 1e-12:
            values.append(round(current, 10))
            current += step
        return values
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def score_metrics(
    rows: list[dict[str, Any]],
    method: str,
    threshold: float,
    positive_labels: set[str],
) -> dict[str, Any]:
    column = SCORE_COLUMNS[method]
    return metrics_from_predictions(
        rows,
        method=method,
        threshold=threshold,
        image_threshold="",
        text_threshold="",
        predictions=[float(row[column]) >= threshold for row in rows],
        positive_labels=positive_labels,
    )


def naive_union_metrics(
    rows: list[dict[str, Any]],
    image_threshold: float,
    text_threshold: float,
    positive_labels: set[str],
) -> dict[str, Any]:
    return metrics_from_predictions(
        rows,
        method="naive_union",
        threshold="",
        image_threshold=image_threshold,
        text_threshold=text_threshold,
        predictions=[
            float(row["image_similarity"]) >= image_threshold or float(row["text_similarity"]) >= text_threshold
            for row in rows
        ],
        positive_labels=positive_labels,
    )


def metrics_from_predictions(
    rows: list[dict[str, Any]],
    method: str,
    threshold: float | str,
    image_threshold: float | str,
    text_threshold: float | str,
    predictions: list[bool],
    positive_labels: set[str],
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row, pred_positive in zip(rows, predictions):
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
    return {
        "method": method,
        "threshold": threshold,
        "image_threshold": image_threshold,
        "text_threshold": text_threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_labels": "|".join(sorted(positive_labels)),
    }


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: (row["f1"], row["precision"], row["recall"], -float(row.get("threshold") or 0)))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def hardware_summary() -> str:
    return f"{platform.system()} {platform.machine()} | Python {platform.python_version()}"


if __name__ == "__main__":
    raise SystemExit(main())
