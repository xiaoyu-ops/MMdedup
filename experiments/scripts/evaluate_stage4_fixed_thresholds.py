"""Evaluate Stage 4 labels with thresholds selected on a separate dev set."""

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
from typing import Any

import yaml


POSITIVE_LABELS = {"duplicate", "near-duplicate", "near_duplicate"}
NEGATIVE_LABELS = {"not-duplicate", "not_duplicate", "negative"}
SCORE_COLUMNS = {
    "image": "image_similarity",
    "text": "text_similarity",
    "joint": "joint_similarity",
    "max": "max_similarity",
}
METHOD_ORDER = ["image", "text", "naive_union", "joint", "max"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", type=Path, required=True)
    parser.add_argument("--dev-metrics-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--positive-labels", default="duplicate,near-duplicate")
    parser.add_argument("--label-column", default="auto")
    parser.add_argument(
        "--manual-thresholds-json",
        default="",
        help=(
            "Optional JSON object with fixed paper-facing thresholds. "
            "When provided, this overrides dev-selected thresholds."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = args.output_dir / "stdout.log"
    stderr_path = args.output_dir / "stderr.log"

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rows = read_labeled_rows(args.annotations_csv, args.label_column)
            positive_labels = {item.strip() for item in args.positive_labels.split(",") if item.strip()}
            dev_metrics = json.loads(args.dev_metrics_json.read_text(encoding="utf-8"))
            thresholds = (
                parse_manual_thresholds(args.manual_thresholds_json)
                if args.manual_thresholds_json
                else fixed_thresholds_from_dev(dev_metrics)
            )
            threshold_policy = (
                "manual paper-facing thresholds aligned with downstream split generation; "
                "no held-out threshold tuning"
                if args.manual_thresholds_json
                else "selected on old 1000-row dev diagnostic only; no held-out threshold tuning"
            )
            evaluation_type = (
                "held_out_manual_fixed_thresholds"
                if args.manual_thresholds_json
                else "held_out_fixed_dev_thresholds"
            )
            notes = (
                "Manual thresholds are fixed before evaluation reporting and match the v2 downstream split policy. "
                "The 3000-row fair annotations are used here only as held-out evaluation labels."
                if args.manual_thresholds_json
                else (
                    "Thresholds are selected only from the old 1000-row dev diagnostic metrics. "
                    "The 3000-row fair annotations are used here only as held-out evaluation labels."
                )
            )

            result_rows = [
                evaluate_method(rows, method, thresholds[method], positive_labels)
                for method in METHOD_ORDER
                if method in thresholds
            ]
            write_csv(args.output_dir / "fixed_threshold_metrics.csv", result_rows)

            by_method = {row["method"]: row for row in result_rows}
            metrics = {
                "experiment_id": args.experiment_id,
                "evaluation_type": evaluation_type,
                "dev_metrics_json": str(args.dev_metrics_json),
                "annotations_csv": str(args.annotations_csv),
                "num_labeled_rows": len(rows),
                "num_positive": sum(1 for row in rows if row["is_positive"]),
                "num_negative": sum(1 for row in rows if not row["is_positive"]),
                "positive_labels": sorted(positive_labels),
                "fixed_thresholds": thresholds,
                "results_by_method": by_method,
                "stage4_vs_naive_union_f1_delta": _delta(by_method, "joint", "naive_union"),
                "stage4_vs_image_f1_delta": _delta(by_method, "joint", "image"),
                "elapsed_seconds": time.time() - started,
                "notes": notes,
                "outputs": {
                    "fixed_threshold_metrics": str(args.output_dir / "fixed_threshold_metrics.csv"),
                },
            }
            (args.output_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            config = {
                "experiment_id": args.experiment_id,
                "annotations_csv": str(args.annotations_csv),
                "dev_metrics_json": str(args.dev_metrics_json),
                "label_column": args.label_column,
                "positive_labels": sorted(positive_labels),
                "fixed_thresholds": thresholds,
                "threshold_selection_policy": threshold_policy,
            }
            (args.output_dir / "config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            manifest = {
                "experiment_id": args.experiment_id,
                "command": " ".join(sys.argv),
                "git_commit": git_commit(),
                "hardware": hardware_summary(),
                "wall_clock_seconds": time.time() - started,
                "outputs": {
                    "config": str(args.output_dir / "config.yaml"),
                    "metrics": str(args.output_dir / "metrics.json"),
                    "fixed_threshold_metrics": str(args.output_dir / "fixed_threshold_metrics.csv"),
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                },
            }
            (args.output_dir / "run_manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    return 0


def fixed_thresholds_from_dev(metrics: dict[str, Any]) -> dict[str, dict[str, float | str]]:
    best = metrics.get("best_by_method", {})
    if not isinstance(best, dict):
        raise ValueError("dev metrics must contain best_by_method")
    thresholds: dict[str, dict[str, float | str]] = {}
    for method in METHOD_ORDER:
        row = best.get(method)
        if not isinstance(row, dict):
            continue
        if method == "naive_union":
            thresholds[method] = {
                "method": method,
                "image_threshold": float(row["image_threshold"]),
                "text_threshold": float(row["text_threshold"]),
                "source": "dev_best_by_method.naive_union",
            }
        else:
            thresholds[method] = {
                "method": method,
                "threshold": float(row["threshold"]),
                "source": f"dev_best_by_method.{method}",
            }
    missing = [method for method in ("image", "text", "naive_union", "joint") if method not in thresholds]
    if missing:
        raise ValueError(f"dev metrics missing fixed thresholds for: {missing}")
    return thresholds


def parse_manual_thresholds(raw: str) -> dict[str, dict[str, float | str]]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("manual thresholds must be a JSON object")
    thresholds: dict[str, dict[str, float | str]] = {}
    for method in METHOD_ORDER:
        row = data.get(method)
        if not isinstance(row, dict):
            continue
        if method == "naive_union":
            thresholds[method] = {
                "method": method,
                "image_threshold": float(row["image_threshold"]),
                "text_threshold": float(row["text_threshold"]),
                "source": str(row.get("source", "manual_paper_thresholds.naive_union")),
            }
        else:
            thresholds[method] = {
                "method": method,
                "threshold": float(row["threshold"]),
                "source": str(row.get("source", f"manual_paper_thresholds.{method}")),
            }
    missing = [method for method in ("image", "text", "naive_union", "joint") if method not in thresholds]
    if missing:
        raise ValueError(f"manual thresholds missing required methods: {missing}")
    return thresholds


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


def resolve_label_column(fieldnames: list[str], label_column: str) -> str:
    if label_column != "auto":
        if label_column not in fieldnames:
            raise ValueError(f"Label column {label_column!r} not found")
        return label_column
    if "final_label" in fieldnames:
        return "final_label"
    if "label" in fieldnames:
        return "label"
    raise ValueError("Annotation CSV must contain label or final_label")


def evaluate_method(
    rows: list[dict[str, Any]],
    method: str,
    threshold_config: dict[str, float | str],
    positive_labels: set[str],
) -> dict[str, Any]:
    if method == "naive_union":
        image_threshold = float(threshold_config["image_threshold"])
        text_threshold = float(threshold_config["text_threshold"])
        predictions = [
            float(row["image_similarity"]) >= image_threshold or float(row["text_similarity"]) >= text_threshold
            for row in rows
        ]
        threshold: float | str = ""
    else:
        threshold = float(threshold_config["threshold"])
        image_threshold = ""
        text_threshold = ""
        predictions = [float(row[SCORE_COLUMNS[method]]) >= threshold for row in rows]
    return metrics_from_predictions(
        rows,
        method=method,
        threshold=threshold,
        image_threshold=image_threshold,
        text_threshold=text_threshold,
        predictions=predictions,
        positive_labels=positive_labels,
        threshold_source=str(threshold_config["source"]),
    )


def metrics_from_predictions(
    rows: list[dict[str, Any]],
    method: str,
    threshold: float | str,
    image_threshold: float | str,
    text_threshold: float | str,
    predictions: list[bool],
    positive_labels: set[str],
    threshold_source: str,
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
    predicted_positive = tp + fp
    return {
        "method": method,
        "threshold": threshold,
        "image_threshold": image_threshold,
        "text_threshold": text_threshold,
        "threshold_source": threshold_source,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive": predicted_positive,
        "predicted_positive_rate": predicted_positive / len(rows) if rows else 0.0,
        "positive_labels": "|".join(sorted(positive_labels)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "method",
        "threshold",
        "image_threshold",
        "text_threshold",
        "threshold_source",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1",
        "predicted_positive",
        "predicted_positive_rate",
        "positive_labels",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _delta(by_method: dict[str, dict[str, Any]], left: str, right: str) -> float | None:
    if left not in by_method or right not in by_method:
        return None
    return float(by_method[left]["f1"]) - float(by_method[right]["f1"])


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def hardware_summary() -> str:
    return f"{platform.system()} {platform.machine()} | Python {platform.python_version()}"


if __name__ == "__main__":
    raise SystemExit(main())
