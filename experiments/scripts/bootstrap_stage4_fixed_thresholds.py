"""Bootstrap confidence intervals for fixed-threshold Stage 4 fair evaluation."""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


METHOD_ORDER = ["image", "text", "naive_union", "joint", "max"]
DEFAULT_DELTAS = [("joint", "naive_union"), ("joint", "image")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", type=Path, required=True)
    parser.add_argument("--base-metrics-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--label-column", default="auto")
    parser.add_argument("--positive-labels", default="duplicate,near-duplicate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = args.output_dir / "stdout.log"
    stderr_path = args.output_dir / "stderr.log"

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            helper = load_eval_helper()
            rows = helper.read_labeled_rows(args.annotations_csv, args.label_column)
            positive_labels = {item.strip() for item in args.positive_labels.split(",") if item.strip()}
            base_metrics = json.loads(args.base_metrics_json.read_text(encoding="utf-8"))
            thresholds = base_metrics["fixed_thresholds"]

            rng = random.Random(args.seed)
            per_method_samples: dict[str, dict[str, list[float]]] = {}
            delta_samples: dict[str, list[float]] = {delta_key(left, right): [] for left, right in DEFAULT_DELTAS}
            progress_every = max(1, args.iterations // 10)

            for iteration_idx in range(args.iterations):
                sample_rows = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
                method_rows: dict[str, dict[str, Any]] = {}
                for method in METHOD_ORDER:
                    if method not in thresholds:
                        continue
                    row = helper.evaluate_method(sample_rows, method, thresholds[method], positive_labels)
                    method_rows[method] = row
                    metric_bucket = per_method_samples.setdefault(
                        method,
                        {
                            "precision": [],
                            "recall": [],
                            "f1": [],
                            "predicted_positive_rate": [],
                        },
                    )
                    for metric_name in metric_bucket:
                        metric_bucket[metric_name].append(float(row[metric_name]))

                for left, right in DEFAULT_DELTAS:
                    if left in method_rows and right in method_rows:
                        delta_samples[delta_key(left, right)].append(
                            float(method_rows[left]["f1"]) - float(method_rows[right]["f1"])
                        )

                if (iteration_idx + 1) % progress_every == 0 or iteration_idx + 1 == args.iterations:
                    print(
                        f"[bootstrap] completed {iteration_idx + 1}/{args.iterations} iterations",
                        flush=True,
                    )

            base_by_method = base_metrics["results_by_method"]
            ci_rows = []
            ci_by_method: dict[str, dict[str, Any]] = {}
            for method in METHOD_ORDER:
                if method not in per_method_samples or method not in base_by_method:
                    continue
                row = build_ci_row(method, base_by_method[method], per_method_samples[method])
                ci_rows.append(row)
                ci_by_method[method] = row
            write_ci_csv(args.output_dir / "bootstrap_ci_by_method.csv", ci_rows)

            delta_rows = []
            delta_summary: dict[str, dict[str, Any]] = {}
            for left, right in DEFAULT_DELTAS:
                key = delta_key(left, right)
                if key not in delta_samples or left not in base_by_method or right not in base_by_method:
                    continue
                point_estimate = float(base_by_method[left]["f1"]) - float(base_by_method[right]["f1"])
                row = build_delta_row(left, right, point_estimate, delta_samples[key])
                delta_rows.append(row)
                delta_summary[key] = row
            write_delta_csv(args.output_dir / "bootstrap_ci_f1_deltas.csv", delta_rows)

            metrics = {
                "experiment_id": args.experiment_id,
                "analysis_type": "bootstrap_ci_fixed_threshold_main_eval",
                "base_metrics_json": str(args.base_metrics_json),
                "annotations_csv": str(args.annotations_csv),
                "num_rows": len(rows),
                "num_positive": sum(1 for row in rows if row["is_positive"]),
                "num_negative": sum(1 for row in rows if not row["is_positive"]),
                "iterations": args.iterations,
                "seed": args.seed,
                "fixed_thresholds": thresholds,
                "confidence_level": 0.95,
                "bootstrap_method": "nonparametric row bootstrap with replacement over the 3000-row stratified fair evaluation set",
                "results_by_method": ci_by_method,
                "f1_delta_ci": delta_summary,
                "elapsed_seconds": time.time() - started,
                "notes": (
                    "Confidence intervals are estimated on the same score-space stratified 3000-row fair evaluation set "
                    "used for the main Stage 4 comparison. These CIs support method comparison, not population-rate extrapolation."
                ),
                "outputs": {
                    "bootstrap_ci_by_method": str(args.output_dir / "bootstrap_ci_by_method.csv"),
                    "bootstrap_ci_f1_deltas": str(args.output_dir / "bootstrap_ci_f1_deltas.csv"),
                },
            }
            (args.output_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            config = {
                "experiment_id": args.experiment_id,
                "annotations_csv": str(args.annotations_csv),
                "base_metrics_json": str(args.base_metrics_json),
                "iterations": args.iterations,
                "seed": args.seed,
                "label_column": args.label_column,
                "positive_labels": sorted(positive_labels),
                "delta_pairs": [f"{left}-minus-{right}" for left, right in DEFAULT_DELTAS],
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
                    "bootstrap_ci_by_method": str(args.output_dir / "bootstrap_ci_by_method.csv"),
                    "bootstrap_ci_f1_deltas": str(args.output_dir / "bootstrap_ci_f1_deltas.csv"),
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


def load_eval_helper():
    script_path = Path(__file__).with_name("evaluate_stage4_fixed_thresholds.py")
    spec = importlib.util.spec_from_file_location("stage4_fixed_eval_helper", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_distribution(samples: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(samples) if samples else 0.0,
        "ci95_low": percentile(samples, 0.025),
        "ci95_high": percentile(samples, 0.975),
    }


def build_ci_row(method: str, point_estimate_row: dict[str, Any], samples: dict[str, list[float]]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": method,
        "threshold": point_estimate_row.get("threshold", ""),
        "image_threshold": point_estimate_row.get("image_threshold", ""),
        "text_threshold": point_estimate_row.get("text_threshold", ""),
        "threshold_source": point_estimate_row.get("threshold_source", ""),
        "point_precision": float(point_estimate_row["precision"]),
        "point_recall": float(point_estimate_row["recall"]),
        "point_f1": float(point_estimate_row["f1"]),
        "point_predicted_positive_rate": float(point_estimate_row["predicted_positive_rate"]),
        "bootstrap_iterations": len(samples["f1"]),
    }
    for metric_name in ("precision", "recall", "f1", "predicted_positive_rate"):
        summary = summarize_distribution(samples[metric_name])
        row[f"{metric_name}_mean"] = summary["mean"]
        row[f"{metric_name}_ci95_low"] = summary["ci95_low"]
        row[f"{metric_name}_ci95_high"] = summary["ci95_high"]
    return row


def delta_key(left: str, right: str) -> str:
    return f"{left}_minus_{right}"


def build_delta_row(left: str, right: str, point_estimate: float, samples: list[float]) -> dict[str, Any]:
    summary = summarize_distribution(samples)
    return {
        "comparison": delta_key(left, right),
        "left_method": left,
        "right_method": right,
        "metric": "f1",
        "point_delta": point_estimate,
        "bootstrap_mean_delta": summary["mean"],
        "ci95_low": summary["ci95_low"],
        "ci95_high": summary["ci95_high"],
        "bootstrap_iterations": len(samples),
    }


def write_ci_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "method",
        "threshold",
        "image_threshold",
        "text_threshold",
        "threshold_source",
        "point_precision",
        "precision_mean",
        "precision_ci95_low",
        "precision_ci95_high",
        "point_recall",
        "recall_mean",
        "recall_ci95_low",
        "recall_ci95_high",
        "point_f1",
        "f1_mean",
        "f1_ci95_low",
        "f1_ci95_high",
        "point_predicted_positive_rate",
        "predicted_positive_rate_mean",
        "predicted_positive_rate_ci95_low",
        "predicted_positive_rate_ci95_high",
        "bootstrap_iterations",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_delta_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "comparison",
        "left_method",
        "right_method",
        "metric",
        "point_delta",
        "bootstrap_mean_delta",
        "ci95_low",
        "ci95_high",
        "bootstrap_iterations",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
