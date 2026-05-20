"""Analyze Stage 4 and baseline errors on adjudicated pair-pair labels."""

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
from typing import Dict, Iterable, List

import yaml


POSITIVE_LABELS = {"duplicate", "near-duplicate"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--image-threshold", type=float, default=0.80)
    parser.add_argument("--text-threshold", type=float, default=0.60)
    parser.add_argument("--joint-threshold", type=float, default=0.85)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = args.output_dir / "stdout.log"
    stderr_path = args.output_dir / "stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rows = _read_rows(args.annotations_csv)
            analyzed = [analyze_row(row, args) for row in rows]
            summary = summarize(analyzed)
            write_csv(analyzed, args.output_dir / "error_cases.csv")
            write_csv([row for row in analyzed if row["joint_error_type"] == "fp"], args.output_dir / "joint_fp_examples.csv")
            write_csv([row for row in analyzed if row["joint_error_type"] == "fn"], args.output_dir / "joint_fn_examples.csv")
            write_csv(
                [row for row in analyzed if row["image_correct"] == "1" and row["joint_correct"] == "0"],
                args.output_dir / "image_wins_joint_loses.csv",
            )
            write_csv(
                [row for row in analyzed if row["joint_correct"] == "1" and row["image_correct"] == "0"],
                args.output_dir / "joint_wins_image_loses.csv",
            )
            config = {
                "experiment_id": args.experiment_id,
                "annotations_csv": str(args.annotations_csv),
                "image_threshold": args.image_threshold,
                "text_threshold": args.text_threshold,
                "joint_threshold": args.joint_threshold,
            }
            (args.output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            summary["elapsed_seconds"] = time.time() - started
            (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            manifest = {
                "experiment_id": args.experiment_id,
                "command": " ".join(sys.argv),
                "git_commit": git_commit(),
                "hardware": hardware_summary(),
                "wall_clock_seconds": time.time() - started,
                "outputs": {
                    "config": str(args.output_dir / "config.yaml"),
                    "metrics": str(args.output_dir / "metrics.json"),
                    "error_cases": str(args.output_dir / "error_cases.csv"),
                    "joint_fp_examples": str(args.output_dir / "joint_fp_examples.csv"),
                    "joint_fn_examples": str(args.output_dir / "joint_fn_examples.csv"),
                    "image_wins_joint_loses": str(args.output_dir / "image_wins_joint_loses.csv"),
                    "joint_wins_image_loses": str(args.output_dir / "joint_wins_image_loses.csv"),
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                },
            }
            (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(json.dumps(summary, indent=2), flush=True)
    return 0


def analyze_row(row: Dict[str, str], args: argparse.Namespace) -> Dict[str, str]:
    label = row.get("final_label") or row.get("label", "")
    is_positive = label in POSITIVE_LABELS
    image_pred = float(row["image_similarity"]) >= args.image_threshold
    text_pred = float(row["text_similarity"]) >= args.text_threshold
    naive_pred = image_pred or text_pred
    joint_pred = float(row["joint_similarity"]) >= args.joint_threshold
    enriched = dict(row)
    enriched.update(
        {
            "is_positive": "1" if is_positive else "0",
            "image_pred": "1" if image_pred else "0",
            "text_pred": "1" if text_pred else "0",
            "naive_pred": "1" if naive_pred else "0",
            "joint_pred": "1" if joint_pred else "0",
            "image_correct": "1" if image_pred == is_positive else "0",
            "text_correct": "1" if text_pred == is_positive else "0",
            "naive_correct": "1" if naive_pred == is_positive else "0",
            "joint_correct": "1" if joint_pred == is_positive else "0",
            "joint_error_type": error_type(joint_pred, is_positive),
            "image_error_type": error_type(image_pred, is_positive),
            "caption_equal": "1" if row.get("caption_a", "").strip() == row.get("caption_b", "").strip() else "0",
            "image_minus_joint": f"{float(row['image_similarity']) - float(row['joint_similarity']):.6f}",
            "text_minus_image": f"{float(row['text_similarity']) - float(row['image_similarity']):.6f}",
        }
    )
    return enriched


def error_type(pred_positive: bool, true_positive: bool) -> str:
    if pred_positive and not true_positive:
        return "fp"
    if not pred_positive and true_positive:
        return "fn"
    return ""


def summarize(rows: List[Dict[str, str]]) -> Dict[str, object]:
    def count(predicate) -> int:
        return sum(1 for row in rows if predicate(row))

    total = len(rows)
    positives = count(lambda row: row["is_positive"] == "1")
    negatives = total - positives
    joint_fp = count(lambda row: row["joint_error_type"] == "fp")
    joint_fn = count(lambda row: row["joint_error_type"] == "fn")
    image_fp = count(lambda row: row["image_error_type"] == "fp")
    image_fn = count(lambda row: row["image_error_type"] == "fn")
    image_wins = count(lambda row: row["image_correct"] == "1" and row["joint_correct"] == "0")
    joint_wins = count(lambda row: row["joint_correct"] == "1" and row["image_correct"] == "0")
    joint_fp_caption_equal = count(lambda row: row["joint_error_type"] == "fp" and row["caption_equal"] == "1")
    return {
        "num_rows": total,
        "num_positive": positives,
        "num_negative": negatives,
        "joint_false_positives": joint_fp,
        "joint_false_negatives": joint_fn,
        "image_false_positives": image_fp,
        "image_false_negatives": image_fn,
        "image_correct_joint_wrong": image_wins,
        "joint_correct_image_wrong": joint_wins,
        "joint_fp_caption_equal": joint_fp_caption_equal,
        "joint_fp_caption_equal_rate": joint_fp_caption_equal / joint_fp if joint_fp else 0.0,
    }


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: Iterable[Dict[str, str]], path: Path) -> None:
    rows = list(rows)
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
