"""Sample a fair Stage 4 annotation set from score-space buckets.

The sampler does not use labels or try to force a duplicate ratio. It only
stratifies candidate pair-pairs by image/text/joint similarity ranges, then
samples as evenly as possible across populated buckets.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


SCORE_COLUMNS = {
    "image": "image_similarity",
    "text": "text_similarity",
    "joint": "joint_similarity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--target-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-csv", type=Path, action="append", default=[])
    parser.add_argument("--id-column", default="candidate_id")
    parser.add_argument("--image-bins", default="-1.0,0.5,0.6,0.7,0.8,0.9,1.01")
    parser.add_argument("--text-bins", default="-1.0,0.5,0.6,0.7,0.8,0.9,1.01")
    parser.add_argument("--joint-bins", default="-1.0,0.3,0.5,0.6,0.7,0.8,0.9,1.01")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    image_bins = parse_bins(args.image_bins)
    text_bins = parse_bins(args.text_bins)
    joint_bins = parse_bins(args.joint_bins)
    excluded_ids = load_excluded_ids(args.exclude_csv, args.id_column)
    rows = read_candidates(args.candidates_csv)
    eligible = [row for row in rows if row.get(args.id_column, "") not in excluded_ids]

    buckets: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        key = (
            bucket_label(float(row["image_similarity"]), image_bins),
            bucket_label(float(row["text_similarity"]), text_bins),
            bucket_label(float(row["joint_similarity"]), joint_bins),
        )
        row = dict(row)
        row["image_bucket"] = key[0]
        row["text_bucket"] = key[1]
        row["joint_bucket"] = key[2]
        row["score_bucket"] = f"image={key[0]}|text={key[1]}|joint={key[2]}"
        buckets[key].append(row)

    sampled = round_robin_sample(buckets, args.target_size, rng)
    sampled.sort(key=lambda row: (row["joint_bucket"], row["image_bucket"], row["text_bucket"], row[args.id_column]))
    for idx, row in enumerate(sampled):
        row["annotation_id"] = f"fair_{idx:04d}"
        row["label"] = ""
        row["annotator"] = ""
        row["audit_label"] = ""
        row["needs_audit"] = "0"
        row["notes"] = ""
        row["label_options"] = "duplicate|near-duplicate|not-duplicate"
        row["annotation_guideline"] = (
            "Label from the image-caption pair semantics; sampling is score-stratified and label-agnostic. "
            "Score-assisted rule: if image_similarity and text_similarity are both >0.85 and <0.95, "
            "near-duplicate is acceptable; if both are >0.95, duplicate is acceptable."
        )

    write_csv(args.output_dir / "fair_eval_candidates_3000.csv", sampled)
    coverage_rows = bucket_coverage_rows(buckets, sampled)
    write_csv(args.output_dir / "bucket_coverage.csv", coverage_rows)

    metrics = {
        "experiment_id": args.experiment_id,
        "source_candidates": len(rows),
        "excluded_candidates": len(excluded_ids),
        "eligible_candidates": len(eligible),
        "target_size": args.target_size,
        "sampled_rows": len(sampled),
        "populated_buckets": len(buckets),
        "sampled_buckets": sum(1 for row in coverage_rows if int(row["sampled_count"]) > 0),
        "image_bins": image_bins,
        "text_bins": text_bins,
        "joint_bins": joint_bins,
        "seed": args.seed,
        "elapsed_seconds": time.time() - started,
        "outputs": {
            "candidates": str(args.output_dir / "fair_eval_candidates_3000.csv"),
            "bucket_coverage": str(args.output_dir / "bucket_coverage.csv"),
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment_id": args.experiment_id,
                "candidates_csv": str(args.candidates_csv),
                "target_size": args.target_size,
                "seed": args.seed,
                "exclude_csv": [str(path) for path in args.exclude_csv],
                "image_bins": image_bins,
                "text_bins": text_bins,
                "joint_bins": joint_bins,
                "note": "Label-agnostic score-space stratified sampling for fair Stage 4 evaluation.",
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "command": " ".join(sys.argv),
                "git_commit": git_commit(),
                "hardware": hardware_summary(),
                "wall_clock_seconds": time.time() - started,
                "outputs": metrics["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


def parse_bins(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) < 2:
        raise ValueError("At least two bin edges are required")
    if values != sorted(values):
        raise ValueError("Bin edges must be sorted")
    return values


def load_excluded_ids(paths: list[Path], id_column: str) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if id_column not in (reader.fieldnames or []):
                continue
            ids.update(row[id_column] for row in reader if row.get(id_column))
    return ids


def read_candidates(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = sorted(set(SCORE_COLUMNS.values()) - set(fieldnames))
        if missing:
            raise ValueError(f"Missing required score columns: {missing}")
        return list(reader)


def bucket_label(value: float, edges: list[float]) -> str:
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= value < hi:
            return f"[{lo:.2f},{hi:.2f})"
    if value == edges[-1]:
        return f"[{edges[-2]:.2f},{edges[-1]:.2f}]"
    return f"out_of_range:{value:.4f}"


def round_robin_sample(
    buckets: dict[tuple[str, str, str], list[dict[str, str]]],
    target_size: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    shuffled: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for key, rows in buckets.items():
        rows = list(rows)
        rng.shuffle(rows)
        shuffled[key] = rows

    keys = sorted(shuffled)
    selected: list[dict[str, str]] = []
    index = 0
    while keys and len(selected) < target_size:
        key = keys[index % len(keys)]
        rows = shuffled[key]
        if rows:
            selected.append(rows.pop())
        if not rows:
            keys.remove(key)
            if not keys:
                break
            index %= len(keys)
        else:
            index += 1
    return selected


def bucket_coverage_rows(
    buckets: dict[tuple[str, str, str], list[dict[str, str]]],
    sampled: list[dict[str, str]],
) -> list[dict[str, Any]]:
    sampled_counts: dict[str, int] = defaultdict(int)
    for row in sampled:
        sampled_counts[row["score_bucket"]] += 1
    output = []
    for key in sorted(buckets):
        bucket = f"image={key[0]}|text={key[1]}|joint={key[2]}"
        output.append(
            {
                "image_bucket": key[0],
                "text_bucket": key[1],
                "joint_bucket": key[2],
                "score_bucket": bucket,
                "available_count": len(buckets[key]),
                "sampled_count": sampled_counts.get(bucket, 0),
            }
        )
    return output


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
