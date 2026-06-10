"""Filter Stage 4 candidate rows by similarity thresholds."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-image", type=float, default=None)
    parser.add_argument("--min-text", type=float, default=None)
    parser.add_argument("--min-joint", type=float, default=None)
    parser.add_argument("--experiment-id", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "stage4_candidate_pairs.csv"
    total = 0
    kept = 0
    with args.candidates_csv.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError("Candidate CSV has no header")
        with output_csv.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                total += 1
                if _keep(row, args):
                    writer.writerow(row)
                    kept += 1
    metrics = {
        "experiment_id": args.experiment_id,
        "source": str(args.candidates_csv),
        "output": str(output_csv),
        "min_image": args.min_image,
        "min_text": args.min_text,
        "min_joint": args.min_joint,
        "num_candidates_input": total,
        "num_candidates": kept,
    }
    elapsed = time.perf_counter() - started_at
    config_text = "\n".join(
        [
            f"experiment_id: {args.experiment_id}",
            f"source_candidates_csv: {args.candidates_csv}",
            f"output_dir: {args.output_dir}",
            f"min_image: {_yaml_scalar(args.min_image)}",
            f"min_text: {_yaml_scalar(args.min_text)}",
            f"min_joint: {_yaml_scalar(args.min_joint)}",
            "",
        ]
    )
    (args.output_dir / "config.yaml").write_text(config_text, encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    run_manifest = {
        "experiment_id": args.experiment_id,
        "command": " ".join(
            [
                "experiments/scripts/filter_stage4_candidates.py",
                "--candidates-csv",
                str(args.candidates_csv),
                "--output-dir",
                str(args.output_dir),
                *(_arg_pair("--min-image", args.min_image)),
                *(_arg_pair("--min-text", args.min_text)),
                *(_arg_pair("--min-joint", args.min_joint)),
                "--experiment-id",
                args.experiment_id,
            ]
        ),
        "git_commit": _git_commit(),
        "hardware": f"{platform.system()} {platform.machine()} | Python {platform.python_version()}",
        "wall_clock_seconds": elapsed,
        "outputs": {
            "config": str(args.output_dir / "config.yaml"),
            "metrics": str(args.output_dir / "metrics.json"),
            "candidates": str(output_csv),
            "stdout": str(args.output_dir / "stdout.log"),
            "stderr": str(args.output_dir / "stderr.log"),
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "stderr.log").touch()
    print(json.dumps(metrics, indent=2), flush=True)
    (args.output_dir / "stdout.log").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


def _keep(row: dict[str, str], args: argparse.Namespace) -> bool:
    checks = [
        ("image_similarity", args.min_image),
        ("text_similarity", args.min_text),
        ("joint_similarity", args.min_joint),
    ]
    for column, threshold in checks:
        if threshold is None:
            continue
        if float(row[column]) < threshold:
            return False
    return True


def _arg_pair(flag: str, value: float | None) -> list[str]:
    if value is None:
        return []
    return [flag, str(value)]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _yaml_scalar(value: float | None) -> str:
    if value is None:
        return "null"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
