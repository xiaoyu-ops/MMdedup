"""Adjudicate primary and collaborator audit labels for Stage 4 annotations."""

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
from typing import Dict, List

import yaml


VALID_LABELS = {"duplicate", "near-duplicate", "not-duplicate"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", default="")
    parser.add_argument(
        "--conflict-policy",
        default="mark",
        choices=["mark", "audit_wins", "primary_wins"],
        help="How to fill final_label when label and audit_label disagree.",
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
            rows = _read_csv(args.annotations_csv)
            adjudicated, metrics = adjudicate(rows, conflict_policy=args.conflict_policy)
            output_csv = args.output_dir / "adjudicated_annotations.csv"
            _write_csv(adjudicated, output_csv)
            config = {
                "experiment_id": args.experiment_id,
                "annotations_csv": str(args.annotations_csv),
                "conflict_policy": args.conflict_policy,
            }
            (args.output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            metrics["elapsed_seconds"] = time.time() - started
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
                    "adjudicated_annotations": str(output_csv),
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                },
            }
            (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(json.dumps(metrics, indent=2), flush=True)
    return 0


def adjudicate(rows: List[Dict[str, str]], conflict_policy: str) -> tuple[List[Dict[str, str]], Dict[str, object]]:
    total_labeled = 0
    audited = 0
    agreements = 0
    conflicts = 0
    needs_manual_resolution = 0
    output: List[Dict[str, str]] = []
    for row in rows:
        primary = _normalize_label(row.get("label", ""))
        audit = _normalize_label(row.get("audit_label", ""))
        final_label = primary
        status = "not_audited"
        if primary:
            total_labeled += 1
        if audit:
            audited += 1
            if primary == audit:
                agreements += 1
                status = "agreed"
            else:
                conflicts += 1
                status = "conflict"
                if conflict_policy == "audit_wins":
                    final_label = audit
                elif conflict_policy == "primary_wins":
                    final_label = primary
                else:
                    final_label = ""
                    needs_manual_resolution += 1
        row = dict(row)
        row["final_label"] = final_label
        row["adjudication_status"] = status
        output.append(row)

    agreement_rate = agreements / audited if audited else None
    return output, {
        "num_rows": len(rows),
        "num_labeled_rows": total_labeled,
        "num_audited_rows": audited,
        "num_agreements": agreements,
        "num_conflicts": conflicts,
        "agreement_rate": agreement_rate,
        "needs_manual_resolution": needs_manual_resolution,
        "conflict_policy": conflict_policy,
    }


def _normalize_label(raw: str) -> str:
    label = (raw or "").strip().lower().replace("_", "-")
    if not label:
        return ""
    if label not in VALID_LABELS:
        raise ValueError(f"Unsupported label {raw!r}; valid labels are {sorted(VALID_LABELS)}")
    return label


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _hardware_summary() -> str:
    return f"{platform.system()} {platform.machine()} | Python {platform.python_version()}"


if __name__ == "__main__":
    raise SystemExit(main())

