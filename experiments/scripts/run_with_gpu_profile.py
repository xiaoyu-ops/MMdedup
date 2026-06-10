"""Run a command while sampling GPU memory and save a small profile JSON."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--sample-seconds", type=float, default=2.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def query_gpu_memory_mib() -> list[int]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        out = subprocess.check_output(
            [
                nvidia_smi,
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return []
    values: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(float(line)))
        except ValueError:
            continue
    return values


def main() -> int:
    args = parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise SystemExit("missing command to run")

    args.profile_json.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    peak_values = query_gpu_memory_mib()
    proc = subprocess.Popen(args.command)
    try:
        while proc.poll() is None:
            peak_values.extend(query_gpu_memory_mib())
            time.sleep(max(0.1, args.sample_seconds))
    finally:
        returncode = proc.wait()
    peak_values.extend(query_gpu_memory_mib())

    profile = {
        "command": args.command,
        "started_at_unix": started,
        "ended_at_unix": time.time(),
        "wall_clock_seconds": time.time() - started,
        "returncode": returncode,
        "gpu_peak_memory_mib": max(peak_values) if peak_values else None,
        "gpu_peak_memory_gib": (max(peak_values) / 1024.0) if peak_values else None,
        "sample_seconds": args.sample_seconds,
    }
    args.profile_json.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(profile, indent=2, ensure_ascii=False))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
