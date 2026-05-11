"""Run Plan B Stage 4 image-caption pair-level deduplication."""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

from pipelines.stage4_pair_dedup import (
    Stage4Config,
    load_pairs_from_csv,
    load_pairs_from_sidecar_dir,
    run_stage4_pair_dedup,
    write_stage4_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, help="Directory with image files and same-stem .txt captions.")
    parser.add_argument("--pairs-csv", type=Path, help="CSV with pair_id,image_path,caption or caption_path.")
    parser.add_argument("--base-dir", type=Path, help="Base directory for relative CSV paths.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", default="auto", choices=["auto", "open_clip", "simple"])
    parser.add_argument("--model-name", default="hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K")
    parser.add_argument("--pretrained")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--joint-method", default="concat", choices=["concat", "weighted_sum"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--tau-cross", type=float, default=0.95)
    parser.add_argument("--image-quality", default="resolution", choices=["resolution", "file_size"])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-exact-pairs", type=int, default=2_000_000)
    parser.add_argument("--experiment-id", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if bool(args.input_dir) == bool(args.pairs_csv):
        raise SystemExit("Provide exactly one of --input-dir or --pairs-csv")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = args.output_dir / "stdout.log"
    stderr_path = args.output_dir / "stderr.log"
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            if args.input_dir:
                pairs = load_pairs_from_sidecar_dir(args.input_dir)
                dataset_ref = str(args.input_dir)
            else:
                pairs = load_pairs_from_csv(args.pairs_csv, base_dir=args.base_dir)
                dataset_ref = str(args.pairs_csv)

            config = Stage4Config(
                embedding_backend=args.backend,
                model_name=args.model_name,
                pretrained=args.pretrained,
                device=args.device,
                batch_size=args.batch_size,
                joint_method=args.joint_method,
                alpha=args.alpha,
                tau_cross=args.tau_cross,
                image_quality=args.image_quality,
                cache_dir=args.cache_dir,
                max_exact_pairs=args.max_exact_pairs,
            )
            (args.output_dir / "config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": args.experiment_id,
                        "dataset": dataset_ref,
                        "stage4": _config_to_dict(config),
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            print(f"Loaded {len(pairs)} image-caption pairs from {dataset_ref}", flush=True)
            result = run_stage4_pair_dedup(pairs, config)
            write_stage4_outputs(result, args.output_dir)
            manifest = {
                "experiment_id": args.experiment_id,
                "dataset": dataset_ref,
                "command": " ".join(sys.argv),
                "git_commit": _git_commit(),
                "hardware": _hardware_summary(),
                "wall_clock_seconds": time.time() - started,
                "outputs": {
                    "config": str(args.output_dir / "config.yaml"),
                    "metrics": str(args.output_dir / "metrics.json"),
                    "summary": str(args.output_dir / "stage4_pair_dedup_summary.json"),
                    "groups": str(args.output_dir / "stage4_duplicate_groups.json"),
                    "keepers": str(args.output_dir / "stage4_keepers.txt"),
                    "drops": str(args.output_dir / "stage4_drops.txt"),
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                },
            }
            (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(json.dumps(result.summary, indent=2), flush=True)
    return 0


def _config_to_dict(config: Stage4Config) -> dict:
    data = dict(config.__dict__)
    if data.get("cache_dir") is not None:
        data["cache_dir"] = str(data["cache_dir"])
    return data


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _hardware_summary() -> str:
    return f"{platform.system()} {platform.machine()} | Python {platform.python_version()}"


if __name__ == "__main__":
    raise SystemExit(main())

