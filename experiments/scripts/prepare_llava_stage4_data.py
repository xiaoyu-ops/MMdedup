"""Convert Stage 4 training manifests into LLaVA-style instruction data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


DEFAULT_PROMPT = "Describe this image in detail."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--instruction", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        help="Path prefix replacement in FROM=TO form. Example: D:\\data=/mnt/d/data",
    )
    parser.add_argument("--limit", type=int, help="Optional row limit for smoke data.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON. Default is compact.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path_maps = parse_path_maps(args.path_map)
    records = list(iter_records(args.manifest_csv, args.instruction, path_maps, args.limit))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(records, ensure_ascii=False, indent=2 if args.pretty else None) + "\n",
        encoding="utf-8",
    )
    summary_path = args.output_json.with_suffix(".summary.json")
    summary = {
        "manifest_csv": str(args.manifest_csv),
        "output_json": str(args.output_json),
        "num_records": len(records),
        "instruction": args.instruction,
        "path_maps": [{"from": src, "to": dst} for src, dst in path_maps],
        "format": "LLaVA conversation JSON list",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def parse_path_maps(items: Iterable[str]) -> list[tuple[str, str]]:
    maps = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--path-map must be FROM=TO, got: {item}")
        src, dst = item.split("=", 1)
        if not src:
            raise ValueError(f"--path-map source cannot be empty: {item}")
        maps.append((src, dst))
    return maps


def iter_records(
    manifest_csv: Path,
    instruction: str,
    path_maps: list[tuple[str, str]],
    limit: int | None,
) -> Iterable[dict[str, object]]:
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"pair_id", "image_path", "caption"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"manifest must contain columns: {sorted(required)}")
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            caption = (row.get("caption") or "").strip()
            if not caption:
                continue
            yield {
                "id": row["pair_id"],
                "image": rewrite_path(row["image_path"], path_maps),
                "conversations": [
                    {"from": "human", "value": f"<image>\n{instruction}"},
                    {"from": "gpt", "value": caption},
                ],
            }


def rewrite_path(raw_path: str, path_maps: list[tuple[str, str]]) -> str:
    path = raw_path.strip()
    for src, dst in path_maps:
        if path.lower().startswith(src.lower()):
            path = dst + path[len(src) :]
            break
    return path.replace("\\", "/")


if __name__ == "__main__":
    raise SystemExit(main())
