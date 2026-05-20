"""Build Plan B split-size and threshold dedup-rate data from Stage 4 candidates."""

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
from typing import Dict, Iterable, List, Sequence, Tuple

import yaml


class DSU:
    def __init__(self, nodes: Iterable[str]) -> None:
        self.parent = {node: node for node in nodes}
        self.size = {node: 1 for node in nodes}

    def find(self, node: str) -> str:
        parent = self.parent.setdefault(node, node)
        self.size.setdefault(node, 1)
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, a: str, b: str) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True

    def drops(self) -> int:
        roots = {self.find(node) for node in self.parent}
        return len(self.parent) - len(roots)

    def components(self) -> Dict[str, List[str]]:
        output: Dict[str, List[str]] = {}
        for node in self.parent:
            output.setdefault(self.find(node), []).append(node)
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--num-pairs", type=int, default=200000)
    parser.add_argument("--thresholds", default="0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--image-threshold", type=float, default=0.80)
    parser.add_argument("--text-threshold", type=float, default=0.60)
    parser.add_argument("--joint-threshold", type=float, default=0.85)
    parser.add_argument("--pair-id-prefix", default="cc3m_")
    parser.add_argument(
        "--write-split-manifests",
        action="store_true",
        help="Write A/B/C/D/E training_manifest.csv, keepers.txt, drops.txt, and duplicate_groups.json.",
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
            rows = read_candidate_edges(args.candidates_csv)
            thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
            pair_ids = infer_pair_ids(rows, args.num_pairs, args.pair_id_prefix, args.manifest_csv)
            scoped_rows, skipped_edges = filter_edges_to_pair_ids(rows, set(pair_ids))
            threshold_rows = build_threshold_rows(scoped_rows, pair_ids, thresholds)
            split_rows = build_split_rows(scoped_rows, pair_ids, args)
            split_outputs = {}
            if args.write_split_manifests:
                manifest_rows = read_manifest_rows(args.manifest_csv) if args.manifest_csv else {}
                split_outputs = write_split_manifests(scoped_rows, pair_ids, manifest_rows, args)
            write_csv(threshold_rows, args.output_dir / "threshold_dedup_rates.csv")
            write_csv(split_rows, args.output_dir / "abcde_split_sizes.csv")
            metrics = summarize(threshold_rows, split_rows, len(rows), len(pair_ids))
            metrics["split_manifests_written"] = bool(args.write_split_manifests)
            metrics["split_outputs"] = split_outputs
            metrics["elapsed_seconds"] = time.time() - started
            config = {
                "experiment_id": args.experiment_id,
                "candidates_csv": str(args.candidates_csv),
                "manifest_csv": str(args.manifest_csv) if args.manifest_csv else None,
                "num_pairs": args.num_pairs,
                "thresholds": thresholds,
                "image_threshold": args.image_threshold,
                "text_threshold": args.text_threshold,
                "joint_threshold": args.joint_threshold,
                "candidate_edges_total": len(rows),
                "candidate_edges_in_manifest": len(scoped_rows),
                "candidate_edges_skipped": skipped_edges,
                "write_split_manifests": bool(args.write_split_manifests),
                "note": "Dedup rates are computed as graph-component drops over mined candidate edges scoped to the manifest/pair universe.",
            }
            (args.output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            manifest = {
                "experiment_id": args.experiment_id,
                "command": " ".join(sys.argv),
                "git_commit": git_commit(),
                "hardware": hardware_summary(),
                "wall_clock_seconds": time.time() - started,
                "outputs": {
                    "config": str(args.output_dir / "config.yaml"),
                    "metrics": str(args.output_dir / "metrics.json"),
                    "threshold_dedup_rates": str(args.output_dir / "threshold_dedup_rates.csv"),
                    "abcde_split_sizes": str(args.output_dir / "abcde_split_sizes.csv"),
                    "split_outputs": split_outputs,
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                },
            }
            (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(json.dumps(metrics, indent=2), flush=True)
    return 0


def read_candidate_edges(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_manifest_rows(path: Path | None) -> Dict[str, Dict[str, str]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "pair_id" not in (reader.fieldnames or []):
            raise ValueError(f"manifest_csv must contain a pair_id column: {path}")
        return {row["pair_id"]: row for row in reader}


def infer_pair_ids(rows: Sequence[Dict[str, str]], num_pairs: int, prefix: str, manifest_csv: Path | None) -> List[str]:
    if manifest_csv:
        with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if "pair_id" not in (reader.fieldnames or []):
                raise ValueError(f"manifest_csv must contain a pair_id column: {manifest_csv}")
            return [row["pair_id"] for row in reader]
    ids = set()
    for row in rows:
        ids.add(row["pair_id_a"])
        ids.add(row["pair_id_b"])
    if prefix and len(ids) < num_pairs:
        for idx in range(num_pairs):
            ids.add(f"{prefix}{idx}")
    return sorted(ids)


def filter_edges_to_pair_ids(rows: Sequence[Dict[str, str]], pair_ids: set[str]) -> Tuple[List[Dict[str, str]], int]:
    scoped = []
    skipped = 0
    for row in rows:
        if row["pair_id_a"] in pair_ids and row["pair_id_b"] in pair_ids:
            scoped.append(row)
        else:
            skipped += 1
    return scoped, skipped


def build_threshold_rows(
    rows: Sequence[Dict[str, str]],
    pair_ids: Sequence[str],
    thresholds: Sequence[float],
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for score_name, column in [
        ("image", "image_similarity"),
        ("text", "text_similarity"),
        ("joint", "joint_similarity"),
        ("max", "max_similarity"),
    ]:
        for threshold in thresholds:
            output.append(rate_row(rows, pair_ids, score_name, threshold, [(column, threshold)]))
    for threshold in thresholds:
        output.append(
            rate_row(
                rows,
                pair_ids,
                "naive_union",
                threshold,
                [("image_similarity", threshold), ("text_similarity", threshold)],
                mode="any",
            )
        )
    return output


def build_split_rows(rows: Sequence[Dict[str, str]], pair_ids: Sequence[str], args: argparse.Namespace) -> List[Dict[str, object]]:
    configs = [
        ("A", "raw", "原始数据，不去重", []),
        ("B", "image_only", "仅图像去重", [("image_similarity", args.image_threshold)]),
        ("C", "text_only", "仅文本去重", [("text_similarity", args.text_threshold)]),
        (
            "D",
            "naive_union",
            "图像 + 文本独立去重并集",
            [("image_similarity", args.image_threshold), ("text_similarity", args.text_threshold)],
        ),
        ("E", "stage4_joint", "Stage 4 跨模态联合去重", [("joint_similarity", args.joint_threshold)]),
    ]
    output = []
    for split, name, desc, checks in configs:
        mode = "any" if name == "naive_union" else "all"
        threshold_label = threshold_label_for(name, args)
        if not checks:
            drops = 0
            edges = 0
        else:
            drops, edges = graph_drops(rows, pair_ids, checks, mode=mode)
        kept = len(pair_ids) - drops
        output.append(
            {
                "split": split,
                "name": name,
                "description": desc,
                "raw_pairs": len(pair_ids),
                "kept_pairs": kept,
                "dropped_pairs": drops,
                "dedup_rate": drops / len(pair_ids) if pair_ids else 0.0,
                "selected_candidate_edges": edges,
                "threshold": threshold_label,
            }
        )
    return output


def split_configs(args: argparse.Namespace) -> List[Tuple[str, str, str, Sequence[Tuple[str, float]], str]]:
    return [
        ("A", "raw", "原始数据，不去重", [], "all"),
        ("B", "image_only", "仅图像去重", [("image_similarity", args.image_threshold)], "all"),
        ("C", "text_only", "仅文本去重", [("text_similarity", args.text_threshold)], "all"),
        (
            "D",
            "naive_union",
            "图像 + 文本独立去重并集",
            [("image_similarity", args.image_threshold), ("text_similarity", args.text_threshold)],
            "any",
        ),
        ("E", "stage4_joint", "Stage 4 跨模态联合去重", [("joint_similarity", args.joint_threshold)], "all"),
    ]


def write_split_manifests(
    rows: Sequence[Dict[str, str]],
    pair_ids: Sequence[str],
    manifest_rows: Dict[str, Dict[str, str]],
    args: argparse.Namespace,
) -> Dict[str, Dict[str, object]]:
    pair_order = {pair_id: idx for idx, pair_id in enumerate(pair_ids)}
    outputs: Dict[str, Dict[str, object]] = {}
    for split, name, desc, checks, mode in split_configs(args):
        split_dir = args.output_dir / "splits" / f"{split}_{name}"
        split_dir.mkdir(parents=True, exist_ok=True)
        components, selected_edges = graph_components(rows, pair_ids, checks, mode=mode)
        keepers, drops, groups = choose_component_outputs(components, pair_order)
        write_pair_ids(split_dir / "keepers.txt", keepers)
        write_pair_ids(split_dir / "drops.txt", drops)
        write_training_manifest(split_dir / "training_manifest.csv", keepers, manifest_rows)
        (split_dir / "duplicate_groups.json").write_text(json.dumps(groups, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = {
            "split": split,
            "name": name,
            "description": desc,
            "raw_pairs": len(pair_ids),
            "kept_pairs": len(keepers),
            "dropped_pairs": len(drops),
            "dedup_rate": len(drops) / len(pair_ids) if pair_ids else 0.0,
            "selected_candidate_edges": selected_edges,
            "threshold": threshold_label_for(name, args),
            "keeper_rule": "first pair_id by manifest order within each connected component",
            "note": (
                "Training manifest materialized from mined candidate graph components. "
                "For final Stage 4 quality tie-breaking, rerun with alignment/quality scores if available."
            ),
        }
        (split_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        outputs[split] = {
            "name": name,
            "training_manifest": str(split_dir / "training_manifest.csv"),
            "keepers": str(split_dir / "keepers.txt"),
            "drops": str(split_dir / "drops.txt"),
            "duplicate_groups": str(split_dir / "duplicate_groups.json"),
            "summary": str(split_dir / "summary.json"),
            "kept_pairs": len(keepers),
            "dropped_pairs": len(drops),
        }
    return outputs


def graph_components(
    rows: Sequence[Dict[str, str]],
    pair_ids: Sequence[str],
    checks: Sequence[Tuple[str, float]],
    mode: str,
) -> Tuple[Dict[str, List[str]], int]:
    dsu = DSU(pair_ids)
    edges = 0
    if not checks:
        return {pair_id: [pair_id] for pair_id in pair_ids}, edges
    for row in rows:
        results = [float(row[column]) >= threshold for column, threshold in checks]
        selected = any(results) if mode == "any" else all(results)
        if not selected:
            continue
        edges += 1
        dsu.union(row["pair_id_a"], row["pair_id_b"])
    return dsu.components(), edges


def choose_component_outputs(
    components: Dict[str, List[str]],
    pair_order: Dict[str, int],
) -> Tuple[List[str], List[str], List[Dict[str, object]]]:
    keepers: List[str] = []
    drops: List[str] = []
    groups: List[Dict[str, object]] = []
    for members in components.values():
        ordered = sorted(members, key=lambda pair_id: (pair_order.get(pair_id, 10**18), pair_id))
        keeper = ordered[0]
        duplicate_ids = ordered[1:]
        keepers.append(keeper)
        drops.extend(duplicate_ids)
        if duplicate_ids:
            groups.append({"keeper": keeper, "duplicates": duplicate_ids})
    return sorted(keepers, key=lambda pair_id: (pair_order.get(pair_id, 10**18), pair_id)), sorted(
        drops, key=lambda pair_id: (pair_order.get(pair_id, 10**18), pair_id)
    ), sorted(groups, key=lambda item: (pair_order.get(str(item["keeper"]), 10**18), str(item["keeper"])))


def write_pair_ids(path: Path, pair_ids: Sequence[str]) -> None:
    path.write_text("\n".join(pair_ids) + ("\n" if pair_ids else ""), encoding="utf-8")


def write_training_manifest(path: Path, keepers: Sequence[str], manifest_rows: Dict[str, Dict[str, str]]) -> None:
    if manifest_rows:
        first_row = next(iter(manifest_rows.values()))
        fieldnames = list(first_row.keys())
    else:
        fieldnames = ["pair_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair_id in keepers:
            writer.writerow(manifest_rows.get(pair_id, {"pair_id": pair_id}))


def threshold_label_for(name: str, args: argparse.Namespace) -> str:
    if name == "image_only":
        return f"image>={args.image_threshold}"
    if name == "text_only":
        return f"text>={args.text_threshold}"
    if name == "naive_union":
        return f"image>={args.image_threshold} OR text>={args.text_threshold}"
    if name == "stage4_joint":
        return f"joint>={args.joint_threshold}"
    return "n/a"


def rate_row(
    rows: Sequence[Dict[str, str]],
    pair_ids: Sequence[str],
    score_name: str,
    threshold: float,
    checks: Sequence[Tuple[str, float]],
    mode: str = "all",
) -> Dict[str, object]:
    drops, edges = graph_drops(rows, pair_ids, checks, mode=mode)
    return {
        "score": score_name,
        "threshold": threshold,
        "raw_pairs": len(pair_ids),
        "selected_candidate_edges": edges,
        "dropped_pairs": drops,
        "kept_pairs": len(pair_ids) - drops,
        "dedup_rate": drops / len(pair_ids) if pair_ids else 0.0,
    }


def graph_drops(
    rows: Sequence[Dict[str, str]],
    pair_ids: Sequence[str],
    checks: Sequence[Tuple[str, float]],
    mode: str,
) -> Tuple[int, int]:
    dsu = DSU(pair_ids)
    edges = 0
    for row in rows:
        results = [float(row[column]) >= threshold for column, threshold in checks]
        selected = any(results) if mode == "any" else all(results)
        if not selected:
            continue
        edges += 1
        dsu.union(row["pair_id_a"], row["pair_id_b"])
    return dsu.drops(), edges


def summarize(threshold_rows: Sequence[Dict[str, object]], split_rows: Sequence[Dict[str, object]], num_candidates: int, num_pairs: int) -> Dict[str, object]:
    return {
        "num_pairs": num_pairs,
        "num_candidates": num_candidates,
        "best_known_split_sizes": {row["split"]: row for row in split_rows},
        "threshold_rows": len(threshold_rows),
        "note": "Rates are graph-component drop rates over mined candidate edges scoped to the manifest/pair universe, not final LLaVA training results.",
    }


def write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
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
