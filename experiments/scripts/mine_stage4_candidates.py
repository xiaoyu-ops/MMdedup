"""Mine candidate image-caption pair-pairs for Stage 4 annotation/evaluation."""

from __future__ import annotations

import argparse
import contextlib
import csv
import heapq
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from pipelines.stage4_pair_dedup import (
    PairRecord,
    Stage4Config,
    build_joint_embeddings,
    encode_pairs,
    load_pairs_from_csv,
    load_pairs_from_sidecar_dir,
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
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--method", default="auto", choices=["auto", "exact", "sklearn", "torch"])
    parser.add_argument("--signals", default="image,text,joint", help="Comma-separated subset of image,text,joint.")
    parser.add_argument("--top-k", type=int, default=20, help="Neighbors retained per pair for each signal.")
    parser.add_argument("--min-similarity", type=float, default=0.0)
    parser.add_argument("--max-exact-pairs", type=int, default=2_000_000)
    parser.add_argument("--max-candidates", type=int, default=5000)
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

            signals = _parse_signals(args.signals)
            config = Stage4Config(
                embedding_backend=args.backend,
                model_name=args.model_name,
                pretrained=args.pretrained,
                device=args.device,
                batch_size=args.batch_size,
                joint_method=args.joint_method,
                alpha=args.alpha,
                cache_dir=args.cache_dir,
            )
            run_config = {
                "experiment_id": args.experiment_id,
                "dataset": dataset_ref,
                "candidate_mining": {
                    "signals": signals,
                    "method": args.method,
                    "top_k": args.top_k,
                    "min_similarity": args.min_similarity,
                    "max_exact_pairs": args.max_exact_pairs,
                    "max_candidates": args.max_candidates,
                },
                "stage4": _config_to_dict(config),
            }
            (args.output_dir / "config.yaml").write_text(
                yaml.safe_dump(run_config, sort_keys=False),
                encoding="utf-8",
            )

            print(f"Loaded {len(pairs)} image-caption pairs from {dataset_ref}", flush=True)
            image_emb, text_emb, backend = encode_pairs(pairs, config)
            joint_emb = build_joint_embeddings(image_emb, text_emb, config)
            candidates = mine_candidates(
                pairs=pairs,
                image_emb=image_emb,
                text_emb=text_emb,
                joint_emb=joint_emb,
                signals=signals,
                method=args.method,
                top_k=args.top_k,
                min_similarity=args.min_similarity,
                max_exact_pairs=args.max_exact_pairs,
                max_candidates=args.max_candidates,
            )
            candidate_path = args.output_dir / "stage4_candidate_pairs.csv"
            write_candidates_csv(candidates, candidate_path)
            metrics = {
                "num_pairs": len(pairs),
                "num_candidates": len(candidates),
                "signals": signals,
                "method": args.method,
                "top_k": args.top_k,
                "min_similarity": args.min_similarity,
                "embedding_backend": backend,
                "elapsed_seconds": time.time() - started,
            }
            (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
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
                    "candidates": str(candidate_path),
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                },
            }
            (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(json.dumps(metrics, indent=2), flush=True)
    return 0


def mine_candidates(
    pairs: Sequence[PairRecord],
    image_emb: np.ndarray,
    text_emb: np.ndarray,
    joint_emb: np.ndarray,
    signals: Sequence[str],
    method: str,
    top_k: int,
    min_similarity: float,
    max_exact_pairs: int,
    max_candidates: int,
) -> List[Dict[str, object]]:
    arrays = {
        "image": _l2_normalize(image_emb),
        "text": _l2_normalize(text_emb),
        "joint": _l2_normalize(joint_emb),
    }
    candidate_keys: Dict[Tuple[int, int], set[str]] = {}
    for signal in signals:
        keys = _candidate_keys_for_signal(
            arrays[signal],
            signal=signal,
            method=method,
            top_k=top_k,
            min_similarity=min_similarity,
            max_exact_pairs=max_exact_pairs,
        )
        for i, j in keys:
            candidate_keys.setdefault((i, j), set()).add(signal)

    selected: list[tuple[float, int, Dict[str, object]]] = []
    for row_index, (i, j) in enumerate(candidate_keys):
        image_similarity = float(np.dot(arrays["image"][i], arrays["image"][j]))
        text_similarity = float(np.dot(arrays["text"][i], arrays["text"][j]))
        joint_similarity = float(np.dot(arrays["joint"][i], arrays["joint"][j]))
        max_similarity = max(image_similarity, text_similarity, joint_similarity)
        row = {
            "candidate_id": f"cand_{row_index:06d}",
            "pair_id_a": pairs[i].pair_id,
            "pair_id_b": pairs[j].pair_id,
            "image_path_a": str(pairs[i].image_path),
            "image_path_b": str(pairs[j].image_path),
            "caption_a": pairs[i].caption,
            "caption_b": pairs[j].caption,
            "image_similarity": image_similarity,
            "text_similarity": text_similarity,
            "joint_similarity": joint_similarity,
            "max_similarity": max_similarity,
            "signals": "|".join(sorted(candidate_keys[(i, j)])),
            "bucket": _bucket(max_similarity),
            "label": "",
            "annotator": "",
            "audit_label": "",
            "notes": "",
        }
        if max_candidates <= 0:
            continue
        item = (max_similarity, row_index, row)
        if len(selected) < max_candidates:
            heapq.heappush(selected, item)
        elif item > selected[0]:
            heapq.heapreplace(selected, item)
    rows = [item[2] for item in selected]
    rows.sort(key=lambda row: (float(row["max_similarity"]), row["candidate_id"]), reverse=True)
    for idx, row in enumerate(rows):
        row["candidate_id"] = f"cand_{idx:06d}"
    return rows


def write_candidates_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    fieldnames = [
        "candidate_id",
        "pair_id_a",
        "pair_id_b",
        "image_path_a",
        "image_path_b",
        "caption_a",
        "caption_b",
        "image_similarity",
        "text_similarity",
        "joint_similarity",
        "max_similarity",
        "signals",
        "bucket",
        "label",
        "annotator",
        "audit_label",
        "notes",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_keys_for_signal(
    embeddings: np.ndarray,
    signal: str,
    method: str,
    top_k: int,
    min_similarity: float,
    max_exact_pairs: int,
) -> List[Tuple[int, int]]:
    n = embeddings.shape[0]
    if n < 2:
        return []
    resolved = method
    exact_pairs = n * (n - 1) // 2
    if resolved == "auto":
        resolved = "exact" if exact_pairs <= max_exact_pairs else "torch"
    if resolved == "exact":
        if exact_pairs > max_exact_pairs:
            raise ValueError(
                f"{signal} exact mining would compare {exact_pairs} pairs; "
                f"use --method sklearn or raise --max-exact-pairs."
            )
        return _exact_topk_keys(embeddings, top_k=top_k, min_similarity=min_similarity)
    if resolved == "sklearn":
        return _sklearn_topk_keys(embeddings, top_k=top_k, min_similarity=min_similarity)
    if resolved == "torch":
        return _torch_topk_keys(embeddings, top_k=top_k, min_similarity=min_similarity)
    raise ValueError(f"Unsupported mining method: {method}")


def _exact_topk_keys(embeddings: np.ndarray, top_k: int, min_similarity: float) -> List[Tuple[int, int]]:
    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -np.inf)
    keys: set[Tuple[int, int]] = set()
    k = min(max(1, top_k), embeddings.shape[0] - 1)
    for i in range(embeddings.shape[0]):
        neighbor_indices = np.argpartition(-similarity[i], kth=k - 1)[:k]
        for j in neighbor_indices:
            if similarity[i, j] < min_similarity:
                continue
            a, b = sorted((int(i), int(j)))
            keys.add((a, b))
    return sorted(keys)


def _sklearn_topk_keys(embeddings: np.ndarray, top_k: int, min_similarity: float) -> List[Tuple[int, int]]:
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for --method sklearn candidate mining") from exc

    k = min(max(2, top_k + 1), embeddings.shape[0])
    model = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)
    keys: set[Tuple[int, int]] = set()
    for i, row in enumerate(indices):
        for position, j in enumerate(row):
            if i == int(j):
                continue
            similarity = 1.0 - float(distances[i][position])
            if similarity < min_similarity:
                continue
            a, b = sorted((int(i), int(j)))
            keys.add((a, b))
    return sorted(keys)


def _torch_topk_keys(
    embeddings: np.ndarray,
    top_k: int,
    min_similarity: float,
    chunk_size: int = 512,
) -> List[Tuple[int, int]]:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for --method torch candidate mining") from exc

    n = embeddings.shape[0]
    k = min(max(1, top_k), n - 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    matrix = torch.as_tensor(np.asarray(embeddings, dtype=np.float32), device=device, dtype=dtype)
    keys: set[Tuple[int, int]] = set()
    with torch.no_grad():
        all_t = matrix.T.contiguous()
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            similarity = matrix[start:end] @ all_t
            row_indices = torch.arange(end - start, device=device)
            similarity[row_indices, torch.arange(start, end, device=device)] = -float("inf")
            values, indices = torch.topk(similarity, k=k, dim=1)
            values_cpu = values.float().cpu().numpy()
            indices_cpu = indices.cpu().numpy()
            for offset in range(end - start):
                i = start + offset
                for position, j in enumerate(indices_cpu[offset]):
                    if values_cpu[offset][position] < min_similarity:
                        continue
                    a, b = sorted((int(i), int(j)))
                    keys.add((a, b))
            if start == 0 or end == n or end % max(chunk_size * 20, 1) == 0:
                print(f"torch top-k progress: {end}/{n}", flush=True)
            del similarity, values, indices
    return sorted(keys)


def _parse_signals(raw: str) -> List[str]:
    signals = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {"image", "text", "joint"}
    invalid = [signal for signal in signals if signal not in allowed]
    if invalid:
        raise ValueError(f"Unsupported signals: {invalid}; allowed={sorted(allowed)}")
    return signals or ["image", "text", "joint"]


def _bucket(score: float) -> str:
    if score >= 0.95:
        return "very_high"
    if score >= 0.85:
        return "high"
    if score >= 0.70:
        return "medium"
    return "low"


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


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
