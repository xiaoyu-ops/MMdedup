"""Stage 4 image-caption pair-level deduplication.

This module is intentionally independent from the legacy modality pipelines so
the Plan B cross-modal contribution can be run, measured, and audited as a
standalone experiment.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # Optional heavy dependencies for real runs.
    import open_clip  # type: ignore
    import torch  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    open_clip = None  # type: ignore
    torch = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    image_path: Path
    caption: str
    caption_path: Optional[Path] = None


@dataclass
class Stage4Config:
    embedding_backend: str = "auto"  # auto | open_clip | simple
    model_name: str = "hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
    pretrained: Optional[str] = None
    device: str = "auto"  # auto | cpu | cuda
    batch_size: int = 32
    joint_method: str = "concat"  # concat | weighted_sum
    alpha: float = 0.5
    tau_cross: float = 0.95
    keeper_tie_eps: float = 0.02
    image_quality: str = "resolution"  # resolution | file_size
    cache_dir: Optional[Path] = None
    max_exact_pairs: int = 2_000_000


@dataclass
class Stage4Result:
    keepers: List[str]
    drops: List[str]
    duplicate_groups: List[Dict[str, object]]
    summary: Dict[str, object]


def load_pairs_from_sidecar_dir(dataset_dir: Path) -> List[PairRecord]:
    """Load image-caption pairs from files sharing the same stem."""

    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {dataset_dir}")

    records: List[PairRecord] = []
    for image_path in sorted(path for path in dataset_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
        caption_path = image_path.with_suffix(".txt")
        if not caption_path.exists():
            continue
        caption = caption_path.read_text(encoding="utf-8", errors="replace").strip()
        if not caption:
            continue
        pair_id = image_path.relative_to(dataset_dir).with_suffix("").as_posix()
        records.append(
            PairRecord(
                pair_id=pair_id,
                image_path=image_path,
                caption=caption,
                caption_path=caption_path,
            )
        )
    return records


def load_pairs_from_csv(csv_path: Path, base_dir: Optional[Path] = None) -> List[PairRecord]:
    """Load pairs from CSV.

    Required columns: `pair_id`, `image_path`, and either `caption` or
    `caption_path`.
    """

    csv_path = Path(csv_path)
    root = Path(base_dir) if base_dir else csv_path.parent
    records: List[PairRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"pair_id", "image_path"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")
        for row in reader:
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = root / image_path
            caption_path: Optional[Path] = None
            caption = (row.get("caption") or "").strip()
            if not caption:
                raw_caption_path = (row.get("caption_path") or "").strip()
                if not raw_caption_path:
                    raise ValueError("Each row must provide caption or caption_path")
                caption_path = Path(raw_caption_path)
                if not caption_path.is_absolute():
                    caption_path = root / caption_path
                caption = caption_path.read_text(encoding="utf-8", errors="replace").strip()
            records.append(
                PairRecord(
                    pair_id=row["pair_id"],
                    image_path=image_path,
                    caption=caption,
                    caption_path=caption_path,
                )
            )
    return records


def run_stage4_pair_dedup(pairs: Sequence[PairRecord], config: Stage4Config) -> Stage4Result:
    """Run Stage 4 pair-level deduplication."""

    start = time.time()
    unique_pairs = _deduplicate_pair_ids(pairs)
    if not unique_pairs:
        return Stage4Result(
            keepers=[],
            drops=[],
            duplicate_groups=[],
            summary={
                "num_pairs": 0,
                "num_keepers": 0,
                "num_drops": 0,
                "dedup_rate": 0.0,
                "config": _jsonable_config(config),
                "elapsed_seconds": time.time() - start,
            },
        )

    image_emb, text_emb, backend = encode_pairs(unique_pairs, config)
    joint_emb = build_joint_embeddings(image_emb, text_emb, config)
    alignment = _cosine_rows(image_emb, text_emb)
    quality = np.asarray([_image_quality_score(pair.image_path, config.image_quality) for pair in unique_pairs])
    keepers, drops, groups = _greedy_dedup(unique_pairs, joint_emb, alignment, quality, config)

    summary: Dict[str, object] = {
        "num_pairs": len(unique_pairs),
        "num_keepers": len(keepers),
        "num_drops": len(drops),
        "dedup_rate": len(drops) / len(unique_pairs),
        "embedding_backend": backend,
        "joint_method": config.joint_method,
        "tau_cross": config.tau_cross,
        "elapsed_seconds": time.time() - start,
        "config": _jsonable_config(config),
    }
    return Stage4Result(keepers=keepers, drops=drops, duplicate_groups=groups, summary=summary)


def encode_pairs(pairs: Sequence[PairRecord], config: Stage4Config) -> Tuple[np.ndarray, np.ndarray, str]:
    backend = config.embedding_backend
    if backend == "auto":
        backend = "open_clip" if open_clip is not None and torch is not None and Image is not None else "simple"
    cached = _load_embedding_cache(pairs, config, backend)
    if cached is not None:
        image_emb, text_emb = cached
        return image_emb, text_emb, f"{backend}:cache"
    if backend == "open_clip":
        image_emb, text_emb = _encode_open_clip(pairs, config)
    elif backend == "simple":
        image_emb, text_emb = _encode_simple(pairs)
    else:
        raise ValueError(f"Unsupported embedding backend: {config.embedding_backend}")
    _save_embedding_cache(pairs, config, backend, image_emb, text_emb)
    return image_emb, text_emb, backend


def build_joint_embeddings(image_emb: np.ndarray, text_emb: np.ndarray, config: Stage4Config) -> np.ndarray:
    image_emb = _l2_normalize(image_emb)
    text_emb = _l2_normalize(text_emb)
    if config.joint_method == "concat":
        return _l2_normalize(np.concatenate([image_emb, text_emb], axis=1))
    if config.joint_method == "weighted_sum":
        if image_emb.shape[1] != text_emb.shape[1]:
            raise ValueError("weighted_sum requires image and text embeddings with equal dimensions")
        alpha = float(config.alpha)
        return _l2_normalize(alpha * image_emb + (1.0 - alpha) * text_emb)
    raise ValueError(f"Unsupported joint_method: {config.joint_method}")


def write_stage4_outputs(result: Stage4Result, output_dir: Path) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary": output_dir / "stage4_pair_dedup_summary.json",
        "metrics": output_dir / "metrics.json",
        "groups": output_dir / "stage4_duplicate_groups.json",
        "keepers": output_dir / "stage4_keepers.txt",
        "drops": output_dir / "stage4_drops.txt",
    }
    paths["summary"].write_text(json.dumps(result.summary, indent=2, sort_keys=True), encoding="utf-8")
    paths["metrics"].write_text(json.dumps(_metrics_from_summary(result.summary), indent=2, sort_keys=True), encoding="utf-8")
    paths["groups"].write_text(json.dumps(result.duplicate_groups, indent=2, sort_keys=True), encoding="utf-8")
    paths["keepers"].write_text("\n".join(result.keepers) + ("\n" if result.keepers else ""), encoding="utf-8")
    paths["drops"].write_text("\n".join(result.drops) + ("\n" if result.drops else ""), encoding="utf-8")
    return paths


def _encode_open_clip(pairs: Sequence[PairRecord], config: Stage4Config) -> Tuple[np.ndarray, np.ndarray]:
    if open_clip is None or torch is None or Image is None:
        raise RuntimeError("open_clip, torch, and Pillow are required for the open_clip backend")

    device = _resolve_device(config.device)
    if config.pretrained:
        model, _, preprocess = open_clip.create_model_and_transforms(
            config.model_name,
            pretrained=config.pretrained,
        )
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(config.model_name)
    tokenizer = open_clip.get_tokenizer(config.model_name)
    model = model.to(device)
    model.eval()

    image_vectors: List[np.ndarray] = []
    text_vectors: List[np.ndarray] = []
    batch_size = max(1, int(config.batch_size))
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            images = []
            for pair in batch:
                try:
                    image = Image.open(pair.image_path).convert("RGB")
                except Exception:
                    image = Image.new("RGB", (224, 224), (0, 0, 0))
                images.append(preprocess(image))
            image_tensor = torch.stack(images).to(device)
            text_tensor = tokenizer([pair.caption for pair in batch]).to(device)
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tensor)
            image_vectors.append(image_features.detach().cpu().float().numpy())
            text_vectors.append(text_features.detach().cpu().float().numpy())
    return _l2_normalize(np.vstack(image_vectors)), _l2_normalize(np.vstack(text_vectors))


def _encode_simple(pairs: Sequence[PairRecord]) -> Tuple[np.ndarray, np.ndarray]:
    image_vectors = []
    text_vectors = []
    for pair in pairs:
        image_vectors.append(_hash_embedding(f"{pair.image_path.stem} {pair.image_path.name}"))
        text_vectors.append(_hash_embedding(pair.caption))
    return _l2_normalize(np.vstack(image_vectors)), _l2_normalize(np.vstack(text_vectors))


def _greedy_dedup(
    pairs: Sequence[PairRecord],
    joint_emb: np.ndarray,
    alignment: np.ndarray,
    quality: np.ndarray,
    config: Stage4Config,
) -> Tuple[List[str], List[str], List[Dict[str, object]]]:
    n = len(pairs)
    exact_pairs = n * (n - 1) // 2
    if exact_pairs > config.max_exact_pairs:
        raise ValueError(
            f"Exact pairwise Stage 4 would compare {exact_pairs} pairs; "
            f"raise max_exact_pairs or add candidate mining first."
        )

    similarity = joint_emb @ joint_emb.T
    alignment_bucket = np.floor(alignment / max(config.keeper_tie_eps, 1e-9))
    order = sorted(
        range(n),
        key=lambda idx: (alignment_bucket[idx], quality[idx], alignment[idx], pairs[idx].pair_id),
        reverse=True,
    )

    removed: set[int] = set()
    keepers: List[str] = []
    drops: List[str] = []
    groups: List[Dict[str, object]] = []
    for idx in order:
        if idx in removed:
            continue
        keepers.append(pairs[idx].pair_id)
        members = []
        for other in order:
            if other == idx or other in removed:
                continue
            if similarity[idx, other] >= config.tau_cross:
                removed.add(other)
                drops.append(pairs[other].pair_id)
                members.append(
                    {
                        "pair_id": pairs[other].pair_id,
                        "similarity": float(similarity[idx, other]),
                        "alignment": float(alignment[other]),
                        "quality": float(quality[other]),
                    }
                )
        if members:
            groups.append(
                {
                    "keeper": pairs[idx].pair_id,
                    "keeper_alignment": float(alignment[idx]),
                    "keeper_quality": float(quality[idx]),
                    "duplicates": members,
                }
            )
    return sorted(keepers), sorted(drops), groups


def _deduplicate_pair_ids(pairs: Sequence[PairRecord]) -> List[PairRecord]:
    seen: set[str] = set()
    unique: List[PairRecord] = []
    for pair in pairs:
        if pair.pair_id in seen:
            raise ValueError(f"Duplicate pair_id in input: {pair.pair_id}")
        seen.add(pair.pair_id)
        unique.append(pair)
    return unique


def _hash_embedding(text: str, dim: int = 64) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().replace("_", " ").replace("-", " ").split()
    if not tokens:
        tokens = [text.lower()]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for offset in range(0, 8, 2):
            index = int.from_bytes(digest[offset : offset + 2], "little") % dim
            sign = 1.0 if digest[offset] % 2 == 0 else -1.0
            vector[index] += sign
    return vector


def _load_embedding_cache(
    pairs: Sequence[PairRecord],
    config: Stage4Config,
    backend: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if config.cache_dir is None:
        return None
    cache_dir = Path(config.cache_dir)
    manifest_path = cache_dir / "stage4_embedding_manifest.json"
    vectors_path = cache_dir / "stage4_embeddings.npz"
    if not manifest_path.exists() or not vectors_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_ids = [pair.pair_id for pair in pairs]
        if manifest.get("pair_ids") != expected_ids:
            return None
        if manifest.get("backend") != backend:
            return None
        if backend == "open_clip" and manifest.get("model_name") != config.model_name:
            return None
        if backend == "open_clip" and manifest.get("pretrained") != config.pretrained:
            return None
        data = np.load(vectors_path)
        return data["image_embeddings"], data["text_embeddings"]
    except Exception:
        return None


def _save_embedding_cache(
    pairs: Sequence[PairRecord],
    config: Stage4Config,
    backend: str,
    image_emb: np.ndarray,
    text_emb: np.ndarray,
) -> None:
    if config.cache_dir is None:
        return
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_dir / "stage4_embeddings.npz",
        image_embeddings=image_emb,
        text_embeddings=text_emb,
    )
    manifest = {
        "pair_ids": [pair.pair_id for pair in pairs],
        "backend": backend,
        "model_name": config.model_name,
        "pretrained": config.pretrained,
    }
    (cache_dir / "stage4_embedding_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _image_quality_score(image_path: Path, mode: str) -> float:
    if mode == "file_size":
        try:
            return float(image_path.stat().st_size)
        except OSError:
            return 0.0
    if mode == "resolution" and Image is not None:
        try:
            with Image.open(image_path) as image:
                width, height = image.size
            return float(width * height)
        except Exception:
            return 0.0
    try:
        return float(image_path.stat().st_size)
    except OSError:
        return 0.0


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape:
        min_dim = min(left.shape[1], right.shape[1])
        left = left[:, :min_dim]
        right = right[:, :min_dim]
    return np.sum(_l2_normalize(left) * _l2_normalize(right), axis=1)


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _jsonable_config(config: Stage4Config) -> Dict[str, object]:
    data = asdict(config)
    if data.get("cache_dir") is not None:
        data["cache_dir"] = str(data["cache_dir"])
    return data


def _metrics_from_summary(summary: Dict[str, object]) -> Dict[str, object]:
    keys = ["num_pairs", "num_keepers", "num_drops", "dedup_rate", "elapsed_seconds", "embedding_backend", "tau_cross"]
    return {key: summary[key] for key in keys if key in summary}
