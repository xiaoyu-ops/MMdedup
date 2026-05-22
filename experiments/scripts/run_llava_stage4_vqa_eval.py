"""Run a LLaVA Stage 4 downstream VQA-style evaluation.

The script evaluates one trained LoRA adapter on a VQA-style JSON/JSONL file
and writes paper-facing source-of-truth outputs:

- config.json / config.yaml
- predictions.jsonl
- metrics.json

It intentionally supports a small generic input format so that we can run a
quick subset before wiring the full VQAv2/TextVQA evaluation data.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
DEFAULT_PROMPT = "USER: <image>\n{question}\nASSISTANT:"


@dataclass
class EvalConfig:
    experiment_id: str
    model_id: str
    adapter_dir: str | None
    eval_json: str
    annotations_json: str | None
    image_root: str | None
    image_template: str | None
    output_dir: str
    max_samples: int | None
    seed: int
    shuffle: bool
    load_in_4bit: bool
    max_new_tokens: int
    max_length: int
    image_size: int | None
    prompt_template: str
    data_only: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--annotations-json", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--image-template", help="Example: COCO_val2014_{image_id:012d}.jpg")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--image-size", type=int, help="Optional square resize for smoke tests.")
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT)
    parser.add_argument("--data-only", action="store_true", help="Validate data and write config/metrics only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = time.time()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = EvalConfig(
        experiment_id=args.experiment_id,
        model_id=args.model_id,
        adapter_dir=str(args.adapter_dir) if args.adapter_dir else None,
        eval_json=str(args.eval_json),
        annotations_json=str(args.annotations_json) if args.annotations_json else None,
        image_root=str(args.image_root) if args.image_root else None,
        image_template=args.image_template,
        output_dir=str(args.output_dir),
        max_samples=args.max_samples,
        seed=args.seed,
        shuffle=args.shuffle,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        image_size=args.image_size,
        prompt_template=args.prompt_template,
        data_only=args.data_only,
    )
    write_config(args.output_dir, asdict(config))

    log_event("loading_eval_records", eval_json=str(args.eval_json), annotations_json=str(args.annotations_json))
    records = load_eval_records(
        eval_json=args.eval_json,
        annotations_json=args.annotations_json,
        image_root=args.image_root,
        image_template=args.image_template,
    )
    if args.shuffle:
        random.shuffle(records)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    log_event("eval_records_loaded", count=len(records))

    image_check = validate_images(records)
    metrics: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "model_id": args.model_id,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "eval_json": str(args.eval_json),
        "annotations_json": str(args.annotations_json) if args.annotations_json else None,
        "num_eval_records": len(records),
        "num_answered_records": sum(1 for record in records if record.answers),
        "image_check": image_check,
        "data_only": args.data_only,
    }
    if image_check["missing_images"] or image_check["bad_images"]:
        metrics["status"] = "failed_data_validation"
        metrics["runtime_seconds"] = time.time() - started_at
        write_json(args.output_dir / "metrics.json", metrics)
        raise SystemExit(f"Image validation failed: {image_check}")

    if args.data_only:
        metrics["status"] = "data_validated"
        metrics["runtime_seconds"] = time.time() - started_at
        write_json(args.output_dir / "metrics.json", metrics)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return 0

    eval_metrics = run_generation_eval(args, records)
    metrics.update(eval_metrics)
    metrics["runtime_seconds"] = time.time() - started_at
    write_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


@dataclass
class EvalRecord:
    question_id: str
    image: str
    question: str
    answers: list[str]
    source: dict[str, Any]


def load_eval_records(
    eval_json: Path,
    annotations_json: Path | None,
    image_root: Path | None,
    image_template: str | None,
) -> list[EvalRecord]:
    payload = read_json_or_jsonl(eval_json)
    annotations = load_annotation_map(annotations_json) if annotations_json else {}

    if isinstance(payload, dict) and isinstance(payload.get("questions"), list):
        raw_records = payload["questions"]
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        raw_records = payload["data"]
    elif isinstance(payload, list):
        raw_records = payload
    else:
        raise ValueError(f"Unsupported eval JSON shape: {eval_json}")

    records: list[EvalRecord] = []
    for idx, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            continue
        question_id = str(raw.get("question_id") or raw.get("id") or idx)
        annotation = annotations.get(question_id, {})
        question = str(raw.get("question") or raw.get("text") or raw.get("prompt") or "").strip()
        if not question:
            continue
        image = resolve_image_path(raw, eval_json.parent, image_root, image_template)
        answers = extract_answers(raw)
        if annotation:
            answers = extract_answers(annotation) or answers
        records.append(
            EvalRecord(
                question_id=question_id,
                image=str(image),
                question=question,
                answers=answers,
                source=raw,
            )
        )
    return records


def read_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def load_annotation_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json_or_jsonl(path)
    if isinstance(payload, dict) and isinstance(payload.get("annotations"), list):
        rows = payload["annotations"]
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        rows = payload["data"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"Unsupported annotations JSON shape: {path}")
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        question_id = row.get("question_id") or row.get("id")
        if question_id is not None:
            mapped[str(question_id)] = row
    return mapped


def resolve_image_path(
    raw: dict[str, Any],
    eval_dir: Path,
    image_root: Path | None,
    image_template: str | None,
) -> Path:
    direct = raw.get("image") or raw.get("image_path") or raw.get("image_file") or raw.get("file_name")
    image_id = raw.get("image_id") or raw.get("imageId")
    if direct:
        path = Path(str(direct))
    elif image_template and image_id is not None:
        format_value: Any = image_id
        try:
            format_value = int(image_id)
        except (TypeError, ValueError):
            pass
        path = Path(image_template.format(image_id=format_value, image=str(image_id)))
    else:
        raise ValueError(f"Cannot resolve image path for record: {raw}")

    if path.is_absolute():
        return path
    if image_root:
        return image_root / path
    return eval_dir / path


def extract_answers(raw: dict[str, Any]) -> list[str]:
    value = raw.get("answers")
    if value is None:
        value = raw.get("answer")
    if value is None:
        value = raw.get("label")
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        answers: list[str] = []
        for item in value:
            if isinstance(item, str):
                answers.append(item)
            elif isinstance(item, dict):
                answer = item.get("answer") or item.get("raw_answer") or item.get("label")
                if answer is not None:
                    answers.append(str(answer))
        return answers
    return [str(value)]


def validate_images(records: list[EvalRecord]) -> dict[str, Any]:
    from PIL import Image

    missing = 0
    bad = 0
    first_size = None
    for record in records:
        image_path = Path(record.image)
        if not image_path.exists():
            missing += 1
            continue
        try:
            with Image.open(image_path) as image:
                if first_size is None:
                    first_size = list(image.size)
                image.verify()
        except Exception:
            bad += 1
    return {
        "checked_records": len(records),
        "missing_images": missing,
        "bad_images": bad,
        "first_image_size": first_size,
    }


def run_generation_eval(args: argparse.Namespace, records: list[EvalRecord]) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

    if args.adapter_dir and not args.adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter directory: {args.adapter_dir}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    log_event(
        "eval_environment",
        cuda_available=torch.cuda.is_available(),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    if args.image_size and hasattr(processor, "image_processor"):
        size_value = {"height": args.image_size, "width": args.image_size}
        processor.image_processor.size = size_value
        if hasattr(processor.image_processor, "crop_size"):
            processor.image_processor.crop_size = size_value
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"torch_dtype": torch.float16}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["device_map"] = "auto"

    log_event("loading_model", model_id=args.model_id, load_in_4bit=args.load_in_4bit)
    model = LlavaForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    if args.adapter_dir:
        log_event("loading_adapter", adapter_dir=str(args.adapter_dir))
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    elif torch.cuda.is_available():
        model.to("cuda")
    model.eval()

    predictions_path = args.output_dir / "predictions.jsonl"
    exact_scores: list[float] = []
    vqa_scores: list[float] = []
    with predictions_path.open("w", encoding="utf-8") as handle:
        for idx, record in enumerate(records):
            prediction = generate_answer(args, model, processor, record)
            exact = exact_match_score(prediction, record.answers) if record.answers else None
            vqa = vqa_consensus_score(prediction, record.answers) if record.answers else None
            if exact is not None:
                exact_scores.append(exact)
            if vqa is not None:
                vqa_scores.append(vqa)
            handle.write(
                json.dumps(
                    {
                        "question_id": record.question_id,
                        "image": record.image,
                        "question": record.question,
                        "prediction": prediction,
                        "answers": record.answers,
                        "exact_match": exact,
                        "vqa_consensus": vqa,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if (idx + 1) % 25 == 0:
                log_event("eval_progress", done=idx + 1, total=len(records))

    return {
        "status": "evaluated",
        "predictions_jsonl": str(predictions_path),
        "exact_match": mean(exact_scores),
        "vqa_consensus": mean(vqa_scores),
        "num_scored_records": len(vqa_scores),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
    }


def generate_answer(args: argparse.Namespace, model: Any, processor: Any, record: EvalRecord) -> str:
    import torch
    from PIL import Image

    image = Image.open(record.image).convert("RGB")
    if args.image_size:
        image = image.resize((args.image_size, args.image_size))
    prompt = args.prompt_template.format(question=record.question)
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    inputs = move_inputs(inputs, model)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = generated[0][prompt_len:]
    text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return clean_prediction(text)


def move_inputs(inputs: dict[str, Any], model: Any) -> dict[str, Any]:
    import torch

    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = None
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {key: value.to(device) for key, value in inputs.items()}


def clean_prediction(text: str) -> str:
    text = text.strip()
    text = text.split("\n")[0].strip()
    text = re.sub(r"^(assistant:|answer:)\s*", "", text, flags=re.IGNORECASE)
    return text


def exact_match_score(prediction: str, answers: list[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    return 1.0 if any(normalized_prediction == normalize_answer(answer) for answer in answers) else 0.0


def vqa_consensus_score(prediction: str, answers: list[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    matches = sum(1 for answer in answers if normalized_prediction == normalize_answer(answer))
    return min(1.0, matches / 3.0)


def normalize_answer(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_config(output_dir: Path, payload: dict[str, Any]) -> None:
    write_json(output_dir / "config.json", payload)
    try:
        import yaml
    except ImportError:
        return
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def log_event(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
