"""Run a small LLaVA LoRA training job from Stage 4 split JSON files.

This script is intentionally conservative. It is meant to validate the
downstream path before launching long A/B/C/D/E runs on the Windows RTX 3090.
It records a config and metrics file in a source-of-truth experiment directory.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


@dataclass
class RunConfig:
    experiment_id: str
    model_id: str
    train_json: str
    output_dir: str
    max_samples: int
    max_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    seed: int
    load_in_4bit: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    max_length: int
    image_size: int | None
    prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--image-size", type=int, help="Optional square resize for tiny-model smoke tests.")
    parser.add_argument("--data-only", action="store_true", help="Only validate JSON/images and write metrics.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = time.time()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.train_json, args.max_samples)
    image_check = validate_images(records)
    config = RunConfig(
        experiment_id=args.experiment_id,
        model_id=args.model_id,
        train_json=str(args.train_json),
        output_dir=str(args.output_dir),
        max_samples=args.max_samples,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        load_in_4bit=args.load_in_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=parse_csv(args.target_modules),
        max_length=args.max_length,
        image_size=args.image_size,
        prompt="LLaVA conversation JSON: human image prompt, assistant caption",
    )
    write_config(args.output_dir, asdict(config))

    metrics: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "train_json": str(args.train_json),
        "model_id": args.model_id,
        "num_loaded_records": len(records),
        "image_check": image_check,
        "data_only": args.data_only,
    }
    if image_check["missing_images"] or image_check["bad_images"]:
        metrics["status"] = "failed_data_validation"
        write_json(args.output_dir / "metrics.json", metrics)
        raise SystemExit(f"Image validation failed: {image_check}")

    if args.data_only:
        metrics.update({"status": "data_validated", "runtime_seconds": time.time() - started_at})
        write_json(args.output_dir / "metrics.json", metrics)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return 0

    train_metrics = run_training(args, records, config)
    metrics.update(train_metrics)
    metrics["runtime_seconds"] = time.time() - started_at
    write_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


def load_records(path: Path, max_samples: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return records[:max_samples]


def validate_images(records: list[dict[str, Any]]) -> dict[str, Any]:
    from PIL import Image

    missing = 0
    bad = 0
    first_size = None
    for record in records:
        image_path = Path(record["image"])
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


def run_training(args: argparse.Namespace, records: list[dict[str, Any]], config: RunConfig) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    processor = AutoProcessor.from_pretrained(args.model_id)
    if args.image_size and hasattr(processor, "image_processor"):
        size_value = {"height": args.image_size, "width": args.image_size}
        processor.image_processor.size = size_value
        if hasattr(processor.image_processor, "crop_size"):
            processor.image_processor.crop_size = size_value
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    model_kwargs: dict[str, Any] = {"torch_dtype": torch.float16}
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"

    model = LlavaForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    elif torch.cuda.is_available():
        model.to("cuda")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()

    dataset = LlavaJsonDataset(records)
    collator = LlavaCollator(
        processor=processor,
        tokenizer=tokenizer,
        max_length=args.max_length,
        image_size=args.image_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    step = 0
    for batch_idx, batch in enumerate(loader):
        batch = move_batch(batch, model)
        outputs = model(**batch)
        loss = outputs.loss / args.gradient_accumulation_steps
        loss.backward()
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            losses.append(float(loss.detach().cpu()) * args.gradient_accumulation_steps)
            print(f"step={step} loss={losses[-1]:.6f}", flush=True)
            if step >= args.max_steps:
                break

    model.save_pretrained(args.output_dir / "adapter")
    processor.save_pretrained(args.output_dir / "processor")
    return {
        "status": "trained",
        "steps": step,
        "losses": losses,
        "final_loss": losses[-1] if losses else None,
        "trainable_parameters": count_trainable_parameters(model),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
    }


class LlavaJsonDataset:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]


class LlavaCollator:
    def __init__(self, processor: Any, tokenizer: Any, max_length: int, image_size: int | None) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        from PIL import Image
        import torch

        encoded_items = []
        for record in records:
            question, answer = extract_question_answer(record)
            image = Image.open(record["image"]).convert("RGB")
            if self.image_size:
                image = image.resize((self.image_size, self.image_size))
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            full_text = f"{prompt} {answer}{self.tokenizer.eos_token or ''}"
            prompt_inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            full_inputs = self.processor(
                text=full_text,
                images=image,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = full_inputs["input_ids"][0]
            attention_mask = full_inputs["attention_mask"][0]
            labels = input_ids.clone()
            prompt_len = min(prompt_inputs["input_ids"].shape[-1], labels.shape[-1])
            labels[:prompt_len] = -100
            labels[attention_mask == 0] = -100
            encoded_items.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "pixel_values": normalize_pixel_values(full_inputs["pixel_values"]),
                }
            )

        max_len = max(item["input_ids"].shape[0] for item in encoded_items)
        pad_id = self.tokenizer.pad_token_id
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "pixel_values": [],
        }
        for item in encoded_items:
            pad_len = max_len - item["input_ids"].shape[0]
            batch["input_ids"].append(pad_tensor(item["input_ids"], pad_len, pad_id))
            batch["attention_mask"].append(pad_tensor(item["attention_mask"], pad_len, 0))
            batch["labels"].append(pad_tensor(item["labels"], pad_len, -100))
            batch["pixel_values"].append(item["pixel_values"])
        return {key: torch.stack(value) for key, value in batch.items()}


def extract_question_answer(record: dict[str, Any]) -> tuple[str, str]:
    conversations = record.get("conversations") or []
    question = "Describe this image in detail."
    answer = ""
    for message in conversations:
        if message.get("from") == "human":
            question = str(message.get("value") or question).replace("<image>", "").strip()
        elif message.get("from") == "gpt":
            answer = str(message.get("value") or "").strip()
    if not answer:
        raise ValueError(f"Missing assistant caption for record {record.get('id')}")
    return question, answer


def pad_tensor(tensor: Any, pad_len: int, value: int) -> Any:
    import torch

    if pad_len <= 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pad_len), value=value)


def normalize_pixel_values(pixel_values: Any) -> Any:
    """Return one image tensor as C x H x W across processor variants."""
    while getattr(pixel_values, "ndim", 0) > 3:
        pixel_values = pixel_values[0]
    return pixel_values


def move_batch(batch: dict[str, Any], model: Any) -> dict[str, Any]:
    import torch

    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = None
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {key: value.to(device) for key, value in batch.items()}


def count_trainable_parameters(model: Any) -> dict[str, int | float]:
    trainable = 0
    total = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return {
        "trainable": trainable,
        "total": total,
        "percent": (trainable / total * 100) if total else 0.0,
    }


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
