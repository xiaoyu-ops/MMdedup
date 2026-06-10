"""Recompute strict and relaxed VQA scores from saved prediction JSONL files."""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.predictions_jsonl)
    metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))

    exact_scores: list[float] = []
    vqa_scores: list[float] = []
    relaxed_contains_scores: list[float] = []
    relaxed_vqa_scores: list[float] = []

    for row in rows:
        prediction = str(row.get("prediction") or "")
        answers = [str(item) for item in row.get("answers") or []]
        if not answers:
            continue
        exact_scores.append(exact_match_score(prediction, answers))
        vqa_scores.append(vqa_consensus_score(prediction, answers))
        relaxed_contains_scores.append(relaxed_contains_score(prediction, answers))
        relaxed_vqa_scores.append(relaxed_vqa_consensus_score(prediction, answers))

    rescored = {
        "status": "rescored",
        "predictions_jsonl": str(args.predictions_jsonl),
        "source_metrics_json": str(args.metrics_json),
        "num_scored_records": len(vqa_scores),
        "strict_exact_match": mean(exact_scores),
        "strict_vqa_consensus": mean(vqa_scores),
        "relaxed_contains": mean(relaxed_contains_scores),
        "relaxed_vqa_consensus": mean(relaxed_vqa_scores),
        "notes": (
            "Relaxed metrics count a record as correct when a normalized gold answer phrase "
            "appears as a token-delimited phrase inside the normalized prediction."
        ),
    }
    metrics["rescored_metrics"] = rescored
    metrics["exact_match"] = rescored["strict_exact_match"]
    metrics["vqa_consensus"] = rescored["strict_vqa_consensus"]
    metrics["relaxed_contains"] = rescored["relaxed_contains"]
    metrics["relaxed_vqa_consensus"] = rescored["relaxed_vqa_consensus"]
    metrics["scoring_note"] = rescored["notes"]

    args.output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(rescored, indent=2, ensure_ascii=False))
    return 0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def exact_match_score(prediction: str, answers: list[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    return 1.0 if any(normalized_prediction == normalize_answer(answer) for answer in answers) else 0.0


def vqa_consensus_score(prediction: str, answers: list[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    matches = sum(1 for answer in answers if normalized_prediction == normalize_answer(answer))
    return min(1.0, matches / 3.0)


def relaxed_contains_score(prediction: str, answers: list[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    return 1.0 if any(answer_mentioned(normalized_prediction, answer) for answer in answers) else 0.0


def relaxed_vqa_consensus_score(prediction: str, answers: list[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    matches = sum(1 for answer in answers if answer_mentioned(normalized_prediction, answer))
    return min(1.0, matches / 3.0)


def answer_mentioned(normalized_prediction: str, answer: str) -> bool:
    normalized_answer = normalize_answer(answer)
    if not normalized_prediction or not normalized_answer:
        return False
    return f" {normalized_answer} " in f" {normalized_prediction} "


def normalize_answer(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


if __name__ == "__main__":
    raise SystemExit(main())
