"""Summarize strict/relaxed VQA rescoring and answer-style stats across experiment dirs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_rescore_helper():
    script_path = Path(__file__).with_name("rescore_llava_vqa_predictions.py")
    spec = importlib.util.spec_from_file_location("stage4_vqa_rescore_helper", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def token_count(text: str) -> int:
    return len([tok for tok in text.strip().split() if tok])


def is_number_like(text: str) -> bool:
    text = text.strip().lower()
    if not text:
        return False
    return text.replace(".", "", 1).isdigit()


def is_yes_no_answer(answers: list[str]) -> bool:
    normalized = {a.strip().lower() for a in answers if a.strip()}
    return bool(normalized) and normalized <= {"yes", "no"}


def main() -> int:
    args = parse_args()
    helper = load_rescore_helper()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for exp_dir in args.experiment_dirs:
        predictions_path = exp_dir / "predictions.jsonl"
        metrics_path = exp_dir / "metrics.json"
        if not predictions_path.exists() or not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows = helper.read_jsonl(predictions_path)
        exact_scores: list[float] = []
        strict_scores: list[float] = []
        relaxed_contains_scores: list[float] = []
        relaxed_vqa_scores: list[float] = []
        token_counts: list[int] = []
        long_answer_flags: list[int] = []
        yes_no_scores: list[float] = []
        number_scores: list[float] = []
        other_scores: list[float] = []

        for row in rows:
            prediction = str(row.get("prediction") or "")
            answers = [str(item) for item in row.get("answers") or []]
            if not answers:
                continue
            exact = helper.exact_match_score(prediction, answers)
            strict = helper.vqa_consensus_score(prediction, answers)
            relaxed_contains = helper.relaxed_contains_score(prediction, answers)
            relaxed_vqa = helper.relaxed_vqa_consensus_score(prediction, answers)
            exact_scores.append(exact)
            strict_scores.append(strict)
            relaxed_contains_scores.append(relaxed_contains)
            relaxed_vqa_scores.append(relaxed_vqa)
            tok_n = token_count(prediction)
            token_counts.append(tok_n)
            long_answer_flags.append(1 if tok_n >= 4 else 0)

            if is_yes_no_answer(answers):
                yes_no_scores.append(strict)
            elif len(answers) == 1 and is_number_like(answers[0]):
                number_scores.append(strict)
            else:
                other_scores.append(strict)

        if not strict_scores:
            continue

        summary = {
            "experiment_dir": str(exp_dir),
            "experiment_id": metrics.get("experiment_id", exp_dir.name),
            "num_scored_records": len(strict_scores),
            "strict_exact_match": statistics.fmean(exact_scores),
            "strict_vqa_consensus": statistics.fmean(strict_scores),
            "relaxed_contains": statistics.fmean(relaxed_contains_scores),
            "relaxed_vqa_consensus": statistics.fmean(relaxed_vqa_scores),
            "strict_to_relaxed_vqa_gap": statistics.fmean(relaxed_vqa_scores) - statistics.fmean(strict_scores),
            "mean_prediction_tokens": statistics.fmean(token_counts),
            "median_prediction_tokens": statistics.median(token_counts),
            "long_answer_rate_ge4": statistics.fmean(long_answer_flags),
            "yes_no_strict_vqa": statistics.fmean(yes_no_scores) if yes_no_scores else None,
            "number_strict_vqa": statistics.fmean(number_scores) if number_scores else None,
            "other_strict_vqa": statistics.fmean(other_scores) if other_scores else None,
        }
        summaries.append(summary)
        rows_out.append(summary)

    fieldnames = [
        "experiment_id",
        "num_scored_records",
        "strict_exact_match",
        "strict_vqa_consensus",
        "relaxed_contains",
        "relaxed_vqa_consensus",
        "strict_to_relaxed_vqa_gap",
        "mean_prediction_tokens",
        "median_prediction_tokens",
        "long_answer_rate_ge4",
        "yes_no_strict_vqa",
        "number_strict_vqa",
        "other_strict_vqa",
        "experiment_dir",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    args.output_json.write_text(
        json.dumps(
            {
                "status": "summarized",
                "num_experiments": len(summaries),
                "summaries": summaries,
                "note": "Strict/relaxed VQA rescoring plus simple answer-style statistics for caption-style analysis.",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output_csv)
    print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
