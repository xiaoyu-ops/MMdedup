"""Write a Windows/WSL queue script for LLaVA Stage 4 downstream evaluation.

The generated queue evaluates A/B/C/D/E adapters sequentially. It is designed
to be launched only after the E training run is complete, so it refuses to run
when a training process is still active unless explicitly overridden.
"""

from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REPO = "/mnt/c/Users/sysu/code/MMdedup"
DEFAULT_ENV = "/home/xiaoyu/stage4_llava_env/bin/activate"
DEFAULT_RESULT_ROOT = "experiments/results/plan_b_stage4"


@dataclass(frozen=True)
class SplitSpec:
    label: str
    train_experiment_id: str


SPLITS = [
    SplitSpec("A_raw", "exp_llava_stage4_train25k_A_raw_25000_2000steps_20260521"),
    SplitSpec("B_image_only", "exp_llava_stage4_train25k_B_image_only_25000_2000steps_20260521"),
    SplitSpec("C_text_only", "exp_llava_stage4_train25k_C_text_only_25000_2000steps_20260521"),
    SplitSpec("D_naive_union", "exp_llava_stage4_train25k_D_naive_union_25000_2000steps_20260521"),
    SplitSpec("E_stage4_joint", "exp_llava_stage4_train25k_E_stage4_joint_25000_2000steps_20260522_rerun4"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", required=True, help="Path as seen inside WSL.")
    parser.add_argument("--annotations-json", help="Optional VQAv2-style annotation JSON path inside WSL.")
    parser.add_argument("--image-root", help="Optional image root path inside WSL.")
    parser.add_argument("--image-template", help="Example: COCO_val2014_{image_id:012d}.jpg")
    parser.add_argument("--eval-name", default="vqav2_quick")
    parser.add_argument("--date-stamp", default="20260522")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--env", default=DEFAULT_ENV)
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-script", type=Path, required=True)
    parser.add_argument("--allow-training-overlap", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script = render_queue(args)
    args.output_script.parent.mkdir(parents=True, exist_ok=True)
    args.output_script.write_text(script, encoding="utf-8")
    args.output_script.chmod(0o755)
    print(args.output_script)
    return 0


def render_queue(args: argparse.Namespace) -> str:
    q = shlex.quote
    lines = [
        "#!/usr/bin/env bash",
        "set -Eeuo pipefail",
        "",
        f"REPO={q(args.repo)}",
        f"RESULT_ROOT=\"$REPO/{args.result_root}\"",
        f"EVAL_JSON={q(args.eval_json)}",
        f"ANNOTATIONS_JSON={q(args.annotations_json or '')}",
        f"IMAGE_ROOT={q(args.image_root or '')}",
        f"IMAGE_TEMPLATE={q(args.image_template or '')}",
        f"EVAL_NAME={q(args.eval_name)}",
        f"DATE_STAMP={q(args.date_stamp)}",
        "QUEUE_LOG=\"$RESULT_ROOT/llava_stage4_vqa_eval_${EVAL_NAME}_${DATE_STAMP}.log\"",
        "",
        "cd \"$REPO\"",
        f"source {q(args.env)}",
        "mkdir -p \"$RESULT_ROOT\"",
        "exec > >(tee -a \"$QUEUE_LOG\") 2>&1",
        "",
        "echo \"VQA_EVAL_QUEUE_START $(date -Is)\"",
        "echo \"REPO=$REPO\"",
        "git rev-parse HEAD || true",
        "nvidia-smi || true",
    ]
    if not args.allow_training_overlap:
        lines.extend(
            [
                "if pgrep -af 'run_llava_stage4_lora.py' >/tmp/stage4_active_training.txt; then",
                "  echo \"QUEUE_REFUSED_ACTIVE_TRAINING $(date -Is)\"",
                "  cat /tmp/stage4_active_training.txt",
                "  exit 2",
                "fi",
                "",
            ]
        )
    lines.extend(
        [
            "run_eval() {",
            "  local split=\"$1\"",
            "  local train_exp=\"$2\"",
            "  local adapter_dir=\"$RESULT_ROOT/$train_exp/adapter\"",
            "  local exp_id=\"exp_llava_stage4_vqa_${EVAL_NAME}_${split}_${DATE_STAMP}\"",
            "  local out_dir=\"$RESULT_ROOT/$exp_id\"",
            "  local code=0",
            "",
            "  mkdir -p \"$out_dir\"",
            "  if [[ ! -d \"$adapter_dir\" ]]; then",
            "    echo \"JOB_BLOCKED_MISSING_ADAPTER $(date -Is) $split adapter=$adapter_dir\"",
            "    return 1",
            "  fi",
            "  if [[ -f \"$out_dir/metrics.json\" ]]; then",
            "    python - \"$out_dir/metrics.json\" <<'PY' && return 0 || true",
            "import json, sys",
            "data = json.load(open(sys.argv[1], 'r', encoding='utf-8'))",
            "raise SystemExit(0 if data.get('status') == 'evaluated' else 1)",
            "PY",
            "  fi",
            "",
            "  echo \"JOB_START $(date -Is) $exp_id split=$split train_exp=$train_exp\"",
            "  cmd=(",
            "    python experiments/scripts/run_llava_stage4_vqa_eval.py",
            "    --experiment-id \"$exp_id\"",
            "    --eval-json \"$EVAL_JSON\"",
            "    --adapter-dir \"$adapter_dir\"",
            "    --output-dir \"$out_dir\"",
            f"    --max-samples {args.max_samples}",
            "    --load-in-4bit",
            "  )",
            "  [[ -n \"$ANNOTATIONS_JSON\" ]] && cmd+=(--annotations-json \"$ANNOTATIONS_JSON\")",
            "  [[ -n \"$IMAGE_ROOT\" ]] && cmd+=(--image-root \"$IMAGE_ROOT\")",
            "  [[ -n \"$IMAGE_TEMPLATE\" ]] && cmd+=(--image-template \"$IMAGE_TEMPLATE\")",
            "  set +e",
            "  \"${cmd[@]}\" >\"$out_dir/stdout.log\" 2>\"$out_dir/stderr.log\"",
            "  code=$?",
            "  set -e",
            "  echo \"$code\" >\"$out_dir/exit_code.txt\"",
            "  if [[ -f \"$out_dir/metrics.json\" ]]; then",
            "    python - \"$out_dir/metrics.json\" <<'PY' || true",
            "import json, sys",
            "data = json.load(open(sys.argv[1], 'r', encoding='utf-8'))",
            "print('METRICS', data.get('experiment_id'), 'status=' + str(data.get('status')), 'vqa=' + str(data.get('vqa_consensus')), 'exact=' + str(data.get('exact_match')), 'n=' + str(data.get('num_scored_records')))",
            "PY",
            "  fi",
            "  echo \"JOB_END $(date -Is) $exp_id code=$code\"",
            "  return \"$code\"",
            "}",
            "",
        ]
    )
    for split in SPLITS:
        lines.append(f"run_eval {q(split.label)} {q(split.train_experiment_id)} || true")
    lines.extend(
        [
            "",
            "echo \"VQA_EVAL_QUEUE_END $(date -Is)\"",
            "nvidia-smi || true",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
