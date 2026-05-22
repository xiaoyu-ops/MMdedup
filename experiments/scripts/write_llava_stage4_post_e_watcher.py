"""Write a Windows/WSL watcher that launches downstream eval after E finishes."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path


DEFAULT_REPO = "/mnt/c/Users/sysu/code/the_work_of_dedup"
DEFAULT_ENV = "/home/xiaoyu/stage4_llava_env/bin/activate"
DEFAULT_RESULT_ROOT = "experiments/results/plan_b_stage4"
DEFAULT_E_EXP = "exp_llava_stage4_train25k_E_stage4_joint_25000_2000steps_20260522_rerun4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--env", default=DEFAULT_ENV)
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--e-experiment-id", default=DEFAULT_E_EXP)
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--annotations-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--image-template", default="COCO_val2014_{image_id:012d}.jpg")
    parser.add_argument("--eval-name", default="vqav2_quick")
    parser.add_argument("--date-stamp", default="20260522")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--expected-steps", type=int, default=2000)
    parser.add_argument("--output-script", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_script.parent.mkdir(parents=True, exist_ok=True)
    args.output_script.write_text(render(args), encoding="utf-8")
    args.output_script.chmod(0o755)
    print(args.output_script)
    return 0


def render(args: argparse.Namespace) -> str:
    q = shlex.quote
    queue_script = (
        f"$RESULT_ROOT/llava_stage4_vqa_eval_queue_{args.eval_name}_{args.date_stamp}.sh"
    )
    lines = [
        "#!/usr/bin/env bash",
        "set -Eeuo pipefail",
        "",
        f"REPO={q(args.repo)}",
        f"RESULT_ROOT=\"$REPO/{args.result_root}\"",
        f"E_EXPERIMENT_ID={q(args.e_experiment_id)}",
        f"E_METRICS=\"$RESULT_ROOT/{args.e_experiment_id}/metrics.json\"",
        f"E_ADAPTER=\"$RESULT_ROOT/{args.e_experiment_id}/adapter\"",
        f"EVAL_JSON={q(args.eval_json)}",
        f"ANNOTATIONS_JSON={q(args.annotations_json)}",
        f"IMAGE_ROOT={q(args.image_root)}",
        f"IMAGE_TEMPLATE={q(args.image_template)}",
        f"EVAL_NAME={q(args.eval_name)}",
        f"DATE_STAMP={q(args.date_stamp)}",
        f"MAX_SAMPLES={args.max_samples}",
        f"POLL_SECONDS={args.poll_seconds}",
        f"EXPECTED_STEPS={args.expected_steps}",
        "WATCH_LOG=\"$RESULT_ROOT/llava_stage4_post_e_watcher_${DATE_STAMP}.log\"",
        f"QUEUE_SCRIPT={queue_script}",
        "",
        "cd \"$REPO\"",
        f"source {q(args.env)}",
        "mkdir -p \"$RESULT_ROOT\"",
        "exec > >(tee -a \"$WATCH_LOG\") 2>&1",
        "",
        "echo \"POST_E_WATCHER_START $(date -Is)\"",
        "echo \"REPO=$REPO\"",
        "git rev-parse HEAD || true",
        "",
        "e_finished() {",
        "  [[ -f \"$E_METRICS\" ]] || return 1",
        "  python - \"$E_METRICS\" \"$EXPECTED_STEPS\" <<'PY'",
        "import json, sys",
        "path = sys.argv[1]",
        "expected_steps = int(sys.argv[2])",
        "data = json.load(open(path, 'r', encoding='utf-8'))",
        "ok = data.get('status') == 'trained' and int(data.get('steps') or 0) >= expected_steps",
        "raise SystemExit(0 if ok else 1)",
        "PY",
        "}",
        "",
        "eval_data_ready() {",
        "  [[ -f \"$EVAL_JSON\" ]] || return 1",
        "  [[ -f \"$ANNOTATIONS_JSON\" ]] || return 1",
        "  [[ -d \"$IMAGE_ROOT\" ]] || return 1",
        "  [[ -d \"$E_ADAPTER\" ]] || return 1",
        "}",
        "",
        "while ! e_finished; do",
        "  latest_step=$(grep -E 'step=[0-9]+ loss=' \"$RESULT_ROOT/$E_EXPERIMENT_ID/stdout.log\" 2>/dev/null | tail -1 || true)",
        "  echo \"WAIT_E $(date -Is) ${latest_step:-no_step_yet}\"",
        "  sleep \"$POLL_SECONDS\"",
        "done",
        "",
        "echo \"E_FINISHED $(date -Is) metrics=$E_METRICS\"",
        "",
        "while pgrep -af 'run_llava_stage4_lora.py' >/tmp/stage4_active_training.txt; do",
        "  echo \"WAIT_TRAINING_PROCESS_EXIT $(date -Is)\"",
        "  cat /tmp/stage4_active_training.txt",
        "  sleep 60",
        "done",
        "",
        "while ! eval_data_ready; do",
        "  echo \"WAIT_EVAL_DATA $(date -Is)\"",
        "  [[ -f \"$EVAL_JSON\" ]] || echo \"MISSING $EVAL_JSON\"",
        "  [[ -f \"$ANNOTATIONS_JSON\" ]] || echo \"MISSING $ANNOTATIONS_JSON\"",
        "  [[ -d \"$IMAGE_ROOT\" ]] || echo \"MISSING $IMAGE_ROOT\"",
        "  [[ -d \"$E_ADAPTER\" ]] || echo \"MISSING $E_ADAPTER\"",
        "  sleep \"$POLL_SECONDS\"",
        "done",
        "",
        "python experiments/scripts/write_llava_stage4_vqa_eval_queue.py \\",
        "  --eval-name \"$EVAL_NAME\" \\",
        "  --date-stamp \"$DATE_STAMP\" \\",
        "  --max-samples \"$MAX_SAMPLES\" \\",
        "  --eval-json \"$EVAL_JSON\" \\",
        "  --annotations-json \"$ANNOTATIONS_JSON\" \\",
        "  --image-root \"$IMAGE_ROOT\" \\",
        "  --image-template \"$IMAGE_TEMPLATE\" \\",
        "  --output-script \"$QUEUE_SCRIPT\"",
        "",
        "echo \"START_DOWNSTREAM_QUEUE $(date -Is) script=$QUEUE_SCRIPT\"",
        "bash \"$QUEUE_SCRIPT\"",
        "echo \"POST_E_WATCHER_DONE $(date -Is)\"",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
