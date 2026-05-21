"""Build static data for the Plan B Stage 4 progress dashboard."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/results/plan_b_stage4"
SYNC = RESULTS / "windows_sync"
DASHBOARD = ROOT / "docs/stage4_dashboard"
DATA_DIR = DASHBOARD / "data"
OUT = DATA_DIR / "status.json"
SPLIT_EXPERIMENT_ID = "exp_stage4_training_manifests_200k_20260520"
SPLIT_EXPERIMENT_DIR = RESULTS / SPLIT_EXPERIMENT_ID


def main() -> int:
    annotation = _annotation_status()
    charts = _charts(annotation)
    status = {
        "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "target": {
            "venue": "CIKM 2026 Full Paper",
            "deadline": "2026-05-23 AoE",
            "branch": "codex/plan-b-stage4-pair-dedup",
            "claim": "面向图文训练语料的 image-caption pair 级跨模态去重。",
        },
        "summary_cards": _summary_cards(),
        "phase_progress": _phase_progress(),
        "charts": charts,
        "plan_requirements": _plan_requirements(),
        "experiments": _experiments(),
        "paper_writing_data": _paper_writing_data(annotation),
        "paper_tables": _paper_tables(annotation),
        "plan_data_matrix": _plan_data_matrix(annotation),
        "data_quality_audit": _data_quality_audit(),
        "annotation": annotation,
        "artifacts": _artifacts(),
        "data_exports": _data_exports(),
        "risks": [
            {
                "level": "high",
                "title": "Stage 4 尚未超过 image-only",
                "detail": "当前 1000 条 high-joint 标注集上，joint F1=0.583，naive_union F1=0.456，但 image-only F1=0.626，需要做误差分析并谨慎写作。",
            },
            {
                "level": "high",
                "title": "Audit agreement 不能当正式合作者一致性",
                "detail": "当前 agreement_rate=1.0 来自 audit_label 默认填主标签，只能说明流程完成，不能作为真实 inter-annotator agreement。",
            },
            {
                "level": "high",
                "title": "当前 GT 是 hard-candidate benchmark",
                "detail": "1000 条标注来自 joint>=0.80 且 image>=0.60 的候选池，不能用来估计原始 CC3M 的重复比例。",
            },
            {
                "level": "medium",
                "title": "embedding cache 仍只在 Windows 端",
                "detail": "200K CLIP embedding cache 约 762 MB，目前尚未完整镜像回 Mac。",
            },
            {
                "level": "medium",
                "title": "LLaVA 正式下游验证尚未完成",
                "detail": "A/B/C/D/E 五组 200K split 已生成，真实 LLaVA-1.5-7B 4-bit LoRA smoke 已跑通 1 step；但完整 LoRA 训练与 VQAv2/TextVQA 指标尚未产生。",
            },
        ],
        "next_steps": [
            "基于误差分析决定论文如何解释：joint 优于 naive union，但 image-only 在当前 high-joint GT 上更强。",
            "基于已验证的 Windows WSL2 环境启动 A/B/C/D/E 五组 LLaVA LoRA 正式训练。",
            "将 Windows 端 embedding cache 和后续 split 结果同步回 Mac source-of-truth。",
            "为每个 LLaVA run 保存 config/metrics/logs，并同步回 Mac source-of-truth。",
        ],
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _write_exports(status, annotation, charts)
    OUT.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


def _summary_cards() -> list[dict[str, str]]:
    prepare = _read_json(SYNC / "cc3m_subset_200k_20260515/prepare_metrics.json")
    candidates = _read_json(SYNC / "exp_stage4_candidates_200k_manifest_20260516/metrics.json")
    high_joint = _read_json(SYNC / "exp_stage4_candidates_200k_high_joint_20260516/metrics.json")
    annotation = _read_json(SYNC / "exp_stage4_annotation_1000_200k_high_joint_20260516/metrics.json")
    return [
        {
            "label": "CC3M pool",
            "value": _fmt_int(prepare.get("saved_pairs", 200000)),
            "note": "Windows RTX 3090 上已准备的图文对数量",
        },
        {
            "label": "候选 pair-pairs",
            "value": _fmt_int(candidates.get("num_candidates", 500000)),
            "note": "由 image/text/joint CLIP top-k 挖掘得到",
        },
        {
            "label": "High-joint pool",
            "value": _fmt_int(high_joint.get("num_candidates", 129139)),
            "note": "筛选条件：joint >= 0.80 且 image >= 0.60",
        },
        {
            "label": "标注目标",
            "value": _fmt_int(annotation.get("num_annotation_rows", 1000)),
            "note": f"其中 {_fmt_int(annotation.get('num_audit_rows', 200))} 条用于合作者抽查",
        },
    ]


def _phase_progress() -> list[dict[str, str | int]]:
    llava = _llava_pilot_status()
    llava_percent = 45 if llava["completed"] >= 5 else 25 + llava["completed"] * 4
    llava_detail = (
        f"A/B/C/D/E 512-sample pilot 已完成 {llava['completed']}/5；"
        f"{llava['current_training']}；完整 VQAv2/TextVQA 指标尚未完成。"
    )
    return [
        {
            "name": "Stage 4 实现",
            "status": "done",
            "percent": 100,
            "detail": "已完成 CLIP 图像/文本 embedding、joint embedding、候选挖掘与 keep/drop 流程。",
        },
        {
            "name": "CC3M 200K 数据准备",
            "status": "done",
            "percent": 100,
            "detail": "Windows 端已准备并验证 200,000 个 image-caption sidecar pairs。",
        },
        {
            "name": "候选挖掘",
            "status": "done",
            "percent": 100,
            "detail": "已挖掘 500,000 个候选；high-joint/high-image 池保留 129,139 个。",
        },
        {
            "name": "人工标注",
            "status": "done",
            "percent": _annotation_percent(),
            "detail": "1000 条 high-joint 主标注已完成；其中 295 条为 duplicate 或 near-duplicate。",
        },
        {
            "name": "Stage 4 主评价",
            "status": "active",
            "percent": 90,
            "detail": "已完成 P/R/F1 主表、阈值扫描和误差分析；joint 优于 naive_union，但尚未超过 image-only，写作需谨慎。",
        },
        {
            "name": "LLaVA 下游验证",
            "status": "active",
            "percent": llava_percent,
            "detail": llava_detail,
        },
    ]


def _plan_requirements() -> list[dict[str, object]]:
    llava_smoke = _read_json(SYNC / "exp_llava_stage4_real_train_smoke_E_20260520/metrics.json")
    llava_peak_memory = _fmt_gib(llava_smoke.get("gpu_peak_memory_bytes"))
    llava = _llava_pilot_status()
    return [
        {
            "name": "Stage 4 实现",
            "status": "complete",
            "required_data": [
                "image-caption pair id",
                "图像路径",
                "caption 文本",
                "CLIP 图像/文本 embeddings",
                "joint embedding",
            ],
            "current_outputs": [
                "keepers",
                "drops",
                "duplicate groups",
                "summary / metrics",
                "Windows 端 embedding cache",
            ],
            "evidence": [
                "pipelines/stage4_pair_dedup.py",
                "experiments/scripts/mine_stage4_candidates.py",
                "experiments/results/plan_b_stage4/windows_sync/exp_stage4_candidates_200k_manifest_20260516/metrics.json",
            ],
        },
        {
            "name": "CC3M Ground Truth 构建",
            "status": "active",
            "required_data": [
                "100K-300K CC3M 数据池",
                "先挖掘 candidate pair-pairs，再人工标注",
                "1000 条已标注 pair-pairs",
                "尽量获得 200 个 positive examples",
                "20% 合作者抽查子集",
            ],
            "current_outputs": [
                "200K CC3M 数据池",
                "500K mined candidates",
                "129,139 个 high-joint/high-image candidates",
                "1000 条已标注主标注表",
                "200 条 audit rows",
                "adjudicated ground truth",
            ],
            "evidence": [
                "experiments/results/plan_b_stage4/windows_sync/cc3m_subset_200k_20260515/validation_summary.json",
                "experiments/results/plan_b_stage4/windows_sync/exp_stage4_candidates_200k_high_joint_20260516/metrics.json",
                "experiments/results/plan_b_stage4/windows_sync/exp_stage4_annotation_1000_200k_high_joint_20260516/annotation_sheet_labeled.csv",
                "experiments/results/plan_b_stage4/exp_stage4_adjudicated_1000_200k_high_joint_20260519/adjudicated_annotations.csv",
            ],
        },
        {
            "name": "Stage 4 主评价",
            "status": "active",
            "required_data": [
                "adjudicated human labels",
                "image-only baseline scores",
                "text-only baseline scores",
                "naive union baseline scores",
                "Stage 4 joint scores",
            ],
            "current_outputs": [
                "1000 条 adjudicated labels",
                "image/text/naive_union/joint/max 阈值扫描结果",
                "per-threshold metrics CSV",
                "metrics JSON",
            ],
            "evidence": [
                "experiments/scripts/evaluate_stage4_groundtruth.py",
                "experiments/scripts/adjudicate_stage4_annotations.py",
                "experiments/results/plan_b_stage4/exp_stage4_eval_1000_200k_high_joint_20260519/metrics.json",
                "experiments/results/plan_b_stage4/exp_stage4_eval_1000_200k_high_joint_20260519/per_threshold_metrics.csv",
            ],
        },
        {
            "name": "MLLM 下游验证",
            "status": "active",
            "required_data": [
                "raw split A",
                "image-only split B",
                "text-only split C",
                "naive union split D",
                "Stage 4 split E",
                "LLaVA LoRA logs 与 VQAv2/TextVQA metrics",
            ],
            "current_outputs": [
                "A/B/C/D/E 200K training manifests",
                "A/B/C/D/E LLaVA JSON data-smoke validated",
                "Stage 4 E 真实 LLaVA-1.5-7B 4-bit LoRA 1-step smoke",
                f"A/B/C/D/E 512-sample LoRA pilot：{llava['completed']}/5 complete",
                f"当前正式训练状态：{llava['current_training']}",
                f"smoke final_loss={_fmt_float(llava_smoke.get('final_loss'), 4)}; peak_memory={llava_peak_memory}",
                "完整 25K/2000-step A/B/C/D/E LoRA 训练与 VQAv2/TextVQA 指标尚未完成",
            ],
            "evidence": [
                "experiments/results/plan_b_stage4/exp_stage4_training_manifests_200k_20260520/metrics.json",
                "experiments/results/plan_b_stage4/exp_llava_stage4_data_smoke_abcde_20260520/metrics.json",
                "experiments/results/plan_b_stage4/windows_sync/exp_llava_stage4_real_train_smoke_E_20260520/metrics.json",
                "experiments/results/plan_b_stage4/windows_sync/exp_llava_stage4_pilot_E_stage4_joint_512_20steps_20260521/metrics.json",
                "experiments/results/plan_b_stage4/windows_sync/llava_stage4_overnight_queue_20260521.log",
                "experiments/results/plan_b_stage4/experiment_ledger.csv",
            ],
        },
        {
            "name": "系统与效率结果",
            "status": "partial",
            "required_data": [
                "CLIP embedding time",
                "candidate search time",
                "端到端 wall-clock time",
                "GPU peak memory",
                "throughput",
            ],
            "current_outputs": [
                "已记录 200K candidate mining runtime",
                f"LLaVA smoke GPU peak memory 已记录：{llava_peak_memory}",
                "完整 Stage 4 embedding/search 分项与正式训练 peak memory 尚未记录",
            ],
            "evidence": [
                "experiments/results/plan_b_stage4/experiment_ledger.csv",
            ],
        },
        {
            "name": "论文修改",
            "status": "pending",
            "required_data": [
                "Stage 4 方法描述",
                "CC3M ground truth 构建过程",
                "主评价表",
                "LLaVA 下游表",
                "效率表",
            ],
            "current_outputs": [
                "已有修改计划",
                "正式结果表等待标注与训练",
            ],
            "evidence": [
                "docs/MMdedup修改计划.md",
                "paper/latex/main.tex",
            ],
        },
    ]


def _experiments() -> list[dict[str, str]]:
    wanted = [
        "cc3m_subset_200k_20260515",
        "exp_stage4_candidates_200k_manifest_20260516",
        "exp_stage4_candidates_200k_high_joint_20260516",
        "exp_stage4_annotation_1000_200k_high_joint_20260516",
        "exp_stage4_adjudicated_1000_200k_high_joint_20260519",
        "exp_stage4_eval_1000_200k_high_joint_20260519",
        "exp_stage4_error_analysis_1000_200k_high_joint_20260520",
        SPLIT_EXPERIMENT_ID,
        "exp_llava_stage4_data_smoke_abcde_20260520",
        "exp_llava_stage4_real_train_smoke_E_20260520",
        "exp_llava_stage4_pilot_E_stage4_joint_512_20steps_20260521",
        "exp_llava_stage4_pilot_D_naive_union_512_20steps_20260521",
        "exp_llava_stage4_pilot_A_raw_512_20steps_20260521",
        "exp_llava_stage4_pilot_B_image_only_512_20steps_20260521",
        "exp_llava_stage4_pilot_C_text_only_512_20steps_20260521",
    ]
    rows = []
    ledger = RESULTS / "experiment_ledger.csv"
    with ledger.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["experiment_id"] in wanted:
                rows.append(
                    {
                        "id": row["experiment_id"],
                        "dataset": row["dataset"],
                        "hardware": row["hardware"],
                        "runtime": _fmt_runtime(row["wall_clock_seconds"]),
                        "numbers": row["paper_numbers"],
                        "notes": row["notes"],
                    }
                )
    order = {name: idx for idx, name in enumerate(wanted)}
    return sorted(rows, key=lambda row: order.get(row["id"], 999))


def _charts(annotation: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    prepare = _read_json(SYNC / "cc3m_subset_200k_20260515/prepare_metrics.json")
    candidates = _read_json(SYNC / "exp_stage4_candidates_200k_manifest_20260516/metrics.json")
    high_joint = _read_json(SYNC / "exp_stage4_candidates_200k_high_joint_20260516/metrics.json")
    annotation_metrics = _read_json(
        SYNC / "exp_stage4_annotation_1000_200k_high_joint_20260516/metrics.json"
    )
    counts = annotation.get("counts", {})
    if not isinstance(counts, dict):
        counts = {}
    return {
        "candidate_funnel": [
            {
                "label": "CC3M 200K 数据池",
                "value": int(prepare.get("saved_pairs", 200000)),
                "unit": "pairs",
                "note": "已下载并验证的 image-caption pairs",
            },
            {
                "label": "候选 pair-pairs",
                "value": int(candidates.get("num_candidates", 500000)),
                "unit": "pair-pairs",
                "note": "image/text/joint top-k 初筛候选",
            },
            {
                "label": "High-joint 候选池",
                "value": int(high_joint.get("num_candidates", 129139)),
                "unit": "pair-pairs",
                "note": "joint >= 0.80 且 image >= 0.60",
            },
            {
                "label": "人工标注目标",
                "value": int(annotation_metrics.get("num_annotation_rows", 1000)),
                "unit": "pair-pairs",
                "note": "当前主标注表规模",
            },
            {
                "label": "已完成标注",
                "value": int(annotation.get("done", 0)),
                "unit": "pair-pairs",
                "note": "来自 annotation_sheet_labeled.csv",
            },
        ],
        "annotation_distribution": [
            {"label": "重复", "key": "duplicate", "value": int(counts.get("duplicate", 0))},
            {
                "label": "近重复",
                "key": "near-duplicate",
                "value": int(counts.get("near-duplicate", 0)),
            },
            {
                "label": "非重复",
                "key": "not-duplicate",
                "value": int(counts.get("not-duplicate", 0)),
            },
            {"label": "未标注", "key": "unlabeled", "value": int(counts.get("unlabeled", 0))},
        ],
        "phase_progress": [
            {
                "label": phase["name"],
                "value": int(phase["percent"]),
                "status": phase["status"],
            }
            for phase in _phase_progress()
        ],
        "experiment_runtime": _runtime_chart(),
        "stage4_eval_best_f1": _stage4_eval_chart(),
        "stage4_abcde_split_sizes": _abcde_split_chart(),
        "threshold_dedup_rates": _threshold_dedup_chart(),
        "llava_pilots": _llava_pilot_chart(),
    }


def _annotation_status() -> dict[str, object]:
    source = SYNC / "exp_stage4_annotation_1000_200k_high_joint_20260516/annotation_sheet.csv"
    labeled = source.with_name("annotation_sheet_labeled.csv")
    path = labeled if labeled.exists() else source
    counts = {label: 0 for label in ["duplicate", "near-duplicate", "not-duplicate", "unlabeled"]}
    audit_rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        label = row.get("label", "").strip()
        if label in counts:
            counts[label] += 1
        else:
            counts["unlabeled"] += 1
        if _truthy(row.get("needs_audit", "")):
            audit_rows += 1
    total = len(rows)
    done = total - counts["unlabeled"]
    positives = counts["duplicate"] + counts["near-duplicate"]
    return {
        "total": total,
        "done": done,
        "remaining": counts["unlabeled"],
        "percent": round(done * 100 / total, 1) if total else 0,
        "positives": positives,
        "audit_rows": audit_rows,
        "counts": counts,
        "source_csv": str(source.relative_to(ROOT)),
        "labeled_csv": str(labeled.relative_to(ROOT)),
        "review_sheets": str((source.parent / "review_sheets").relative_to(ROOT)),
    }


def _artifacts() -> list[dict[str, str]]:
    return [
        {
            "title": "Primary annotation CSV",
            "path": "experiments/results/plan_b_stage4/windows_sync/exp_stage4_annotation_1000_200k_high_joint_20260516/annotation_sheet.csv",
        },
        {
            "title": "Labeled annotation CSV",
            "path": "experiments/results/plan_b_stage4/windows_sync/exp_stage4_annotation_1000_200k_high_joint_20260516/annotation_sheet_labeled.csv",
        },
        {
            "title": "Review sheets",
            "path": "experiments/results/plan_b_stage4/windows_sync/exp_stage4_annotation_1000_200k_high_joint_20260516/review_sheets/",
        },
        {
            "title": "Experiment ledger",
            "path": "experiments/results/plan_b_stage4/experiment_ledger.csv",
        },
        {
            "title": "Stage 4 evaluation metrics",
            "path": "experiments/results/plan_b_stage4/exp_stage4_eval_1000_200k_high_joint_20260519/metrics.json",
        },
        {
            "title": "Daily log",
            "path": "experiments/results/plan_b_stage4/daily_logs/2026-05-19.md",
        },
        {
            "title": "Stage 4 error analysis",
            "path": "experiments/results/plan_b_stage4/exp_stage4_error_analysis_1000_200k_high_joint_20260520/metrics.json",
        },
        {
            "title": "A/B/C/D/E split sizes",
            "path": f"experiments/results/plan_b_stage4/{SPLIT_EXPERIMENT_ID}/abcde_split_sizes.csv",
        },
    ]


def _runtime_chart() -> list[dict[str, object]]:
    rows = []
    for exp in _experiments():
        runtime = exp["runtime"]
        seconds = _runtime_to_seconds(runtime)
        if seconds is None:
            continue
        rows.append(
            {
                "label": exp["id"],
                "value": round(seconds / 60, 2),
                "unit": "minutes",
                "note": exp["numbers"],
            }
        )
    return rows


def _stage4_eval_chart() -> list[dict[str, object]]:
    metrics = _read_json(RESULTS / "exp_stage4_eval_1000_200k_high_joint_20260519/metrics.json")
    best_by_score = metrics.get("best_by_score", {})
    if not isinstance(best_by_score, dict):
        return []
    labels = {
        "image": "Image-only",
        "text": "Text-only",
        "naive_union": "Naive union",
        "joint": "Stage 4 joint",
        "max": "Max score",
    }
    rows = []
    for key in ["image", "text", "naive_union", "joint", "max"]:
        item = best_by_score.get(key, {})
        if not isinstance(item, dict) or "f1" not in item:
            continue
        rows.append(
            {
                "label": labels[key],
                "value": round(float(item["f1"]), 3),
                "unit": "F1",
                "note": f"threshold={item.get('threshold')}; P={float(item.get('precision', 0)):.3f}; R={float(item.get('recall', 0)):.3f}",
            }
        )
    return rows


def _abcde_split_chart() -> list[dict[str, object]]:
    path = SPLIT_EXPERIMENT_DIR / "abcde_split_sizes.csv"
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "label": f"{row['split']} {row['name']}",
                    "value": int(row["kept_pairs"]),
                    "unit": "kept pairs",
                    "note": f"dropped={_fmt_int(row['dropped_pairs'])}; rate={float(row['dedup_rate']):.3f}; {row['threshold']}",
                }
            )
    return rows


def _threshold_dedup_chart() -> list[dict[str, object]]:
    path = SPLIT_EXPERIMENT_DIR / "threshold_dedup_rates.csv"
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["score"] not in {"image", "text", "joint", "naive_union"}:
                continue
            rows.append(
                {
                    "label": row["score"],
                    "threshold": float(row["threshold"]),
                    "value": float(row["dedup_rate"]),
                    "unit": "dedup rate",
                    "note": f"dropped={_fmt_int(row['dropped_pairs'])}; edges={_fmt_int(row['selected_candidate_edges'])}",
                }
            )
    return rows


def _llava_pilot_status() -> dict[str, object]:
    pilots = [
        ("A", "raw", "exp_llava_stage4_pilot_A_raw_512_20steps_20260521"),
        ("B", "image-only", "exp_llava_stage4_pilot_B_image_only_512_20steps_20260521"),
        ("C", "text-only", "exp_llava_stage4_pilot_C_text_only_512_20steps_20260521"),
        ("D", "naive union", "exp_llava_stage4_pilot_D_naive_union_512_20steps_20260521"),
        ("E", "Stage 4 joint", "exp_llava_stage4_pilot_E_stage4_joint_512_20steps_20260521"),
    ]
    rows = []
    for split, name, exp_id in pilots:
        metrics = _read_json(SYNC / f"{exp_id}/metrics.json")
        done = metrics.get("status") == "trained" and int(metrics.get("steps") or 0) >= 20
        rows.append(
            {
                "split": split,
                "name": name,
                "experiment_id": exp_id,
                "status": "complete" if done else "missing",
                "steps": int(metrics.get("steps") or 0),
                "samples": int(metrics.get("num_loaded_records") or 0),
                "final_loss": metrics.get("final_loss"),
                "runtime_seconds": metrics.get("runtime_seconds"),
                "gpu_peak_gb": (float(metrics.get("gpu_peak_memory_bytes") or 0) / 1024**3)
                if metrics
                else None,
                "metrics_path": str((SYNC / f"{exp_id}/metrics.json").relative_to(ROOT)),
            }
        )
    completed = sum(1 for row in rows if row["status"] == "complete")
    summary_parts = [
        f"{row['split']} loss={_fmt_float(row['final_loss'], 4)}"
        for row in rows
        if row["status"] == "complete"
    ]
    queue_log = SYNC / "llava_stage4_overnight_queue_20260521.log"
    current_training = "25K/2000-step 正式队列已启动"
    if queue_log.exists():
        for line in reversed(queue_log.read_text(encoding="utf-8", errors="replace").splitlines()):
            if line.startswith("JOB_START") and "train25k" in line:
                current_training = line.strip()
                break
    return {
        "completed": completed,
        "total": len(rows),
        "pilots": rows,
        "pilot_summary": "; ".join(summary_parts) if summary_parts else "pilot metrics pending",
        "current_training": current_training,
        "queue_log": str(queue_log.relative_to(ROOT)),
    }


def _llava_pilot_chart() -> list[dict[str, object]]:
    rows = []
    for row in _llava_pilot_status()["pilots"]:
        rows.append(
            {
                "label": f"{row['split']} {row['name']}",
                "value": round(float(row["final_loss"]), 4)
                if row.get("final_loss") is not None
                else None,
                "unit": "final loss",
                "note": f"steps={row['steps']}; samples={row['samples']}; peak={_fmt_float(row.get('gpu_peak_gb'), 3)} GiB",
            }
        )
    return rows


def _data_exports() -> list[dict[str, str]]:
    return [
        {
            "title": "Dashboard 当前状态",
            "href": "data/status.json",
            "description": "Dashboard 使用的完整状态快照。",
        },
        {
            "title": "图表数据源",
            "href": "data/charts.json",
            "description": "candidate funnel、标注分布、阶段进度和实验耗时的图表数据。",
        },
        {
            "title": "方案数据需求",
            "href": "data/plan_requirements.json",
            "description": "每个 Plan B 产物需要的数据、当前产物和证据路径。",
        },
        {
            "title": "最新标注状态",
            "href": "data/latest_annotation_status.json",
            "description": "当前标签数量、标注进度和输出路径。",
        },
        {
            "title": "实验 ledger CSV",
            "href": "data/experiment_ledger.csv",
            "description": "source-of-truth 实验台账快照。",
        },
        {
            "title": "论文写作证据目录",
            "href": "data/paper_writing_data.json",
            "description": "按论文段落和表格组织的可引用数字、解释和证据文件链接。",
        },
        {
            "title": "论文表格工作台数据源",
            "href": "data/paper_tables.json",
            "description": "按论文表/图/段落组织的可直接写作视图，包含推荐表头、可填行、写法和风险。",
        },
        {
            "title": "方案完整数据清单",
            "href": "data/plan_data_matrix.json",
            "description": "对照 MMdedup 修改计划列出的所有实验表格、当前状态、已有来源和缺口。",
        },
        {
            "title": "数据合理性审核",
            "href": "data/data_quality_audit.json",
            "description": "当前数据是否适合写进论文、哪些数字必须加限定、哪些还不能作为最终 claim。",
        },
    ]


def _write_exports(status: dict, annotation: dict[str, object], charts: dict[str, object]) -> None:
    (DATA_DIR / "charts.json").write_text(
        json.dumps(charts, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (DATA_DIR / "plan_requirements.json").write_text(
        json.dumps(status["plan_requirements"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (DATA_DIR / "latest_annotation_status.json").write_text(
        json.dumps(annotation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (DATA_DIR / "paper_writing_data.json").write_text(
        json.dumps(status["paper_writing_data"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (DATA_DIR / "paper_tables.json").write_text(
        json.dumps(status["paper_tables"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (DATA_DIR / "plan_data_matrix.json").write_text(
        json.dumps(status["plan_data_matrix"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (DATA_DIR / "data_quality_audit.json").write_text(
        json.dumps(status["data_quality_audit"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    ledger = RESULTS / "experiment_ledger.csv"
    if ledger.exists():
        shutil.copyfile(ledger, DATA_DIR / "experiment_ledger.csv")
    _copy_paper_source_files()


def _data_quality_audit() -> dict[str, object]:
    audit_path = RESULTS / "data_audits/2026-05-20_data_reasonableness_audit.json"
    return _read_json(audit_path)


def _paper_writing_data(annotation: dict[str, object]) -> list[dict[str, object]]:
    prepare = _read_json(SYNC / "cc3m_subset_200k_20260515/prepare_metrics.json")
    validation = _read_json(SYNC / "cc3m_subset_200k_20260515/validation_summary.json")
    candidates = _read_json(SYNC / "exp_stage4_candidates_200k_manifest_20260516/metrics.json")
    high_joint = _read_json(SYNC / "exp_stage4_candidates_200k_high_joint_20260516/metrics.json")
    annotation_metrics = _read_json(SYNC / "exp_stage4_annotation_1000_200k_high_joint_20260516/metrics.json")
    adjudication = _read_json(RESULTS / "exp_stage4_adjudicated_1000_200k_high_joint_20260519/metrics.json")
    evaluation = _read_json(RESULTS / "exp_stage4_eval_1000_200k_high_joint_20260519/metrics.json")
    error_analysis = _read_json(RESULTS / "exp_stage4_error_analysis_1000_200k_high_joint_20260520/metrics.json")
    split_metrics = _read_json(SPLIT_EXPERIMENT_DIR / "metrics.json")
    llava_smoke = _read_json(SYNC / "exp_llava_stage4_real_train_smoke_E_20260520/metrics.json")
    llava = _llava_pilot_status()
    best_by_score = evaluation.get("best_by_score", {})
    if not isinstance(best_by_score, dict):
        best_by_score = {}

    image = best_by_score.get("image", {})
    text = best_by_score.get("text", {})
    naive = best_by_score.get("naive_union", {})
    joint = best_by_score.get("joint", {})

    return [
        {
            "title": "CC3M 数据池与候选挖掘",
            "status": "complete",
            "paper_use": "用于论文 Dataset / Ground-truth Construction 段落，说明我们不是随机抽 pair-pairs，而是先在 200K CC3M 图文对中挖掘候选。",
            "key_numbers": [
                {"label": "CC3M pool", "value": _fmt_int(prepare.get("saved_pairs", 200000)), "note": "已准备 image-caption pairs"},
                {"label": "候选 pair-pairs", "value": _fmt_int(candidates.get("num_candidates", 500000)), "note": "image/text/joint top-k 初筛"},
                {"label": "High-joint pool", "value": _fmt_int(high_joint.get("num_candidates", 129139)), "note": "joint >= 0.80 且 image >= 0.60"},
                {"label": "候选挖掘耗时", "value": _fmt_runtime(str(candidates.get("elapsed_seconds", ""))), "note": "Windows RTX 3090"},
            ],
            "sources": [
                _source("200K 数据准备 metrics", "data/paper/cc3m_subset_200k_prepare_metrics.json", "下载/准备 CC3M 200K 的源记录。"),
                _source("200K 数据验证 summary", "data/paper/cc3m_subset_200k_validation_summary.json", "jpg/txt/manifest 验证记录。"),
                _source("500K 候选挖掘 metrics", "data/paper/stage4_candidates_200k_metrics.json", "候选挖掘规模、signals、top_k、runtime。"),
                _source("High-joint 筛选 metrics", "data/paper/stage4_candidates_200k_high_joint_metrics.json", "high-joint/high-image 过滤条件和保留数量。"),
            ],
        },
        {
            "title": "人工标注与 Ground Truth",
            "status": "complete",
            "paper_use": "用于论文 Annotation Protocol / Ground Truth Dataset 段落，说明标注规模、正负例数量、标签定义和 audit 状态。",
            "key_numbers": [
                {"label": "标注总数", "value": _fmt_int(annotation.get("done", 0)), "note": "已完成 pair-pairs"},
                {"label": "正例", "value": _fmt_int(annotation.get("positives", 0)), "note": "duplicate + near-duplicate"},
                {"label": "负例", "value": _fmt_int(annotation.get("counts", {}).get("not-duplicate", 0)), "note": "not-duplicate"},
                {"label": "Audit 一致率", "value": _fmt_float(adjudication.get("agreement_rate", 0), 3), "note": "当前内部默认 audit run"},
            ],
            "sources": [
                _source("标注表 metrics", "data/paper/stage4_annotation_1000_high_joint_metrics.json", "1000 条标注表和 200 条 audit rows 的生成记录。"),
                _source("已标注 CSV", "data/paper/stage4_annotation_1000_high_joint_labeled.csv", "人工标注后的原始 CSV，可查看每条 pair-pair 标签。"),
                _source("Adjudicated labels CSV", "data/paper/stage4_adjudicated_annotations.csv", "带 final_label 和 adjudication_status 的最终评价标签。"),
                _source("Adjudication metrics", "data/paper/stage4_adjudication_metrics.json", "audit 数量、一致率、冲突数量。"),
            ],
        },
        {
            "title": "Stage 4 主评价表",
            "status": "active",
            "paper_use": "用于论文 Main Results 表。当前结论是 Stage 4 joint 优于 naive union，但 image-only 在这批候选集上更强；误差分析显示 joint false positives 中约 42.5% 是 caption 完全相同的模板型样本。",
            "key_numbers": [
                {"label": "Image-only best F1", "value": _score_text(image), "note": _threshold_note(image)},
                {"label": "Text-only best F1", "value": _score_text(text), "note": _threshold_note(text)},
                {"label": "Naive union best F1", "value": _score_text(naive), "note": _threshold_note(naive)},
                {"label": "Stage 4 joint best F1", "value": _score_text(joint), "note": _threshold_note(joint)},
                {"label": "Joint false positives", "value": _fmt_int(error_analysis.get("joint_false_positives")), "note": f"caption_equal_rate={_fmt_float(error_analysis.get('joint_fp_caption_equal_rate'), 3)}"},
                {"label": "Image correct / joint wrong", "value": _fmt_int(error_analysis.get("image_correct_joint_wrong")), "note": "解释 image-only 当前更强的样本池证据"},
            ],
            "sources": [
                _source("主评价 metrics JSON", "data/paper/stage4_eval_metrics.json", "各 score 的 best precision/recall/F1。"),
                _source("阈值扫描 CSV", "data/paper/stage4_eval_per_threshold_metrics.csv", "image/text/naive_union/joint/max 的完整 threshold sweep。"),
                _source("误差分析 metrics JSON", "data/paper/stage4_error_analysis_metrics.json", "joint/image 的 FP/FN 和互胜样本统计。"),
                _source("Joint FP examples CSV", "data/paper/stage4_joint_fp_examples.csv", "Stage 4 false positive 样例。"),
                _source("Image wins / joint loses CSV", "data/paper/stage4_image_wins_joint_loses.csv", "image-only 正确但 joint 错误的样例。"),
                _source("实验 ledger CSV", "data/experiment_ledger.csv", "所有可引用实验的 source-of-truth ledger。"),
            ],
        },
        {
            "title": "效率与系统开销",
            "status": "partial",
            "paper_use": "用于论文 Efficiency / System Overhead 表。已有数据准备和候选挖掘耗时，也已有真实 LLaVA smoke 的 GPU peak memory；完整 Stage 4 embedding/search 分项与正式训练峰值仍缺。",
            "key_numbers": [
                {"label": "200K 数据准备耗时", "value": _fmt_runtime(str(prepare.get("elapsed_seconds", ""))), "note": "下载/保存 image-caption sidecars"},
                {"label": "候选挖掘耗时", "value": _fmt_runtime(str(candidates.get("elapsed_seconds", ""))), "note": "500K candidates"},
                {"label": "Stage 4 评价耗时", "value": _fmt_runtime(str(evaluation.get("elapsed_seconds", ""))), "note": "Mac 上纯指标计算"},
                {"label": "LLaVA smoke peak memory", "value": _fmt_gib(llava_smoke.get("gpu_peak_memory_bytes")), "note": "真实模型 1-step 工程 smoke，不代表完整训练"},
                {"label": "完整 Stage 4 GPU peak memory", "value": "缺失", "note": "后续 Windows 正式实验需要记录"},
            ],
            "sources": [
                _source("200K 数据准备 metrics", "data/paper/cc3m_subset_200k_prepare_metrics.json", "数据准备 wall-clock。"),
                _source("候选挖掘 metrics", "data/paper/stage4_candidates_200k_metrics.json", "候选挖掘 runtime 和配置。"),
                _source("主评价 metrics JSON", "data/paper/stage4_eval_metrics.json", "评价脚本 runtime。"),
                _source("LLaVA smoke metrics", "data/paper/llava_stage4_real_train_smoke_E_metrics.json", "真实 LLaVA-1.5-7B 4-bit LoRA 1-step smoke 的 runtime 和 GPU peak memory。"),
            ],
        },
        {
            "title": "LLaVA 下游验证",
            "status": "partial",
            "paper_use": "用于论文 Downstream Validation 表。当前已有 A/B/C/D/E 在 200K manifest 上的训练 manifest、五组数据 smoke、Stage 4 E 真实 LLaVA smoke，以及 A/B/C/D/E 五组 512-sample pilot；完整 25K/2000-step A/B/C/D/E LoRA 训练日志和 VQAv2/TextVQA 指标仍未完成。",
            "key_numbers": _split_key_numbers(split_metrics)
            + [
                {"label": "Data smoke", "value": "A/B/C/D/E", "note": "每组检查 32 条，missing_images=0，bad_images=0"},
                {"label": "Real LLaVA smoke", "value": f"loss={_fmt_float(llava_smoke.get('final_loss'), 4)}", "note": f"steps={llava_smoke.get('steps', 'n/a')}; peak={_fmt_gib(llava_smoke.get('gpu_peak_memory_bytes'))}"},
                {"label": "Pilot 训练", "value": f"{llava['completed']}/5 complete", "note": llava["pilot_summary"]},
                {"label": "正式训练队列", "value": "running", "note": llava["current_training"]},
            ],
            "sources": [
                _source("A/B/C/D/E split sizes CSV", "data/paper/stage4_abcde_split_sizes.csv", "200K manifest 上的五组训练数据规模。"),
                _source("200K 阈值去重率 CSV", "data/paper/stage4_threshold_dedup_rates.csv", "image/text/joint/naive threshold vs dedup rate。"),
                _source("A/B/C/D/E data smoke metrics", "data/paper/llava_stage4_data_smoke_abcde_metrics.json", "五组 LLaVA JSON 的路径和图片可读性检查。"),
                _source("LLaVA E real smoke metrics", "data/paper/llava_stage4_real_train_smoke_E_metrics.json", "Stage 4 E 上真实模型 1-step LoRA 训练 smoke。"),
                _source("LLaVA pilot metrics", "data/paper/llava_stage4_pilot_metrics.json", "A/B/C/D/E 五组 512-sample / 20-step pilot 汇总。"),
                _source("LLaVA overnight queue log", "data/paper/llava_stage4_overnight_queue_20260521.log", "Windows 任务计划程序启动的 overnight queue 日志快照。"),
                _source("实验设计规则", "data/plan_requirements.json", "保留 A/B/C/D/E 设计，不默认收缩。"),
                _source("实验 ledger CSV", "data/experiment_ledger.csv", "训练完成后每组结果必须进入 ledger。"),
            ],
        },
    ]


def _paper_tables(annotation: dict[str, object]) -> list[dict[str, object]]:
    prepare = _read_json(SYNC / "cc3m_subset_200k_20260515/prepare_metrics.json")
    candidates = _read_json(SYNC / "exp_stage4_candidates_200k_manifest_20260516/metrics.json")
    high_joint = _read_json(SYNC / "exp_stage4_candidates_200k_high_joint_20260516/metrics.json")
    adjudication = _read_json(RESULTS / "exp_stage4_adjudicated_1000_200k_high_joint_20260519/metrics.json")
    evaluation = _read_json(RESULTS / "exp_stage4_eval_1000_200k_high_joint_20260519/metrics.json")
    error_analysis = _read_json(RESULTS / "exp_stage4_error_analysis_1000_200k_high_joint_20260520/metrics.json")
    split_metrics = _read_json(SPLIT_EXPERIMENT_DIR / "metrics.json")
    llava_smoke = _read_json(SYNC / "exp_llava_stage4_real_train_smoke_E_20260520/metrics.json")
    llava = _llava_pilot_status()
    best_by_score = evaluation.get("best_by_score", {})
    if not isinstance(best_by_score, dict):
        best_by_score = {}

    image = best_by_score.get("image", {})
    text = best_by_score.get("text", {})
    naive = best_by_score.get("naive_union", {})
    joint = best_by_score.get("joint", {})
    split_rows = _split_table_rows(split_metrics)

    return [
        {
            "id": "dataset-ground-truth",
            "title": "数据集与 Ground Truth 构建",
            "paper_location": "论文位置：Dataset construction；表：CC3M 候选挖掘与标注统计",
            "status": "ready",
            "what_it_answers": "我们用了什么真实数据、为什么不是随机抽样、标注集有多少正负例。",
            "recommended_claim": (
                "We construct a real CC3M-based hard-candidate benchmark by first mining likely duplicate "
                "image-caption pair-pairs from a 200K pool and then manually annotating 1,000 candidate pairs."
            ),
            "do_not_write": "不要写 raw CC3M 的自然重复率就是 29.5%；这个比例只属于 high-joint hard-candidate 标注集。",
            "table_columns": ["项目", "当前数字", "论文写法"],
            "rows": [
                ["CC3M 数据池", _fmt_int(prepare.get("saved_pairs", 200000)), "从 CC3M 准备 200K image-caption pairs。"],
                ["初筛候选 pair-pairs", _fmt_int(candidates.get("num_candidates", 500000)), "人工标注前先挖掘可能重复的候选 pair-pairs。"],
                ["High-joint 候选池", _fmt_int(high_joint.get("num_candidates", 129139)), "按 joint similarity 和 image similarity 过滤，用于标注采样。"],
                ["已标注 pair-pairs", _fmt_int(annotation.get("done", 0)), "用于 Stage 4 评价的人工标注样本。"],
                ["正例数量", _fmt_int(annotation.get("positives", 0)), "duplicate + near-duplicate。"],
                ["负例数量", _fmt_int(annotation.get("counts", {}).get("not-duplicate", 0)), "not-duplicate。"],
            ],
            "evidence": [
                _source("CC3M 200K prepare", "data/paper/cc3m_subset_200k_prepare_metrics.json", "200K pool preparation metrics."),
                _source("Candidate mining", "data/paper/stage4_candidates_200k_metrics.json", "500K mined candidate pair-pairs."),
                _source("High-joint filter", "data/paper/stage4_candidates_200k_high_joint_metrics.json", "Filtered candidate pool for annotation."),
                _source("Labeled annotations", "data/paper/stage4_annotation_1000_high_joint_labeled.csv", "Human-labeled 1,000 row benchmark."),
            ],
            "gap": "如果要正式报告 inter-annotator agreement，需要真实合作者 audit；当前 audit 只是流程占位。",
        },
        {
            "id": "main-stage4-evaluation",
            "title": "主评价：Stage 4 与 Baselines 对比",
            "paper_location": "论文表：CC3M hard-candidate benchmark 上的 Precision / Recall / F1",
            "status": "ready_with_caution",
            "what_it_answers": "Stage 4 是否比三个单模态拼接/naive union 更合理。",
            "recommended_claim": (
                "On the current hard-candidate benchmark, Stage 4 improves over the naive union baseline, "
                "but image-only remains stronger; the paper should present this honestly and use error analysis to explain why."
            ),
            "do_not_write": "不要写 Stage 4 全面超过所有单模态 baseline；当前 image-only F1 更高。",
            "table_columns": ["方法", "阈值", "Precision", "Recall", "F1", "论文备注"],
            "rows": [
                _eval_row("Image-only", image, "当前 GT 上最强 baseline。"),
                _eval_row("Text-only", text, "在 high-joint 候选集上明显过度预测正例。"),
                _eval_row("Naive union", naive, "单模态结果并集，代表简单拼接式多模态 baseline。"),
                _eval_row("Stage 4 joint", joint, "超过 naive union，但未超过 image-only。"),
            ],
            "evidence": [
                _source("Main metrics", "data/paper/stage4_eval_metrics.json", "Best P/R/F1 by score."),
                _source("Threshold sweep", "data/paper/stage4_eval_per_threshold_metrics.csv", "Complete threshold sweep."),
                _source("Experiment ledger", "data/experiment_ledger.csv", "Traceable experiment record."),
            ],
            "gap": "如果想增强主 claim，下一步不是改口径，而是按预定方案跑 fair operating point / 下游验证。",
        },
        {
            "id": "error-analysis",
            "title": "误差分析",
            "paper_location": "论文图/表：Stage 4 失败样例与 image-only 对比",
            "status": "partial",
            "what_it_answers": "为什么 Stage 4 没超过 image-only，以及它错在哪里。",
            "recommended_claim": (
                "A large fraction of Stage 4 false positives are caption-template cases, suggesting that identical or near-identical "
                "captions can over-amplify joint similarity even when image-level evidence is weaker."
            ),
            "do_not_write": "不要把误差分析写成最终算法失败；这里应写成当前表示选择的边界和后续改进方向。",
            "table_columns": ["诊断项", "当前数字", "解释"],
            "rows": [
                ["Joint false positives", _fmt_int(error_analysis.get("joint_false_positives")), "joint 判为重复但人工标签为负例。"],
                ["Joint false negatives", _fmt_int(error_analysis.get("joint_false_negatives")), "人工正例中被 joint 阈值漏掉的样本。"],
                ["Caption 完全相同的 joint FP 比例", _fmt_float(error_analysis.get("joint_fp_caption_equal_rate"), 3), "模板化 caption 是主要 FP 来源之一。"],
                ["Image 正确 / Joint 错误", _fmt_int(error_analysis.get("image_correct_joint_wrong")), "解释为什么 image-only 当前更强。"],
                ["Joint 正确 / Image 错误", _fmt_int(error_analysis.get("joint_correct_image_wrong")), "说明跨模态信号确实有增益样本。"],
            ],
            "evidence": [
                _source("Error metrics", "data/paper/stage4_error_analysis_metrics.json", "Aggregated FP/FN diagnostics."),
                _source("Joint FP examples", "data/paper/stage4_joint_fp_examples.csv", "Concrete false positive rows."),
                _source("Image wins joint loses", "data/paper/stage4_image_wins_joint_loses.csv", "Cases where image-only is correct and joint is wrong."),
                _source("Joint wins image loses", "data/paper/stage4_joint_wins_image_loses.csv", "Cases where Stage 4 is correct and image-only is wrong."),
            ],
            "gap": "需要抽几组可视化例子放进论文或 appendix，而不是只给统计数。",
        },
        {
            "id": "abcde-downstream",
            "title": "下游训练 A/B/C/D/E 数据划分",
            "paper_location": "论文表：LLaVA 训练数据规模与下游结果",
            "status": "partial",
            "what_it_answers": "原始、单模态、naive union、Stage 4 五组训练数据规模分别是多少，后续 LLaVA 怎么跑。",
            "recommended_claim": "At this stage, the dashboard supports reporting materialized A/B/C/D/E training manifests and a real LLaVA-1.5 4-bit LoRA smoke test, but not downstream benchmark performance.",
            "do_not_write": "不要写 VQAv2/TextVQA 有结果；当前 manifest 与真实模型 smoke 可证明训练链路可用，但正式下游结果还没有产生。",
            "table_columns": ["组别", "方法", "保留 pairs", "删除 pairs", "去重率", "阈值"],
            "rows": split_rows,
            "evidence": [
                _source("A/B/C/D/E split sizes", "data/paper/stage4_abcde_split_sizes.csv", "Current 200K materialized training split sizes."),
                _source("Threshold dedup rates", "data/paper/stage4_threshold_dedup_rates.csv", "Threshold vs dedup rate table."),
                _source("Split metrics", "data/paper/stage4_split_threshold_metrics.json", "Notes and source metrics."),
                _source("Experiment ledger", "data/experiment_ledger.csv", "Includes data smoke and real LLaVA-1.5 4-bit LoRA smoke records."),
            ],
            "gap": "下一步必须在 Windows 3090 上跑完整 A/B/C/D/E LLaVA LoRA，并产出 VQAv2/TextVQA 指标。",
        },
        {
            "id": "efficiency-overhead",
            "title": "系统效率与 Stage 4 开销",
            "paper_location": "论文表：Stage 4 overhead",
            "status": "partial",
            "what_it_answers": "Stage 4 增加了多少计算开销，是否可接受。",
            "recommended_claim": "Only partial timing can be written now; the real LLaVA smoke memory is recorded, while full Stage 4 component timing and full training memory are still missing.",
            "do_not_write": "不要写完整系统开销表已经完成；LLaVA smoke 的 GPU memory 不能替代完整 Stage 4 embedding/search 或正式训练开销。",
            "table_columns": ["指标", "当前数字", "状态"],
            "rows": [
                ["CC3M 200K 数据准备 wall-clock", _fmt_runtime(str(prepare.get("elapsed_seconds", ""))), "已有"],
                ["候选挖掘 wall-clock", _fmt_runtime(str(candidates.get("elapsed_seconds", ""))), "已有"],
                ["评价脚本 wall-clock", _fmt_runtime(str(evaluation.get("elapsed_seconds", ""))), "已有，但不是 GPU 开销"],
                ["LLaVA smoke GPU peak memory", _fmt_gib(llava_smoke.get("gpu_peak_memory_bytes")), "已有，工程 smoke"],
                ["完整 Stage 4 / 正式训练 GPU peak memory", "缺失", "必须在 Windows RTX 3090 上记录"],
                ["Embedding / search 分项耗时", "缺失", "完整系统表需要补"],
            ],
            "evidence": [
                _source("Prepare metrics", "data/paper/cc3m_subset_200k_prepare_metrics.json", "Data preparation timing."),
                _source("Candidate mining metrics", "data/paper/stage4_candidates_200k_metrics.json", "Candidate mining timing."),
                _source("Evaluation metrics", "data/paper/stage4_eval_metrics.json", "Evaluation runtime."),
                _source("LLaVA smoke metrics", "data/paper/llava_stage4_real_train_smoke_E_metrics.json", "Real 1-step LLaVA smoke memory/runtime."),
            ],
            "gap": "补 Windows 侧 GPU memory、CLIP embedding time、nearest-neighbor/search time、end-to-end throughput。",
        },
        {
            "id": "audit-safety",
            "title": "论文数字安全与 Claim 控制",
            "paper_location": "写作检查清单：把数字复制进论文前必须核对",
            "status": "active",
            "what_it_answers": "哪些数字现在能写，哪些必须加限定，哪些还不能写。",
            "recommended_claim": "Use the dashboard as writing view, but use experiment files and ledger as source-of-truth.",
            "do_not_write": "不要把 dashboard JSON 当 source-of-truth；它只是前端快照。",
            "table_columns": ["风险点", "当前判断", "必须动作"],
            "rows": [
                ["Agreement rate", "不是正式合作者一致性", "真实合作者 audit 前不要报告。"],
                ["Raw duplicate prevalence", "尚未估计", "不能从 high-joint benchmark 推断。"],
                ["Stage 4 superiority", "目前只超过 naive union", "必须说明 image-only 在当前 benchmark 更强。"],
                ["Downstream performance", "缺失", "LLaVA 跑完前不能宣称训练效果。"],
            ],
            "evidence": [
                _source("Data quality audit", "data/data_quality_audit.json", "Current paper-safety audit."),
                _source("Experiment ledger", "data/experiment_ledger.csv", "Source-of-truth index."),
            ],
            "gap": "写论文前逐表核对：每个数字必须有 experiment id。",
        },
    ]


def _eval_row(name: str, row: object, note: str) -> list[str]:
    if not isinstance(row, dict):
        return [name, "n/a", "n/a", "n/a", "n/a", note]
    return [
        name,
        str(row.get("threshold", "n/a")),
        _fmt_float(row.get("precision"), 3),
        _fmt_float(row.get("recall"), 3),
        _fmt_float(row.get("f1"), 3),
        note,
    ]


def _split_table_rows(metrics: dict[str, object]) -> list[list[str]]:
    splits = metrics.get("best_known_split_sizes", {})
    if not isinstance(splits, dict):
        return []
    rows = []
    for key in ["A", "B", "C", "D", "E"]:
        item = splits.get(key, {})
        if not isinstance(item, dict):
            continue
        rows.append(
            [
                key,
                str(item.get("name", "")),
                _fmt_int(item.get("kept_pairs")),
                _fmt_int(item.get("dropped_pairs")),
                _fmt_float(item.get("dedup_rate"), 3),
                str(item.get("threshold", "")),
            ]
        )
    return rows


def _plan_data_matrix(annotation: dict[str, object]) -> list[dict[str, object]]:
    evaluation = _read_json(RESULTS / "exp_stage4_eval_1000_200k_high_joint_20260519/metrics.json")
    best_by_score = evaluation.get("best_by_score", {})
    if not isinstance(best_by_score, dict):
        best_by_score = {}
    image = best_by_score.get("image", {})
    text = best_by_score.get("text", {})
    naive = best_by_score.get("naive_union", {})
    joint = best_by_score.get("joint", {})
    prepare = _read_json(SYNC / "cc3m_subset_200k_20260515/prepare_metrics.json")
    candidates = _read_json(SYNC / "exp_stage4_candidates_200k_manifest_20260516/metrics.json")
    high_joint = _read_json(SYNC / "exp_stage4_candidates_200k_high_joint_20260516/metrics.json")
    adjudication = _read_json(RESULTS / "exp_stage4_adjudicated_1000_200k_high_joint_20260519/metrics.json")
    error_analysis = _read_json(RESULTS / "exp_stage4_error_analysis_1000_200k_high_joint_20260520/metrics.json")
    split_metrics = _read_json(SPLIT_EXPERIMENT_DIR / "metrics.json")
    llava_smoke = _read_json(SYNC / "exp_llava_stage4_real_train_smoke_E_20260520/metrics.json")
    llava = _llava_pilot_status()

    paper_eval = [_source("主评价 metrics", "data/paper/stage4_eval_metrics.json", "")]
    threshold_csv = [_source("阈值扫描 CSV", "data/paper/stage4_eval_per_threshold_metrics.csv", "")]
    dedup_rate_csv = [_source("200K 阈值去重率 CSV", "data/paper/stage4_threshold_dedup_rates.csv", "")]
    split_csv = [_source("A/B/C/D/E split sizes CSV", "data/paper/stage4_abcde_split_sizes.csv", "")]
    llava_smoke_sources = [
        _source("A/B/C/D/E data smoke metrics", "data/paper/llava_stage4_data_smoke_abcde_metrics.json", ""),
        _source("LLaVA E real smoke metrics", "data/paper/llava_stage4_real_train_smoke_E_metrics.json", ""),
        _source("实验 ledger", "data/experiment_ledger.csv", ""),
    ]
    error_sources = [
        _source("误差分析 metrics", "data/paper/stage4_error_analysis_metrics.json", ""),
        _source("Joint FP examples", "data/paper/stage4_joint_fp_examples.csv", ""),
        _source("Image wins joint loses", "data/paper/stage4_image_wins_joint_loses.csv", ""),
    ]
    annotation_sources = [
        _source("已标注 CSV", "data/paper/stage4_annotation_1000_high_joint_labeled.csv", ""),
        _source("Adjudicated CSV", "data/paper/stage4_adjudicated_annotations.csv", ""),
    ]
    candidate_sources = [
        _source("200K prepare", "data/paper/cc3m_subset_200k_prepare_metrics.json", ""),
        _source("candidate metrics", "data/paper/stage4_candidates_200k_metrics.json", ""),
        _source("high-joint metrics", "data/paper/stage4_candidates_200k_high_joint_metrics.json", ""),
    ]

    return [
        {
            "experiment": "实验 1：跨模态去重模块 Stage 4",
            "status": "active",
            "purpose": "核心 novelty：用 image-caption pair 级 joint embedding 做跨模态去重。",
            "items": [
                _matrix_item(
                    "表 1.1 Joint embedding 方式对比",
                    "partial",
                    "当前只有 concat/joint 第一版；weighted sum α=0.3/0.5/0.7 未跑",
                    paper_eval,
                    "需要补 weighted sum 或在方案中说明本轮只采用 concat。",
                    "concat vs weighted sum 在 1000 条 ground truth 上的 P/R/F1。",
                ),
                _matrix_item(
                    "表 1.2 τ_cross 阈值扫描",
                    "complete",
                    "joint best F1=0.583@0.85",
                    threshold_csv,
                    "",
                    "τ_cross 对 P/R/F1 的影响。",
                ),
                _matrix_item(
                    "表 1.3 最优配置最终性能",
                    "complete",
                    f"Stage 4 joint: F1={_score_text(joint)}, {_threshold_note(joint)}",
                    paper_eval,
                    "",
                    "论文主表可引用的 Stage 4 最优配置。",
                ),
                _matrix_item(
                    "表 1.4 与 naive multimodal baseline 对比",
                    "complete",
                    f"naive F1={_score_text(naive)}; Stage 4 F1={_score_text(joint)}; joint_fp_caption_equal_rate={_fmt_float(error_analysis.get('joint_fp_caption_equal_rate'), 3)}",
                    paper_eval + error_sources,
                    "Stage 4 打过 naive union，但没打过 image-only，写作需说明。",
                    "证明跨模态联合处理相对简单拼接/并集的增量价值。",
                ),
                _matrix_item(
                    "表 1.5 计算开销",
                    "partial",
                    f"candidate mining={_fmt_runtime(str(candidates.get('elapsed_seconds', '')))}",
                    candidate_sources,
                    "缺 GPU peak memory、embedding time、cluster/search 分项、100K end-to-end。",
                    "100K/200K 图文对上的效率数据。",
                ),
            ],
        },
        {
            "experiment": "实验 2：CC3M 真实 Ground Truth 标注",
            "status": "complete",
            "purpose": "替换合成数据评估，回应 synthetic benchmark 不被认可的问题。",
            "items": [
                _matrix_item(
                    "表 2.1 数据集来源与采样",
                    "complete",
                    f"CC3M pool={_fmt_int(prepare.get('saved_pairs', 200000))}; candidates={_fmt_int(candidates.get('num_candidates', 500000))}; high-joint={_fmt_int(high_joint.get('num_candidates', 129139))}",
                    candidate_sources,
                    "",
                    "CC3M 来源、下载规模、候选挖掘和采样策略。",
                ),
                _matrix_item(
                    "表 2.2 标注汇总",
                    "complete",
                    f"labeled={_fmt_int(annotation.get('done', 0))}; audit_rows={_fmt_int(annotation.get('audit_rows', 0))}",
                    annotation_sources,
                    "当前 audit 为内部默认完成；若正式写双人一致性，仍需真实合作者抽查。",
                    "标注数量、标注人、audit/adjudication。",
                ),
                _matrix_item(
                    "表 2.3 标签分布",
                    "complete",
                    f"duplicate={annotation.get('counts', {}).get('duplicate', 0)}; near={annotation.get('counts', {}).get('near-duplicate', 0)}; not={annotation.get('counts', {}).get('not-duplicate', 0)}",
                    annotation_sources,
                    "",
                    "三类标签数量与占比。",
                ),
                _matrix_item(
                    "表 2.4 标注一致性指标",
                    "partial",
                    f"agreement_rate={_fmt_float(adjudication.get('agreement_rate', 0), 3)}; conflicts={adjudication.get('num_conflicts', 0)}",
                    [_source("adjudication metrics", "data/paper/stage4_adjudication_metrics.json", "")],
                    "Cohen's kappa/Fleiss' kappa 还未计算；当前不是严格双盲两人标注。",
                    "标注质量控制指标。",
                ),
            ],
        },
        {
            "experiment": "实验 3：SSCD Baseline 补充",
            "status": "pending",
            "purpose": "回应图像 baseline 不够强的问题；在 ImageNet-Expanded 和 CC3M GT 上补 SSCD。",
            "items": [
                _matrix_item("表 3.1 SSCD 在 ImageNet-Expanded 上的阈值扫描", "pending", "", [], "需要下载 SSCD 权重并跑 ImageNet-Expanded。", "阈值、dedup rate、P/R/F1、下游 Acc。"),
                _matrix_item("表 3.2 SSCD 在 CC3M Ground Truth 上的阈值扫描", "pending", "", [], "需要在当前 1000 条 CC3M GT 上跑 SSCD image similarity。", "SSCD 在真实 CC3M GT 上的 P/R/F1。"),
                _matrix_item("表 3.3 SSCD 与原稿方法对比", "pending", "", [], "需要整合原 Table 5 与 SSCD 新结果。", "更新后的 image baseline 主表。"),
            ],
        },
        {
            "experiment": "实验 4：MLLM 下游训练验证",
            "status": "active",
            "purpose": "最重要的下游验证：证明去重对 LLaVA-1.5-7B LoRA 训练有实际收益或不伤性能。",
            "items": [
                _matrix_item("表 4.0 LLaVA 训练链路 smoke", "partial", f"A/B/C/D/E data smoke 通过；E real smoke steps={llava_smoke.get('steps', 'n/a')}; final_loss={_fmt_float(llava_smoke.get('final_loss'), 4)}; peak={_fmt_gib(llava_smoke.get('gpu_peak_memory_bytes'))}", llava_smoke_sources, "只有 Stage 4 E 跑了 1 step；还不是正式 A/B/C/D/E 训练或下游评测。", "验证 Windows 3090 上真实 LLaVA-1.5-7B 4-bit LoRA 训练入口可用。"),
                _matrix_item("表 4.0b LLaVA A/B/C/D/E pilot", "complete", f"{llava['completed']}/5 pilot complete; {llava['pilot_summary']}; {llava['current_training']}", [_source("pilot metrics", "data/paper/llava_stage4_pilot_metrics.json", ""), _source("queue log", "data/paper/llava_stage4_overnight_queue_20260521.log", ""), _source("ledger", "data/experiment_ledger.csv", "")], "这仍不是 VQAv2/TextVQA 下游性能，只能证明五组 split 均可完成真实 LoRA 训练 pilot。", "A/B/C/D/E 五组 512-sample / 20-step pilot 训练状态。"),
                _matrix_item("表 4.1 五组训练数据规模 A/B/C/D/E", "complete", _split_summary(split_metrics), split_csv, "训练 manifest 已生成，且 data smoke 已验证；下一步是 Windows 3090 上实际训练和评测。", "原始样本数、去重后样本数、去重率。"),
                _matrix_item("表 4.2 五组训练时间", "pending", "", [], "需要 Windows 3090 上记录完整 A/B/C/D/E GPU-hour 和 wall-clock；当前只有 E 组 1-step smoke runtime。", "训练效率收益。"),
                _matrix_item("表 4.3 VQAv2 评测结果", "pending", "", [], "需要每组至少一个 seed；理想 2 seeds。", "配置、seed、accuracy。"),
                _matrix_item("表 4.4 TextVQA 评测结果", "pending", "", [], "时间允许再跑；不允许虚构。", "配置、seed、accuracy。"),
                _matrix_item("表 4.5 汇总性能表", "pending", "", [], "依赖 A/B/C/D/E 训练和评测完成。", "VQAv2/TextVQA mean ± std。"),
            ],
        },
        {
            "experiment": "实验 5：阈值敏感性分析",
            "status": "partial",
            "purpose": "更新原 Figure 3，展示跨模态阈值和单模态阈值对去重率/性能的影响。",
            "items": [
                _matrix_item("表 5.1 各模态阈值 vs 去重率", "complete", "已完成 200K manifest 上 image/text/joint/naive threshold vs dedup-rate", dedup_rate_csv, "音频不属于当前 CIKM Plan B 主线，暂不补。", "图像、文本、音频、跨模态阈值曲线。"),
                _matrix_item("表 5.2 各模态最优阈值与去重率", "complete", f"image best={_score_text(image)}@{image.get('threshold')}; joint best={_score_text(joint)}@{joint.get('threshold')}", paper_eval + dedup_rate_csv, "音频不属于当前 CIKM Plan B 主线，暂不补。", "最优阈值选择依据。"),
                _matrix_item("表 5.3 跨模态与单模态阈值组合", "pending", "", [], "需要组合扫描 image/text/cross thresholds。", "联合去重率或相关分析。"),
            ],
        },
        {
            "experiment": "实验 6：消融研究",
            "status": "pending",
            "purpose": "更新原 Table 8，量化 Stage 4 以及各单模态组件贡献。",
            "items": [
                _matrix_item("表 6.1 各 ablation 配置去重率", "pending", "", [], "需要 Full/w-o Stage4/w-o Image/w-o Text 等配置跑完。", "Mixed-Test 或 CC3M 上的各组件去重率。"),
                _matrix_item("表 6.2 各 ablation 配置 MLLM 下游性能", "pending", "", [], "依赖 LLaVA 下游训练。", "VQAv2/TextVQA acc。"),
                _matrix_item("表 6.3 Stage 4 设计选择对比", "partial", f"concat/joint F1={_score_text(joint)}", paper_eval, "weighted sum α=0.3/0.5/0.7 未跑。", "concat vs weighted sum 的消融。"),
            ],
        },
        {
            "experiment": "统一实验记录与论文数字一致性",
            "status": "active",
            "purpose": "保证摘要、正文、表格、讨论里的数字都能追溯到 source-of-truth。",
            "items": [
                _matrix_item("实验 ledger", "complete", "已建立并持续更新", [_source("ledger CSV", "data/experiment_ledger.csv", "")], "", "每个可引用实验一行。"),
                _matrix_item("论文写作数据入口", "complete", "已上线", [_source("paper writing data", "data/paper_writing_data.json", ""), _source("plan matrix", "data/plan_data_matrix.json", "")], "", "合作者可点击查看当前数字和源文件。"),
                _matrix_item("论文正文数字替换", "pending", "", [], "需要开始修改 paper/latex/main.tex，并给每个表格绑定 experiment id。", "避免数字不一致。"),
            ],
        },
    ]


def _matrix_item(
    name: str,
    status: str,
    current_numbers: str,
    sources: list[dict[str, str]],
    gap: str,
    description: str,
) -> dict[str, object]:
    paper_table_status = _paper_table_status(name, status)
    has_sources = bool(sources)
    return {
        "name": name,
        "status": status,
        "data_status": status,
        "paper_table_status": paper_table_status,
        "description": description,
        "current_numbers": current_numbers,
        "existing_data": _split_matrix_text(current_numbers) if current_numbers else [],
        "missing_data": _split_matrix_text(gap) if gap else [],
        "next_action": _next_action_for_matrix_item(name, status, gap, has_sources),
        "sources": sources,
        "gap": gap,
    }


def _paper_table_status(name: str, data_status: str) -> str:
    if not name.startswith("表 "):
        return "ready" if data_status == "complete" else data_status
    if data_status == "pending":
        return "missing_data"
    return "data_ready_table_missing"


def _next_action_for_matrix_item(name: str, status: str, gap: str, has_sources: bool) -> str:
    if status == "pending":
        return gap or "先运行对应实验并写入 source-of-truth。"
    if status == "partial":
        return gap or "补齐缺失数据后再生成论文表。"
    if name.startswith("表 "):
        return "已有部分/全部数据；下一步是生成论文表格草稿并绑定 experiment id。"
    if has_sources:
        return "保持随实验更新。"
    return ""


def _split_matrix_text(value: str) -> list[str]:
    return [part.strip() for part in value.split(";") if part.strip()]


def _copy_paper_source_files() -> None:
    paper_dir = DATA_DIR / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    copies = {
        SYNC / "cc3m_subset_200k_20260515/prepare_metrics.json": "cc3m_subset_200k_prepare_metrics.json",
        SYNC / "cc3m_subset_200k_20260515/validation_summary.json": "cc3m_subset_200k_validation_summary.json",
        SYNC / "exp_stage4_candidates_200k_manifest_20260516/metrics.json": "stage4_candidates_200k_metrics.json",
        SYNC / "exp_stage4_candidates_200k_high_joint_20260516/metrics.json": "stage4_candidates_200k_high_joint_metrics.json",
        SYNC / "exp_stage4_annotation_1000_200k_high_joint_20260516/metrics.json": "stage4_annotation_1000_high_joint_metrics.json",
        SYNC / "exp_stage4_annotation_1000_200k_high_joint_20260516/annotation_sheet_labeled.csv": "stage4_annotation_1000_high_joint_labeled.csv",
        RESULTS / "exp_stage4_adjudicated_1000_200k_high_joint_20260519/adjudicated_annotations.csv": "stage4_adjudicated_annotations.csv",
        RESULTS / "exp_stage4_adjudicated_1000_200k_high_joint_20260519/metrics.json": "stage4_adjudication_metrics.json",
        RESULTS / "exp_stage4_eval_1000_200k_high_joint_20260519/metrics.json": "stage4_eval_metrics.json",
        RESULTS / "exp_stage4_eval_1000_200k_high_joint_20260519/per_threshold_metrics.csv": "stage4_eval_per_threshold_metrics.csv",
        RESULTS / "exp_stage4_error_analysis_1000_200k_high_joint_20260520/metrics.json": "stage4_error_analysis_metrics.json",
        RESULTS / "exp_stage4_error_analysis_1000_200k_high_joint_20260520/joint_fp_examples.csv": "stage4_joint_fp_examples.csv",
        RESULTS / "exp_stage4_error_analysis_1000_200k_high_joint_20260520/joint_fn_examples.csv": "stage4_joint_fn_examples.csv",
        RESULTS / "exp_stage4_error_analysis_1000_200k_high_joint_20260520/image_wins_joint_loses.csv": "stage4_image_wins_joint_loses.csv",
        RESULTS / "exp_stage4_error_analysis_1000_200k_high_joint_20260520/joint_wins_image_loses.csv": "stage4_joint_wins_image_loses.csv",
        SPLIT_EXPERIMENT_DIR / "metrics.json": "stage4_split_threshold_metrics.json",
        SPLIT_EXPERIMENT_DIR / "abcde_split_sizes.csv": "stage4_abcde_split_sizes.csv",
        SPLIT_EXPERIMENT_DIR / "threshold_dedup_rates.csv": "stage4_threshold_dedup_rates.csv",
        RESULTS / "exp_llava_stage4_data_smoke_abcde_20260520/metrics.json": "llava_stage4_data_smoke_abcde_metrics.json",
        SYNC / "exp_llava_stage4_real_train_smoke_E_20260520/metrics.json": "llava_stage4_real_train_smoke_E_metrics.json",
        SYNC / "llava_stage4_pilot_metrics.json": "llava_stage4_pilot_metrics.json",
        SYNC / "llava_stage4_overnight_queue_20260521.log": "llava_stage4_overnight_queue_20260521.log",
    }
    for src, name in copies.items():
        if src.exists():
            dest = paper_dir / name
            shutil.copyfile(src, dest)
            if dest.suffix == ".csv":
                dest.write_text(dest.read_text(encoding="utf-8").replace("\r\n", "\n"), encoding="utf-8")


def _source(title: str, href: str, description: str) -> dict[str, str]:
    return {"title": title, "href": href, "description": description}


def _fmt_float(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _score_text(row: object) -> str:
    if not isinstance(row, dict):
        return "n/a"
    return _fmt_float(row.get("f1"), 3)


def _threshold_note(row: object) -> str:
    if not isinstance(row, dict):
        return "暂无结果"
    return (
        f"tau={row.get('threshold')}; "
        f"P={_fmt_float(row.get('precision'), 3)}; "
        f"R={_fmt_float(row.get('recall'), 3)}"
    )


def _split_key_numbers(metrics: dict[str, object]) -> list[dict[str, str]]:
    splits = metrics.get("best_known_split_sizes", {})
    if not isinstance(splits, dict) or not splits:
        return [
            {"label": "Raw A", "value": "待生成", "note": "no dedup"},
            {"label": "Image-only B", "value": "待生成", "note": "image-only dedup"},
            {"label": "Naive D", "value": "待生成", "note": "image + text union"},
            {"label": "Stage 4 E", "value": "待生成", "note": "pair-level cross-modal dedup"},
        ]
    labels = {
        "A": "Raw A",
        "B": "Image-only B",
        "C": "Text-only C",
        "D": "Naive union D",
        "E": "Stage 4 E",
    }
    rows = []
    for key in ["A", "B", "C", "D", "E"]:
        row = splits.get(key, {})
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "label": labels[key],
                "value": _fmt_int(row.get("kept_pairs")),
                "note": f"dropped={_fmt_int(row.get('dropped_pairs'))}; rate={_fmt_float(row.get('dedup_rate'), 3)}; {row.get('threshold', '')}",
            }
        )
    return rows


def _split_summary(metrics: dict[str, object]) -> str:
    return "; ".join(f"{item['label']} kept={item['value']}" for item in _split_key_numbers(metrics))


def _annotation_percent() -> int:
    status = _annotation_status()
    return int(status["percent"])


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_int(value: object) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_gib(value: object) -> str:
    try:
        return f"{float(value) / (1024**3):.3f} GiB"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_runtime(raw: str) -> str:
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return "n/a"
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f} s"


def _runtime_to_seconds(runtime: str) -> float | None:
    if runtime == "n/a":
        return None
    try:
        value_raw, unit = runtime.split(" ", 1)
        value = float(value_raw)
    except ValueError:
        return None
    if unit == "h":
        return value * 3600
    if unit == "min":
        return value * 60
    if unit == "s":
        return value
    return None


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    raise SystemExit(main())
