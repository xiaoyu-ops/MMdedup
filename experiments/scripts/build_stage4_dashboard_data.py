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
                "level": "medium",
                "title": "embedding cache 仍只在 Windows 端",
                "detail": "200K CLIP embedding cache 约 762 MB，目前尚未完整镜像回 Mac。",
            },
            {
                "level": "medium",
                "title": "LLaVA 下游验证尚未开始",
                "detail": "A/B/C/D/E 五组 LoRA 训练需要在标注和 Stage 4 阈值选择之后继续推进。",
            },
        ],
        "next_steps": [
            "基于当前 1000 条结果做误差分析：找出 image-only 赢在哪里、joint 误杀/漏检在哪里。",
            "固定 Stage 4 候选阈值，并准备用于 A/B/C/D/E 的训练数据划分。",
            "将 Windows 端 embedding cache 和后续 split 结果同步回 Mac source-of-truth。",
            "确定阈值后准备 A/B/C/D/E 五组 LLaVA 训练数据划分。",
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
            "percent": 70,
            "detail": "已完成第一版 P/R/F1：joint 优于 naive_union，但尚未超过 image-only，需要误差分析。",
        },
        {
            "name": "LLaVA 下游验证",
            "status": "pending",
            "percent": 0,
            "detail": "阈值确定后继续准备 A/B/C/D/E 五组数据并运行 LoRA。",
        },
    ]


def _plan_requirements() -> list[dict[str, object]]:
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
            "status": "pending",
            "required_data": [
                "raw split A",
                "image-only split B",
                "text-only split C",
                "naive union split D",
                "Stage 4 split E",
                "LLaVA LoRA logs 与 VQAv2/TextVQA metrics",
            ],
            "current_outputs": [
                "尚未开始",
            ],
            "evidence": [
                "AGENTS.md",
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
                "GPU peak memory 尚未记录",
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


def _data_exports() -> list[dict[str, str]]:
    return [
        {
            "title": "Dashboard status JSON",
            "href": "data/status.json",
            "description": "Dashboard 使用的完整状态快照。",
        },
        {
            "title": "图表数据 JSON",
            "href": "data/charts.json",
            "description": "candidate funnel、标注分布、阶段进度和实验耗时的图表数据。",
        },
        {
            "title": "方案数据需求 JSON",
            "href": "data/plan_requirements.json",
            "description": "每个 Plan B 产物需要的数据、当前产物和证据路径。",
        },
        {
            "title": "最新标注状态 JSON",
            "href": "data/latest_annotation_status.json",
            "description": "当前标签数量、标注进度和输出路径。",
        },
        {
            "title": "实验 ledger CSV",
            "href": "data/experiment_ledger.csv",
            "description": "source-of-truth 实验台账快照。",
        },
        {
            "title": "论文写作数据 JSON",
            "href": "data/paper_writing_data.json",
            "description": "按论文段落和表格组织的可引用数字、解释和证据文件链接。",
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
    ledger = RESULTS / "experiment_ledger.csv"
    if ledger.exists():
        shutil.copyfile(ledger, DATA_DIR / "experiment_ledger.csv")
    _copy_paper_source_files()


def _paper_writing_data(annotation: dict[str, object]) -> list[dict[str, object]]:
    prepare = _read_json(SYNC / "cc3m_subset_200k_20260515/prepare_metrics.json")
    validation = _read_json(SYNC / "cc3m_subset_200k_20260515/validation_summary.json")
    candidates = _read_json(SYNC / "exp_stage4_candidates_200k_manifest_20260516/metrics.json")
    high_joint = _read_json(SYNC / "exp_stage4_candidates_200k_high_joint_20260516/metrics.json")
    annotation_metrics = _read_json(SYNC / "exp_stage4_annotation_1000_200k_high_joint_20260516/metrics.json")
    adjudication = _read_json(RESULTS / "exp_stage4_adjudicated_1000_200k_high_joint_20260519/metrics.json")
    evaluation = _read_json(RESULTS / "exp_stage4_eval_1000_200k_high_joint_20260519/metrics.json")
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
            "paper_use": "用于论文 Main Results 表。当前结论是 Stage 4 joint 优于 naive union，但 image-only 在这批候选集上更强，写作时必须如实说明。",
            "key_numbers": [
                {"label": "Image-only best F1", "value": _score_text(image), "note": _threshold_note(image)},
                {"label": "Text-only best F1", "value": _score_text(text), "note": _threshold_note(text)},
                {"label": "Naive union best F1", "value": _score_text(naive), "note": _threshold_note(naive)},
                {"label": "Stage 4 joint best F1", "value": _score_text(joint), "note": _threshold_note(joint)},
            ],
            "sources": [
                _source("主评价 metrics JSON", "data/paper/stage4_eval_metrics.json", "各 score 的 best precision/recall/F1。"),
                _source("阈值扫描 CSV", "data/paper/stage4_eval_per_threshold_metrics.csv", "image/text/naive_union/joint/max 的完整 threshold sweep。"),
                _source("实验 ledger CSV", "data/experiment_ledger.csv", "所有可引用实验的 source-of-truth ledger。"),
            ],
        },
        {
            "title": "效率与系统开销",
            "status": "partial",
            "paper_use": "用于论文 Efficiency / System Overhead 表。已有数据准备和候选挖掘耗时，GPU peak memory 还缺。",
            "key_numbers": [
                {"label": "200K 数据准备耗时", "value": _fmt_runtime(str(prepare.get("elapsed_seconds", ""))), "note": "下载/保存 image-caption sidecars"},
                {"label": "候选挖掘耗时", "value": _fmt_runtime(str(candidates.get("elapsed_seconds", ""))), "note": "500K candidates"},
                {"label": "Stage 4 评价耗时", "value": _fmt_runtime(str(evaluation.get("elapsed_seconds", ""))), "note": "Mac 上纯指标计算"},
                {"label": "GPU peak memory", "value": "缺失", "note": "后续 Windows 实验需要记录"},
            ],
            "sources": [
                _source("200K 数据准备 metrics", "data/paper/cc3m_subset_200k_prepare_metrics.json", "数据准备 wall-clock。"),
                _source("候选挖掘 metrics", "data/paper/stage4_candidates_200k_metrics.json", "候选挖掘 runtime 和配置。"),
                _source("主评价 metrics JSON", "data/paper/stage4_eval_metrics.json", "评价脚本 runtime。"),
            ],
        },
        {
            "title": "LLaVA 下游验证",
            "status": "pending",
            "paper_use": "用于论文 Downstream Validation 表。当前尚未开始，后续要放 A/B/C/D/E 五组 LoRA 训练日志和 VQAv2/TextVQA 指标。",
            "key_numbers": [
                {"label": "Raw A", "value": "待生成", "note": "no dedup"},
                {"label": "Image-only B", "value": "待生成", "note": "image-only dedup"},
                {"label": "Naive D", "value": "待生成", "note": "image + text union"},
                {"label": "Stage 4 E", "value": "待生成", "note": "pair-level cross-modal dedup"},
            ],
            "sources": [
                _source("实验设计规则", "data/plan_requirements.json", "保留 A/B/C/D/E 设计，不默认收缩。"),
                _source("实验 ledger CSV", "data/experiment_ledger.csv", "训练完成后每组结果必须进入 ledger。"),
            ],
        },
    ]


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
    }
    for src, name in copies.items():
        if src.exists():
            shutil.copyfile(src, paper_dir / name)


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
