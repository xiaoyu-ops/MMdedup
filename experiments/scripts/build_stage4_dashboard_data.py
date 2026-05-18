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
        "annotation": annotation,
        "artifacts": _artifacts(),
        "data_exports": _data_exports(),
        "risks": [
            {
                "level": "high",
                "title": "人工标注尚未完成",
                "detail": "1000 条 high-joint 标注表完成前，Checkpoint 3 的 P/R/F1 不能正式计算。",
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
            "完成 high-joint 1000 条主标注表的人工标注。",
            "让合作者抽查 200 条 audit rows，并计算一致率。",
            "基于 adjudicated labels 计算 Stage 4 vs image-only/text-only/naive-union 的 P/R/F1。",
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
            "status": "active",
            "percent": _annotation_percent(),
            "detail": "1000 条主标注表是当前正式 P/R/F1 评价的关键阻塞项。",
        },
        {
            "name": "Stage 4 主评价",
            "status": "blocked",
            "percent": 0,
            "detail": "等待人工标注、合作者抽查和 adjudication。",
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
                "1000 条主标注表",
                "200 条 audit rows",
            ],
            "evidence": [
                "experiments/results/plan_b_stage4/windows_sync/cc3m_subset_200k_20260515/validation_summary.json",
                "experiments/results/plan_b_stage4/windows_sync/exp_stage4_candidates_200k_high_joint_20260516/metrics.json",
                "experiments/results/plan_b_stage4/windows_sync/exp_stage4_annotation_1000_200k_high_joint_20260516/annotation_sheet.csv",
            ],
        },
        {
            "name": "Stage 4 主评价",
            "status": "blocked",
            "required_data": [
                "adjudicated human labels",
                "image-only baseline scores",
                "text-only baseline scores",
                "naive union baseline scores",
                "Stage 4 joint scores",
            ],
            "current_outputs": [
                "评价脚本已存在",
                "正式指标等待标注结果",
            ],
            "evidence": [
                "experiments/scripts/evaluate_stage4_groundtruth.py",
                "experiments/scripts/adjudicate_stage4_annotations.py",
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
            "title": "Daily log",
            "path": "experiments/results/plan_b_stage4/daily_logs/2026-05-16.md",
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
    ledger = RESULTS / "experiment_ledger.csv"
    if ledger.exists():
        shutil.copyfile(ledger, DATA_DIR / "experiment_ledger.csv")


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
