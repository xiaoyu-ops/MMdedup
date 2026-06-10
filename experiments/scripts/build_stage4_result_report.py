"""Build the Stage 4 result report as Markdown, HTML, and PDF.

The report is conservative by design: it only fills numbers traceable to the
Plan B source-of-truth files. Pending E/downstream cells are left blank so they
can be filled after the Windows RTX 3090 runs finish.
"""

from __future__ import annotations

import csv
import html
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "experiments/results/plan_b_stage4"
REPORT_DIR = ROOT / "docs/reports"
MD_PATH = REPORT_DIR / "stage4_final_result_report.md"
HTML_PATH = REPORT_DIR / "stage4_final_result_report.html"
PDF_PATH = REPORT_DIR / "stage4_final_result_report.pdf"


@dataclass
class ReportTable:
    title: str
    columns: list[str]
    rows: list[list[Any]]
    note: str = ""


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_report_data()
    tables = build_tables(data)
    MD_PATH.write_text(build_markdown(data, tables), encoding="utf-8")
    HTML_PATH.write_text(build_html(data, tables), encoding="utf-8")
    build_pdf_from_html()
    print(MD_PATH)
    print(HTML_PATH)
    print(PDF_PATH)
    return 0


def load_report_data() -> dict[str, Any]:
    with (RESULT_ROOT / "experiment_ledger.csv").open(encoding="utf-8") as f:
        ledger_rows = list(csv.DictReader(f))
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ledger_rows": ledger_rows,
        "adjudication": read_json("exp_stage4_adjudicated_1000_200k_high_joint_20260519/metrics.json"),
        "stage4_eval": read_json("exp_stage4_eval_1000_200k_high_joint_20260519/metrics.json"),
        "error_analysis": read_json("exp_stage4_error_analysis_1000_200k_high_joint_20260520/metrics.json"),
        "training_manifests": read_json("exp_stage4_training_manifests_200k_20260520/metrics.json"),
        "data_audit": read_json("data_audits/2026-05-20_data_reasonableness_audit.json"),
        "threshold_metrics": read_csv("exp_stage4_eval_1000_200k_high_joint_20260519/per_threshold_metrics.csv"),
        "threshold_dedup_rates": read_csv("exp_stage4_split_threshold_200k_20260520/threshold_dedup_rates.csv"),
        "cc3m_prepare": read_json("windows_sync/cc3m_subset_200k_20260515/prepare_metrics.json"),
        "candidate_mining": read_json("windows_sync/exp_stage4_candidates_200k_manifest_20260516/metrics.json"),
        "split_threshold": read_json("exp_stage4_split_threshold_200k_20260520/metrics.json"),
        "llava_metrics": load_llava_train25k_metrics(),
    }


def read_json(rel_path: str) -> dict[str, Any]:
    with (RESULT_ROOT / rel_path).open(encoding="utf-8") as f:
        return json.load(f)


def read_csv(rel_path: str) -> list[dict[str, str]]:
    with (RESULT_ROOT / rel_path).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_llava_train25k_metrics() -> dict[str, dict[str, Any]]:
    mapping = {
        "A": "windows_sync/exp_llava_stage4_train25k_A_raw_25000_2000steps_20260521/metrics.json",
        "B": "windows_sync/exp_llava_stage4_train25k_B_image_only_25000_2000steps_20260521/metrics.json",
        "C": "windows_sync/exp_llava_stage4_train25k_C_text_only_25000_2000steps_20260521/metrics.json",
        "D": "windows_sync/exp_llava_stage4_train25k_D_naive_union_25000_2000steps_20260521/metrics.json",
    }
    out: dict[str, dict[str, Any]] = {}
    for split, rel_path in mapping.items():
        path = RESULT_ROOT / rel_path
        if path.exists():
            with path.open(encoding="utf-8") as f:
                out[split] = json.load(f)
    return out


def build_tables(data: dict[str, Any]) -> dict[str, ReportTable]:
    audit = data["data_audit"]
    checks = audit["hard_consistency_checks"]
    labels = audit["current_key_numbers"]["label_distribution"]
    best = data["stage4_eval"]["best_by_score"]
    err = data["error_analysis"]
    splits = data["training_manifests"]["best_known_split_sizes"]
    llava = data["llava_metrics"]

    data_inventory = ReportTable(
        "表 1. 方案所需数据清单",
        ["数据项", "当前值", "论文用途", "Source-of-truth"],
        [
            ["CC3M pool", fmt_int(checks["manifest_rows"]), "数据构建与训练池", "data audit"],
            ["Candidate pair-pairs", fmt_int(checks["candidate_rows"]), "hard-candidate mining", "data audit"],
            ["High-joint candidates", fmt_int(checks["high_joint_rows"]), "标注候选池", "data audit"],
            ["Annotated pair-pairs", fmt_int(checks["annotation_rows"]), "Stage 4 GT benchmark", "adjudication metrics"],
            ["Positive labels", fmt_int(labels["positive_total"]), "duplicate + near-duplicate", "data audit"],
            ["Negative labels", fmt_int(labels["not_duplicate"]), "not-duplicate", "data audit"],
        ],
        "标注集来自 mined hard candidates，不能解释为原始 CC3M 的自然重复率。",
    )

    annotation = ReportTable(
        "表 2. 人工标注与 Audit 状态",
        ["指标", "数值", "解释"],
        [
            ["duplicate", labels["duplicate"], "严格重复"],
            ["near-duplicate", labels["near_duplicate"], "语义或视觉近重复"],
            ["not-duplicate", labels["not_duplicate"], "负类"],
            ["audit rows", data["adjudication"]["num_audited_rows"], "当前内部 audit 行数"],
            ["agreement rate", "暂不报告", "audit label 当前默认等于 primary label，不是真实合作者复核"],
        ],
        "真实合作者一致性统计尚未完成，论文不能声称 agreement_rate=1.0。",
    )

    main_eval = ReportTable(
        "表 3. 1,000 条标注候选对上的 Stage 4 主评价",
        ["方法", "阈值", "Precision", "Recall", "F1", "TP", "FP", "TN", "FN"],
        [
            make_eval_row("Image-only", best["image"]),
            make_eval_row("Text-only", best["text"]),
            make_eval_row("Naive union", best["naive_union"]),
            make_eval_row("Stage 4 joint", best["joint"]),
        ],
        "Stage 4 joint 优于 text-only/naive union，但当前 image-only 的 F1 最高，写作时必须如实说明。",
    )

    error_analysis = ReportTable(
        "表 4. 误差分析摘要",
        ["指标", "数值", "论文中如何使用"],
        [
            ["Stage 4 false positives", err["joint_false_positives"], "分析过删风险"],
            ["Stage 4 false negatives", err["joint_false_negatives"], "分析漏检风险"],
            ["Image correct / joint wrong", err["image_correct_joint_wrong"], "解释 image-only 当前更强的原因"],
            ["Joint correct / image wrong", err["joint_correct_image_wrong"], "说明 Stage 4 的增量价值"],
            [
                "Joint FP with identical captions",
                f"{err['joint_fp_caption_equal']} ({pct(err['joint_fp_caption_equal_rate'])})",
                "caption-template failure mode",
            ],
        ],
    )

    split_rows = []
    for split in ["A", "B", "C", "D", "E"]:
        row = splits[split]
        split_rows.append(
            [
                split,
                row["name"],
                fmt_int(row["raw_pairs"]),
                fmt_int(row["kept_pairs"]),
                fmt_int(row["dropped_pairs"]),
                pct(row["dedup_rate"]),
                row["threshold"],
            ]
        )
    split_sizes = ReportTable(
        "表 5. 200K 图文池上的 A/B/C/D/E 训练数据规模",
        ["配置", "名称", "原始样本", "保留", "删除", "去重率", "规则"],
        split_rows,
        "这些 split sizes 用于定义下游训练 manifest；E 是 Stage 4 joint 对照组。",
    )

    threshold_scan = ReportTable(
        "表 6. Stage 4 joint 阈值扫描展开表",
        ["τ_cross", "Precision", "Recall", "F1", "200K 去重率", "备注"],
        make_joint_threshold_rows(data),
        "评价指标来自 1,000 条人工标注 hard candidates；去重率来自 200K CC3M mined candidate graph。",
    )

    efficiency = ReportTable(
        "表 7. 当前可报告效率表",
        ["阶段", "规模", "时间", "Throughput", "硬件/备注"],
        make_efficiency_rows(data),
        "这是当前已有 source-of-truth 下可报告的效率版本；GPU peak memory 还需要后续补充到正式效率表。",
    )

    descriptions = {
        "A": "raw",
        "B": "image-only",
        "C": "text-only",
        "D": "naive union",
        "E": "Stage 4 joint",
    }
    training_rows = []
    for split in ["A", "B", "C", "D", "E"]:
        metric = llava.get(split)
        if metric:
            training_rows.append(
                [
                    split,
                    descriptions[split],
                    metric.get("status"),
                    fmt_int(metric.get("num_loaded_records")),
                    metric.get("steps"),
                    f"{metric.get('final_loss'):.4f}",
                    f"{metric.get('runtime_seconds') / 3600:.2f}",
                    f"{metric.get('gpu_peak_memory_bytes') / 1024**3:.3f}",
                ]
            )
        else:
            training_rows.append([split, descriptions[split], "pending", "", "2000 target", "", "", ""])
    llava_training = ReportTable(
        "表 8. LLaVA-1.5-7B LoRA 训练状态",
        ["配置", "名称", "状态", "样本数", "步数", "最终 loss", "Wall-clock h", "Peak GB"],
        training_rows,
        "Training loss 只是工程和收敛证据，不是 VQAv2/TextVQA 下游性能。",
    )

    vqav2 = ReportTable(
        "表 9. VQAv2 评测结果",
        ["配置", "Seed 1 Acc", "Seed 2 Acc", "Mean +/- Std", "状态"],
        [[split, "", "", "", "pending"] for split in ["A", "B", "C", "D", "E"]],
        "保持空白，直到 VQAv2 evaluation metrics.json 生成。",
    )
    textvqa = ReportTable(
        "表 10. TextVQA 评测结果",
        ["配置", "Seed 1 Acc", "Seed 2 Acc", "Mean +/- Std", "状态"],
        [[split, "", "", "", "pending"] for split in ["A", "B", "C", "D", "E"]],
        "时间允许再补，不能虚构。",
    )
    pending = ReportTable(
        "表 11. 缺失数据与后续补数位置",
        ["缺失项", "用途", "预期 Source-of-truth"],
        [
            ["E 25K/2000-step metrics", "补全表 6", "Windows E rerun metrics.json mirrored to windows_sync"],
            ["VQAv2 A/B/C/D/E metrics", "补全表 7 与下游 claim", "exp_llava_stage4_vqa_vqav2_quick_* metrics.json"],
            ["TextVQA A/B/C/D/E metrics", "补全表 8，可选", "TextVQA eval metrics.json"],
            ["真实合作者 audit labels", "inter-annotator agreement", "adjudicated annotation CSV with collaborator labels"],
            ["完整效率计时", "system overhead table", "Stage 4 embedding/search/runtime metrics on Windows RTX 3090"],
            ["SSCD baseline", "回应强 image baseline", "SSCD eval metrics on CC3M GT / image benchmarks"],
        ],
    )

    return {
        "data_inventory": data_inventory,
        "annotation": annotation,
        "main_eval": main_eval,
        "error_analysis": error_analysis,
        "split_sizes": split_sizes,
        "threshold_scan": threshold_scan,
        "efficiency": efficiency,
        "llava_training": llava_training,
        "vqav2": vqav2,
        "textvqa": textvqa,
        "pending": pending,
    }


def make_eval_row(label: str, row: dict[str, Any]) -> list[Any]:
    return [
        label,
        row["threshold"],
        pct(row["precision"]),
        pct(row["recall"]),
        f"{row['f1']:.4f}",
        row["tp"],
        row["fp"],
        row["tn"],
        row["fn"],
    ]


def make_joint_threshold_rows(data: dict[str, Any]) -> list[list[Any]]:
    dedup_by_threshold = {
        row["threshold"]: row
        for row in data["threshold_dedup_rates"]
        if row["score"] == "joint"
    }
    wanted = {"0.6", "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95"}
    out: list[list[Any]] = []
    for row in data["threshold_metrics"]:
        if row["score"] != "joint" or row["threshold"] not in wanted:
            continue
        dedup = dedup_by_threshold.get(row["threshold"], {})
        note = "selected" if row["threshold"] == "0.85" else ""
        out.append(
            [
                row["threshold"],
                pct(float(row["precision"])),
                pct(float(row["recall"])),
                f"{float(row['f1']):.4f}",
                pct(float(dedup["dedup_rate"])) if dedup else "",
                note,
            ]
        )
    return out


def make_efficiency_rows(data: dict[str, Any]) -> list[list[Any]]:
    cc3m = data["cc3m_prepare"]
    mining = data["candidate_mining"]
    split = data["split_threshold"]
    manifests = data["training_manifests"]
    eval_metrics = data["stage4_eval"]
    return [
        [
            "CC3M 200K prepare",
            fmt_int(cc3m["saved_pairs"]) + " pairs",
            fmt_duration(cc3m["elapsed_seconds"]),
            f"{cc3m['saved_pairs'] / cc3m['elapsed_seconds']:.2f} pairs/s",
            "Windows data preparation; not counted as Stage 4 algorithm overhead",
        ],
        [
            "OpenCLIP candidate mining",
            fmt_int(mining["num_pairs"]) + " pairs / " + fmt_int(mining["num_candidates"]) + " edges",
            fmt_duration(mining["elapsed_seconds"]),
            f"{mining['num_pairs'] / mining['elapsed_seconds']:.2f} pairs/s",
            "Windows RTX 3090; includes embedding/backend search pipeline",
        ],
        [
            "Threshold-to-split graph computation",
            fmt_int(split["num_pairs"]) + " pairs / " + fmt_int(split["num_candidates"]) + " edges",
            fmt_duration(split["elapsed_seconds"]),
            f"{split['num_candidates'] / split['elapsed_seconds']:.2f} edges/s",
            "Mac-side source-of-truth consolidation",
        ],
        [
            "A/B/C/D/E manifest writing",
            fmt_int(manifests["num_pairs"]) + " pairs",
            fmt_duration(manifests["elapsed_seconds"]),
            f"{manifests['num_pairs'] / manifests['elapsed_seconds']:.2f} pairs/s",
            "Generates LLaVA train manifests",
        ],
        [
            "1K labeled evaluation",
            fmt_int(eval_metrics["num_labeled_rows"]) + " labeled pair-pairs",
            fmt_duration(eval_metrics["elapsed_seconds"]),
            f"{eval_metrics['num_labeled_rows'] / eval_metrics['elapsed_seconds']:.2f} rows/s",
            "Metric computation only",
        ],
    ]


def build_markdown(data: dict[str, Any], tables: dict[str, ReportTable]) -> str:
    lines = [
        "# MMdedup Plan B Stage 4 Final Result Report",
        "",
        f"Generated at: {data['generated_at']} Asia/Shanghai",
        "",
        "## 0. Summary",
        "",
        "- 目标：服务 CIKM 2026 Full Paper 的 Plan B 修订。",
        "- 核心回应：新增 Stage 4，使用 CLIP joint embedding 做图文对级别跨模态去重。",
        "- 当前安全结论：在 1,000 条 hard-candidate 标注集上，Stage 4 优于 text-only 和 naive union，但 image-only 的 F1 仍更高。",
        "- 当前下游状态：A/B/C/D 的 25K/2000-step LLaVA LoRA 已完成；E rerun 与 VQAv2/TextVQA 评测仍待完成，不能提前填数。",
        "- 当前补齐项：Stage 4 joint 阈值扫描展开表与现有可报告效率表已加入。",
        "- 本报告不包含进度面板逐项对照，只列可直接用于论文写作的数据与 source-of-truth。",
        "",
        "## 1. Experiment Design",
        "",
        *implementation_markdown(),
        "",
        "## 2. Complete Data Inventory",
        "",
        *tables_to_markdown(list(tables.values())),
        "",
        "## 3. Source-of-truth Files",
        "",
        *source_files_markdown(),
    ]
    return "\n".join(lines) + "\n"


def implementation_markdown() -> list[str]:
    blocks = [
        (
            "1.1 Stage 4 跨模态图文对去重",
            [
                "输入带 pair id、image path、caption text 的图文对。",
                "使用 CLIP/OpenCLIP image encoder 与 text encoder 分别得到 e_img 和 e_txt。",
                "主方案使用 concat([e_img; e_txt]) 作为 joint embedding；weighted sum 作为可选消融。",
                "joint similarity 超过 tau_cross 时判为重复图文对。",
                "保留 CLIP 图文对齐分数更高的 pair；打平时再用图像质量或 manifest 顺序。",
            ],
        ),
        (
            "1.2 CC3M hard-candidate ground truth 构建",
            [
                "不从 CC3M 随机抽 pair-pairs，因为自然重复比例太低。",
                "先构建 200K CC3M 图文对池，再用 image/text/joint 相似度挖掘 500K candidate edges。",
                "抽 1,000 条 hard candidates 人工标注为 duplicate、near-duplicate、not-duplicate。",
                "评价时 positive class = duplicate + near-duplicate。",
            ],
        ),
        (
            "1.3 Stage 4 主评价",
            [
                "Baselines 包括 image-only、text-only、image/text independent drops 的 naive union。",
                "Ours 使用 joint similarity threshold 的 Stage 4 joint pair dedup。",
                "指标包括 precision、recall、F1、TP/FP/TN/FN 与 threshold sweep。",
            ],
        ),
        (
            "1.4 LLaVA 下游验证",
            [
                "构建 A raw、B image-only、C text-only、D naive union、E Stage 4 joint 五组训练数据。",
                "在单张 RTX 3090 上用 LoRA/QLoRA 微调 LLaVA-1.5-7B，五组超参保持一致。",
                "优先评测 VQAv2；TextVQA 视时间补充。",
                "training loss 不能当作下游性能。",
            ],
        ),
        (
            "1.5 效率与消融",
            [
                "效率表记录 CLIP embedding time、candidate search/clustering time、wall-clock time、GPU peak memory 与 throughput。",
                "阈值敏感性报告 image/text/joint/naive threshold 下的 dedup rate 与 P/R/F1。",
                "完整消融在下游评测完成后覆盖 raw、单模态、naive union 与 Stage 4。",
            ],
        ),
    ]
    lines: list[str] = []
    for title, bullets in blocks:
        lines.extend([f"### {title}", ""])
        lines.extend([f"- {item}" for item in bullets])
        lines.append("")
    return lines


def tables_to_markdown(tables: list[ReportTable]) -> list[str]:
    out: list[str] = []
    for table in tables:
        out.extend([f"### {table.title}", ""])
        out.append("| " + " | ".join(table.columns) + " |")
        out.append("| " + " | ".join(["---"] * len(table.columns)) + " |")
        for row in table.rows:
            out.append("| " + " | ".join(str(cell) for cell in row) + " |")
        if table.note:
            out.extend(["", f"Note: {table.note}"])
        out.append("")
    return out


def source_files_markdown() -> list[str]:
    return [
        "- `experiments/results/plan_b_stage4/experiment_ledger.csv`",
        "- `experiments/results/plan_b_stage4/data_audits/2026-05-20_data_reasonableness_audit.json`",
        "- `experiments/results/plan_b_stage4/exp_stage4_eval_1000_200k_high_joint_20260519/metrics.json`",
        "- `experiments/results/plan_b_stage4/exp_stage4_eval_1000_200k_high_joint_20260519/per_threshold_metrics.csv`",
        "- `experiments/results/plan_b_stage4/exp_stage4_training_manifests_200k_20260520/abcde_split_sizes.csv`",
        "- `experiments/results/plan_b_stage4/windows_sync/exp_llava_stage4_train25k_*_25000_2000steps_20260521/metrics.json`",
        "- Pending E: Windows 原始实验目录完成后同步到 `experiments/results/plan_b_stage4/windows_sync/`。",
    ]


def build_html(data: dict[str, Any], tables: dict[str, ReportTable]) -> str:
    body_parts = [
        '<section class="cover">',
        "<p class=\"kicker\">MMdedup Plan B / CIKM 2026 Full Paper</p>",
        "<h1>Stage 4 最终结果报告</h1>",
        "<p class=\"subtitle\">方案所需数据清单、实验实现思路与当前 source-of-truth 快照</p>",
        f"<p class=\"meta\">生成时间：{h(data['generated_at'])} Asia/Shanghai</p>",
        '<div class="summary">',
        "<h2>当前摘要</h2>",
        bullet_html(
            [
                "核心回应：新增 Stage 4，使用 CLIP joint embedding 做图文对级别跨模态去重。",
                "当前安全结论：Stage 4 joint 优于 text-only 和 naive union；但 image-only 的 F1 仍更高。",
                "A/B/C/D 的 25K/2000-step LLaVA LoRA 已完成；E rerun 与 VQAv2/TextVQA 暂不填数。",
                "Stage 4 joint 阈值扫描展开表与当前可报告效率表已加入。",
                "本报告不包含进度面板逐项对照，只保留论文写作需要的数据与 source-of-truth。",
            ]
        ),
        "</div>",
        "</section>",
        '<section class="page-break">',
        "<h2>1. 各实验实现思路</h2>",
        implementation_html(),
        "</section>",
        '<section class="page-break">',
        "<h2>2. 完整数据清单</h2>",
        *[table_html(table) for table in tables.values()],
        "</section>",
        '<section class="page-break">',
        "<h2>3. Source-of-truth 文件</h2>",
        bullet_html(source_files_plain()),
        "</section>",
    ]
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>MMdedup Stage 4 最终结果报告</title>
  <style>{report_css()}</style>
</head>
<body>
  <main>
    {''.join(body_parts)}
  </main>
</body>
</html>
"""


def implementation_html() -> str:
    blocks = [
        ("Stage 4 跨模态图文对去重", [
            "输入带 pair id、image path、caption text 的图文对。",
            "使用 CLIP/OpenCLIP image encoder 与 text encoder 分别得到 e_img 和 e_txt。",
            "主方案使用 concat([e_img; e_txt]) 作为 joint embedding；weighted sum 作为可选消融。",
            "joint similarity 超过 tau_cross 时判为重复图文对。",
            "保留 CLIP 图文对齐分数更高的 pair；打平时再用图像质量或 manifest 顺序。",
        ]),
        ("CC3M hard-candidate ground truth 构建", [
            "不从 CC3M 随机抽 pair-pairs，因为自然重复比例太低。",
            "先构建 200K CC3M 图文对池，再用 image/text/joint 相似度挖掘 500K candidate edges。",
            "抽 1,000 条 hard candidates 人工标注为 duplicate、near-duplicate、not-duplicate。",
            "评价时 positive class = duplicate + near-duplicate。",
        ]),
        ("Stage 4 主评价", [
            "Baselines 包括 image-only、text-only、image/text independent drops 的 naive union。",
            "Ours 使用 joint similarity threshold 的 Stage 4 joint pair dedup。",
            "指标包括 precision、recall、F1、TP/FP/TN/FN 与 threshold sweep。",
        ]),
        ("LLaVA 下游验证", [
            "构建 A raw、B image-only、C text-only、D naive union、E Stage 4 joint 五组训练数据。",
            "在单张 RTX 3090 上用 LoRA/QLoRA 微调 LLaVA-1.5-7B，五组超参保持一致。",
            "优先评测 VQAv2；TextVQA 视时间补充。",
            "training loss 不能当作下游性能。",
        ]),
        ("效率与消融", [
            "效率表记录 CLIP embedding time、candidate search/clustering time、wall-clock time、GPU peak memory 与 throughput。",
            "阈值敏感性报告 image/text/joint/naive threshold 下的 dedup rate 与 P/R/F1。",
            "完整消融在下游评测完成后覆盖 raw、单模态、naive union 与 Stage 4。",
        ]),
    ]
    return "".join(
        f'<article class="method"><h3>{h(title)}</h3>{bullet_html(items)}</article>'
        for title, items in blocks
    )


def table_html(table: ReportTable) -> str:
    header = "".join(f"<th>{h(col)}</th>" for col in table.columns)
    rows = []
    for row in table.rows:
        rows.append("<tr>" + "".join(f"<td>{h(cell)}</td>" for cell in row) + "</tr>")
    note = f'<p class="note">注：{h(table.note)}</p>' if table.note else ""
    return f"""
<article class="table-block">
  <h3>{h(table.title)}</h3>
  <table>
    <thead><tr>{header}</tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  {note}
</article>
"""


def bullet_html(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{h(item)}</li>" for item in items) + "</ul>"


def source_files_plain() -> list[str]:
    return [
        "experiments/results/plan_b_stage4/experiment_ledger.csv",
        "experiments/results/plan_b_stage4/data_audits/2026-05-20_data_reasonableness_audit.json",
        "experiments/results/plan_b_stage4/exp_stage4_eval_1000_200k_high_joint_20260519/metrics.json",
        "experiments/results/plan_b_stage4/exp_stage4_eval_1000_200k_high_joint_20260519/per_threshold_metrics.csv",
        "experiments/results/plan_b_stage4/exp_stage4_training_manifests_200k_20260520/abcde_split_sizes.csv",
        "experiments/results/plan_b_stage4/windows_sync/exp_llava_stage4_train25k_*_25000_2000steps_20260521/metrics.json",
        "Pending E: Windows 原始实验目录完成后同步到 experiments/results/plan_b_stage4/windows_sync/。",
    ]


def report_css() -> str:
    return """
@page {
  size: A4;
  margin: 15mm 14mm 16mm;
}
* {
  box-sizing: border-box;
}
body {
  margin: 0;
  color: #172033;
  background: #ffffff;
  font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Heiti SC", "Microsoft YaHei", Arial, sans-serif;
  font-size: 11px;
  line-height: 1.52;
}
main {
  width: 100%;
}
.page-break {
  break-before: page;
}
.cover {
  padding-top: 12mm;
}
.kicker {
  margin: 0 0 8px;
  color: #2563eb;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}
h1 {
  margin: 0;
  font-size: 28px;
  line-height: 1.18;
}
h2 {
  margin: 0 0 12px;
  padding-bottom: 6px;
  border-bottom: 1px solid #d7dde8;
  font-size: 18px;
}
h3 {
  margin: 0 0 7px;
  font-size: 12.5px;
}
.subtitle {
  margin: 10px 0 2px;
  color: #475569;
  font-size: 13px;
}
.meta {
  color: #64748b;
  font-size: 10px;
}
.summary {
  margin-top: 22px;
  padding: 14px 16px;
  border: 1px solid #dbe3ef;
  border-radius: 8px;
  background: #f8fafc;
}
.method,
.table-block {
  break-inside: avoid;
  margin: 0 0 13px;
}
ul {
  margin: 5px 0 0 18px;
  padding: 0;
}
li {
  margin: 2px 0;
}
table {
  width: 100%;
  border-collapse: collapse;
  table-layout: auto;
  margin-top: 5px;
}
th {
  background: #1e293b;
  color: #ffffff;
  font-weight: 700;
}
th,
td {
  border: 1px solid #cbd5e1;
  padding: 5px 6px;
  vertical-align: top;
  word-break: break-word;
}
tr:nth-child(even) td {
  background: #f8fafc;
}
.note {
  margin: 5px 0 0;
  color: #475569;
  font-size: 9.5px;
}
"""


def build_pdf_from_html() -> None:
    cmd = [
        "npx",
        "--yes",
        "playwright",
        "pdf",
        "--paper-format",
        "A4",
        "--channel",
        "chrome",
        "--wait-for-timeout",
        "500",
        HTML_PATH.resolve().as_uri(),
        str(PDF_PATH),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def h(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fmt_int(value: Any) -> str:
    if value in (None, ""):
        return ""
    return f"{int(value):,}"


def fmt_duration(seconds: Any) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.2f} s"
    if seconds < 3600:
        return f"{seconds / 60:.2f} min"
    return f"{seconds / 3600:.2f} h"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())
