"""Check that Stage 4 dashboard data matches current source-of-truth state."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/results/plan_b_stage4"
DASHBOARD_DATA = ROOT / "docs/stage4_dashboard/data"
STATUS_JSON = DASHBOARD_DATA / "status.json"
LEDGER = RESULTS / "experiment_ledger.csv"
DASHBOARD_LEDGER = DASHBOARD_DATA / "experiment_ledger.csv"


def main() -> int:
    issues: list[str] = []
    ledger_ids = _ledger_ids(LEDGER)
    status = _read_json(STATUS_JSON, issues)

    if LEDGER.exists() and DASHBOARD_LEDGER.exists():
        if LEDGER.read_text(encoding="utf-8") != DASHBOARD_LEDGER.read_text(encoding="utf-8"):
            issues.append("dashboard experiment_ledger.csv does not match source-of-truth ledger")
    else:
        issues.append("missing source or dashboard experiment_ledger.csv")

    if "exp_stage4_error_analysis_1000_200k_high_joint_20260520" in ledger_ids:
        stage4 = _find_by_name(status.get("phase_progress", []), "Stage 4 主评价")
        if stage4.get("percent", 0) < 90:
            issues.append("Stage 4 主评价 phase percent is stale; expected >=90 after error analysis")

    if "exp_llava_stage4_real_train_smoke_E_20260520" in ledger_ids:
        _check_llava_smoke_state(status, issues)

    stale_phrases = [
        "LoRA 训练日志和 VQAv2/TextVQA 指标仍未开始",
        "GPU peak memory 尚未记录",
    ]
    dashboard_text = json.dumps(status, ensure_ascii=False)
    for phrase in stale_phrases:
        if phrase in dashboard_text:
            issues.append(f"stale dashboard phrase remains: {phrase}")

    if issues:
        print("dashboard consistency failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("dashboard consistency ok")
    return 0


def _check_llava_smoke_state(status: dict, issues: list[str]) -> None:
    llava_phase = _find_by_name(status.get("phase_progress", []), "LLaVA 下游验证")
    if llava_phase.get("status") == "pending" or llava_phase.get("percent", 0) <= 0:
        issues.append("LLaVA phase is stale; smoke exists but phase is pending/0")

    llava_requirement = _find_by_name(status.get("plan_requirements", []), "MLLM 下游验证")
    if llava_requirement.get("status") == "pending":
        issues.append("MLLM 下游验证 requirement is stale; smoke exists but status is pending")
    outputs = llava_requirement.get("current_outputs", [])
    if outputs == ["尚未开始"] or "尚未开始" in outputs:
        issues.append("MLLM 下游验证 requirement still says 尚未开始")
    if not _contains(outputs, "真实 LLaVA"):
        issues.append("MLLM 下游验证 requirement does not mention real LLaVA smoke")

    writing = _find_by_title(status.get("paper_writing_data", []), "LLaVA 下游验证")
    if "真实 LLaVA" not in str(writing.get("paper_use", "")):
        issues.append("paper_writing_data LLaVA block does not mention real LLaVA smoke")

    matrix = _find_by_experiment(status.get("plan_data_matrix", []), "实验 4：MLLM 下游训练验证")
    if matrix.get("status") == "pending":
        issues.append("plan_data_matrix experiment 4 is stale; smoke exists but experiment status is pending")
    item_names = [str(item.get("name", "")) for item in matrix.get("items", []) if isinstance(item, dict)]
    if "表 4.0 LLaVA 训练链路 smoke" not in item_names:
        issues.append("plan_data_matrix missing 表 4.0 LLaVA 训练链路 smoke")

    required_exports = [
        DASHBOARD_DATA / "paper/llava_stage4_data_smoke_abcde_metrics.json",
        DASHBOARD_DATA / "paper/llava_stage4_real_train_smoke_E_metrics.json",
    ]
    for path in required_exports:
        if not path.exists():
            issues.append(f"missing dashboard paper export: {path.relative_to(ROOT)}")


def _ledger_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["experiment_id"] for row in csv.DictReader(handle) if row.get("experiment_id")}


def _read_json(path: Path, issues: list[str]) -> dict:
    if not path.exists():
        issues.append(f"missing {path.relative_to(ROOT)}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _find_by_name(rows: object, name: str) -> dict:
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and row.get("name") == name:
            return row
    return {}


def _find_by_title(rows: object, title: str) -> dict:
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and row.get("title") == title:
            return row
    return {}


def _find_by_experiment(rows: object, experiment: str) -> dict:
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and row.get("experiment") == experiment:
            return row
    return {}


def _contains(values: object, needle: str) -> bool:
    if not isinstance(values, list):
        return False
    return any(needle in str(value) for value in values)


if __name__ == "__main__":
    raise SystemExit(main())
