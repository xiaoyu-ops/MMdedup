#!/usr/bin/env python3
"""Regenerate the two Fig. 6 PDF panels with larger text for IEEE layout."""

from __future__ import annotations

import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


ROOT = Path(__file__).resolve().parents[2]
PAIR_CSV = ROOT / "experiments/results/plan_b_stage4/exp_stage4_fair_eval_3000_conservative_and_20260601/fixed_threshold_metrics_with_conservative_and.csv"
COCO_CSV = ROOT / "experiments/results/plan_b_stage4/icdm_revision/summary_20260530/llava_coco_caption_val2014_5k_ckpt1500_20260601.csv"
OUT_DIR = ROOT / "Cleaning_up_the_Digital_Swamp__An_End_to_End_Pipeline_for_Sorting_and_Deduplicating_Mixed_Modality_Data/IEEE-conference-template-062824/figures/stage4_candidates/paper_style"

BLACK = colors.HexColor("#1f2937")
GRID = colors.HexColor("#d8dee8")
GRAY = colors.HexColor("#b9c0ca")
TEXT_TAN = colors.HexColor("#c5b5a5")
RED = colors.HexColor("#df6f67")
BLUE = colors.HexColor("#4f86c6")
GREEN = colors.HexColor("#3f8f5a")
TIMES_NEW_ROMAN_BOLD = "TimesNewRoman-Bold"
TIMES_NEW_ROMAN_BOLD_PATH = Path("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf")

if TIMES_NEW_ROMAN_BOLD_PATH.exists():
    pdfmetrics.registerFont(TTFont(TIMES_NEW_ROMAN_BOLD, str(TIMES_NEW_ROMAN_BOLD_PATH)))
else:
    TIMES_NEW_ROMAN_BOLD = "Times-Bold"

LABELS = {
    "image": "Image",
    "text": "Text",
    "naive_union": "Union",
    "joint": "Joint",
    "conservative_and": "Cons.",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def sx(value: float, left: float, width: float, low: float, high: float) -> float:
    return left + (value - low) / (high - low) * width


def sy(value: float, bottom: float, height: float, low: float, high: float) -> float:
    return bottom + (value - low) / (high - low) * height


def draw_text(c: canvas.Canvas, x: float, y: float, text: str, size: int, font: str = TIMES_NEW_ROMAN_BOLD, anchor: str = "middle", fill=BLACK) -> None:
    c.setFillColor(fill)
    c.setFont(font, size)
    if anchor == "middle":
        c.drawCentredString(x, y, text)
    elif anchor == "end":
        c.drawRightString(x, y, text)
    else:
        c.drawString(x, y, text)


def draw_axes(
    c: canvas.Canvas,
    left: float,
    bottom: float,
    width: float,
    height: float,
    x_ticks: list[float],
    y_ticks: list[float],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    x_label: str | None,
    y_label: str | None,
) -> None:
    top = bottom + height
    c.setStrokeColor(GRID)
    c.setLineWidth(1.4)
    for tick in x_ticks:
        x = sx(tick, left, width, *x_range)
        c.line(x, top, x, bottom)
    for tick in y_ticks:
        y = sy(tick, bottom, height, *y_range)
        c.line(left, y, left + width, y)

    c.setStrokeColor(BLACK)
    c.setLineWidth(3)
    c.line(left, bottom, left + width, bottom)
    c.line(left, top, left, bottom)

    for tick in x_ticks:
        x = sx(tick, left, width, *x_range)
        c.line(x, bottom - 9, x, bottom)
        draw_text(c, x, bottom - 36, f"{tick:.1f}", 34)
    for tick in y_ticks:
        y = sy(tick, bottom, height, *y_range)
        c.line(left - 8, y, left, y)
        draw_text(c, left - 17, y - 12, f"{tick:.1f}", 34, anchor="end")

    if x_label:
        draw_text(c, left + width / 2, bottom - 82, x_label, 44)
    if y_label:
        c.saveState()
        c.translate(left - 68, bottom + height / 2)
        c.rotate(90)
        draw_text(c, 0, 0, y_label, 40)
        c.restoreState()


def precision_recall_panel(rows: list[dict[str, str]]) -> None:
    path = OUT_DIR / "03_precision_recall_tradeoff_paper.pdf"
    c = canvas.Canvas(str(path), pagesize=(720, 500))

    left, bottom, width, height = 108, 112, 555, 345
    x_range = (0.15, 0.94)
    y_range = (0.05, 0.82)
    draw_axes(
        c,
        left,
        bottom,
        width,
        height,
        x_ticks=[0.2, 0.4, 0.6, 0.8],
        y_ticks=[0.2, 0.4, 0.6, 0.8],
        x_range=x_range,
        y_range=y_range,
        x_label="Recall",
        y_label="Precision",
    )

    palette = {
        "image": GRAY,
        "text": TEXT_TAN,
        "naive_union": RED,
        "joint": BLUE,
        "conservative_and": GREEN,
    }
    offsets = {
        "image": (-20, 48, "end"),
        "text": (32, -8, "start"),
        "naive_union": (10, 48, "start"),
        "joint": (48, -3, "start"),
        "conservative_and": (-34, -6, "end"),
    }
    for row in rows:
        method = row["method"]
        x = sx(f(row, "recall"), left, width, *x_range)
        y = sy(f(row, "precision"), bottom, height, *y_range)
        radius = 16 + 35 * f(row, "predicted_positive_rate")
        c.setFillColor(palette[method])
        c.setStrokeColor(BLACK)
        c.setLineWidth(4)
        c.circle(x, y, radius, stroke=1, fill=1)
        dx, dy, anchor = offsets[method]
        draw_text(c, x + dx, y + dy, LABELS[method], 42, anchor=anchor)

    c.showPage()
    c.save()


def coco_panel(rows: list[dict[str, str]]) -> None:
    path = OUT_DIR / "06_coco_retention_caption_metrics_paper.pdf"
    c = canvas.Canvas(str(path), pagesize=(720, 500))

    left, bottom, width, height = 105, 96, 492, 325
    y_range = (0, 210)
    draw_axes(
        c,
        left,
        bottom,
        width,
        height,
        x_ticks=[],
        y_ticks=[0, 50, 100, 150, 200],
        x_range=(0, 1),
        y_range=y_range,
        x_label=None,
        y_label=None,
    )

    group_w = width / len(rows)
    xs: list[float] = []
    bar_w = 54
    for i, row in enumerate(rows):
        x = left + group_w * (i + 0.5)
        xs.append(x)
        kept_k = f(row, "kept_pairs") / 1000
        y = sy(kept_k, bottom, height, *y_range)
        color = GREEN if row["split"] == "E" else RED if row["split"] == "D" else GRAY
        c.setFillColor(color)
        c.setStrokeColor(BLACK)
        c.setLineWidth(2.5)
        c.rect(x - bar_w / 2, bottom, bar_w, y - bottom, stroke=1, fill=1)
        draw_text(c, x, bottom - 49, row["split"], 42)

    right_range = (0.12, 0.80)
    c.setStrokeColor(BLACK)
    c.setLineWidth(3)
    c.line(left + width, bottom, left + width, bottom + height)
    for tick in [0.2, 0.4, 0.6, 0.8]:
        y = sy(tick, bottom, height, *right_range)
        c.line(left + width, y, left + width + 8, y)
        draw_text(c, left + width + 18, y - 12, f"{tick:.1f}", 34, anchor="start")

    def draw_poly(points: list[tuple[float, float]], color, dashed: bool = False) -> None:
        c.setStrokeColor(color)
        c.setLineWidth(7)
        c.setDash(9, 6) if dashed else c.setDash()
        path = c.beginPath()
        path.moveTo(*points[0])
        for point in points[1:]:
            path.lineTo(*point)
        c.drawPath(path)
        c.setDash()

    cider = [(xs[i], sy(f(row, "CIDEr"), bottom, height, *right_range)) for i, row in enumerate(rows)]
    bleu = [(xs[i], sy(f(row, "BLEU_4"), bottom, height, *right_range)) for i, row in enumerate(rows)]
    draw_poly(cider, BLACK)
    draw_poly(bleu, BLUE, dashed=True)
    for x, y in cider:
        c.setFillColor(colors.white)
        c.setStrokeColor(BLACK)
        c.setLineWidth(4)
        c.circle(x, y, 10, stroke=1, fill=1)
    for x, y in bleu:
        c.setFillColor(colors.white)
        c.setStrokeColor(BLUE)
        c.setLineWidth(4)
        c.rect(x - 10, y - 10, 20, 20, stroke=1, fill=1)

    # Large legend placed above the plotting area to avoid covering bars.
    legend_y = 462
    entries = [("Kept", GRAY, "bar"), ("CIDEr", BLACK, "line"), ("BLEU-4", BLUE, "dash")]
    legend_x = 54
    for label, color, kind in entries:
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(5)
        if kind == "bar":
            c.rect(legend_x, legend_y - 17, 34, 24, stroke=1, fill=1)
        else:
            if kind == "dash":
                c.setDash(9, 6)
            c.line(legend_x, legend_y - 4, legend_x + 48, legend_y - 4)
            c.setDash()
        draw_text(c, legend_x + 58, legend_y - 16, label, 34, anchor="start")
        legend_x += 214

    c.showPage()
    c.save()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    precision_recall_panel(read_csv(PAIR_CSV))
    coco_panel(read_csv(COCO_CSV))
    print(OUT_DIR / "03_precision_recall_tradeoff_paper.pdf")
    print(OUT_DIR / "06_coco_retention_caption_metrics_paper.pdf")


if __name__ == "__main__":
    main()
