"""Build candidate Stage 4 paper figures from source-of-truth results.

The script intentionally uses only the Python standard library so the figures
can be regenerated on machines without plotting packages installed.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAIR_CSV = ROOT / "experiments/results/plan_b_stage4/exp_stage4_fair_eval_3000_conservative_and_20260601/fixed_threshold_metrics_with_conservative_and.csv"
COCO_CSV = ROOT / "experiments/results/plan_b_stage4/icdm_revision/summary_20260530/llava_coco_caption_val2014_5k_ckpt1500_20260601.csv"
CI_JSON = ROOT / "experiments/results/plan_b_stage4/exp_stage4_fair_eval_3000_bootstrap_ci_20260531/metrics.json"
OUT_DIR = ROOT / "paper/ieee/figures/stage4_candidates"
OUT_PAPER_DIR = OUT_DIR / "paper_style"


COLORS = {
    "navy": "#1f4e79",
    "teal": "#2a9d8f",
    "orange": "#e76f51",
    "gold": "#e9c46a",
    "gray": "#8d99ae",
    "dark": "#1f2933",
    "light": "#eef2f7",
    "grid": "#d8dee9",
    "joint": "#1f4e79",
    "cons": "#2a9d8f",
    "union": "#e76f51",
    "image": "#8d99ae",
    "text": "#b7a99a",
}

PAPER = {
    "blue": "#1f77b4",
    "red": "#d62728",
    "green": "#2ca02c",
    "black": "#222222",
    "gray": "#7f7f7f",
    "light_blue": "#b7d4ea",
    "light_red": "#f0a3a3",
    "light_green": "#a8d8a8",
    "grid": "#d9d9d9",
}

LABELS = {
    "image": "Image",
    "text": "Text",
    "naive_union": "Union",
    "joint": "Joint",
    "conservative_and": "Cons.",
    "max": "Max",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def esc(text: str) -> str:
    return html.escape(text, quote=True)


class Svg:
    def __init__(self, width: int = 1080, height: int = 560):
        self.width = width
        self.height = height
        self.items: list[str] = []

    def add(self, value: str) -> None:
        self.items.append(value)

    def line(self, x1, y1, x2, y2, stroke=COLORS["dark"], width=1, dash: str | None = None) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>')

    def rect(self, x, y, w, h, fill, stroke="none", width=1, opacity=1.0, rx=0) -> None:
        self.add(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"/>'
        )

    def circle(self, x, y, r, fill, stroke=COLORS["dark"], width=1, opacity=1.0) -> None:
        self.add(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"/>')

    def polyline(self, points: list[tuple[float, float]], stroke, width=2, dash: str | None = None) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.add(f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>')

    def text(self, x, y, text, size=16, anchor="middle", weight="400", fill=COLORS["dark"], rotate: float | None = None) -> None:
        transform = f' transform="rotate({rotate:.1f} {x:.2f} {y:.2f})"' if rotate is not None else ""
        self.add(
            f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" font-family="Times New Roman, Times, serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}"{transform}>{esc(text)}</text>'
        )

    def save(self, path: Path) -> None:
        body = "\n".join(self.items)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">\n'
            f'<rect width="100%" height="100%" fill="white"/>\n{body}\n</svg>\n'
        )
        path.write_text(svg, encoding="utf-8")


class Plot:
    def __init__(self, svg: Svg, x: int, y: int, w: int, h: int, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0):
        self.svg = svg
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def sx(self, value: float) -> float:
        return self.x + (value - self.x_min) / (self.x_max - self.x_min) * self.w

    def sy(self, value: float) -> float:
        return self.y + self.h - (value - self.y_min) / (self.y_max - self.y_min) * self.h

    def axes(self, x_ticks: list[float] | None = None, y_ticks: list[float] | None = None, x_label="", y_label="") -> None:
        if y_ticks:
            for t in y_ticks:
                yy = self.sy(t)
                self.svg.line(self.x, yy, self.x + self.w, yy, COLORS["grid"], 1)
                self.svg.text(self.x - 10, yy + 5, fmt_tick(t), 13, "end", fill=COLORS["dark"])
        if x_ticks:
            for t in x_ticks:
                xx = self.sx(t)
                self.svg.line(xx, self.y, xx, self.y + self.h, COLORS["grid"], 1)
                self.svg.text(xx, self.y + self.h + 22, fmt_tick(t), 13)
        self.svg.line(self.x, self.y + self.h, self.x + self.w, self.y + self.h, COLORS["dark"], 1.2)
        self.svg.line(self.x, self.y, self.x, self.y + self.h, COLORS["dark"], 1.2)
        if x_label:
            self.svg.text(self.x + self.w / 2, self.y + self.h + 48, x_label, 16)
        if y_label:
            self.svg.text(self.x - 55, self.y + self.h / 2, y_label, 16, rotate=-90)


def fmt_tick(v: float) -> str:
    if abs(v) >= 10:
        return f"{v:.0f}"
    return f"{v:.1f}"


def legend(svg: Svg, x: int, y: int, entries: list[tuple[str, str]], size=14) -> None:
    for i, (name, color) in enumerate(entries):
        yy = y + i * 22
        svg.rect(x, yy - 11, 14, 14, color, COLORS["dark"], 0.8)
        svg.text(x + 22, yy, name, size, "start")


def draw_title(svg: Svg, title: str, subtitle: str | None = None) -> None:
    svg.text(svg.width / 2, 34, title, 22, weight="700")
    if subtitle:
        svg.text(svg.width / 2, 58, subtitle, 14, fill="#52616f")


def method_color(method: str) -> str:
    if method in COLORS:
        return COLORS[method]
    return COLORS["gray"]


def pair_grouped_bars(pair_rows: list[dict[str, str]]) -> None:
    svg = Svg()
    draw_title(svg, "Stage 4 Pair-Level Detection", "Precision, recall, and F1 on the 3,000-label CC3M-PairEval set")
    plot = Plot(svg, 90, 86, 780, 380, y_min=0, y_max=0.86)
    plot.axes(y_ticks=[0, 0.2, 0.4, 0.6, 0.8], y_label="Score")
    metrics = [("precision", "Precision", "#d9e2ec"), ("recall", "Recall", "#9fb3c8"), ("f1", "F1", COLORS["navy"])]
    n = len(pair_rows)
    group_w = plot.w / n
    bar_w = 19
    for i, row in enumerate(pair_rows):
        center = plot.x + group_w * (i + 0.5)
        for j, (key, _label, color) in enumerate(metrics):
            val = f(row, key)
            x = center + (j - 1) * (bar_w + 4) - bar_w / 2
            y = plot.sy(val)
            svg.rect(x, y, bar_w, plot.y + plot.h - y, color, COLORS["dark"], 0.8)
        svg.text(center, plot.y + plot.h + 28, LABELS[row["method"]], 15, rotate=-22)
    legend(svg, 650, 92, [(label, color) for _key, label, color in metrics])
    svg.save(OUT_DIR / "01_pair_precision_recall_f1.svg")


def f1_ci_bars(pair_rows: list[dict[str, str]]) -> None:
    svg = Svg()
    draw_title(svg, "Pair-Level F1 with 95% CI", "Joint matching separates itself from modality-wise baselines")
    plot = Plot(svg, 90, 88, 780, 380, y_min=0, y_max=0.70)
    plot.axes(y_ticks=[0, 0.2, 0.4, 0.6], y_label="F1")
    n = len(pair_rows)
    group_w = plot.w / n
    for i, row in enumerate(pair_rows):
        method = row["method"]
        val = f(row, "f1")
        lo = f(row, "f1_ci95_low")
        hi = f(row, "f1_ci95_high")
        center = plot.x + group_w * (i + 0.5)
        bar_w = 54
        y = plot.sy(val)
        svg.rect(center - bar_w / 2, y, bar_w, plot.y + plot.h - y, method_color(method), COLORS["dark"], 0.8, opacity=0.88)
        svg.line(center, plot.sy(lo), center, plot.sy(hi), COLORS["dark"], 2)
        svg.line(center - 12, plot.sy(lo), center + 12, plot.sy(lo), COLORS["dark"], 2)
        svg.line(center - 12, plot.sy(hi), center + 12, plot.sy(hi), COLORS["dark"], 2)
        svg.text(center, y - 10, f"{val:.3f}", 14, weight="700")
        svg.text(center, plot.y + plot.h + 28, LABELS[method], 15, rotate=-20)
    svg.save(OUT_DIR / "02_pair_f1_confidence_intervals.svg")


def precision_recall_tradeoff(pair_rows: list[dict[str, str]]) -> None:
    svg = Svg()
    draw_title(svg, "Precision-Recall Tradeoff", "Bubble size encodes predicted duplicate rate")
    plot = Plot(svg, 105, 88, 735, 385, x_min=0.15, x_max=0.86, y_min=0.15, y_max=0.80)
    plot.axes(x_ticks=[0.2, 0.4, 0.6, 0.8], y_ticks=[0.2, 0.4, 0.6, 0.8], x_label="Recall", y_label="Precision")
    for row in pair_rows:
        method = row["method"]
        x = plot.sx(f(row, "recall"))
        y = plot.sy(f(row, "precision"))
        r = 9 + 30 * f(row, "predicted_positive_rate")
        svg.circle(x, y, r, method_color(method), COLORS["dark"], 1.2, opacity=0.82)
        dx = 12 if method != "conservative_and" else -12
        anchor = "start" if dx > 0 else "end"
        svg.text(x + dx, y - r - 8, LABELS[method], 15, anchor, weight="700")
    svg.text(plot.sx(0.74), plot.sy(0.31), "Union: many positives", 13, "middle", fill=COLORS["union"])
    svg.text(plot.sx(0.671), plot.sy(0.569) - 45, "Joint: balanced", 13, "middle", fill=COLORS["joint"])
    svg.save(OUT_DIR / "03_precision_recall_tradeoff.svg")


def error_decomposition(pair_rows: list[dict[str, str]]) -> None:
    svg = Svg()
    draw_title(svg, "Duplicate-Detection Error Decomposition", "True positives, false positives, and missed positives")
    plot = Plot(svg, 90, 88, 780, 380, y_min=0, y_max=1900)
    plot.axes(y_ticks=[0, 500, 1000, 1500], y_label="Pair count")
    parts = [("tp", "TP", COLORS["teal"]), ("fp", "FP", COLORS["orange"]), ("fn", "FN", "#b7a99a")]
    n = len(pair_rows)
    group_w = plot.w / n
    bar_w = 58
    for i, row in enumerate(pair_rows):
        center = plot.x + group_w * (i + 0.5)
        bottom = plot.y + plot.h
        for key, _label, color in parts:
            val = f(row, key)
            y = plot.sy(val + (plot.y_max - plot.y_max))
            h = plot.y + plot.h - plot.sy(val)
            bottom -= h
            svg.rect(center - bar_w / 2, bottom, bar_w, h, color, COLORS["dark"], 0.5, opacity=0.88)
        svg.text(center, plot.y + plot.h + 28, LABELS[row["method"]], 15, rotate=-20)
    legend(svg, 690, 92, [(label, color) for _key, label, color in parts])
    svg.save(OUT_DIR / "04_error_decomposition_tp_fp_fn.svg")


def f1_delta(ci: dict) -> None:
    svg = Svg()
    draw_title(svg, "Stage 4 F1 Improvement", "Bootstrap 95% CI for the main method comparison")
    deltas = list(ci["f1_delta_ci"].values())
    plot = Plot(svg, 110, 100, 730, 330, x_min=0, x_max=0.36, y_min=0, y_max=len(deltas))
    plot.axes(x_ticks=[0.0, 0.1, 0.2, 0.3], x_label="F1 delta")
    for i, row in enumerate(deltas):
        y = plot.y + 90 + i * 120
        lo = row["ci95_low"]
        hi = row["ci95_high"]
        val = row["point_delta"]
        svg.line(plot.sx(lo), y, plot.sx(hi), y, COLORS["dark"], 3)
        svg.line(plot.sx(lo), y - 12, plot.sx(lo), y + 12, COLORS["dark"], 2)
        svg.line(plot.sx(hi), y - 12, plot.sx(hi), y + 12, COLORS["dark"], 2)
        svg.circle(plot.sx(val), y, 10, COLORS["joint"], COLORS["dark"], 1)
        label = f"{LABELS[row['left_method']]} - {LABELS[row['right_method']]}"
        svg.text(plot.x - 12, y + 5, label, 16, "end", weight="700")
        svg.text(plot.sx(val), y - 22, f"+{val:.3f}", 15, weight="700")
    svg.line(plot.sx(0), plot.y, plot.sx(0), plot.y + plot.h, COLORS["grid"], 1)
    svg.save(OUT_DIR / "05_f1_delta_bootstrap_ci.svg")


def coco_retention_metrics(coco_rows: list[dict[str, str]]) -> None:
    svg = Svg()
    draw_title(svg, "Downstream Captioning Tradeoff", "Retention bars with CIDEr and BLEU-4 overlay")
    plot = Plot(svg, 90, 86, 780, 380, y_min=0, y_max=210)
    plot.axes(y_ticks=[0, 50, 100, 150, 200], y_label="Kept pairs (K)")
    n = len(coco_rows)
    group_w = plot.w / n
    bar_w = 62
    xs = []
    for i, row in enumerate(coco_rows):
        center = plot.x + group_w * (i + 0.5)
        xs.append(center)
        val = f(row, "kept_pairs") / 1000
        y = plot.sy(val)
        color = COLORS["cons"] if row["split"] == "E" else COLORS["gray"] if row["split"] != "D" else COLORS["union"]
        svg.rect(center - bar_w / 2, y, bar_w, plot.y + plot.h - y, color, COLORS["dark"], 0.8, opacity=0.75)
        svg.text(center, plot.y + plot.h + 28, row["split"], 16, weight="700")
    right = Plot(svg, plot.x, plot.y, plot.w, plot.h, y_min=0.12, y_max=0.80)
    cider = [(xs[i], right.sy(f(row, "CIDEr"))) for i, row in enumerate(coco_rows)]
    bleu = [(xs[i], right.sy(f(row, "BLEU_4"))) for i, row in enumerate(coco_rows)]
    svg.polyline(cider, COLORS["dark"], 2.4)
    svg.polyline(bleu, COLORS["navy"], 2.4, dash="5 4")
    for x, y in cider:
        svg.circle(x, y, 5, "white", COLORS["dark"], 2)
    for x, y in bleu:
        svg.rect(x - 5, y - 5, 10, 10, "white", COLORS["navy"], 1.8)
    for t in [0.2, 0.4, 0.6, 0.8]:
        yy = right.sy(t)
        svg.text(plot.x + plot.w + 12, yy + 5, fmt_tick(t), 13, "start")
    svg.line(plot.x + plot.w, plot.y, plot.x + plot.w, plot.y + plot.h, COLORS["dark"], 1.2)
    svg.text(plot.x + plot.w + 54, plot.y + plot.h / 2, "Caption score", 16, rotate=90)
    legend(svg, 630, 92, [("Kept pairs", COLORS["gray"]), ("CIDEr", COLORS["dark"]), ("BLEU-4", COLORS["navy"])])
    svg.save(OUT_DIR / "06_coco_retention_caption_metrics.svg")


def dedup_vs_caption(coco_rows: list[dict[str, str]]) -> None:
    svg = Svg()
    draw_title(svg, "Deduplication Rate vs Caption Transfer", "More aggressive cleaning is not the only useful operating point")
    plot = Plot(svg, 95, 86, 765, 385, x_min=-0.01, x_max=0.25, y_min=0.64, y_max=0.77)
    plot.axes(x_ticks=[0, 0.05, 0.10, 0.15, 0.20, 0.25], y_ticks=[0.65, 0.70, 0.75], x_label="Deduplication rate", y_label="CIDEr")
    for row in coco_rows:
        split = row["split"]
        color = COLORS["cons"] if split == "E" else COLORS["union"] if split == "D" else COLORS["gray"]
        x = plot.sx(f(row, "dedup_rate"))
        y = plot.sy(f(row, "CIDEr"))
        r = 8 + f(row, "kept_pairs") / 200000 * 16
        svg.circle(x, y, r, color, COLORS["dark"], 1.2, opacity=0.83)
        svg.text(x + 13, y - r - 3, split, 16, "start", weight="700")
    svg.text(plot.sx(0.046), plot.sy(0.742) - 46, "E: high retention", 14, "middle", fill=COLORS["cons"])
    svg.text(plot.sx(0.224), plot.sy(0.749) + 42, "D: strongest CIDEr", 14, "middle", fill=COLORS["union"])
    svg.save(OUT_DIR / "07_dedup_rate_vs_cider.svg")


def kept_vs_multi_metric(coco_rows: list[dict[str, str]]) -> None:
    svg = Svg()
    draw_title(svg, "Retention vs Caption Metrics", "A/B/C/D/E split comparison at checkpoint 1500")
    plot = Plot(svg, 95, 86, 765, 385, x_min=145, x_max=205, y_min=0.12, y_max=0.78)
    plot.axes(x_ticks=[150, 170, 190, 200], y_ticks=[0.2, 0.4, 0.6], x_label="Kept pairs (K)", y_label="Score")
    metrics = [("CIDEr", COLORS["dark"]), ("BLEU_4", COLORS["navy"]), ("ROUGE_L", COLORS["teal"])]
    for key, color in metrics:
        pts = [(plot.sx(f(row, "kept_pairs") / 1000), plot.sy(f(row, key))) for row in coco_rows]
        svg.polyline(pts, color, 2.2, dash="5 4" if key == "BLEU_4" else None)
        for i, (x, y) in enumerate(pts):
            svg.circle(x, y, 5, "white", color, 2)
            if key == "CIDEr":
                svg.text(x, y - 13, coco_rows[i]["split"], 14, weight="700")
    legend(svg, 670, 92, [("CIDEr", COLORS["dark"]), ("BLEU-4", COLORS["navy"]), ("ROUGE-L", COLORS["teal"])])
    svg.save(OUT_DIR / "08_kept_pairs_vs_caption_metrics.svg")


def paper_axes(plot: Plot, x_ticks: list[float] | None = None, y_ticks: list[float] | None = None, x_label="", y_label="") -> None:
    if y_ticks:
        for t in y_ticks:
            yy = plot.sy(t)
            plot.svg.line(plot.x, yy, plot.x + plot.w, yy, PAPER["grid"], 1, dash="4 3")
            plot.svg.text(plot.x - 10, yy + 5, fmt_tick(t), 18, "end", weight="700", fill=PAPER["black"])
    if x_ticks:
        for t in x_ticks:
            xx = plot.sx(t)
            plot.svg.line(xx, plot.y, xx, plot.y + plot.h, PAPER["grid"], 1, dash="4 3")
            plot.svg.text(xx, plot.y + plot.h + 25, fmt_tick(t), 18, weight="700", fill=PAPER["black"])
    plot.svg.line(plot.x, plot.y + plot.h, plot.x + plot.w, plot.y + plot.h, PAPER["black"], 1.8)
    plot.svg.line(plot.x, plot.y, plot.x, plot.y + plot.h, PAPER["black"], 1.8)
    if x_label:
        plot.svg.text(plot.x + plot.w / 2, plot.y + plot.h + 58, x_label, 24, weight="700", fill=PAPER["black"])
    if y_label:
        plot.svg.text(plot.x - 63, plot.y + plot.h / 2, y_label, 24, weight="700", fill=PAPER["black"], rotate=-90)


def paper_legend(svg: Svg, x: int, y: int, entries: list[tuple[str, str]], size=19) -> None:
    box_h = 24 * len(entries) + 12
    box_w = max(120, max(len(name) for name, _ in entries) * 10 + 48)
    svg.rect(x - 10, y - 22, box_w, box_h, "white", "#888888", 1, opacity=0.92)
    for i, (name, color) in enumerate(entries):
        yy = y + i * 24
        svg.rect(x, yy - 13, 18, 15, color, PAPER["black"], 0.9)
        svg.text(x + 30, yy, name, size, "start", weight="700", fill=PAPER["black"])


def paper_legend_horizontal(svg: Svg, x: int, y: int, entries: list[tuple[str, str]], size=17) -> None:
    cursor = x
    for name, color in entries:
        svg.rect(cursor, y - 13, 18, 15, color, PAPER["black"], 0.9)
        svg.text(cursor + 28, y, name, size, "start", weight="700", fill=PAPER["black"])
        cursor += 130


def paper_title(svg: Svg, title: str) -> None:
    svg.text(svg.width / 2, 38, title, 28, weight="700", fill=PAPER["black"])


def paper_pair_grouped_bars(pair_rows: list[dict[str, str]]) -> None:
    svg = Svg(width=720, height=500)
    paper_title(svg, "Stage 4 Pair-Level Detection")
    plot = Plot(svg, 82, 105, 585, 300, y_min=0, y_max=0.86)
    paper_axes(plot, y_ticks=[0, 0.2, 0.4, 0.6, 0.8], y_label="Score")
    metrics = [("precision", "Precision", PAPER["blue"]), ("recall", "Recall", PAPER["red"]), ("f1", "F1", PAPER["green"])]
    group_w = plot.w / len(pair_rows)
    bar_w = 12
    for i, row in enumerate(pair_rows):
        center = plot.x + group_w * (i + 0.5)
        for j, (key, _label, color) in enumerate(metrics):
            val = f(row, key)
            x = center + (j - 1) * (bar_w + 3) - bar_w / 2
            y = plot.sy(val)
            svg.rect(x, y, bar_w, plot.y + plot.h - y, color, PAPER["black"], 0.8)
        svg.text(center, plot.y + plot.h + 30, LABELS[row["method"]], 18, rotate=-18, weight="700", fill=PAPER["black"])
    paper_legend_horizontal(svg, 205, 76, [(label, color) for _key, label, color in metrics], size=17)
    svg.save(OUT_PAPER_DIR / "01_pair_precision_recall_f1_paper.svg")


def paper_f1_ci(pair_rows: list[dict[str, str]]) -> None:
    svg = Svg(width=720, height=500)
    paper_title(svg, "Pair-Level F1 with 95% CI")
    plot = Plot(svg, 82, 72, 585, 330, y_min=0, y_max=0.70)
    paper_axes(plot, y_ticks=[0, 0.2, 0.4, 0.6], y_label="F1")
    colors = {
        "image": PAPER["gray"],
        "text": "#b7a99a",
        "naive_union": PAPER["red"],
        "joint": PAPER["blue"],
        "conservative_and": PAPER["green"],
    }
    group_w = plot.w / len(pair_rows)
    for i, row in enumerate(pair_rows):
        method = row["method"]
        val = f(row, "f1")
        lo = f(row, "f1_ci95_low")
        hi = f(row, "f1_ci95_high")
        center = plot.x + group_w * (i + 0.5)
        bar_w = 36
        y = plot.sy(val)
        svg.rect(center - bar_w / 2, y, bar_w, plot.y + plot.h - y, colors[method], PAPER["black"], 0.8, opacity=0.9)
        svg.line(center, plot.sy(lo), center, plot.sy(hi), PAPER["black"], 2)
        svg.line(center - 10, plot.sy(lo), center + 10, plot.sy(lo), PAPER["black"], 2)
        svg.line(center - 10, plot.sy(hi), center + 10, plot.sy(hi), PAPER["black"], 2)
        svg.text(center, y - 9, f"{val:.2f}", 15, weight="700", fill=PAPER["black"])
        svg.text(center, plot.y + plot.h + 30, LABELS[method], 18, rotate=-18, weight="700", fill=PAPER["black"])
    svg.save(OUT_PAPER_DIR / "02_pair_f1_confidence_intervals_paper.svg")


def paper_precision_recall(pair_rows: list[dict[str, str]]) -> None:
    svg = Svg(width=720, height=500)
    paper_title(svg, "Precision-Recall Tradeoff")
    plot = Plot(svg, 88, 70, 575, 330, x_min=0.15, x_max=0.86, y_min=0.15, y_max=0.80)
    paper_axes(plot, x_ticks=[0.2, 0.4, 0.6, 0.8], y_ticks=[0.2, 0.4, 0.6, 0.8], x_label="Recall", y_label="Precision")
    colors = {
        "image": PAPER["gray"],
        "text": "#c4b7aa",
        "naive_union": PAPER["red"],
        "joint": PAPER["blue"],
        "conservative_and": PAPER["green"],
    }
    for row in pair_rows:
        method = row["method"]
        x = plot.sx(f(row, "recall"))
        y = plot.sy(f(row, "precision"))
        r = 7 + 24 * f(row, "predicted_positive_rate")
        svg.circle(x, y, r, colors[method], PAPER["black"], 1.6, opacity=0.72)
        dx = 10 if method != "conservative_and" else -10
        anchor = "start" if dx > 0 else "end"
        svg.text(x + dx, y - r - 8, LABELS[method], 18, anchor, weight="700", fill=PAPER["black"])
    svg.text(plot.sx(0.70), plot.sy(0.32), "Union: many positives", 16, "middle", weight="700", fill=PAPER["red"])
    svg.text(plot.sx(0.66), plot.sy(0.65), "Joint: balanced", 16, "middle", weight="700", fill=PAPER["blue"])
    svg.save(OUT_PAPER_DIR / "03_precision_recall_tradeoff_paper.svg")


def paper_error_decomposition(pair_rows: list[dict[str, str]]) -> None:
    svg = Svg(width=720, height=500)
    paper_title(svg, "Error Decomposition")
    plot = Plot(svg, 88, 72, 575, 330, y_min=0, y_max=1900)
    paper_axes(plot, y_ticks=[0, 500, 1000, 1500], y_label="Pair Count")
    parts = [("tp", "TP", PAPER["blue"]), ("fp", "FP", PAPER["red"]), ("fn", "FN", PAPER["green"])]
    group_w = plot.w / len(pair_rows)
    bar_w = 42
    for i, row in enumerate(pair_rows):
        center = plot.x + group_w * (i + 0.5)
        bottom = plot.y + plot.h
        for key, _label, color in parts:
            val = f(row, key)
            h = plot.y + plot.h - plot.sy(val)
            bottom -= h
            svg.rect(center - bar_w / 2, bottom, bar_w, h, color, PAPER["black"], 0.6, opacity=0.82)
        svg.text(center, plot.y + plot.h + 30, LABELS[row["method"]], 18, rotate=-18, weight="700", fill=PAPER["black"])
    paper_legend(svg, 510, 96, [(label, color) for _key, label, color in parts], size=17)
    svg.save(OUT_PAPER_DIR / "04_error_decomposition_tp_fp_fn_paper.svg")


def paper_f1_delta(ci: dict) -> None:
    svg = Svg(width=720, height=410)
    paper_title(svg, "Stage 4 F1 Improvement")
    deltas = list(ci["f1_delta_ci"].values())
    plot = Plot(svg, 220, 80, 420, 230, x_min=0, x_max=0.36, y_min=0, y_max=len(deltas))
    paper_axes(plot, x_ticks=[0, 0.1, 0.2, 0.3], x_label="F1 Delta")
    for i, row in enumerate(deltas):
        y = plot.y + 70 + i * 92
        lo = row["ci95_low"]
        hi = row["ci95_high"]
        val = row["point_delta"]
        svg.line(plot.sx(lo), y, plot.sx(hi), y, PAPER["black"], 3)
        svg.line(plot.sx(lo), y - 12, plot.sx(lo), y + 12, PAPER["black"], 2)
        svg.line(plot.sx(hi), y - 12, plot.sx(hi), y + 12, PAPER["black"], 2)
        svg.circle(plot.sx(val), y, 9, PAPER["blue"], PAPER["black"], 1.2)
        label = f"{LABELS[row['left_method']]} - {LABELS[row['right_method']]}"
        svg.text(plot.x - 18, y + 6, label, 18, "end", weight="700", fill=PAPER["black"])
        svg.text(plot.sx(val), y - 18, f"+{val:.2f}", 16, weight="700", fill=PAPER["black"])
    svg.save(OUT_PAPER_DIR / "05_f1_delta_bootstrap_ci_paper.svg")


def paper_coco_retention(coco_rows: list[dict[str, str]]) -> None:
    svg = Svg(width=720, height=500)
    paper_title(svg, "COCO Caption Transfer")
    plot = Plot(svg, 82, 105, 555, 295, y_min=0, y_max=210)
    paper_axes(plot, y_ticks=[0, 50, 100, 150, 200], y_label="Kept Pairs (K)")
    group_w = plot.w / len(coco_rows)
    xs = []
    bar_w = 42
    for i, row in enumerate(coco_rows):
        center = plot.x + group_w * (i + 0.5)
        xs.append(center)
        kept_k = f(row, "kept_pairs") / 1000
        y = plot.sy(kept_k)
        color = PAPER["green"] if row["split"] == "E" else PAPER["red"] if row["split"] == "D" else PAPER["gray"]
        svg.rect(center - bar_w / 2, y, bar_w, plot.y + plot.h - y, color, PAPER["black"], 0.8, opacity=0.72)
        svg.text(center, plot.y + plot.h + 28, row["split"], 20, weight="700", fill=PAPER["black"])
    right = Plot(svg, plot.x, plot.y, plot.w, plot.h, y_min=0.12, y_max=0.80)
    cider = [(xs[i], right.sy(f(row, "CIDEr"))) for i, row in enumerate(coco_rows)]
    bleu = [(xs[i], right.sy(f(row, "BLEU_4"))) for i, row in enumerate(coco_rows)]
    svg.polyline(cider, PAPER["black"], 3)
    svg.polyline(bleu, PAPER["blue"], 3, dash="5 4")
    for x, y in cider:
        svg.circle(x, y, 5.5, "white", PAPER["black"], 2)
    for x, y in bleu:
        svg.rect(x - 5.5, y - 5.5, 11, 11, "white", PAPER["blue"], 2)
    for t in [0.2, 0.4, 0.6, 0.8]:
        yy = right.sy(t)
        svg.text(plot.x + plot.w + 12, yy + 5, fmt_tick(t), 18, "start", weight="700", fill=PAPER["black"])
    svg.line(plot.x + plot.w, plot.y, plot.x + plot.w, plot.y + plot.h, PAPER["black"], 1.8)
    svg.text(plot.x + plot.w + 63, plot.y + plot.h / 2, "Caption Score", 22, weight="700", fill=PAPER["black"], rotate=90)
    paper_legend_horizontal(svg, 178, 76, [("Kept Pairs", PAPER["gray"]), ("CIDEr", PAPER["black"]), ("BLEU-4", PAPER["blue"])], size=16)
    svg.save(OUT_PAPER_DIR / "06_coco_retention_caption_metrics_paper.svg")


def paper_dedup_vs_cider(coco_rows: list[dict[str, str]]) -> None:
    svg = Svg(width=720, height=500)
    paper_title(svg, "Dedup Rate vs. CIDEr")
    plot = Plot(svg, 90, 72, 570, 330, x_min=-0.01, x_max=0.25, y_min=0.64, y_max=0.77)
    paper_axes(plot, x_ticks=[0, 0.05, 0.10, 0.15, 0.20, 0.25], y_ticks=[0.65, 0.70, 0.75], x_label="Deduplication Rate", y_label="CIDEr")
    for row in coco_rows:
        split = row["split"]
        color = PAPER["green"] if split == "E" else PAPER["red"] if split == "D" else PAPER["gray"]
        x = plot.sx(f(row, "dedup_rate"))
        y = plot.sy(f(row, "CIDEr"))
        r = 7 + f(row, "kept_pairs") / 200000 * 13
        svg.circle(x, y, r, color, PAPER["black"], 1.5, opacity=0.78)
        svg.text(x + 10, y - r - 4, split, 18, "start", weight="700", fill=PAPER["black"])
    svg.text(plot.sx(0.046), plot.sy(0.742) - 38, "E: high retention", 16, "middle", weight="700", fill=PAPER["green"])
    svg.text(plot.sx(0.205), plot.sy(0.749) + 34, "D: strongest CIDEr", 16, "middle", weight="700", fill=PAPER["red"])
    svg.save(OUT_PAPER_DIR / "07_dedup_rate_vs_cider_paper.svg")


def build_paper_style_figures(pair_rows: list[dict[str, str]], coco_rows: list[dict[str, str]], ci: dict) -> None:
    OUT_PAPER_DIR.mkdir(parents=True, exist_ok=True)
    paper_pair_grouped_bars(pair_rows)
    paper_f1_ci(pair_rows)
    paper_precision_recall(pair_rows)
    paper_error_decomposition(pair_rows)
    paper_f1_delta(ci)
    paper_coco_retention(coco_rows)
    paper_dedup_vs_cider(coco_rows)


def gallery(files: list[Path]) -> None:
    cards = []
    for path in files:
        display = path.relative_to(OUT_DIR).as_posix()
        cards.append(
            f'<section><h2>{esc(display)}</h2><img src="{esc(display)}" alt="{esc(display)}"></section>'
        )
    html_doc = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Stage 4 Candidate Figures</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; background: #f7f9fc; color: #1f2933; }
h1 { font-size: 24px; }
section { background: white; border: 1px solid #d8dee9; margin: 18px 0; padding: 16px; }
h2 { font-size: 15px; margin: 0 0 12px; color: #52616f; }
img { width: 100%; max-width: 980px; display: block; }
</style>
</head>
<body>
<h1>Stage 4 Candidate Figures</h1>
""" + "\n".join(cards) + "\n</body>\n</html>\n"
    (OUT_DIR / "gallery.html").write_text(html_doc, encoding="utf-8")


def readme(files: list[Path]) -> None:
    lines = [
        "# Stage 4 Candidate Figures",
        "",
        "These figures are generated from real experiment data and are kept as paper candidates.",
        "",
        "## Data Sources",
        "",
        f"- `{PAIR_CSV.relative_to(ROOT)}`",
        f"- `{COCO_CSV.relative_to(ROOT)}`",
        f"- `{CI_JSON.relative_to(ROOT)}`",
        "",
        "## Candidate Figures",
        "",
    ]
    descriptions = {
        "01_pair_precision_recall_f1.svg": "Main-result candidate showing pair-level precision, recall, and F1.",
        "02_pair_f1_confidence_intervals.svg": "Main-result candidate showing F1 and 95% confidence intervals.",
        "03_precision_recall_tradeoff.svg": "Precision-recall tradeoff candidate; bubble size encodes predicted positive rate.",
        "04_error_decomposition_tp_fp_fn.svg": "Error decomposition candidate showing TP, FP, and FN counts.",
        "05_f1_delta_bootstrap_ci.svg": "Robustness candidate showing F1 deltas and bootstrap confidence intervals.",
        "06_coco_retention_caption_metrics.svg": "Downstream candidate showing kept pairs versus CIDEr/BLEU-4.",
        "07_dedup_rate_vs_cider.svg": "Downstream tradeoff candidate showing dedup rate versus CIDEr.",
        "08_kept_pairs_vs_caption_metrics.svg": "Downstream multi-metric candidate showing kept pairs versus CIDEr/BLEU-4/ROUGE-L.",
    }
    for path in files:
        rel = path.relative_to(OUT_DIR).as_posix()
        desc = descriptions.get(path.name, "paper-style 版本：更接近原文 Matplotlib/IEEE 图风格。")
        lines.append(f"- `{rel}`: {desc}")
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pair_rows = read_csv(PAIR_CSV)
    coco_rows = read_csv(COCO_CSV)
    ci = json.loads(CI_JSON.read_text(encoding="utf-8"))
    pair_grouped_bars(pair_rows)
    f1_ci_bars(pair_rows)
    precision_recall_tradeoff(pair_rows)
    error_decomposition(pair_rows)
    f1_delta(ci)
    coco_retention_metrics(coco_rows)
    dedup_vs_caption(coco_rows)
    kept_vs_multi_metric(coco_rows)
    build_paper_style_figures(pair_rows, coco_rows, ci)
    files = sorted(OUT_DIR.glob("*.svg")) + sorted(OUT_PAPER_DIR.glob("*.svg"))
    gallery(files)
    readme(files)
    print(f"Wrote {len(files)} SVG figures to {OUT_DIR}")
    for path in files:
        print(path)


if __name__ == "__main__":
    main()
