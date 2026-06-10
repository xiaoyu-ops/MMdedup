"""Build visual review contact sheets for Stage 4 annotation rows."""

from __future__ import annotations

import argparse
import csv
import math
import textwrap
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows-per-sheet", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(args.annotations_csv)
    sheets_dir = args.output_dir / "review_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    rows_per_sheet = max(1, args.rows_per_sheet)
    for sheet_idx in range(math.ceil(len(rows) / rows_per_sheet)):
        chunk = rows[sheet_idx * rows_per_sheet : (sheet_idx + 1) * rows_per_sheet]
        sheet = _draw_sheet(chunk, image_size=args.image_size)
        sheet.save(sheets_dir / f"review_sheet_{sheet_idx:03d}.jpg", quality=88)

    summary = {
        "annotation_rows": str(len(rows)),
        "rows_per_sheet": str(rows_per_sheet),
        "num_sheets": str(math.ceil(len(rows) / rows_per_sheet)),
        "review_sheets_dir": str(sheets_dir),
    }
    (args.output_dir / "review_assets_summary.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )
    print(summary)
    return 0


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _draw_sheet(rows: List[Dict[str, str]], image_size: int) -> Image.Image:
    font = ImageFont.load_default()
    margin = 16
    gutter = 14
    row_h = max(250, image_size + 76)
    width = 1500
    height = margin * 2 + row_h * len(rows)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, row in enumerate(rows):
        top = margin + idx * row_h
        draw.rectangle((margin, top, width - margin, top + row_h - 8), outline=(210, 210, 210), width=1)
        text_x = margin + 8
        img_a_x = 420
        img_b_x = img_a_x + image_size + gutter
        img_y = top + 42

        header = (
            f"{row.get('candidate_id', '')} | bucket={row.get('bucket', '')} | "
            f"img={_fmt(row.get('image_similarity'))} text={_fmt(row.get('text_similarity'))} "
            f"joint={_fmt(row.get('joint_similarity'))} | audit={row.get('needs_audit', '')}"
        )
        draw.text((text_x, top + 10), header, fill=(0, 0, 0), font=font)
        _paste_image(canvas, row.get("image_path_a", ""), (img_a_x, img_y), image_size)
        _paste_image(canvas, row.get("image_path_b", ""), (img_b_x, img_y), image_size)
        draw.text((img_a_x, img_y + image_size + 4), row.get("pair_id_a", "")[:32], fill=(0, 0, 0), font=font)
        draw.text((img_b_x, img_y + image_size + 4), row.get("pair_id_b", "")[:32], fill=(0, 0, 0), font=font)

        caption_x = img_b_x + image_size + gutter
        caption_w = 68
        caption_a = "A: " + row.get("caption_a", "")
        caption_b = "B: " + row.get("caption_b", "")
        wrapped = textwrap.wrap(caption_a, width=caption_w)[:5] + [""] + textwrap.wrap(caption_b, width=caption_w)[:5]
        draw.multiline_text((caption_x, img_y), "\n".join(wrapped), fill=(0, 0, 0), font=font, spacing=4)
    return canvas


def _paste_image(canvas: Image.Image, path_raw: str, xy: tuple[int, int], size: int) -> None:
    draw = ImageDraw.Draw(canvas)
    box = (xy[0], xy[1], xy[0] + size, xy[1] + size)
    try:
        with Image.open(path_raw) as image:
            image = image.convert("RGB")
            image.thumbnail((size, size))
            background = Image.new("RGB", (size, size), (245, 245, 245))
            offset = ((size - image.width) // 2, (size - image.height) // 2)
            background.paste(image, offset)
            canvas.paste(background, xy)
    except Exception as exc:
        draw.rectangle(box, fill=(245, 245, 245), outline=(200, 0, 0))
        draw.text((xy[0] + 6, xy[1] + 6), f"image error\n{exc}", fill=(160, 0, 0))
    draw.rectangle(box, outline=(180, 180, 180), width=1)


def _fmt(value: str | None) -> str:
    try:
        return f"{float(value or 0.0):.3f}"
    except ValueError:
        return "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
