"""Serve a local web app for Stage 4 image-caption pair annotation.

The app edits a copy of the annotation CSV instead of overwriting the source
sheet. It uses the pre-rendered review sheets and crops the current row, so it
can run on the Mac mirror without the full CC3M image cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from PIL import Image


LABELS = ("duplicate", "near-duplicate", "not-duplicate")
DEFAULT_ANNOTATION_CSV = Path(
    "experiments/results/plan_b_stage4/windows_sync/"
    "exp_stage4_annotation_1000_200k_high_joint_20260516/annotation_sheet.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", type=Path, default=DEFAULT_ANNOTATION_CSV)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--annotator", default="wzy")
    parser.add_argument("--mode", choices=("primary", "audit"), default="primary")
    parser.add_argument("--rows-per-sheet", type=int, default=8)
    return parser.parse_args()


class AnnotationStore:
    def __init__(
        self,
        annotations_csv: Path,
        output_csv: Path | None,
        annotator: str,
        mode: str,
        rows_per_sheet: int,
    ) -> None:
        self.annotations_csv = annotations_csv
        self.output_csv = output_csv or annotations_csv.with_name(
            annotations_csv.stem + "_labeled.csv"
        )
        self.annotator = annotator
        self.mode = mode
        self.rows_per_sheet = rows_per_sheet
        self.review_sheets_dir = annotations_csv.parent / "review_sheets"
        if not self.annotations_csv.exists():
            raise FileNotFoundError(f"annotation CSV not found: {self.annotations_csv}")
        if not self.output_csv.exists():
            shutil.copyfile(self.annotations_csv, self.output_csv)
        self.rows, self.fieldnames = self._read_rows()
        self._ensure_columns()

    def _read_rows(self) -> tuple[list[dict[str, str]], list[str]]:
        with self.output_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"annotation CSV has no header: {self.output_csv}")
            return list(reader), list(reader.fieldnames)

    def _ensure_columns(self) -> None:
        changed = False
        for column in ("label", "annotator", "audit_label", "notes"):
            if column not in self.fieldnames:
                self.fieldnames.append(column)
                changed = True
        for row in self.rows:
            for column in self.fieldnames:
                row.setdefault(column, "")
        if changed:
            self._write_rows()

    def _write_rows(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=self.output_csv.parent,
            delete=False,
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
            tmp_path = Path(handle.name)
        tmp_path.replace(self.output_csv)

    def status(self) -> dict[str, Any]:
        target_column = "audit_label" if self.mode == "audit" else "label"
        eligible = [
            row
            for row in self.rows
            if self.mode == "primary" or _is_truthy(row.get("needs_audit", ""))
        ]
        done = sum(1 for row in eligible if row.get(target_column, "").strip())
        counts = {label: 0 for label in LABELS}
        for row in eligible:
            label = row.get(target_column, "").strip()
            if label in counts:
                counts[label] += 1
        return {
            "mode": self.mode,
            "annotator": self.annotator,
            "total": len(eligible),
            "done": done,
            "remaining": len(eligible) - done,
            "counts": counts,
            "output_csv": str(self.output_csv),
        }

    def row_payload(self, index: int) -> dict[str, Any]:
        index = min(max(index, 0), len(self.rows) - 1)
        row = dict(self.rows[index])
        return {
            "index": index,
            "total_rows": len(self.rows),
            "row": row,
            "status": self.status(),
            "prev_unlabeled": self.find_unlabeled(index - 1, -1),
            "next_unlabeled": self.find_unlabeled(index + 1, 1),
        }

    def find_unlabeled(self, start: int, step: int) -> int | None:
        target_column = "audit_label" if self.mode == "audit" else "label"
        idx = start
        while 0 <= idx < len(self.rows):
            row = self.rows[idx]
            if self.mode == "audit" and not _is_truthy(row.get("needs_audit", "")):
                idx += step
                continue
            if not row.get(target_column, "").strip():
                return idx
            idx += step
        return None

    def first_unlabeled(self) -> int:
        found = self.find_unlabeled(0, 1)
        return 0 if found is None else found

    def update_label(
        self,
        index: int,
        label: str,
        notes: str | None,
        annotator: str | None,
    ) -> dict[str, Any]:
        if label not in LABELS and label != "":
            raise ValueError(f"invalid label: {label}")
        row = self.rows[index]
        if self.mode == "audit":
            row["audit_label"] = label
            if notes is not None:
                row["notes"] = notes
        else:
            row["label"] = label
            row["annotator"] = "" if label == "" else annotator or self.annotator
            if notes is not None:
                row["notes"] = notes
        self._write_rows()
        return self.row_payload(index)

    def crop_review_row(self, index: int, output_path: Path) -> None:
        sheet_idx = index // self.rows_per_sheet
        local_idx = index % self.rows_per_sheet
        sheet_path = self.review_sheets_dir / f"review_sheet_{sheet_idx:03d}.jpg"
        if not sheet_path.exists():
            raise FileNotFoundError(f"review sheet not found: {sheet_path}")
        with Image.open(sheet_path) as image:
            width, height = image.size
            margin = 16
            row_h = max(1, (height - margin * 2) // self.rows_per_sheet)
            top = max(0, margin + local_idx * row_h)
            bottom = min(height, top + row_h)
            crop = image.crop((0, top, width, bottom))
            crop.save(output_path, quality=92)


def make_handler(store: AnnotationStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html()
            elif parsed.path == "/api/row":
                query = parse_qs(parsed.query)
                index = int(query.get("index", [store.first_unlabeled()])[0])
                self._send_json(store.row_payload(index))
            elif parsed.path == "/api/status":
                self._send_json(store.status())
            elif parsed.path == "/api/first-unlabeled":
                self._send_json({"index": store.first_unlabeled()})
            elif parsed.path == "/assets/review-row.jpg":
                query = parse_qs(parsed.query)
                index = int(query.get("index", [0])[0])
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
                    tmp_path = Path(handle.name)
                try:
                    store.crop_review_row(index, tmp_path)
                    body = tmp_path.read_bytes()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                finally:
                    tmp_path.unlink(missing_ok=True)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/label":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            try:
                result = store.update_label(
                    index=int(payload["index"]),
                    label=str(payload.get("label", "")),
                    notes=payload.get("notes"),
                    annotator=payload.get("annotator"),
                )
            except Exception as exc:  # pragma: no cover - server guard
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result)

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self) -> None:
            body = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stage 4 Annotation</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #667085;
      --line: #d8dde6;
      --accent: #2563eb;
      --good: #0f766e;
      --warn: #b45309;
      --bad: #b91c1c;
      --shadow: 0 8px 24px rgba(16, 24, 40, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 3;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 12px 20px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: center;
    }
    .title { font-size: 16px; font-weight: 700; }
    .meta { color: var(--muted); font-size: 13px; margin-top: 3px; }
    .progress-wrap {
      min-width: 280px;
      display: grid;
      gap: 6px;
    }
    .progress {
      height: 8px;
      background: #e8ebf0;
      border-radius: 999px;
      overflow: hidden;
    }
    .progress > div { height: 100%; background: var(--accent); width: 0%; }
    main {
      display: grid;
      grid-template-columns: minmax(560px, 1fr) 380px;
      gap: 18px;
      padding: 18px;
      max-width: 1480px;
      margin: 0 auto;
    }
    section, aside {
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      border-radius: 8px;
    }
    .viewer { padding: 16px; }
    .row-image {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      display: block;
    }
    .captions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 12px;
    }
    .caption-box {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 12px;
      min-height: 96px;
      background: #fbfcfe;
      line-height: 1.45;
      font-size: 14px;
    }
    .caption-box b { display: block; margin-bottom: 4px; color: var(--muted); }
    aside { padding: 16px; align-self: start; position: sticky; top: 78px; }
    .candidate { font-size: 15px; font-weight: 700; }
    .score-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin: 12px 0;
    }
    .score {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fbfcfe;
    }
    .score span { display: block; color: var(--muted); font-size: 12px; }
    .score strong { font-size: 16px; }
    .labels { display: grid; gap: 10px; margin: 14px 0; }
    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 7px;
      min-height: 42px;
      padding: 10px 12px;
      font-size: 14px;
      font-weight: 650;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    button.active { color: #fff; border-color: transparent; }
    .duplicate.active { background: var(--bad); }
    .near-duplicate.active { background: var(--warn); }
    .not-duplicate.active { background: var(--good); }
    .nav {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 12px;
    }
    .wide { grid-column: 1 / -1; }
    input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
      background: #fff;
    }
    textarea { min-height: 84px; resize: vertical; margin-top: 8px; }
    .field-label { font-size: 12px; color: var(--muted); margin: 12px 0 5px; }
    .hint {
      margin-top: 12px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .path {
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      word-break: break-all;
    }
    @media (max-width: 980px) {
      header { grid-template-columns: 1fr; }
      main { grid-template-columns: 1fr; padding: 12px; }
      aside { position: static; }
      .captions { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="title">Stage 4 Pair Annotation</div>
      <div id="subtitle" class="meta">loading</div>
    </div>
    <div class="progress-wrap">
      <div class="meta" id="progressText">0 / 0</div>
      <div class="progress"><div id="progressBar"></div></div>
    </div>
  </header>
  <main>
    <section class="viewer">
      <img id="rowImage" class="row-image" alt="current review row" />
      <div class="captions">
        <div class="caption-box"><b>Caption A</b><span id="captionA"></span></div>
        <div class="caption-box"><b>Caption B</b><span id="captionB"></span></div>
      </div>
    </section>
    <aside>
      <div class="candidate" id="candidateId">candidate</div>
      <div class="meta" id="indexText">row</div>
      <div class="score-grid">
        <div class="score"><span>image</span><strong id="scoreImage">-</strong></div>
        <div class="score"><span>text</span><strong id="scoreText">-</strong></div>
        <div class="score"><span>joint</span><strong id="scoreJoint">-</strong></div>
      </div>
      <div class="field-label">annotator</div>
      <input id="annotator" />
      <div class="labels">
        <button class="duplicate" data-label="duplicate">1 · Duplicate</button>
        <button class="near-duplicate" data-label="near-duplicate">2 · Near Duplicate</button>
        <button class="not-duplicate" data-label="not-duplicate">3 · Not Duplicate</button>
      </div>
      <div class="field-label">notes</div>
      <textarea id="notes" placeholder="不确定、caption 相同但图片不同等情况写这里"></textarea>
      <div class="nav">
        <button id="prevBtn">上一条</button>
        <button id="nextBtn">下一条</button>
        <button id="prevTodoBtn">上一条未标</button>
        <button id="nextTodoBtn">下一条未标</button>
        <button id="clearBtn" class="wide">清空当前标签</button>
      </div>
      <div class="hint">
        快捷键：1 duplicate，2 near-duplicate，3 not-duplicate，←/→ 切换。标注会立即写入输出 CSV。
        辅助规则：image/text 两个相似度均 >0.85 且 <0.95 可标 near-duplicate；均 >0.95 可标 duplicate。
      </div>
      <div class="path" id="outputPath"></div>
    </aside>
  </main>
<script>
let current = 0;
let payload = null;

async function loadRow(index = null) {
  const url = index === null ? "/api/row" : `/api/row?index=${index}`;
  const res = await fetch(url);
  payload = await res.json();
  current = payload.index;
  render();
}

function fmt(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(3) : "-";
}

function render() {
  const row = payload.row;
  const st = payload.status;
  const pct = st.total ? Math.round(st.done * 100 / st.total) : 0;
  document.getElementById("subtitle").textContent = `${st.mode} · ${st.annotator} · ${st.remaining} remaining`;
  document.getElementById("progressText").textContent = `${st.done} / ${st.total} (${pct}%)`;
  document.getElementById("progressBar").style.width = `${pct}%`;
  document.getElementById("candidateId").textContent = row.candidate_id;
  document.getElementById("indexText").textContent = `row ${current + 1} / ${payload.total_rows} · bucket ${row.bucket} · audit ${row.needs_audit}`;
  document.getElementById("scoreImage").textContent = fmt(row.image_similarity);
  document.getElementById("scoreText").textContent = fmt(row.text_similarity);
  document.getElementById("scoreJoint").textContent = fmt(row.joint_similarity);
  document.getElementById("captionA").textContent = row.caption_a || "";
  document.getElementById("captionB").textContent = row.caption_b || "";
  document.getElementById("notes").value = row.notes || "";
  document.getElementById("annotator").value = row.annotator || st.annotator;
  document.getElementById("outputPath").textContent = `output: ${st.output_csv}`;
  document.getElementById("rowImage").src = `/assets/review-row.jpg?index=${current}&t=${Date.now()}`;
  const currentLabel = st.mode === "audit" ? row.audit_label : row.label;
  document.querySelectorAll("[data-label]").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.label === currentLabel);
  });
}

async function save(label, advance = true) {
  const body = {
    index: current,
    label,
    notes: document.getElementById("notes").value,
    annotator: document.getElementById("annotator").value,
  };
  const res = await fetch("/api/label", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  payload = await res.json();
  if (payload.error) {
    alert(payload.error);
    return;
  }
  if (advance && payload.next_unlabeled !== null) {
    await loadRow(payload.next_unlabeled);
  } else {
    render();
  }
}

document.querySelectorAll("[data-label]").forEach((btn) => {
  btn.addEventListener("click", () => save(btn.dataset.label));
});
document.getElementById("clearBtn").addEventListener("click", () => save("", false));
document.getElementById("prevBtn").addEventListener("click", () => loadRow(Math.max(0, current - 1)));
document.getElementById("nextBtn").addEventListener("click", () => loadRow(Math.min(payload.total_rows - 1, current + 1)));
document.getElementById("prevTodoBtn").addEventListener("click", () => {
  if (payload.prev_unlabeled !== null) loadRow(payload.prev_unlabeled);
});
document.getElementById("nextTodoBtn").addEventListener("click", () => {
  if (payload.next_unlabeled !== null) loadRow(payload.next_unlabeled);
});
document.addEventListener("keydown", (event) => {
  if (event.target.tagName === "TEXTAREA" || event.target.tagName === "INPUT") return;
  const key = event.key || event.code;
  if (key === "1" || event.code === "Digit1" || event.code === "Numpad1") save("duplicate");
  if (key === "2" || event.code === "Digit2" || event.code === "Numpad2") save("near-duplicate");
  if (key === "3" || event.code === "Digit3" || event.code === "Numpad3") save("not-duplicate");
  if (event.key === "ArrowLeft") loadRow(Math.max(0, current - 1));
  if (event.key === "ArrowRight") loadRow(Math.min(payload.total_rows - 1, current + 1));
});
loadRow();
</script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    store = AnnotationStore(
        annotations_csv=args.annotations_csv,
        output_csv=args.output_csv,
        annotator=args.annotator,
        mode=args.mode,
        rows_per_sheet=args.rows_per_sheet,
    )
    handler = make_handler(store)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving Stage 4 annotation app at http://{args.host}:{args.port}")
    print(f"Source CSV: {store.annotations_csv}")
    print(f"Output CSV: {store.output_csv}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
