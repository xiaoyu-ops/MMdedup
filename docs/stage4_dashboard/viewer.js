function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function params() {
  const search = new URLSearchParams(window.location.search);
  return {
    file: search.get("file") || "",
    title: search.get("title") || "数据文件",
  };
}

function safeFilePath(path) {
  return path.startsWith("data/") && !path.includes("..") && !path.startsWith("data/../");
}

async function main() {
  const { file, title } = params();
  document.getElementById("viewerTitle").textContent = title;
  document.getElementById("viewerPath").textContent = file || "未指定";
  document.getElementById("rawLink").href = file || "#";

  if (!safeFilePath(file)) {
    throw new Error("只能查看 data/ 目录下的前端数据文件。");
  }

  const response = await fetch(file);
  if (!response.ok) throw new Error(`读取失败：${response.status} ${response.statusText}`);
  const text = await response.text();
  const root = document.getElementById("viewerRoot");
  root.innerHTML = "";

  if (file.endsWith(".csv")) {
    const rows = parseCsv(text);
    document.getElementById("viewerMeta").textContent = `${Math.max(0, rows.length - 1)} 行`;
    root.append(renderCsvTable(rows));
    return;
  }

  if (file.endsWith(".json")) {
    const data = JSON.parse(text);
    document.getElementById("viewerMeta").textContent = Array.isArray(data) ? `${data.length} 项` : "JSON";
    root.append(renderJsonValue(data));
    return;
  }

  document.getElementById("viewerMeta").textContent = "文本";
  const pre = el("pre", "raw-pre", text);
  root.append(pre);
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (quoted) {
      if (ch === '"' && next === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }
  if (field || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  return rows.filter((item) => item.some((value) => value !== ""));
}

function renderCsvTable(rows) {
  if (!rows.length) return el("p", "paper-data-purpose", "空文件。");
  const header = rows[0];
  const body = rows.slice(1);
  const wrap = el("div", "table-wrap viewer-table-wrap");
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const tr = document.createElement("tr");
  header.forEach((name) => tr.append(el("th", "", name)));
  thead.append(tr);
  table.append(thead);
  const tbody = document.createElement("tbody");
  body.forEach((row) => {
    const bodyRow = document.createElement("tr");
    header.forEach((_, idx) => {
      bodyRow.append(el("td", "", row[idx] || ""));
    });
    tbody.append(bodyRow);
  });
  table.append(tbody);
  wrap.append(table);
  return wrap;
}

function renderJsonValue(value) {
  if (Array.isArray(value)) return renderJsonArray(value);
  if (value && typeof value === "object") return renderJsonObject(value);
  return el("span", "json-scalar", String(value));
}

function renderJsonArray(items) {
  if (!items.length) return el("p", "paper-data-purpose", "空数组。");
  if (items.every((item) => item && typeof item === "object" && !Array.isArray(item))) {
    const keys = [...new Set(items.flatMap((item) => Object.keys(item)))];
    const wrap = el("div", "table-wrap viewer-table-wrap");
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    keys.forEach((key) => headRow.append(el("th", "", key)));
    thead.append(headRow);
    table.append(thead);
    const tbody = document.createElement("tbody");
    items.forEach((item) => {
      const row = document.createElement("tr");
      keys.forEach((key) => row.append(el("td", "", summarizeValue(item[key]))));
      tbody.append(row);
    });
    table.append(tbody);
    wrap.append(table);
    return wrap;
  }
  const list = el("div", "json-list");
  items.forEach((item, idx) => {
    const card = el("article", "json-card");
    card.append(el("strong", "", `#${idx + 1}`));
    card.append(renderJsonValue(item));
    list.append(card);
  });
  return list;
}

function renderJsonObject(obj) {
  const wrap = el("div", "json-object");
  Object.entries(obj).forEach(([key, value]) => {
    const item = el("article", "json-field");
    item.append(el("strong", "", labelForKey(key)));
    if (Array.isArray(value) || (value && typeof value === "object")) {
      item.append(renderJsonValue(value));
    } else {
      item.append(el("span", "json-scalar", String(value)));
    }
    wrap.append(item);
  });
  return wrap;
}

function summarizeValue(value) {
  if (Array.isArray(value)) return `${value.length} 项`;
  if (value && typeof value === "object") return Object.entries(value).map(([k, v]) => `${k}: ${summarizeValue(v)}`).join("; ");
  if (value === undefined || value === null) return "";
  return String(value);
}

function labelForKey(key) {
  const labels = {
    audit_id: "审核 ID",
    overall_assessment: "总体判断",
    findings: "审核发现",
    paper_safety_labels: "论文安全标签",
    safe_to_report_with_qualification: "可带限定写入",
    not_safe_as_final_claim_yet: "暂不能作为最终结论",
    title: "标题",
    detail: "说明",
    action: "建议动作",
    severity: "严重程度",
    current_key_numbers: "当前关键数字",
    hard_consistency_checks: "硬一致性检查",
  };
  return labels[key] || key;
}

main().catch((error) => {
  document.getElementById("viewerRoot").append(el("p", "paper-data-purpose", error.message));
});
