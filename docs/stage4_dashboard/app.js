async function loadStatus() {
  const response = await fetch("data/status.json");
  if (!response.ok) throw new Error("failed to load status.json");
  return response.json();
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function pct(value) {
  const n = Number(value || 0);
  return `${Math.max(0, Math.min(100, n))}%`;
}

function formatNumber(value) {
  return Number(value || 0).toLocaleString("en-US");
}

function render(data) {
  document.getElementById("claim").textContent = data.target.claim;
  document.getElementById("targetVenue").textContent = data.target.venue;
  document.getElementById("targetDeadline").textContent = data.target.deadline;
  document.getElementById("updatedAt").textContent = `更新于 ${data.updated_at.replace("T", " ")}`;
  renderCards(data.summary_cards);
  renderCharts(data.charts);
  renderPhases(data.phase_progress);
  renderAnnotation(data.annotation);
  renderExperiments(data.experiments);
  renderPaperData(data.paper_writing_data || []);
  renderPlanDataMatrix(data.plan_data_matrix || []);
  renderRequirements(data.plan_requirements);
  renderRisks(data.risks);
  renderNextSteps(data.next_steps);
  renderArtifacts(data.artifacts);
  renderExports(data.data_exports);
}

function renderCharts(charts) {
  renderBarChart("candidateFunnelChart", charts.candidate_funnel, {
    valueFormatter: formatNumber,
    maxMode: "local",
  });
  renderStackedChart(
    "annotationDistributionChart",
    "annotationDistributionLegend",
    charts.annotation_distribution
  );
  renderBarChart("phaseProgressChart", charts.phase_progress, {
    valueFormatter: (value) => `${value}%`,
    maxMode: "fixed100",
  });
  renderBarChart("stage4EvalChart", charts.stage4_eval_best_f1 || [], {
    valueFormatter: (value) => value.toFixed(3),
    fixedMax: 1,
  });
  renderBarChart("runtimeChart", charts.experiment_runtime, {
    valueFormatter: (value) => `${value} min`,
    maxMode: "local",
  });
}

function renderBarChart(rootId, rows, options = {}) {
  const root = document.getElementById(rootId);
  root.innerHTML = "";
  const maxValue = options.fixedMax
    ? options.fixedMax
    : options.maxMode === "fixed100"
      ? 100
      : Math.max(1, ...rows.map((row) => Number(row.value || 0)));
  rows.forEach((row) => {
    const value = Number(row.value || 0);
    const item = el("div", "chart-row");
    const top = el("div", "chart-row-top");
    top.append(el("span", "", row.label));
    top.append(el("strong", "", (options.valueFormatter || formatNumber)(value)));
    const track = el("div", "chart-track");
    const fill = el("div", `chart-fill ${row.status || row.key || ""}`);
    fill.style.width = pct((value / maxValue) * 100);
    track.append(fill);
    item.append(top);
    item.append(track);
    if (row.note) item.append(el("small", "", row.note));
    root.append(item);
  });
}

function renderStackedChart(rootId, legendId, rows) {
  const root = document.getElementById(rootId);
  const legend = document.getElementById(legendId);
  root.innerHTML = "";
  legend.innerHTML = "";
  const total = Math.max(1, rows.reduce((sum, row) => sum + Number(row.value || 0), 0));
  rows.forEach((row) => {
    const value = Number(row.value || 0);
    const segment = el("div", `stack-segment ${row.key}`);
    segment.style.width = pct((value / total) * 100);
    segment.title = `${row.label}: ${formatNumber(value)}`;
    root.append(segment);

    const item = el("span", "legend-item");
    item.append(el("b", row.key));
    item.append(document.createTextNode(`${row.label} ${formatNumber(value)}`));
    legend.append(item);
  });
}

function renderCards(cards) {
  const root = document.getElementById("summaryCards");
  root.innerHTML = "";
  cards.forEach((card) => {
    const node = el("article", "card");
    node.append(el("span", "", card.label));
    node.append(el("strong", "", card.value));
    node.append(el("p", "", card.note));
    root.append(node);
  });
}

function renderPhases(phases) {
  const root = document.getElementById("phaseList");
  root.innerHTML = "";
  phases.forEach((phase) => {
    const node = el("div", "phase");
    const copy = el("div");
    copy.append(el("strong", "", phase.name));
    copy.append(el("small", "", phase.detail));
    const bar = el("div", "bar");
    const fill = el("div");
    fill.style.width = pct(phase.percent);
    bar.append(fill);
    node.append(copy);
    node.append(bar);
    node.append(el("span", `pill ${phase.status}`, statusText(phase.status)));
    root.append(node);
  });
}

function renderAnnotation(annotation) {
  document.getElementById("annotationPercent").textContent = `${annotation.percent}%`;
  document.getElementById("annotationBar").style.width = pct(annotation.percent);
  const root = document.getElementById("annotationStats");
  root.innerHTML = "";
  [
    ["已标注", annotation.done],
    ["剩余", annotation.remaining],
    ["正例数量", annotation.positives],
    ["抽查行数", annotation.audit_rows],
    ["重复", annotation.counts.duplicate],
    ["近重复", annotation.counts["near-duplicate"]],
    ["非重复", annotation.counts["not-duplicate"]],
    ["未标注", annotation.counts.unlabeled],
  ].forEach(([label, value]) => {
    const item = el("div", "stat");
    item.append(el("span", "", label));
    item.append(el("strong", "", String(value)));
    root.append(item);
  });
}

function renderExperiments(experiments) {
  const root = document.getElementById("experimentRows");
  root.innerHTML = "";
  experiments.forEach((exp) => {
    const tr = document.createElement("tr");
    [exp.id, exp.dataset, exp.runtime, exp.numbers, exp.notes].forEach((value, index) => {
      const td = document.createElement("td");
      if (index === 0) {
        const code = document.createElement("code");
        code.textContent = value;
        td.append(code);
      } else {
        td.textContent = value;
      }
      tr.append(td);
    });
    root.append(tr);
  });
}

function renderPaperData(sections) {
  const root = document.getElementById("paperDataList");
  root.innerHTML = "";
  sections.forEach((section) => {
    const node = el("article", "paper-data-card");
    const head = el("div", "paper-data-head");
    head.append(el("div", "", section.title));
    head.append(el("span", `pill ${section.status}`, statusText(section.status)));
    node.append(head);
    node.append(el("p", "paper-data-purpose", section.paper_use));

    const facts = el("div", "paper-facts");
    section.key_numbers.forEach((item) => {
      const fact = el("div", "paper-fact");
      fact.append(el("span", "", item.label));
      fact.append(el("strong", "", String(item.value)));
      if (item.note) fact.append(el("small", "", item.note));
      facts.append(fact);
    });
    node.append(facts);

    const links = el("div", "paper-links");
    section.sources.forEach((source) => {
      const link = el("a", "source-link");
      link.href = source.href;
      link.target = "_blank";
      link.rel = "noopener";
      link.append(el("strong", "", source.title));
      link.append(el("span", "", source.description));
      link.append(el("code", "", source.href));
      links.append(link);
    });
    node.append(links);
    root.append(node);
  });
}

function renderPlanDataMatrix(groups) {
  const root = document.getElementById("planDataMatrix");
  root.innerHTML = "";
  groups.forEach((group) => {
    const section = el("article", "matrix-group");
    const head = el("div", "matrix-head");
    head.append(el("div", "", group.experiment));
    head.append(el("span", `pill ${group.status}`, statusText(group.status)));
    section.append(head);
    section.append(el("p", "paper-data-purpose", group.purpose));

    const table = document.createElement("table");
    table.className = "matrix-table";
    const thead = document.createElement("thead");
    const header = document.createElement("tr");
    ["数据 / 表格", "当前状态", "已有数字", "来源 / 缺口"].forEach((title) => {
      header.append(el("th", "", title));
    });
    thead.append(header);
    table.append(thead);

    const tbody = document.createElement("tbody");
    group.items.forEach((item) => {
      const tr = document.createElement("tr");
      const name = document.createElement("td");
      name.append(el("strong", "", item.name));
      if (item.description) name.append(el("small", "", item.description));
      tr.append(name);
      const status = document.createElement("td");
      status.append(el("span", `pill ${item.status}`, statusText(item.status)));
      tr.append(status);
      const numbers = document.createElement("td");
      numbers.textContent = item.current_numbers || "待生成";
      tr.append(numbers);
      const sources = document.createElement("td");
      if (item.sources && item.sources.length) {
        item.sources.forEach((source) => {
          const link = el("a", "inline-source", source.title);
          link.href = source.href;
          link.target = "_blank";
          link.rel = "noopener";
          sources.append(link);
        });
      }
      if (item.gap) sources.append(el("small", "", item.gap));
      tr.append(sources);
      tbody.append(tr);
    });
    table.append(tbody);
    const wrap = el("div", "table-wrap");
    wrap.append(table);
    section.append(wrap);
    root.append(section);
  });
}

function renderRisks(risks) {
  const root = document.getElementById("riskList");
  root.innerHTML = "";
  risks.forEach((risk) => {
    const node = el("div", "risk");
    node.append(el("strong", "", `${riskLevelText(risk.level)} · ${risk.title}`));
    node.append(el("p", "", risk.detail));
    root.append(node);
  });
}

function renderRequirements(requirements) {
  const root = document.getElementById("requirementsList");
  root.innerHTML = "";
  requirements.forEach((item) => {
    const node = el("article", "requirement");
    const head = el("div", "requirement-head");
    head.append(el("strong", "", item.name));
    head.append(el("span", `pill ${item.status}`, statusText(item.status)));
    node.append(head);
    node.append(renderMiniList("需要的数据", item.required_data));
    node.append(renderMiniList("当前产物", item.current_outputs));
    node.append(renderMiniList("证据文件", item.evidence, true));
    root.append(node);
  });
}

function renderMiniList(title, values, code = false) {
  const wrap = el("div", "mini-list");
  wrap.append(el("span", "", title));
  const ul = document.createElement("ul");
  values.forEach((value) => {
    const li = document.createElement("li");
    if (code) {
      const codeNode = document.createElement("code");
      codeNode.textContent = value;
      li.append(codeNode);
    } else {
      li.textContent = value;
    }
    ul.append(li);
  });
  wrap.append(ul);
  return wrap;
}

function renderNextSteps(steps) {
  const root = document.getElementById("nextSteps");
  root.innerHTML = "";
  steps.forEach((step) => {
    root.append(el("li", "", step));
  });
}

function renderArtifacts(artifacts) {
  const root = document.getElementById("artifactList");
  root.innerHTML = "";
  artifacts.forEach((artifact) => {
    const node = el("div", "artifact");
    node.append(el("strong", "", artifact.title));
    const path = el("p");
    const code = document.createElement("code");
    code.textContent = artifact.path;
    path.append(code);
    node.append(path);
    root.append(node);
  });
}

function renderExports(exports) {
  const root = document.getElementById("exportList");
  root.innerHTML = "";
  exports.forEach((entry) => {
    const node = el("a", "export-card");
    node.href = entry.href;
    node.download = "";
    node.append(el("strong", "", entry.title));
    node.append(el("p", "", entry.description));
    node.append(el("code", "", entry.href));
    root.append(node);
  });
}

loadStatus().then(render).catch((error) => {
  document.getElementById("claim").textContent = error.message;
});

function statusText(status) {
  return {
    done: "已完成",
    complete: "已完成",
    active: "进行中",
    blocked: "阻塞",
    pending: "待开始",
    partial: "部分完成",
  }[status] || status;
}

function riskLevelText(level) {
  return {
    high: "高风险",
    medium: "中风险",
    low: "低风险",
  }[level] || level;
}
