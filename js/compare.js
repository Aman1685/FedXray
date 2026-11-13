/* compare.js: the Model Comparison section.
   It shows REAL evaluation numbers only. They come from results-data.js (or later from the backend).
   If there are none, the page says so instead of showing anything made up. */

/* ============ DATA (edit here) ============ */

// The metrics this page knows how to show. "key" must match the names used in results-data.js.
// To show another metric (for example specificity), add a line here and add that key to your data.
const METRIC_DEFINITIONS = [
  { key: "accuracy", label: "Per-label Accuracy" },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "f1", label: "F1-score" },
  { key: "auc", label: "AUC-ROC" }
];

/* ============ STATE AND PAGE ELEMENTS ============ */

let comparisonData = { source: "none", description: "", series: [], warnings: [] };
let availableMetrics = [];        // metrics that at least one entry actually has
let selectedMetricKey = null;

const compareStatus = document.getElementById("compare-status");
const compareWarnings = document.getElementById("compare-warnings");
const compareEmpty = document.getElementById("compare-empty");
const compareContent = document.getElementById("compare-content");
const compareLeaders = document.getElementById("compare-leaders");
const metricToggle = document.getElementById("metric-toggle");
const compareChart = document.getElementById("compare-chart");
const compareLegend = document.getElementById("compare-legend");
const compareTable = document.getElementById("compare-table");
const compareNote = document.getElementById("compare-note");

/* ============ LOADING AND CHECKING THE DATA ============ */

// Keeps only valid entries and values, and collects a warning for everything it ignores.
function cleanResults(data) {
  const warnings = [];
  const series = [];
  const items = data && Array.isArray(data.series) ? data.series : [];

  items.forEach((item, index) => {
    const name = item && item.label ? String(item.label) : "Entry " + (index + 1);

    if (!item || typeof item.metrics !== "object" || item.metrics === null) {
      warnings.push(name + ": has no metrics, so it was skipped.");
      return;
    }

    const metrics = {};
    METRIC_DEFINITIONS.forEach(definition => {
      const value = item.metrics[definition.key];
      if (value === undefined || value === null) return;            // not provided: simply left out
      if (typeof value === "number" && value >= 0 && value <= 1) {
        metrics[definition.key] = value;
      } else {
        warnings.push(name + ": " + definition.label + " must be a number from 0 to 1 (for 85.2% write 0.852), so it was ignored.");
      }
    });

    if (Object.keys(metrics).length === 0) {
      warnings.push(name + ": has no valid metrics, so it was skipped.");
      return;
    }
    series.push({ model: item.model || "", label: name, metrics: metrics });
  });

  return { description: data && data.description ? String(data.description) : "", series: series, warnings: warnings };
}

async function loadEvaluationResults() {
  const fromFile = cleanResults(EVALUATION_RESULTS);

  return {
    ...fromFile,
    source: fromFile.series.length > 0 ? "file" : "none"
  };
}

/* ============ DRAWING THE PAGE ============ */

function modelColorVar(modelId) {
  const model = MODELS.find(item => item.id === modelId);
  return model ? model.colorVar : "--color-primary";
}

function renderStatus() {
  const messages = {
    none: "No evaluation results are loaded yet, so there is nothing to compare. The steps below explain how to add them.",
    file: "Results loaded from js/results-data.js.",
    backend: "Results loaded from the backend."
  };
  compareStatus.dataset.source = comparisonData.source;     // CSS colours the banner by source
  compareStatus.textContent = messages[comparisonData.source];
  compareWarnings.innerHTML = comparisonData.warnings.map(text => `<li>${escapeHtml(text)}</li>`).join("");
}

// One card per metric showing which entry scored highest
function renderLeaders() {
  compareLeaders.innerHTML = availableMetrics.map(metric => {
    const candidates = comparisonData.series.filter(item => metric.key in item.metrics);
    const best = candidates.reduce((top, item) => item.metrics[metric.key] > top.metrics[metric.key] ? item : top);
    return `
      <article class="cp-leader" style="--model-color: var(${modelColorVar(best.model)})">
        <small>Highest ${metric.label}</small>
        <strong>${escapeHtml(best.label)}</strong>
        <span class="cp-leader__value">${best.metrics[metric.key].toFixed(3)}</span>
      </article>`;
  }).join("");
}

function renderMetricToggle() {
  metricToggle.innerHTML = availableMetrics.map(metric => `
    <button type="button" data-metric="${metric.key}" class="${metric.key === selectedMetricKey ? "is-active" : ""}"
            aria-pressed="${metric.key === selectedMetricKey}">${metric.label}</button>`).join("");
}

// Bars for the selected metric, highest first. Every bar starts at 0 and ends at 1.0.
function renderChart() {
  const rows = comparisonData.series
    .filter(item => selectedMetricKey in item.metrics)
    .sort((a, b) => b.metrics[selectedMetricKey] - a.metrics[selectedMetricKey]);

  compareChart.innerHTML = rows.map(item => {
    const value = item.metrics[selectedMetricKey];
    return `
      <li class="cp-bar" style="--model-color: var(${modelColorVar(item.model)})">
        <span>${escapeHtml(item.label)}</span>
        <span class="cp-bar__track"><span class="cp-bar__fill" style="width: ${value * 100}%"></span></span>
        <span class="cp-bar__value">${value.toFixed(3)}</span>
      </li>`;
  }).join("");

  const modelIds = [...new Set(comparisonData.series.map(item => item.model))];
  compareLegend.innerHTML = modelIds.map(id => {
    const model = MODELS.find(item => item.id === id);
    return `<li><span class="cp-dot" style="background: var(${modelColorVar(id)})"></span>${escapeHtml(model ? model.name : (id || "Other"))}</li>`;
  }).join("");
}

// Every entry and every metric. The best value in each column is highlighted.
function renderTable() {
  const bestValues = {};
  availableMetrics.forEach(metric => {
    bestValues[metric.key] = Math.max(...comparisonData.series.filter(item => metric.key in item.metrics).map(item => item.metrics[metric.key]));
  });

  const head = availableMetrics.map(metric => `<th>${metric.label}</th>`).join("");
  const body = comparisonData.series.map(item => {
    const cells = availableMetrics.map(metric => {
      if (!(metric.key in item.metrics)) return `<td>-</td>`;           // this metric was not provided
      const value = item.metrics[metric.key];
      return `<td class="${value === bestValues[metric.key] ? "is-best" : ""}">${value.toFixed(3)}</td>`;
    }).join("");
    return `<tr><th scope="row"><span class="cp-dot" style="background: var(${modelColorVar(item.model)})"></span>${escapeHtml(item.label)}</th>${cells}</tr>`;
  }).join("");

  compareTable.innerHTML = `<thead><tr><th>Model</th>${head}</tr></thead><tbody>${body}</tbody>`;
}

function renderComparison() {
  const hasData = comparisonData.series.length > 0;
  compareEmpty.hidden = hasData;
  compareContent.hidden = !hasData;
  renderStatus();
  if (!hasData) return;

  availableMetrics = METRIC_DEFINITIONS.filter(metric => comparisonData.series.some(item => metric.key in item.metrics));
  if (!availableMetrics.some(metric => metric.key === selectedMetricKey)) selectedMetricKey = availableMetrics[0].key;

  renderLeaders();
  renderMetricToggle();
  renderChart();
  renderTable();

  compareNote.textContent = (comparisonData.description ||
    "How these numbers were computed was not described. Add a description in results-data.js.") +
    " These are research results on a fixed evaluation split, not clinical performance.";
}

/* ============ STARTING UP ============ */

// Called once from app.js
async function initComparison() {
  // One listener on the button row handles every metric button
  metricToggle.addEventListener("click", event => {
    const button = event.target.closest("button[data-metric]");
    if (!button) return;
    selectedMetricKey = button.dataset.metric;
    renderMetricToggle();
    renderChart();
    metricToggle.querySelector(".is-active").focus();     // buttons were redrawn: keep keyboard focus
  });

  comparisonData = await loadEvaluationResults();
  renderComparison();
}