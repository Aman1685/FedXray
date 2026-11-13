/* dashboard.js: fills the dashboard with content from config.js.
   We only insert our own fixed text with innerHTML here.
   Later, for text that comes from the backend, use textContent instead. */

// Four statistic cards at the top
function renderKpis() {
  document.getElementById("kpi-grid").innerHTML = DATASET_FACTS.map(fact => `
    <article class="kpi">
      <span class="kpi__icon">${iconSvg(fact.icon)}</span>
      <span>
        <span class="kpi__value">${fact.value.toLocaleString("en-US")}</span>
        <span class="kpi__label">${fact.label}</span>
        <span class="kpi__caption">${fact.caption}</span>
      </span>
    </article>
  `).join("");
}

// Model overview cards. The coloured top edge comes from --model-color.
function renderModels() {
  document.getElementById("model-grid").innerHTML = MODELS.map(model => `
    <article class="model-card" style="--model-color: var(${model.colorVar})">
      <div class="model-card__top">
        <h4>${model.name}</h4>
        <span class="badge">${model.kind}</span>
      </div>
      <p>${model.summary}</p>
      <p><strong>Training:</strong> ${model.training}</p>
      <!-- Placeholder. Once the backend lists its models, this will show real availability. -->
      <span class="badge badge--muted">Availability pending</span>
      <a class="button button--outline button--small" href="#models" data-select-model="${model.id}">Select model</a>
    </article>
  `).join("");
}

// The 14 label names (the list lives in dataset.js)
function renderLabelChips() {
  document.getElementById("label-chips").innerHTML = LABEL_COUNTS
    .map(item => `<li class="chip">${item.label}</li>`)
    .join("");
}

function renderWorkflow() {
  document.getElementById("workflow-steps").innerHTML = WORKFLOW_STEPS.map((step, index) => `
    <li class="step">
      <span class="step__number">${index + 1}</span>
      <h4>${step.title}</h4>
      <p>${step.text}</p>
    </li>
  `).join("");
}

// Placeholder chart: same-height dashed bars, one per model, for each metric.
// It deliberately contains NO values. Real data arrives in the Model Comparison step.
function renderComparisonPreview() {
  const groups = COMPARISON_METRICS.map(() => `
    <div class="compare-group">
      ${MODELS.map(model => `<span class="compare-bar" style="--bar-color: var(${model.colorVar})"></span>`).join("")}
    </div>`).join("");

  const labels = COMPARISON_METRICS.map(name => `<span>${name}</span>`).join("");

  const legend = MODELS.map(model =>
    `<li><span class="legend__dot" style="background: var(${model.colorVar})"></span>${model.name}</li>`
  ).join("");

  document.getElementById("compare-preview").innerHTML = `
    <div class="compare-chart" role="img" aria-label="Empty chart placeholder. Real metrics appear once the backend provides them.">${groups}</div>
    <div class="compare-labels">${labels}</div>
    <ul class="legend">${legend}</ul>`;
}

// Recent project activity (entries come from ACTIVITY in config.js)
function renderActivity() {
  const stateLabels = { done: "Done", next: "Up next", planned: "Planned" };

  document.getElementById("activity-list").innerHTML = ACTIVITY.map(item => `
    <li data-state="${item.state}">
      <span class="timeline__title">${item.title}</span>
      <span class="timeline__detail">${item.detail}</span>
      <span class="timeline__state">${stateLabels[item.state]}</span>
    </li>
  `).join("");
}

// Quick navigation: every section except the dashboard itself
function renderQuickLinks() {
  document.getElementById("quick-links").innerHTML = SECTIONS
    .filter(section => section.id !== "dashboard")
    .map(section => `
      <li>
        <a class="nav-tile" href="#${section.id}">
          <span class="nav-tile__icon">${iconSvg(section.icon)}</span>
          <span><strong>${section.title}</strong><small>${section.description}</small></span>
        </a>
      </li>
    `).join("");
}