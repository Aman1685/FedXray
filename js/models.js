/* models.js: the Model Selection section.
   Other sections can call getModelSelection() to learn what the user picked,
   or listen for the "modelselectionchange" event on document. */

/* ============ DATA (edit here) ============ */

const SELECTION_STORAGE_KEY = "fedxray-selection";

// DEMO configuration options. They are shown ONLY while the backend is not connected,
// and the page labels them as previews. The real list comes from the backend.
const FEDERATED_DEMO_OPTIONS = [
  {
    id: "iid",
    label: "IID split",
    description: "Data divided randomly across the clients."
  },
  {
    id: "non-iid",
    label: "Non-IID split",
    description: "Each client holds a different mix of labels."
  },
  {
    id: "balanced-non-iid",
    label: "Balanced Non-IID split",
    description: "Clients have different label distributions while maintaining balanced data sizes."
  }
];

const DEMO_CONFIGURATIONS = {
  centralized: [
    {
      id: "standard",
      label: "Standard training",
      description: "All X-rays pooled on one machine."
    }
  ],
  fedavg: FEDERATED_DEMO_OPTIONS,
  fedprox: FEDERATED_DEMO_OPTIONS
};
/* ============ STATE ============ */

let modelCatalogue = { source: "demo", message: "", models: [] };   // what exists
let modelSelection = { modelId: null, configId: null };            // what the user picked

function findModel(modelId) {
  return modelCatalogue.models.find(model => model.id === modelId) || null;
}

function defaultConfigId(model) {
  return model.configurations.length > 0 ? model.configurations[0].id : null;
}

// Other sections call this. Returns { model, configuration, source } (model is null if nothing is chosen).
function getModelSelection() {
  const model = findModel(modelSelection.modelId);
  const configuration = model ? model.configurations.find(item => item.id === modelSelection.configId) || null : null;
  return { model, configuration, source: modelCatalogue.source };
}

/* ============ LOADING THE CATALOGUE ============ */

function buildDemoCatalogue(message) {
  return {
    source: "demo",
    message: message,
    models: MODELS.map(model => ({ ...model, available: true, configurations: DEMO_CONFIGURATIONS[model.id] || [] }))
  };
}

// Turns the backend's answer into the shape this page uses.
// Names, colours and descriptions come from MODELS (config.js) when the backend does not send them.
function normaliseBackendModels(data) {
  if (!data || !Array.isArray(data.models)) throw new Error("Unexpected model list format");

  return data.models.map(item => {
    const known = MODELS.find(model => model.id === item.id) || {};
    return {
      id: String(item.id),
      name: item.name || known.name || String(item.id),
      kind: known.kind || "Model",
      colorVar: known.colorVar || "--color-primary",
      summary: item.description || known.summary || "",
      training: known.training || "Not described",
      available: item.available !== false,
      configurations: Array.isArray(item.configurations) ? item.configurations.filter(config => config && config.id !== undefined) : []
    };
  });
}

async function loadModelCatalogue() {
  if (!ENABLE_BACKEND_CHECK) {
    return buildDemoCatalogue("Demo catalogue: the backend is not connected, so these models and options are interface previews.");
  }
  try {
    const data = await fetchModelCatalogue();
    return { source: "backend", message: "Model list loaded from the backend.", models: normaliseBackendModels(data) };
  } catch (error) {
    return buildDemoCatalogue("The model list could not be loaded from the backend, so a demo catalogue is shown instead.");
  }
}

/* ============ SAVING THE CHOICE ============ */

function saveSelection() {
  try {
    localStorage.setItem(SELECTION_STORAGE_KEY, JSON.stringify(modelSelection));
  } catch (error) {
    /* storage blocked: the choice just will not be remembered */
  }
}

// Bring back the saved choice, but only if it still exists in the current catalogue
function restoreSelection() {
  modelSelection = { modelId: null, configId: null };
  try {
    const saved = JSON.parse(localStorage.getItem(SELECTION_STORAGE_KEY));
    const model = saved ? findModel(saved.modelId) : null;
    if (model && model.available) {
      const configuration = model.configurations.find(item => item.id === saved.configId);
      modelSelection = { modelId: model.id, configId: configuration ? configuration.id : defaultConfigId(model) };
    }
  } catch (error) {
    /* nothing saved, or unreadable: start with no selection */
  }
}

/* ============ CHANGING THE SELECTION ============ */

function afterSelectionChange() {
  saveSelection();
  renderModelSelection();
  document.dispatchEvent(new CustomEvent("modelselectionchange", { detail: getModelSelection() }));
}

function selectModel(modelId) {
  const model = findModel(modelId);
  if (!model || !model.available) return;
  modelSelection = { modelId: model.id, configId: defaultConfigId(model) };   // first configuration is pre-selected
  afterSelectionChange();
}

function selectConfiguration(configId) {
  if (!modelSelection.modelId) return;
  modelSelection.configId = configId;
  afterSelectionChange();
}

function clearSelection() {
  modelSelection = { modelId: null, configId: null };
  afterSelectionChange();
}

/* ============ DRAWING THE PAGE ============ */

function renderCatalogueNotice() {
  const notice = document.getElementById("catalogue-notice");
  notice.dataset.source = modelCatalogue.source;     // CSS colours the banner by source
  notice.textContent = modelCatalogue.message;
}

function renderModelCards() {
  const container = document.getElementById("model-options");

  if (modelCatalogue.models.length === 0) {
    container.innerHTML = `<p class="ms-empty">No models are available.</p>`;
    return;
  }

  container.innerHTML = modelCatalogue.models.map(model => {
    const isSelected = model.id === modelSelection.modelId;
    const availability = modelCatalogue.source === "demo" ? "Demo preview" : (model.available ? "Available" : "Not available");
    return `
      <label class="select-card ${isSelected ? "is-selected" : ""} ${model.available ? "" : "is-disabled"}"
             style="--model-color: var(${model.colorVar})">
        <input type="radio" name="model" value="${escapeHtml(model.id)}" ${isSelected ? "checked" : ""} ${model.available ? "" : "disabled"}>
        <span class="select-card__top"><strong>${escapeHtml(model.name)}</strong><span class="badge">${escapeHtml(model.kind)}</span></span>
        <span class="select-card__text">${escapeHtml(model.summary)}</span>
        <span class="select-card__text"><strong>Training:</strong> ${escapeHtml(model.training)}</span>
        <span class="select-card__foot">
          <span class="badge badge--muted">${availability}</span>
          <span class="select-card__picked">Selected</span>
        </span>
      </label>`;
  }).join("");
}

function renderConfigurations() {
  const container = document.getElementById("config-options");
  const model = findModel(modelSelection.modelId);

  if (!model) {
    container.innerHTML = `<p class="ms-empty">Choose a model first.</p>`;
    return;
  }
  if (model.configurations.length === 0) {
    container.innerHTML = `<p class="ms-empty">${escapeHtml(model.name)} has no selectable training configurations.</p>`;
    return;
  }

  container.innerHTML = model.configurations.map(config => {
    const isSelected = config.id === modelSelection.configId;
    return `
      <label class="config-option ${isSelected ? "is-selected" : ""}">
        <input type="radio" name="configuration" value="${escapeHtml(config.id)}" ${isSelected ? "checked" : ""}>
        <span><strong>${escapeHtml(config.label || config.id)}</strong><small>${escapeHtml(config.description || "")}</small></span>
      </label>`;
  }).join("");
}

function renderSummary() {
  const { model, configuration, source } = getModelSelection();
  const none = `<span class="badge badge--muted">None selected</span>`;

  let configurationText = none;
  if (model) configurationText = configuration ? escapeHtml(configuration.label || configuration.id) : "Not applicable";

  document.getElementById("selection-summary").innerHTML = `
    <div><dt>Model</dt><dd>${model ? escapeHtml(model.name) : none}</dd></div>
    <div><dt>Configuration</dt><dd>${configurationText}</dd></div>
    <div><dt>Data source</dt><dd>${source === "demo" ? "Demo catalogue" : "Backend"}</dd></div>`;

  // The Continue link looks and behaves disabled until a model is chosen
  const continueButton = document.getElementById("continue-button");
  continueButton.setAttribute("aria-disabled", String(!model));
  continueButton.tabIndex = model ? 0 : -1;
}

function renderModelSelection() {
  renderCatalogueNotice();
  renderModelCards();
  renderConfigurations();
  renderSummary();
}

/* ============ STARTING UP ============ */

async function loadAndRenderCatalogue() {
  const container = document.getElementById("model-options");
  container.setAttribute("aria-busy", "true");
  container.innerHTML = `<p class="ms-empty">Loading models</p>`;

  modelCatalogue = await loadModelCatalogue();
  restoreSelection();

   container.removeAttribute("aria-busy");
  renderModelSelection();
  document.dispatchEvent(new CustomEvent("modelselectionchange", { detail: getModelSelection() }));
}


// After a radio list is redrawn, put keyboard focus back on the checked option
function refocusChecked(containerId) {
  const checked = document.querySelector(`#${containerId} input:checked`);
  if (checked) checked.focus();
}

// Called once from app.js
function initModelSelection() {
  document.getElementById("model-options").addEventListener("change", event => {
    if (event.target.name === "model") {
      selectModel(event.target.value);
      refocusChecked("model-options");
    }
  });
  document.getElementById("config-options").addEventListener("change", event => {
    if (event.target.name === "configuration") {
      selectConfiguration(event.target.value);
      refocusChecked("config-options");
    }
  });
  document.getElementById("clear-selection").addEventListener("click", clearSelection);
  document.getElementById("reload-catalogue").addEventListener("click", loadAndRenderCatalogue);

  // Buttons elsewhere (for example "Select model" on the dashboard) can preselect a model
  // by carrying data-select-model="fedavg". The link itself still takes the user to this page.
  document.addEventListener("click", event => {
    const trigger = event.target.closest("[data-select-model]");
    if (trigger) selectModel(trigger.dataset.selectModel);
  });

  loadAndRenderCatalogue();
}