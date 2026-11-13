/* predict.js: the X-ray Prediction section.
   Flow: choose an image -> check it -> preview it -> run the prediction on the backend -> show results.

   IMPORTANT: this file never makes up probabilities. Numbers appear only when the backend sends them,
   or when the user presses "Preview with sample data" (which is clearly marked as sample). */

/* ============ DATA (edit here) ============ */

const ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png"];
const ALLOWED_MIME_TYPES = ["image/jpeg", "image/png"];
const MAX_UPLOAD_MB = 10;

// PLACEHOLDER numbers for the "Preview with sample data" button ONLY.
// They are deliberately round, and they are not model output.
const SAMPLE_PROBABILITIES = [0.9, 0.7, 0.55, 0.45, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05];
const SAMPLE_THRESHOLD = 0.5;

/* ============ PAGE ELEMENTS AND STATE ============ */

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const previewPanel = document.getElementById("preview-panel");
const previewImage = document.getElementById("preview-image");
const previewMeta = document.getElementById("preview-meta");
const uploadMessage = document.getElementById("upload-message");
const selectedModelBox = document.getElementById("selected-model");
const thresholdModeSelect = document.getElementById("threshold-mode");
const runButton = document.getElementById("run-button");
const runHint = document.getElementById("run-hint");
const sampleButton = document.getElementById("sample-button");
const resultsCard = document.getElementById("results-card");
const resultsBody = document.getElementById("results-body");

let selectedImageFile = null;   // the validated image, or null
let previewUrl = null;          // temporary browser address used to show the preview
let isPredicting = false;
let lastResult = null;          // the results currently on screen (used when sorting)
let resultSort = "probability";

/* ============ SMALL HELPERS ============ */

function formatPercent(value) {
  return (value * 100).toFixed(1) + "%";
}

function formatFileSize(bytes) {
  return bytes < 1024 * 1024 ? Math.round(bytes / 1024) + " KB" : (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function formatThresholdMode(mode) {
  if (mode === "tuned") return "Tuned (validation optimized)";
  if (mode === "default") return "Default (0.5)";
  return "Sample preview";
}

function showUploadMessage(text) {
  uploadMessage.textContent = text;
}

/* ============ UPLOAD AND PREVIEW ============ */

// Returns a message if the file is not acceptable, or "" if it is fine.
// These checks are a convenience only: the backend must validate files again.
function validateImageFile(file) {
  const fileName = file.name.toLowerCase();
  const hasAllowedExtension = ALLOWED_EXTENSIONS.some(extension => fileName.endsWith(extension));
  const hasAllowedType = file.type === "" || ALLOWED_MIME_TYPES.includes(file.type);

  if (!hasAllowedExtension || !hasAllowedType) return "Please choose a JPG, JPEG or PNG image.";
  if (file.size === 0) return "That file is empty.";
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) return "That file is larger than " + MAX_UPLOAD_MB + " MB.";
  return "";
}

function clearImage() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);   // free the memory used by the old preview
  previewUrl = null;
  selectedImageFile = null;
  previewImage.removeAttribute("src");
}

function handleFile(file) {
  const problem = validateImageFile(file);
  if (problem) {
    showUploadMessage(problem);
    return;
  }
  showUploadMessage("");

  // Load the image once to make sure it really is a readable picture
  const url = URL.createObjectURL(file);
  const testImage = new Image();

  testImage.onload = () => {
    clearImage();
    selectedImageFile = file;
    previewUrl = url;
    previewImage.src = url;
    previewMeta.innerHTML = `
      <div><dt>File</dt><dd>${escapeHtml(file.name)}</dd></div>
      <div><dt>Size</dt><dd>${formatFileSize(file.size)}</dd></div>
      <div><dt>Dimensions</dt><dd>${testImage.naturalWidth} × ${testImage.naturalHeight} px</dd></div>`;
    dropzone.hidden = true;
    previewPanel.hidden = false;
    showIdleResults();
    updateRunButton();
  };
  testImage.onerror = () => {
    URL.revokeObjectURL(url);
    showUploadMessage("That file could not be read as an image.");
  };
  testImage.src = url;
}

function removeImage() {
  clearImage();
  previewPanel.hidden = true;
  dropzone.hidden = false;
  showUploadMessage("");
  showIdleResults();
  updateRunButton();
}

/* ============ SELECTED MODEL AND RUN BUTTON ============ */

// getModelSelection() comes from models.js
function renderSelectedModel() {
  const { model, configuration, source } = getModelSelection();

  if (!model) {
    selectedModelBox.innerHTML = `<p class="ms-empty">No model selected yet. <a class="text-link" href="#models">Choose one</a></p>`;
    return;
  }
  const configurationText = configuration ? escapeHtml(configuration.label || configuration.id) : "No configuration options";
  selectedModelBox.innerHTML = `
    <div class="px-model" style="--model-color: var(${model.colorVar})">
      <strong>${escapeHtml(model.name)}</strong>
      <span class="badge">${escapeHtml(model.kind)}</span>
      <small>${configurationText}${source === "demo" ? " (demo catalogue)" : ""}</small>
    </div>`;
}

function updateRunButton() {
  const { model } = getModelSelection();
  runButton.disabled = isPredicting || !selectedImageFile || !model;

  let hint = "Ready to run.";
  if (!selectedImageFile) hint = "Upload an X-ray first.";
  else if (!model) hint = "Choose a model on the Model selection page.";
  else if (!ENABLE_BACKEND_CHECK) hint = "The backend is not connected, so no prediction can be made yet.";
  runHint.textContent = hint;
}

/* ============ RESULTS: STATES ============ */

function showResultsMessage(kind, title, text) {
  lastResult = null;
  resultsCard.classList.remove("is-sample");
  resultsBody.innerHTML = `<div class="px-state px-state--${kind}"><strong></strong><p></p></div>`;
  resultsBody.querySelector("strong").textContent = title;   // textContent: safe for any text
  resultsBody.querySelector("p").textContent = text;
}

function showIdleResults() {
  showResultsMessage("idle", "No results yet", "Upload an X-ray, check the selected model, then run a prediction.");
}

function showLoading() {
  lastResult = null;
  resultsCard.classList.remove("is-sample");
  resultsBody.innerHTML = `<div role="status"><p>Running the model on the backend</p>${'<div class="px-skeleton"></div>'.repeat(8)}</div>`;
}

/* ============ RESULTS: THE BACKEND ANSWER ============ */

// Checks the backend's answer and turns it into the shape the page uses.
// There is NO default threshold: if the backend sends none, no label is marked as predicted.
function normalisePredictionResponse(data) {
  if (!data || !Array.isArray(data.predictions) || data.predictions.length === 0) {
    throw new Error("The backend response contained no predictions.");
  }

  const predictions = data.predictions.map(item => {
   const probability = item.probability > 1
  ? item.probability / 100
  : item.probability;
    const hasThreshold = item.threshold !== undefined && item.threshold !== null;
    const threshold = hasThreshold ? item.threshold : null;

    const probabilityIsValid = typeof probability === "number" && probability >= 0 && probability <= 1;
    const thresholdIsValid = threshold === null || (typeof threshold === "number" && threshold >= 0 && threshold <= 1);
    const label = item.label || item.disease;

if (!label || !probabilityIsValid || !thresholdIsValid) {
      throw new Error("The backend response contained an invalid prediction.");
    }

    // Use the backend's own decision if it sends one, otherwise compare with ITS threshold
    const predicted = typeof item.predicted === "boolean" ? item.predicted : (threshold !== null && probability >= threshold);
    return { label: String(label), probability, threshold, predicted };
  });

  return { predictions };
}

async function runPrediction() {
  const { model, configuration } = getModelSelection();
  if (!selectedImageFile || !model || isPredicting) return;

  if (!ENABLE_BACKEND_CHECK) {
    showResultsMessage(
      "info",
      "No prediction was made",
      "The backend is not connected. Enable the backend check in config.js."
    );
    return;
  }

  isPredicting = true;
  updateRunButton();
  showLoading();

  try {
    const thresholdMode = thresholdModeSelect
      ? thresholdModeSelect.value
      : "default";

    const data = await predictXray(
      selectedImageFile,
      model.id,
      configuration ? configuration.id : null,
      thresholdMode
    );

    showResults(normalisePredictionResponse(data), {
      isSample: false,
      modelName: model.name,
      configName: configuration
        ? (configuration.label || configuration.id)
        : "Not applicable",
      imageName: selectedImageFile.name,
      colorVar: model.colorVar,
      thresholdMode: data.threshold_mode || thresholdMode
    });

  } catch (error) {
    showResultsMessage("error", "The prediction failed", error.message);
  } finally {
    isPredicting = false;
    updateRunButton();
  }
}

/* ============ RESULTS: DRAWING THEM ============ */

function showResults(result, context) {
  lastResult = result;
  resultsCard.classList.toggle("is-sample", context.isSample);
  resultsCard.style.setProperty("--model-color", `var(${context.colorVar})`);

  const predictedItems = result.predictions.filter(item => item.predicted).sort((a, b) => b.probability - a.probability);
  const hasThresholds = result.predictions.some(item => item.threshold !== null);

  let predictedHtml = `<p class="px-note">No label reached its threshold.</p>`;
  if (predictedItems.length > 0) {
    predictedHtml = `<ul class="px-chips">${predictedItems.map(item =>
      `<li class="px-chip">${escapeHtml(item.label)} <span>${formatPercent(item.probability)}</span></li>`).join("")}</ul>`;
  } else if (!hasThresholds) {
    predictedHtml = `<p class="px-note">The backend did not send thresholds, so no labels are marked as predicted.</p>`;
  }

  const countWarning = result.predictions.length !== LABEL_COUNTS.length
    ? `<p class="px-note px-note--warn">Expected ${LABEL_COUNTS.length} labels but received ${result.predictions.length}.</p>` : "";

  const sampleBanner = context.isSample
    ? `<div class="px-banner"><strong>Sample data.</strong> These numbers are placeholders for previewing the layout. No model produced them, and they say nothing about your image.</div>` : "";

  resultsBody.innerHTML = `
    ${sampleBanner}
    <dl class="px-meta">
      <div><dt>Model</dt><dd>${escapeHtml(context.modelName)}</dd></div>
      <div><dt>Configuration</dt><dd>${escapeHtml(context.configName)}</dd></div>
      <div><dt>Threshold Mode</dt><dd>${escapeHtml(formatThresholdMode(context.thresholdMode))}</dd></div>
      <div><dt>Image</dt><dd>${escapeHtml(context.imageName)}</dd></div>
    </dl>
    <h4>Predicted labels</h4>
    ${predictedHtml}
    ${countWarning}
    <div class="px-rows-head">
      <h4>All ${result.predictions.length} probabilities</h4>
      <label>Sort
        <select class="px-sort" id="sort-select">
          <option value="probability">Highest first</option>
          <option value="name">Label name (A to Z)</option>
        </select>
      </label>
    </div>
    <p class="px-legend">The bar shows the probability. The vertical line marks the threshold sent by the backend.</p>
    <ul class="px-rows" id="px-rows"></ul>
    <p class="card__note">Research use only. These outputs are not a medical diagnosis and must not guide clinical decisions.</p>`;

  document.getElementById("sort-select").value = resultSort;
  renderRows();
}

// Draws the 14 rows. Called again when the sort option changes.
function renderRows() {
  if (!lastResult) return;

  const items = [...lastResult.predictions];
  if (resultSort === "name") {
    items.sort((a, b) => a.label.localeCompare(b.label));
  } else {
    items.sort((a, b) => b.probability - a.probability);
  }

  document.getElementById("px-rows").innerHTML = items.map(item => {
    const thresholdText = item.threshold === null ? "No threshold" : "Threshold " + formatPercent(item.threshold);
    const marker = item.threshold === null ? "" : `<span class="px-row__threshold" style="left: ${item.threshold * 100}%"></span>`;
    return `
      <li class="px-row ${item.predicted ? "is-predicted" : ""}">
        <span class="px-row__label">${escapeHtml(item.label)}</span>
        <span class="px-row__track">
          <span class="px-row__fill" style="width: ${item.probability * 100}%"></span>${marker}
        </span>
        <span class="px-row__value">${formatPercent(item.probability)}</span>
        <span class="px-row__status">${item.predicted ? "Predicted" : "Not predicted"}<small>${thresholdText}</small></span>
      </li>`;
  }).join("");
}

// SAMPLE result for the layout preview. Alphabetical order, so the order says nothing about any image.
function showSamplePreview() {
  const labels = LABEL_COUNTS.map(item => item.label).sort();
  const predictions = labels.map((label, index) => ({
    label: label,
    probability: SAMPLE_PROBABILITIES[index],
    threshold: SAMPLE_THRESHOLD,
    predicted: SAMPLE_PROBABILITIES[index] >= SAMPLE_THRESHOLD
  }));}

  showResults({ predictions }, {
    isSample: true,
    modelName: "Sample (no model)",
    configName: "Not applicable",
    imageName: selectedImageFile ? selectedImageFile.name : "None",
    thresholdMode: "sample",
    colorVar: "--color-primary"
  });

/* ============ STARTING UP ============ */

// Called once from app.js
function initPrediction() {
  // Choosing a file with the browse dialog (the label around the hidden input opens it)
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
    fileInput.value = "";           // lets the same file be chosen again later
  });

  // Drag and drop
  ["dragenter", "dragover"].forEach(eventName => dropzone.addEventListener(eventName, event => {
    event.preventDefault();
    dropzone.classList.add("is-dragover");
  }));
  ["dragleave", "drop"].forEach(eventName => dropzone.addEventListener(eventName, event => {
    event.preventDefault();
    dropzone.classList.remove("is-dragover");
  }));
  dropzone.addEventListener("drop", event => {
    if (event.dataTransfer.files[0]) handleFile(event.dataTransfer.files[0]);
  });

  document.getElementById("replace-image").addEventListener("click", () => fileInput.click());
  document.getElementById("remove-image").addEventListener("click", removeImage);
  runButton.addEventListener("click", runPrediction);
  sampleButton.addEventListener("click", showSamplePreview);

  // The sort menu is redrawn with the results, so listen on the container that stays
  resultsBody.addEventListener("change", event => {
    if (event.target.id === "sort-select") {
      resultSort = event.target.value;
      renderRows();
    }
  });

  // Keep the model card up to date when the choice on the Model selection page changes
  document.addEventListener("modelselectionchange", () => {
    renderSelectedModel();
    updateRunButton();
  });

  renderSelectedModel();
  showIdleResults();
  updateRunButton();
}