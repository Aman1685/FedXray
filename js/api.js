
/* api.js: All communication with the Flask backend lives here. */

// Escape backend text before inserting it into HTML.
function escapeHtml(text) {
  const replacements = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  };

  return String(text).replace(
    /[&<>"']/g,
    character => replacements[character]
  );
}


// Shared fetch helper with timeout.
async function fetchWithTimeout(path, options = {}, timeoutMs = 5000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(API_BASE_URL + path, {
      ...options,
      signal: controller.signal
    });
  } finally {
    clearTimeout(timeoutId);
  }
}


// Check backend status.
async function checkBackendStatus() {
  try {
    const response = await fetchWithTimeout(
      API_ENDPOINTS.health,
      {},
      3000
    );

    return response.ok;
  } catch (error) {
    return false;
  }
}


// Fetch available models and configurations.
async function fetchModelCatalogue() {
  const response = await fetchWithTimeout(API_ENDPOINTS.models);

  if (!response.ok) {
    throw new Error(
      "Model list request failed with status " + response.status
    );
  }

  return response.json();
}


// Send X-ray image for prediction.
//
// thresholdMode:
//   "default" = fixed threshold 0.5
//   "tuned"   = validation-optimized disease-specific thresholds
//
async function predictXray(
  imageFile,
  modelId,
  configurationId,
  thresholdMode = "default"
) {

  const formData = new FormData();

  // Image
  formData.append("image", imageFile);

  // Selected model
  formData.append("model_id", modelId);

  // Selected configuration
  formData.append("config_id", configurationId || "standard");

  // Selected threshold mode
  formData.append("threshold_mode", thresholdMode);

  let response;

  try {
    response = await fetchWithTimeout(
      API_ENDPOINTS.predict,
      {
        method: "POST",
        body: formData
      },
      60000
    );

  } catch (error) {

    throw new Error(
      error.name === "AbortError"
        ? "The backend took too long to answer."
        : "Could not reach the backend. Is the server running?"
    );
  }

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}


// Read backend error messages.
async function readErrorMessage(response) {
  try {
    const body = await response.json();

    if (typeof body.detail === "string") {
      return body.detail;
    }

    if (typeof body.message === "string") {
      return body.message;
    }

  } catch (error) {
    // Fall through to generic error.
  }

  return "The server answered with status " + response.status + ".";
}


// Fetch evaluation metrics.
async function fetchMetrics() {
  const response = await fetchWithTimeout(API_ENDPOINTS.metrics);

  if (!response.ok) {
    throw new Error(
      "Metrics request failed with status " + response.status
    );
  }

  return response.json();
}
