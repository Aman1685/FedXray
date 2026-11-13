/* app.js: the starting point. Runs last, after every other script has loaded. */

const statusPill = document.getElementById("backend-status");
const statusText = document.getElementById("backend-status-text");

// Update the pill in the header (text, tooltip and screen-reader label)
function setBackendStatus(state, message) {
  statusPill.dataset.state = state;
  statusText.textContent = message;
  statusPill.title = message + ". Click to check again.";
  statusPill.setAttribute("aria-label", statusPill.title);
}

// Ask the backend if it is running
async function refreshBackendStatus() {
  // Backend check switched off in config.js: show demo mode and make no request
  if (!ENABLE_BACKEND_CHECK) {
    setBackendStatus("demo", "Demo mode");
    statusPill.title = "Backend check is off. Set ENABLE_BACKEND_CHECK to true in config.js when the backend runs.";
    return;
  }

  setBackendStatus("checking", "Checking backend");
  const isOnline = await checkBackendStatus();
  setBackendStatus(isOnline ? "online" : "offline", isOnline ? "Backend connected" : "Backend offline");
}

statusPill.addEventListener("click", refreshBackendStatus);

initThemeToggle();
fillStaticIcons();
renderKpis();
renderModels();
renderLabelChips();
renderWorkflow();
renderComparisonPreview();
renderActivity();
renderQuickLinks();
renderDataset();
renderFederated();
initModelSelection();
initPrediction();
initComparison();
refreshBackendStatus();