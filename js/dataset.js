/* dataset.js: data and drawing code for the Dataset section. */

/* ============ DATA (edit here) ============ */

// Rows of the "Dataset information" panel
const DATASET_INFO = [
  { term: "Source", detail: "NIH Clinical Center (Wang et al., CVPR 2017)" },
  { term: "Images", detail: "112,120 frontal-view chest X-rays" },
  { term: "Patients", detail: "30,805 unique patients" },
  { term: "Image format", detail: "PNG, 1024 × 1024 pixels" },
  { term: "Labels", detail: "14 thoracic disease labels, plus \"No Finding\"" },
  { term: "Label source", detail: "Text-mined from radiology reports" },
  { term: "Task", detail: "Multi-label classification" }
];

// Published image counts per label for the FULL dataset.
// Check them against the dataset's own label file (Data_Entry_2017.csv) and replace them
// with the counts from your project's training split once you know them.
const LABEL_COUNTS = [
  { label: "Infiltration", count: 19894 },
  { label: "Effusion", count: 13317 },
  { label: "Atelectasis", count: 11559 },
  { label: "Nodule", count: 6331 },
  { label: "Mass", count: 5782 },
  { label: "Pneumothorax", count: 5302 },
  { label: "Consolidation", count: 4667 },
  { label: "Pleural thickening", count: 3385 },
  { label: "Cardiomegaly", count: 2776 },
  { label: "Emphysema", count: 2516 },
  { label: "Edema", count: 2303 },
  { label: "Fibrosis", count: 1686 },
  { label: "Pneumonia", count: 1431 },
  { label: "Hernia", count: 227 }
];

// ILLUSTRATION ONLY: the share of four example labels on each client (each row adds up to 1)
const CLIENT_NAMES = ["Client A", "Client B", "Client C"];
const IID_MIXES = [[0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]];
const NON_IID_MIXES = [[0.7, 0.15, 0.1, 0.05], [0.1, 0.65, 0.15, 0.1], [0.15, 0.1, 0.2, 0.55]];

// Describe your real client split here once the team confirms it
const CLIENT_SPLIT_NOTE = "";

/* ============ DRAWING CODE ============ */

function renderDatasetInfo() {
  document.getElementById("dataset-info").innerHTML = DATASET_INFO
    .map(row => `<div><dt>${row.term}</dt><dd>${row.detail}</dd></div>`)
    .join("");
}

// One row per label. The bar width is the count as a percentage of the largest count.
function renderLabelChart() {
  const sortedLabels = [...LABEL_COUNTS].sort((a, b) => b.count - a.count);
  const largestCount = sortedLabels[0].count;

  document.getElementById("label-chart").innerHTML = sortedLabels.map(item => `
    <li>
      <span>${item.label}</span>
      <span class="bar-chart__track">
        <span class="bar-chart__fill" style="width: ${(item.count / largestCount) * 100}%"></span>
      </span>
      <span class="bar-chart__value">${item.count.toLocaleString("en-US")}</span>
    </li>
  `).join("");
}

// Stacked bar per client; each coloured segment is one example label
function renderMixChart(containerId, clientMixes, description) {
  const container = document.getElementById(containerId);
  container.setAttribute("role", "img");
  container.setAttribute("aria-label", description);

  container.innerHTML = clientMixes.map((mix, clientIndex) => {
    const segments = mix
      .map((share, labelIndex) => `<span style="width: ${share * 100}%; background: var(--mix-${labelIndex + 1})"></span>`)
      .join("");
    return `<div class="mix__row"><span>${CLIENT_NAMES[clientIndex]}</span><div class="mix__bar">${segments}</div></div>`;
  }).join("");
}

function renderClientSplitNote() {
  document.getElementById("client-split-note").textContent = CLIENT_SPLIT_NOTE ||
    "This project's own client split is not set yet. Add the client count and split method to CLIENT_SPLIT_NOTE in dataset.js once your team confirms it.";
}

function renderDataset() {
  renderDatasetInfo();
  renderLabelChart();
  renderMixChart("mix-iid", IID_MIXES, "Three clients with the same label mix");
  renderMixChart("mix-non-iid", NON_IID_MIXES, "Three clients with very different label mixes");
  renderClientSplitNote();
}