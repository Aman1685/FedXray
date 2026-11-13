/* federated.js: data and behaviour for the Federated Learning section. */

/* ============ DATA (edit here) ============ */

// The four steps of one round. "points" are the bullet points shown for each step.
const FL_STEPS = [
  {
    title: "Send the global model",
    summary: "The server sends the current global model to every client.",
    points: [
      "In the first round the model starts from initial weights.",
      "Every client receives exactly the same weights."
    ]
  },
  {
    title: "Train on local data",
    summary: "Each client trains its own copy of the model on its own X-rays for a few local epochs.",
    points: [
      "Images and labels never leave the client.",
      "FedProx adds a proximal penalty to the local loss at this step.",
      "Clients can hold different label mixes (non-IID data)."
    ]
  },
  {
    title: "Return the updates",
    summary: "Each client sends its updated weights back to the server.",
    points: [
      "Only model weights travel, along with how many samples the client used.",
      "Raw X-rays are never transmitted."
    ]
  },
  {
    title: "Aggregate",
    summary: "The server combines the client models into a new global model.",
    points: [
      "FedAvg and FedProx both use a weighted average.",
      "Clients with more samples get more weight.",
      "The new global model starts the next round."
    ]
  }
];

// EXAMPLE sample counts for the aggregation demo. These are NOT project data.
const AGGREGATION_EXAMPLE = [
  { name: "Client A", samples: 600, colorVar: "--mix-1" },
  { name: "Client B", samples: 300, colorVar: "--mix-2" },
  { name: "Client C", samples: 100, colorVar: "--mix-3" }
];

// Your real training settings. Leave value as null until your partner confirms it,
// then replace null with text, for example: value: "5 clients"

const FL_PROJECT_SETTINGS = [
  { label: "Number of clients", value: "5" },
  { label: "Communication rounds", value: "5" },
  { label: "Local epochs per round", value: "1" },
  { label: "FedProx strength (μ)", value: "0.01" },
  { label: "Data split between clients", value: "IID, Non-IID, Balanced Non-IID" }
];


/* ============ BEHAVIOUR ============ */

let currentFlStep = 1;

// Show one step: update the diagram, the buttons and the explanation text
function showFlStep(stepNumber) {
  currentFlStep = stepNumber;
  document.getElementById("fl-diagram").dataset.step = stepNumber;   // CSS highlights the right part

  document.querySelectorAll(".fl-step").forEach(button => {
    const isActive = Number(button.dataset.step) === stepNumber;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  });

  const step = FL_STEPS[stepNumber - 1];
  document.getElementById("fl-explain").innerHTML = `
    <h4>${stepNumber}. ${step.title}</h4>
    <p>${step.summary}</p>
    <ul class="fl-list">${step.points.map(point => `<li>${point}</li>`).join("")}</ul>`;
}

function renderRoundExplorer() {
  const stepList = document.getElementById("fl-steps");

  stepList.innerHTML = FL_STEPS.map((step, index) => `
    <li>
      <button class="fl-step" data-step="${index + 1}" aria-pressed="false">
        <span class="fl-step__num">${index + 1}</span><span>${step.title}</span>
      </button>
    </li>`).join("");

  // One click listener on the list handles all four buttons
  stepList.addEventListener("click", event => {
    const button = event.target.closest(".fl-step");
    if (button) showFlStep(Number(button.dataset.step));
  });

  // "Next step" cycles 1, 2, 3, 4 and back to 1
  document.getElementById("fl-next").addEventListener("click", () => {
    showFlStep((currentFlStep % FL_STEPS.length) + 1);
  });

  showFlStep(1);
}

// Sliders change each client's sample count; the bars show its share of the average
function renderAggregationDemo() {
  const sliders = document.getElementById("fl-sliders");
  const shares = document.getElementById("fl-shares");

  sliders.innerHTML = AGGREGATION_EXAMPLE.map((client, index) => `
    <label class="fl-slider">
      <span>${client.name}</span>
      <input type="range" min="50" max="1000" step="50" value="${client.samples}" data-index="${index}">
      <output>${client.samples}</output>
    </label>`).join("");

  shares.innerHTML = AGGREGATION_EXAMPLE.map(client => `
    <div class="fl-share">
      <span>${client.name}</span>
      <span class="fl-share__track"><span class="fl-share__fill" style="background: var(${client.colorVar})"></span></span>
      <span class="fl-share__pct"></span>
    </div>`).join("");

  function updateShares() {
    const sampleCounts = Array.from(sliders.querySelectorAll("input")).map(input => Number(input.value));
    const totalSamples = sampleCounts.reduce((sum, count) => sum + count, 0);

    sliders.querySelectorAll("output").forEach((output, index) => {
      output.textContent = sampleCounts[index];
    });
    shares.querySelectorAll(".fl-share").forEach((row, index) => {
      const percent = (sampleCounts[index] / totalSamples) * 100;
      row.querySelector(".fl-share__fill").style.width = percent + "%";
      row.querySelector(".fl-share__pct").textContent = Math.round(percent) + "%";
    });
  }

  sliders.addEventListener("input", updateShares);   // runs every time a slider moves
  updateShares();
}

function renderProjectSettings() {
  document.getElementById("fl-settings").innerHTML = FL_PROJECT_SETTINGS.map(setting => `
    <div>
      <dt>${setting.label}</dt>
      <dd>${setting.value === null ? '<span class="badge badge--muted">Not set yet</span>' : setting.value}</dd>
    </div>`).join("");
}

function renderFederated() {
  renderRoundExplorer();
  renderAggregationDemo();
  renderProjectSettings();
}