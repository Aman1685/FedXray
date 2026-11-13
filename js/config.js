/* config.js: project settings and shared data.
   Change values here instead of hunting through other files. */

// Flask backend address.
const API_BASE_URL = "http://127.0.0.1:5000";

// Flask API endpoint paths.
const API_ENDPOINTS = {
  health: "/api/health",
  models: "/api/models",
  predict: "/api/predict",
  metrics: "/api/metrics"
};

// Enable communication with the Flask backend.
const ENABLE_BACKEND_CHECK = true;
// All sections of the app. The sidebar and Quick navigation are both built from this list.
// "icon" must be a name from icons.js.
const SECTIONS = [
  { id: "dashboard", title: "Dashboard", icon: "dashboard", description: "Project overview and model summary" },
  { id: "dataset", title: "Dataset", icon: "dataset", description: "NIH ChestX-ray14, its 14 labels and IID vs non-IID splits" },
  { id: "federated", title: "Federated learning", icon: "federated", description: "How clients, server and aggregation fit together" },
  { id: "models", title: "Model selection", icon: "models", description: "Choose a model and training configuration" },
  { id: "predict", title: "X-ray prediction", icon: "predict", description: "Upload a chest X-ray and view label probabilities" },
  { id: "compare", title: "Model comparison", icon: "compare", description: "Evaluation metrics for the three models" },
  { id: "about", title: "About project", icon: "about", description: "Objectives, technologies and team" }
];

// Published NIH ChestX-ray14 figures. Check them against the split your team actually used.
const DATASET_FACTS = [
  { value: 112120, label: "Chest X-ray images", caption: "NIH ChestX-ray14", icon: "images" },
  { value: 30805, label: "Unique patients", caption: "Published figure", icon: "users" },
  { value: 14, label: "Disease labels", caption: "Multi-label task", icon: "tag" },
  { value: 3, label: "Model approaches", caption: "Centralized, FedAvg, FedProx", icon: "layers" }
];

// Descriptions of the three approaches. No performance numbers belong here.
const MODELS = [
  {
    id: "centralized",
    name: "Centralized CNN",
    kind: "Baseline",
    colorVar: "--color-centralized",
    summary: "A single CNN trained on all the data pooled in one place. It shows what is possible when privacy is not a constraint.",
    training: "All data on one machine"
  },
  {
    id: "fedavg",
    name: "FedAvg",
    kind: "Federated",
    colorVar: "--color-fedavg",
    summary: "Each client trains on its own data and sends only model weights. The server averages them into a new global model every round.",
    training: "Local training, averaging on the server"
  },
  {
    id: "fedprox",
    name: "FedProx",
    kind: "Federated",
    colorVar: "--color-fedprox",
    summary: "Works like FedAvg but adds a proximal term that keeps each client close to the global model. This helps when client data differs (non-IID).",
    training: "Local training with a proximal penalty"
  }
];

// Metrics shown on the comparison placeholder. Add or remove names to match your partner's evaluation.
const COMPARISON_METRICS = ["Accuracy", "Precision", "Recall", "F1-score"];

// The four steps of one federated learning round
const WORKFLOW_STEPS = [
  {
    title: "Share the global model",
    text: "The server distributes the current global model weights to the simulated clients."
  },
  {
    title: "Train locally",
    text: "Each simulated client trains the model using its assigned local dataset partition. FedProx adds a proximal term."
  },
  {
    title: "Send updates",
    text: "Clients return model updates to the server instead of combining their training data into one dataset."
  },
  {
    title: "Aggregate",
    text: "The server combines client updates using weighted averaging to create the next global model."
  }
];

// Project activity. Written by hand: this is NOT read from the backend.
// state is "done", "next" or "planned".
const ACTIVITY = [
  { title: "Dashboard and navigation built", detail: "Sidebar, header and dashboard layout.", state: "done" },
  { title: "Dataset section added", detail: "Labels, distribution chart, IID and non-IID explanation.", state: "done" },
  { title: "Light and dark theme added", detail: "Your choice is saved in the browser.", state: "done" },
  { title: "Federated Learning section added", detail: "Interactive round, aggregation demo, FedAvg vs FedProx.", state: "done" },
    { title: "Model Selection section added", detail: "Choose a model and training configuration.", state: "done" },
    { title: "X-ray Prediction section added", detail: "Upload, preview, model display and probability results.", state: "done" },
    { title: "Model Comparison section added", detail: "Metric cards, ranked chart and table (real numbers only).", state: "done" },
  { title: "About Project section", detail: "Objectives, technologies and team.", state: "done" },
{ title: "Flask backend integration", detail: "Model catalogue and X-ray prediction connected.", state: "done" },
{ title: "Official model evaluation", detail: "Requires the official test dataset and evaluation results.", state: "planned" }
];