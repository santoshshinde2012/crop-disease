# 🖥️ Streamlit App

[← Back to Home](Home.md)

This page explains how the web-based crop disease detection app works, how it loads models, and how the user interface is structured.

---

## What Is Streamlit?

[Streamlit](https://streamlit.io/) is a Python framework for building data-science web apps with minimal code. Instead of writing HTML/CSS/JavaScript, you write Python and Streamlit handles the UI.

```python
import streamlit as st

st.title("Hello World")
uploaded = st.file_uploader("Upload image")
if uploaded:
    st.image(uploaded)
```

This alone creates a web page with a title and file uploader. Our app builds on this with model inference and disease information.

---

## Running the App

```bash
cd crop-disease
streamlit run app/streamlit_app.py
```

This opens `http://localhost:8501` in your browser.

> **Prerequisite:** You need a trained model checkpoint in the `models/` directory (created at runtime by training). If none exists, the app loads an untrained model and shows a warning.

---

## App Architecture

The Streamlit app is decomposed into **5 focused modules** under the `app/` directory:

| Module | Responsibility |
|--------|---------------|
| `streamlit_app.py` | Entry point — orchestrates page layout, wires sidebar + main area |
| `config.py` | Constants: model directory, checkpoint names, confidence threshold (70%), UI colours, supported file extensions |
| `disease_info.py` | `DISEASE_INFO` database (12 entries) + `DEFAULT_INFO` fallback + lookup helpers |
| `model_service.py` | `load_model()` with `@st.cache_resource`, `Prediction` dataclass, `predict()` function |
| `ui_components.py` | `render_sidebar()`, `render_results()`, `render_confidence_chart()` widgets |

```
┌─────────────────────────────────────────────────────┐
│                  app/ (5 modules)                      │
│                                                        │
│  ┌──────────────────────────────────────────────┐  │
│  │    streamlit_app.py (Entry Point)                 │  │
│  │    Uses: config, disease_info,                    │  │
│  │          model_service, ui_components              │  │
│  └──────────────────────────────────────────────┘  │
│         │            │             │                    │
│         ▼            ▼             ▼                    │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐     │
│  │ config   │  │ disease_  │  │model_service │     │
│  │   .py    │  │  info.py  │  │predict()     │     │
│  └──────────┘  └───────────┘  └──────────────┘     │
│                                                        │
│  ┌─────────────────────────┐                          │
│  │  ui_components.py        │                          │
│  │  render_sidebar()        │                          │
│  │  render_results()        │                          │
│  │  render_confidence_chart()│                          │
│  └─────────────────────────┘                          │
└─────────────────────────────────────────────────────┘
```

---

## Model Loading

### How the app loads the model

The `load_model()` function in `app/model_service.py` is decorated with `@st.cache_resource`, which means:
- The model is loaded **once** when the app starts
- Subsequent requests reuse the cached model (no reloading)
- This makes predictions fast (~50ms instead of ~5s)

### Model selection priority

The app tries to load checkpoints in this order:

| Priority | Model | File | Why |
|----------|-------|------|-----|
| 1st | EfficientNet-B0 | `efficientnet_b0_best.pth` | Best accuracy/size tradeoff (recommended) |
| 2nd | ResNet-50 | `resnet50_best.pth` | Highest accuracy |
| 3rd | MobileNetV3-Small | `mobilenetv3_best.pth` | Smallest model |
| Fallback | EfficientNet-B0 (untrained) | None | Uses pretrained ImageNet weights |

### Class mapping

The app loads `models/class_mapping.json` to translate prediction indices to class names:

```json
{
  "0": "Pepper__bell___Bacterial_spot",
  "1": "Pepper__bell___healthy",
  "2": "Potato___Early_blight",
  ...
  "11": "Tomato_healthy"
}
```

If this file doesn't exist, the app generates the mapping from `config.py`.

---

## Prediction Pipeline

When a user uploads an image, the following happens:

```
1. Image loaded as PIL.Image → .convert("RGB")
                │
                ▼
2. get_val_transforms(224) applied:
   Resize(256) → CenterCrop(224) → ToTensor → Normalize
                │
                ▼
3. Add batch dimension: (3, 224, 224) → (1, 3, 224, 224)
                │
                ▼
4. model(tensor) → logits (1, 12)
                │
                ▼
5. softmax(logits) → probabilities (1, 12)
                │
                ▼
6. topk(3) → top-3 class indices + probabilities
                │
                ▼
7. Map indices to class names via class_mapping
                │
                ▼
8. Return: [("Tomato_Early_blight", 0.923), ("Tomato_Late_blight", 0.045), ...]
```

**Important:** The same `get_val_transforms()` function is used as during evaluation. This ensures the model sees images exactly as it was tested on — same resize, crop, and normalization.

---

## Confidence Thresholding

### The 70% threshold

| Confidence | App Behavior |
|-----------|-------------|
| **≥ 70%** | Green success banner with disease name, treatment, and product recommendation |
| **< 70%** | Yellow warning: "Low confidence. Retake photo with better lighting." |

### Why 70%?

- The model's incorrect predictions typically have lower confidence (60-80%)
- A 70% threshold catches most unreliable predictions
- Below 70%, showing "Tomato Early Blight" when it might actually be Late Blight could lead to **wrong treatment** — dangerous in agriculture
- It's better to say "I'm not sure, please retake" than to give a wrong diagnosis

### Color-coded confidence chart

The top-3 predictions bar chart uses:
- **Teal (#4ECDC4)** for predictions ≥ 70% — confident
- **Red (#FF6B6B)** for predictions < 70% — uncertain

---

## Disease Information Database

The `DISEASE_INFO` dictionary (in `app/disease_info.py`) maps each of the 12 classes to actionable information:

```python
DISEASE_INFO = {
    "Tomato_Early_blight": {
        "crop": "Tomato",
        "disease": "Early Blight",
        "severity": "Moderate",
        "action": "Apply fungicide preventively and curatively. Mulch to prevent splash. Rotate crops.",
        "product": "Score® (Difenoconazole)",
    },
    # ... 11 more entries
}
```

### Fields explained

| Field | What It Shows | Example |
|-------|--------------|---------|
| `crop` | Plant type | "Tomato", "Potato", "Pepper (Bell)" |
| `disease` | Disease name (human-readable) | "Early Blight", "Healthy" |
| `severity` | How urgent is treatment? | "None", "Moderate", "High — Urgent" |
| `action` | What the farmer should do | "Remove infected leaves. Apply fungicide." |
| `product` | Product recommendation | "Score® (Difenoconazole)" |

### Healthy plants

For healthy plants, the app shows:
- Severity: None
- Action: "No treatment needed. Continue regular monitoring."
- Product: "N/A — Preventive: Revus® for protection"

---

## UI Layout

### Sidebar

| Element | Purpose |
|---------|---------|
| Title & description | Explains what the app does |
| Model info | Shows which model is loaded and how many classes |
| Supported crops | Lists Tomato, Potato, Pepper (Bell) |
| Disclaimer | "This tool is for guidance only. Consult a local agronomist." |

### Main Area — Before Upload

Shows a placeholder with tips for taking good leaf photos:
- Use clear, well-lit photos
- Focus on a single leaf
- Include both healthy and affected areas
- Avoid shadows and glare

### Main Area — After Upload

| Left Column | Right Column |
|------------|-------------|
| Uploaded image (full size) | Analysis results: prediction, disease info, treatment recommendation |

Below both columns: **Top-3 predictions chart** (horizontal bar chart with Plotly)

---

## Key Implementation Details

### Why Plotly instead of Matplotlib?

Plotly charts are **interactive** — users can hover over bars to see exact values. Matplotlib generates static images which are less engaging in a web app.

### Why `use_container_width=True`?

Both the image and chart use `use_container_width=True` so they automatically resize to fill the available space. This makes the app look good on different screen sizes.

### Error handling

| Scenario | Behavior |
|----------|---------|
| No checkpoint found | Loads untrained pretrained model, shows `st.warning()` |
| Invalid image format | Streamlit's file_uploader filters by extension |
| Image not RGB | `.convert("RGB")` handles grayscale, RGBA, palette images |
| Class not in DISEASE_INFO | Falls back to `DEFAULT_INFO` with "consult agronomist" |

---

## Customizing the App

### Streamlit configuration (`.streamlit/config.toml`)

The project includes `.streamlit/config.toml` with these settings:

| Setting | Value | Purpose |
|---------|-------|---------|
| `server.headless` | `true` | Required for cloud deployment (no browser prompt) |
| `server.maxUploadSize` | `10` | Max upload size in MB |
| `server.enableXsrfProtection` | `true` | Cross-site request forgery protection |
| `browser.gatherUsageStats` | `false` | Opt out of Streamlit telemetry |
| `theme.primaryColor` | `#2E7D32` | Green theme for agriculture |

This file is auto-detected by Streamlit Cloud and Hugging Face Spaces.

### Adding new diseases

1. Add the class to `config.data.selected_classes`
2. Retrain the model (notebook §3)
3. Add an entry to `DISEASE_INFO` in `app/disease_info.py`
4. The class mapping updates automatically

### Changing the confidence threshold

In `app/config.py`, update the `CONFIDENCE_THRESHOLD` constant (default `0.70`). Change to `0.80` for stricter, `0.60` for more lenient.

### Using a different model

Place your checkpoint as `models/your_model_best.pth` and update the `model_preferences` list in `load_model()`.

---

## Best Practices Applied

| Practice | How We Apply It | Why It Matters |
|----------|----------------|----------------|
| **Modular architecture** | 5 focused files instead of one monolith | Each module is testable and maintainable independently |
| **Cached model loading** | `@st.cache_resource` decorator | Model loaded once; subsequent requests are instant (~50 ms) |
| **Same transforms as evaluation** | Uses `get_val_transforms()` from training code | Predictions match what the model was tested on |
| **Confidence thresholding** | 70% cutoff with visual warning | Prevents misdiagnosis — better to say "unsure" than give wrong advice |
| **Graceful fallbacks** | Fallback model, `DEFAULT_INFO` for unknown classes | App works even without trained checkpoints or with unseen classes |
| **Responsive layout** | `use_container_width=True` for images and charts | Works on different screen sizes |
| **Interactive charts** | Plotly instead of Matplotlib | Users can hover for exact values; better web experience |
| **Medical disclaimer** | Sidebar disclaimer text | Protects against misuse; directs to professional agronomists |

---

## Next Steps

| What | Where |
|------|-------|
| Common Streamlit issues and fixes | [FAQ & Troubleshooting](FAQ-and-Troubleshooting.md) |
| Setup instructions for running the app | [Getting Started](Getting-Started.md) |
| Deploy as a production API or mobile app | [Deployment Guide](Deployment-Guide.md) |
| How the model was trained and evaluated | [Model Training](Model-Training.md) → [Evaluation & Metrics](Evaluation-and-Metrics.md) |
| See how this maps to assignment requirements | [Task Walkthrough — Part 4](Task-Walkthrough.md) |
