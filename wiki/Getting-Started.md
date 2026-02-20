# 🚀 Getting Started

[← Back to Home](Home.md)

This guide walks you through setting up and running the Crop Disease Classification project from scratch.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.11 |
| RAM | 8 GB | 16 GB |
| Disk Space | 2 GB (code + dataset) | 5 GB (with models) |
| GPU | Not required | NVIDIA CUDA GPU (10x faster training) |
| OS | macOS, Linux, Windows | Any |

---

## Step 1: Clone / Download the Project

If you received this as a ZIP file, extract it. Otherwise:

```bash
cd /path/to/your/workspace
# The project folder is: crop-disease/
```

---

## Step 2: Download the Dataset

1. Go to [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
2. Download and extract the dataset
3. Place it **alongside** the project folder:

```
your-workspace/
├── PlantVillage Dataset/
│   └── PlantVillage/              ← This folder has the class subfolders
│       ├── Tomato_Bacterial_spot/
│       ├── Tomato_Early_blight/
│       ├── Potato___Early_blight/
│       ├── Pepper__bell___healthy/
│       └── ... (other class folders)
└── crop-disease/         ← The project
```

> **Note:** The notebook auto-detects the dataset path. If your dataset is in a different location, update `DATASET_ROOT` in the first few cells of the notebook.

---

## Step 3: Create a Virtual Environment

```bash
cd crop-disease

# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (Command Prompt)
# .venv\Scripts\Activate.ps1      # Windows (PowerShell)
```

**Why a virtual environment?** It keeps project dependencies isolated from your system Python, preventing version conflicts with other projects.

---

## Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all required packages:

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Deep learning framework (model training & inference) |
| `torchmetrics` | Efficient metric computation (accuracy, F1) |
| `numpy`, `pandas` | Numerical computing & data manipulation |
| `matplotlib`, `seaborn` | Static plots and charts |
| `Pillow` | Image loading and processing |
| `scikit-learn` | Stratified splitting, classification report |
| `tqdm` | Progress bars during training |
| `streamlit`, `plotly` | Interactive web app and charts |
| `jupyter`, `ipykernel` | Jupyter notebook support |

### Platform-Specific Notes

**macOS (Apple Silicon M1/M2/M3):**
- PyTorch uses MPS (Metal Performance Shaders) for GPU acceleration
- AMP (mixed precision) is automatically disabled on MPS
- Training is ~3-5x faster than CPU

**Linux/Windows with NVIDIA GPU:**
- Ensure CUDA toolkit is installed
- Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Full AMP support for fastest training

**CPU Only:**
- Everything works, just slower (~3-4 hours for full training)
- Reduce `batch_size` to 16 if RAM is limited

---

## Step 5: Run the Jupyter Notebook

```bash
# From the project root
jupyter notebook notebooks/crop_disease_classification.ipynb
```

Or if you prefer JupyterLab:
```bash
jupyter lab notebooks/crop_disease_classification.ipynb
```

Or in **VS Code**:
1. Open the project folder in VS Code
2. Open `notebooks/crop_disease_classification.ipynb`
3. Select the `.venv` Python kernel
4. Run cells sequentially with `Shift+Enter`

### Notebook Execution Order

The notebook has **8 sections** that must be run **in order**:

| Section | What It Does | Approx. Time |
|---------|-------------|--------------|
| §0 — Setup | Imports, seed, device detection | < 1 min |
| §1 — Data Exploration | Load dataset, visualize, analyze stats | 2–5 min |
| §2 — Data Pipeline | Split data, show augmentations, create loaders | 1–2 min |
| §3 — Model Training | Train 3 models through 3 stages each | 1–3 hours (GPU) / 3–6 hours (CPU) |
| §4 — Evaluation | Confusion matrices, predictions, error analysis | 5–10 min |
| §5 — Model Comparison | Profile models, comparison table and charts | 5–10 min |
| §6 — Business Recommendation | Markdown analysis (no code execution) | — |
| §7 — Export | Save checkpoints, class mapping, figures | < 1 min |

> **Tip:** If training is too slow, you can train only one model (e.g., EfficientNet-B0) by modifying the `model_names` list in §3 to `['efficientnet_b0']`.

---

## Step 6: Run the Streamlit App (Optional)

After training (so that model checkpoints exist in `models/`):

```bash
cd crop-disease
streamlit run app/streamlit_app.py
```

This opens a web browser with the disease detection app:
1. Upload a leaf photo
2. Get disease prediction with confidence score
3. See treatment recommendations

> **Note:** If no trained checkpoint exists, the app loads an untrained model for demo purposes and displays a warning.

---

## Step 7: Run the REST API (Optional)

After training (so that model checkpoints exist in `models/`):

```bash
cd crop-disease

# Development (auto-reload)
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

# Or with Docker
docker-compose up --build
```

- **Swagger UI docs:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health
- **Test prediction:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@leaf_photo.jpg" | python -m json.tool
```

> **Architecture:** The API uses SOLID principles — see [Deployment Guide](Deployment-Guide.md) for full details.

---

## Step 8: Verify Everything Worked

After running the full notebook, two runtime directories are created with the following outputs:

```
crop-disease/
├── models/                             ← created at runtime
│   ├── resnet50_best.pth               ← ~98 MB
│   ├── efficientnet_b0_best.pth        ← ~20 MB
│   ├── mobilenetv3_best.pth            ← ~10 MB
│   ├── class_mapping.json              ← Class index ↔ name mapping
│   └── training_config.json            ← Full config + results
└── outputs/                            ← created at runtime
    ├── sample_images_grid.png
    ├── class_distribution.png
    ├── training_curves.png
    ├── confusion_matrix_resnet50.png
    ├── confusion_matrix_efficientnet_b0.png
    ├── confusion_matrix_mobilenetv3.png
    ├── correct_predictions.png
    ├── incorrect_predictions.png
    └── model_comparison.png
```

---

## Folder Structure Explained

```
crop-disease/
│
├── src/                              # Source code (modular, reusable)
│   ├── config.py                     # ALL hyperparameters live here
│   ├── data/
│   │   ├── dataset.py                # Custom PyTorch Dataset class
│   │   ├── transforms.py             # Image augmentation pipelines
│   │   ├── splitter.py               # Train/val/test stratified splitting
│   │   └── loader.py                 # DataLoader creation
│   ├── models/
│   │   ├── factory.py                # Model creation & param utilities
│   │   └── freeze.py                 # Layer freezing / unfreezing
│   ├── training/
│   │   ├── trainer.py                # Training loop & checkpointing
│   │   ├── scheduler.py              # LR scheduler factory
│   │   └── early_stopping.py         # EarlyStopping callback
│   ├── evaluation/
│   │   ├── metrics.py                # Predictions, classification report
│   │   ├── confusion.py              # Confusion matrix heatmap
│   │   ├── predictions.py            # Correct/incorrect prediction grids
│   │   └── profiler.py               # Model latency & size measurement
│   └── utils/
│       ├── seed.py                   # Reproducibility (random seed management)
│       ├── text_helpers.py           # Class-name shortening & crop extraction
│       ├── plot_data.py              # Data exploration plots
│       ├── plot_training.py          # Training & comparison plots
│       └── export.py                 # ONNX model export
│
├── notebooks/
│   └── crop_disease_classification.ipynb   # Main notebook — runs everything
│
├── app/                              # Streamlit web application (5 modules)
│   ├── streamlit_app.py              # Entry point
│   ├── config.py                     # App constants & thresholds
│   ├── disease_info.py               # Disease database & helpers
│   ├── model_service.py              # Checkpoint loading & prediction
│   └── ui_components.py              # Sidebar, results, chart widgets
│
├── api/                              # FastAPI REST API (SOLID architecture)
│   ├── app.py                        # Application factory
│   ├── schemas.py                    # Pydantic request/response models
│   ├── protocols.py                  # Abstract interfaces (DIP)
│   ├── dependencies.py               # FastAPI dependency injection
│   ├── middleware.py                 # CORS, request logging
│   ├── routes/                       # Endpoint handlers
│   │   ├── health.py                 # GET /health, GET /model/version
│   │   └── predict.py                # POST /predict
│   └── services/                     # Business logic
│       ├── inference_service.py      # PyTorch model inference
│       └── disease_service.py        # Disease info enrichment
├── scripts/                          # CLI utilities
│   └── export_model.py              # PyTorch → ONNX → TFLite pipeline
├── tests/                            # Unit tests (78 passing, 1 skipped)
├── wiki/                             # Documentation (you're reading it!)
├── pyproject.toml                    # Package config & tool settings
├── requirements.txt                  # Python package list
├── requirements-api.txt              # Lean API-only dependencies
├── Dockerfile                        # Multi-stage API container
├── Dockerfile.hf                     # Hugging Face Spaces container
├── docker-compose.yml                # Local development setup
├── render.yaml                       # Render IaC Blueprint
├── .streamlit/config.toml            # Streamlit headless config & theme
├── .dockerignore                     # Docker build exclusions
├── .gitignore                        # Files excluded from version control
├── DEPLOYMENT.md                     # End-to-end cloud deployment guide
└── README.md                         # Project summary
```

> **Note:** `models/` and `outputs/` directories are created at runtime during training.

---

## Next Steps

| What | Where |
|------|-------|
| Understand the project structure | [Architecture Overview](Architecture-Overview.md) |
| Walk through every requirement | [Task Walkthrough](Task-Walkthrough.md) |
| Learn how data flows through the system | [Data Pipeline](Data-Pipeline.md) |
| Understand the training strategy | [Model Training](Model-Training.md) |
| Interpret evaluation results | [Evaluation & Metrics](Evaluation-and-Metrics.md) |
| Deploy to production (API or mobile) | [Deployment Guide](Deployment-Guide.md) |
| Deploy for free on the cloud | [Cloud Deployment](Cloud-Deployment.md) |
| Step-by-step sharing plan | [Sharing Plan](Sharing-Plan.md) |
| Fix common errors | [FAQ & Troubleshooting](FAQ-and-Troubleshooting.md) |
