# Crop Disease Classification

A deep learning solution for identifying plant diseases from leaf images, built to support digital agriculture and precision farming initiatives.

## Overview

This project classifies plant diseases across **12 classes** from 3 crops (Tomato, Potato, Pepper) using transfer learning with PyTorch. Three architectures are compared — **ResNet-50**, **EfficientNet-B0**, and **MobileNetV3-Small** — using a three-stage progressive fine-tuning strategy.

## Project Structure

```
crop-disease/
├── README.md
├── DEPLOYMENT.md                            # End-to-end cloud deployment guide
├── pyproject.toml                           # Package config & tool settings
├── requirements.txt                         # Pinned dependencies
├── requirements-api.txt                     # Lean API-only dependencies
├── .gitignore
├── Dockerfile                               # Multi-stage production container
├── Dockerfile.hf                            # Hugging Face Spaces container
├── docker-compose.yml                       # Local development setup
├── .dockerignore                            # Docker build exclusions
├── render.yaml                              # Render IaC blueprint
├── .streamlit/
│   └── config.toml                          # Streamlit headless config & theme
├── notebooks/
│   └── crop_disease_classification.ipynb    # Main notebook (all parts)
├── src/                                     # Core ML library
│   ├── __init__.py
│   ├── config.py                            # Central configuration
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                       # Custom PyTorch Dataset
│   │   ├── transforms.py                    # Augmentation pipelines
│   │   ├── splitter.py                      # Stratified train/val/test split
│   │   └── loader.py                        # DataLoader factory
│   ├── models/
│   │   ├── __init__.py
│   │   ├── factory.py                       # Model creation & param utilities
│   │   └── freeze.py                        # Layer freezing / unfreezing
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                       # Training engine
│   │   ├── scheduler.py                     # LR scheduler factory
│   │   └── early_stopping.py                # EarlyStopping callback
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                       # Classification metrics
│   │   ├── confusion.py                     # Confusion matrix
│   │   ├── predictions.py                   # Prediction visualization
│   │   └── profiler.py                      # Model profiling
│   └── utils/
│       ├── __init__.py
│       ├── seed.py                          # Reproducibility
│       ├── text_helpers.py                  # Class-name shortening utilities
│       ├── plot_data.py                     # Data exploration plots
│       ├── plot_training.py                 # Training & comparison plots
│       └── export.py                        # ONNX model export
├── app/                                     # Streamlit web application
│   ├── __init__.py
│   ├── streamlit_app.py                     # Streamlit entry point
│   ├── config.py                            # App constants & thresholds
│   ├── disease_info.py                      # Disease database & helpers
│   ├── model_service.py                     # Checkpoint loading & prediction
│   └── ui_components.py                     # Sidebar, results, chart widgets
├── api/                                     # FastAPI REST API (SOLID architecture)
│   ├── __init__.py                          # Package version
│   ├── app.py                               # Application factory
│   ├── schemas.py                           # Pydantic request/response models
│   ├── protocols.py                         # Abstract interfaces (DIP)
│   ├── dependencies.py                      # FastAPI dependency injection
│   ├── middleware.py                        # CORS, request logging
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py                        # GET /health, GET /model/version
│   │   └── predict.py                       # POST /predict
│   └── services/
│       ├── __init__.py
│       ├── inference_service.py             # PyTorch model inference
│       └── disease_service.py               # Disease info enrichment
├── scripts/                                 # CLI utilities
│   ├── __init__.py
│   └── export_model.py                      # PyTorch → ONNX → TFLite pipeline
├── tests/                                   # Unit tests (pytest — 78 passing)
│   ├── __init__.py
│   ├── test_api.py                          # API endpoint tests
│   ├── test_config.py
│   ├── test_dataset.py
│   ├── test_transforms.py
│   ├── test_splitter.py
│   ├── test_model_factory.py
│   ├── test_export.py
│   ├── test_early_stopping.py
│   ├── test_scheduler.py
│   └── test_text_helpers.py
└── wiki/                                    # Project documentation (12 pages)
    ├── Home.md
    ├── Getting-Started.md
    ├── Architecture-Overview.md
    ├── Data-Pipeline.md
    ├── Model-Training.md
    ├── Evaluation-and-Metrics.md
    ├── Streamlit-App.md
    ├── Deployment-Guide.md
    ├── Cloud-Deployment.md
    ├── Sharing-Plan.md
    ├── Task-Walkthrough.md
    └── FAQ-and-Troubleshooting.md
```

> **Note:** `models/` and `outputs/` directories are created at runtime by the training pipeline. They contain saved checkpoints (`.pth`) and generated figures (`.png`) respectively.

## Setup Instructions

### 1. Prerequisites

- Python 3.10+
- pip or conda package manager
- (Optional) CUDA-capable GPU for faster training

### 2. Install Dependencies

```bash
cd crop-disease

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install packages
pip install -r requirements.txt
```

### 3. Dataset Setup

Download the [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) from Kaggle and place it alongside the project:

```
assignment/
├── PlantVillage Dataset/
│   └── PlantVillage/          # ← Contains class folders
│       ├── Tomato_Bacterial_spot/
│       ├── Tomato_healthy/
│       └── ...
└── crop-disease/     # ← This project
```

Or update `DATASET_ROOT` in the notebook to point to your dataset location.

### 4. Run the Notebook

```bash
cd crop-disease
jupyter notebook notebooks/crop_disease_classification.ipynb
```

Execute all cells sequentially. Training takes approximately 3-4 hours on CPU, ~1 hour on GPU.

### 5. Run the Streamlit App

```bash
cd crop-disease
streamlit run app/streamlit_app.py
```

### 6. Run the REST API

```bash
cd crop-disease

# Development (auto-reload)
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

# Or with Docker
docker-compose up --build
```

API docs available at: http://localhost:8000/docs

### 7. Run Tests

```bash
python -m pytest tests/ -v
```

## Model Performance Summary

| Model | Accuracy | F1 (Macro) | F1 (Weighted) | Model Size | CPU Latency |
|-------|----------|------------|---------------|------------|-------------|
| ResNet-50 | ~95-97% | ~0.95-0.97 | ~0.95-0.97 | ~98 MB | ~80-120 ms |
| EfficientNet-B0 | ~94-96% | ~0.94-0.96 | ~0.94-0.96 | ~20 MB | ~30-50 ms |
| MobileNetV3-Small | ~91-94% | ~0.91-0.94 | ~0.91-0.94 | ~10 MB | ~15-25 ms |

*Note: Exact metrics depend on hardware and training run. Values shown are expected ranges based on the architecture and dataset.*

## Training Strategy

**Three-Stage Progressive Fine-Tuning:**

| Stage | Epochs | Learning Rate | Trainable Params | Objective |
|-------|--------|---------------|-------------------|-----------|
| 1 — Feature Extraction | 5 | 1×10⁻³ | Head only | Warm up classifier on frozen features |
| 2 — Adaptation | 10 | 1×10⁻⁴ | Top backbone + head | Adapt high-level features to leaf diseases |
| 3 — Full Refinement | 10 | 1×10⁻⁵ / 5×10⁻⁵ | All (differential LR) | End-to-end refinement |

**Regularization:** Weight decay (1e-4), label smoothing (0.1), dropout (0.3/0.15), gradient clipping (1.0), data augmentation, early stopping (patience=5).

## Business Recommendation

**Deploy EfficientNet-B0** for a mobile crop disease detection app. It achieves the optimal balance between accuracy (~95%), model size (~20 MB → ~5 MB after INT8 quantization), and inference speed (~30-50 ms on CPU). ResNet-50's marginal accuracy gain doesn't justify its 5× larger size, while MobileNetV3's 2-3% accuracy drop risks missing diseases — a costly false negative in agriculture. The deployment pipeline would be PyTorch → ONNX → TFLite with INT8 quantization, running entirely on-device for offline functionality in areas with limited connectivity.

## Selected Classes (12)

| Crop | Classes | Selection Rationale |
|------|---------|---------------------|
| Tomato (7) | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Target Spot, Septoria Leaf Spot, Healthy | Multiple diseases per crop — tests intra-crop discrimination |
| Potato (3) | Early Blight, Late Blight, Healthy | Shared disease names with Tomato — tests cross-crop confusion |
| Pepper (2) | Bacterial Spot, Healthy | Extends classification to another crop type |

## Tech Stack

- **Framework:** PyTorch 2.0+, torchvision, torchmetrics
- **Data Science:** pandas, NumPy, scikit-learn
- **Visualization:** matplotlib, seaborn, Plotly
- **App:** Streamlit
- **API:** FastAPI, uvicorn, Pydantic
- **Deployment:** Docker, Docker Compose, ONNX export
- **Cloud:** Streamlit Community Cloud, Hugging Face Spaces, Render
- **Testing:** pytest (78 tests)
- **Notebook:** Jupyter

## Cloud Deployment

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for end-to-end instructions to deploy on:
- **Streamlit Community Cloud** — interactive demo (~5 min setup)
- **Hugging Face Spaces** — Docker-based Streamlit app (~10 min setup)
- **Render** — FastAPI REST API (~10 min setup)

All three platforms are **free** and auto-deploy on `git push`.
