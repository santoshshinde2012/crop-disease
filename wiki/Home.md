# 🌿 Crop Disease Classification — Project Wiki

Welcome to the **Crop Disease Classification** project wiki. This documentation is designed to help beginners understand the application, the technical approach, and how to get everything running.

---

## 📚 Wiki Pages

| Page | Description |
|------|-------------|
| [Getting Started](Getting-Started.md) | Environment setup, installation, running the notebook and app |
| [Architecture Overview](Architecture-Overview.md) | Project structure, module responsibilities, data flow diagram |
| [Task Walkthrough](Task-Walkthrough.md) | End-to-end guide — every requirement mapped to code, plus Deployment & MLOps |
| [Data Pipeline](Data-Pipeline.md) | Dataset loading, augmentation, splitting, and DataLoaders explained |
| [Model Training](Model-Training.md) | Three-stage fine-tuning, freezing strategy, regularization |
| [Evaluation & Metrics](Evaluation-and-Metrics.md) | Confusion matrix, predictions, profiling, business recommendation |
| [Streamlit App](Streamlit-App.md) | How the demo web app works, UI layout, confidence thresholding |
| [Deployment Guide](Deployment-Guide.md) | Online API & offline mobile deployment — two approaches |
| [Cloud Deployment](Cloud-Deployment.md) | Free hosting on Streamlit Cloud, Hugging Face Spaces, Render |
| [Sharing Plan](Sharing-Plan.md) | Step-by-step plan to share this project for evaluation |
| [FAQ & Troubleshooting](FAQ-and-Troubleshooting.md) | Common errors, performance tips, platform-specific notes |

---

## 🎯 What This Project Does

This project solves a **real-world agricultural problem**: identifying crop diseases from photos of plant leaves.

**In simple terms:**
1. A farmer takes a photo of a plant leaf with their phone
2. The app analyzes the photo using a trained deep learning model
3. It tells the farmer what disease (if any) the plant has
4. It recommends a specific treatment and product

**Technical summary:**
- Uses **transfer learning** with three pretrained CNN architectures (ResNet-50, EfficientNet-B0, MobileNetV3-Small)
- Trains on **12 disease classes** across Tomato, Potato, and Pepper crops from the PlantVillage dataset
- Employs a **three-stage progressive fine-tuning** strategy for optimal accuracy
- Includes a **Streamlit web app** for interactive disease prediction
- Supports **online deployment** (FastAPI REST API + Docker) and **offline deployment** (React Native + TFLite)
- Covers **MLOps best practices** — CI/CD, experiment tracking, model registry, monitoring, and automated retraining

---

## 🏗️ Quick Project Overview

```
crop-disease/
├── notebooks/                  ← Jupyter notebook (main deliverable)
│   └── crop_disease_classification.ipynb
├── src/                        ← Modular Python source code
│   ├── config.py               ← All hyperparameters in one place
│   ├── data/                   ← Dataset, transforms, splitting, loading
│   ├── models/                 ← Model creation & layer freezing
│   │   ├── factory.py          ← Architecture registry + custom heads
│   │   └── freeze.py           ← Backbone freeze / partial-unfreeze / full-unfreeze
│   ├── training/               ← Training engine
│   │   ├── trainer.py          ← Training loop + checkpointing
│   │   ├── scheduler.py        ← LR scheduler factory
│   │   └── early_stopping.py   ← EarlyStopping callback
│   ├── evaluation/             ← Metrics, confusion matrix, profiling
│   └── utils/                  ← Seed, plots, text helpers, ONNX export
│       ├── text_helpers.py     ← Class-name shortening & crop extraction
│       ├── plot_data.py        ← Sample images, class distribution, augmentation plots
│       └── plot_training.py    ← Training curves & model comparison charts
├── app/                        ← Streamlit web application (5 modules)
│   ├── streamlit_app.py        ← Entry point
│   ├── config.py               ← App constants & thresholds
│   ├── disease_info.py         ← Disease database & helpers
│   ├── model_service.py        ← Checkpoint loading & prediction logic
│   └── ui_components.py        ← Sidebar, results, chart widgets
├── api/                        ← FastAPI REST API (SOLID architecture)
│   ├── app.py                  ← Application factory
│   ├── schemas.py              ← Pydantic request/response models
│   ├── protocols.py            ← Abstract interfaces (DIP)
│   ├── dependencies.py         ← FastAPI dependency injection
│   ├── middleware.py            ← CORS, request logging
│   ├── routes/                 ← Endpoint handlers (ISP)
│   │   ├── health.py           ← GET /health, GET /model/version
│   │   └── predict.py          ← POST /predict
│   └── services/               ← Business logic (SRP, OCP)
│       ├── inference_service.py← PyTorch model loading & prediction
│       └── disease_service.py  ← Disease info enrichment
├── scripts/                    ← CLI utilities
│   └── export_model.py         ← PyTorch → ONNX → TFLite pipeline
├── tests/                      ← Unit tests (pytest — 78 passing)
├── wiki/                       ← This documentation (12 pages)
├── Dockerfile                  ← Multi-stage API container (Render)
├── Dockerfile.hf               ← HF Spaces Streamlit container
├── docker-compose.yml          ← Local development setup
├── render.yaml                 ← Render IaC Blueprint
├── .streamlit/config.toml      ← Streamlit headless config & theme
├── .dockerignore               ← Docker build exclusions
├── pyproject.toml              ← Package config & tool settings
├── requirements.txt            ← Full Python dependencies
├── requirements-api.txt        ← Lean API-only dependencies
├── DEPLOYMENT.md               ← End-to-end cloud deployment guide
└── README.md                   ← Project summary
```

> **Note:** `models/` and `outputs/` directories are created at runtime during training.

---

## 🔑 Key Concepts for Beginners

| Concept | What It Means |
|---------|---------------|
| **Transfer Learning** | Instead of training a model from scratch, we start with a model that already learned to recognize patterns from millions of images (ImageNet). We then fine-tune it on our specific task (plant diseases). |
| **Fine-Tuning** | Adjusting a pretrained model's weights for a new task. We do this progressively in 3 stages to avoid destroying the useful patterns the model already knows. |
| **Data Augmentation** | Artificially creating variations of training images (flipping, rotating, changing brightness) so the model learns to be robust to different conditions. |
| **Confusion Matrix** | A table showing which classes the model confuses with each other. Helps identify where the model struggles. |
| **F1 Score** | A metric that balances precision (how many predictions are correct) and recall (how many actual cases are found). Better than accuracy for imbalanced datasets. |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
cd crop-disease
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the notebook
jupyter notebook notebooks/crop_disease_classification.ipynb

# 3. (After training) Run the web app
streamlit run app/streamlit_app.py
```

For detailed instructions, see [Getting Started](Getting-Started.md).

---

## 📖 Reading Order

If you're new to the project, follow this path:

1. **[Getting Started](Getting-Started.md)** — set up your environment
2. **[Architecture Overview](Architecture-Overview.md)** — understand the project structure
3. **[Task Walkthrough](Task-Walkthrough.md)** — end-to-end guide through all 6 parts
4. **[Data Pipeline](Data-Pipeline.md)** → **[Model Training](Model-Training.md)** → **[Evaluation & Metrics](Evaluation-and-Metrics.md)** — deep dives
5. **[Streamlit App](Streamlit-App.md)** — interactive demo
6. **[Deployment Guide](Deployment-Guide.md)** — take the model to production
7. **[Cloud Deployment](Cloud-Deployment.md)** — deploy for free on Streamlit Cloud / Hugging Face
8. **[Sharing Plan](Sharing-Plan.md)** — checklist for sharing this project
9. **[FAQ & Troubleshooting](FAQ-and-Troubleshooting.md)** — if anything goes wrong
