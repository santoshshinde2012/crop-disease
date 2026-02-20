# 📋 Sharing Plan — Prepare & Share for Evaluation

[← Back to Home](Home.md)

This page provides a **step-by-step plan** to prepare and share the Crop Disease Classification project for evaluation (e.g., portfolio sharing, or team review).

---

## Overview

| Phase | What | Time Estimate |
|-------|------|---------------|
| **Phase 1** | Pre-flight checks (local) | ~15 min |
| **Phase 2** | Push to GitHub | ~10 min |
| **Phase 3** | Deploy live demo | ~15 min |
| **Phase 4** | Prepare submission package | ~10 min |
| **Total** | | **~50 min** |

---

## Phase 1 — Pre-Flight Checks (Local)

Run these checks locally before sharing to ensure everything works end to end.

### 1.1 — Verify Tests Pass

```bash
cd crop-disease
source .venv/bin/activate
python -m pytest tests/ -v --tb=short
```

**Expected:** `78 passed, 1 skipped` (ONNX test skipped if `onnx` not installed).

### 1.2 — Verify Notebook Runs

Open and run the notebook **top to bottom** in a fresh kernel:

```bash
jupyter notebook notebooks/crop_disease_classification.ipynb
```

- `Kernel → Restart & Run All`
- All cells should execute without errors
- Training outputs (checkpoints, figures) should appear in `models/` and `outputs/`

> **Tip:** If time is limited, you can skip retraining — just verify the notebook cells up to §2 (Data Pipeline) execute cleanly, then jump to evaluation cells that use existing checkpoints.

### 1.3 — Verify Streamlit App

```bash
streamlit run app/streamlit_app.py
```

- Upload a test leaf image
- Confirm prediction + confidence + treatment recommendation appear
- Check sidebar shows model name and class count

### 1.4 — Verify REST API

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

- Open http://localhost:8000/docs (Swagger UI)
- Test `GET /health` → should return `{"status": "healthy"}`
- Test `POST /predict` with a leaf image → should return predictions

### 1.5 — Verify Docker Build

```bash
docker-compose up --build
```

- API should be accessible at http://localhost:8000/docs
- `GET /health` should return healthy

### 1.6 — Check Generated Outputs

Verify these files exist after a full notebook run:

```
models/
├── efficientnet_b0_best.pth       ✓
├── resnet50_best.pth              ✓
├── mobilenetv3_best.pth           ✓
├── class_mapping.json             ✓
└── training_config.json           ✓

outputs/
├── sample_images_grid.png         ✓
├── class_distribution.png         ✓
├── training_curves.png            ✓
├── confusion_matrix_*.png         ✓ (3 files)
├── correct_predictions.png        ✓
├── incorrect_predictions.png      ✓
└── model_comparison.png           ✓
```

---

## Phase 2 — Push to GitHub

### 2.1 — Create a GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `crop-disease`
3. Visibility: **Public** (required for free Streamlit Cloud deployment)
4. Click **"Create repository"**

### 2.2 — Initialize and Push

```bash
cd crop-disease

# Initialize git
git init
git branch -M main

# Configure Git LFS for model checkpoints
git lfs install
git lfs track "models/*.pth"

# Add files
git add .gitattributes
git add .

# Verify .gitignore is working (these should NOT be staged)
git status | grep -E "\.venv|__pycache__|outputs/"
# Should show nothing — these are properly ignored

# Commit and push
git commit -m "Crop Disease Classification - Complete submission"
git remote add origin https://github.com/<your-username>/crop-disease.git
git push -u origin main
```

### 2.3 — Verify GitHub Repository

Check the repository page on GitHub:
- [ ] README renders correctly with project structure, setup instructions
- [ ] All directories visible: `src/`, `app/`, `api/`, `tests/`, `wiki/`, `scripts/`, `notebooks/`
- [ ] Wiki pages visible in `wiki/` folder
- [ ] Model checkpoint tracked via Git LFS (click on file — should show "Stored with Git LFS")
- [ ] `class_mapping.json` present in `models/`

---

## Phase 3 — Deploy Live Demo

Deploy to **Streamlit Community Cloud** so evaluators can test without local setup.

### 3.1 — Deploy Streamlit App

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"Create app"**
4. Select:
   - Repository: `<your-username>/crop-disease`
   - Branch: `main`
   - Main file: `app/streamlit_app.py`
5. Click **"Deploy!"**

**Result:** Live app at `https://<your-username>-crop-disease-appstreamlit-app-<hash>.streamlit.app`

### 3.2 — (Optional) Deploy API on Render

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **New → Web Service → Connect GitHub repo**
3. Select `crop-disease`, branch `main`
4. Runtime: Docker, Instance: Free
5. Deploy

**Result:** Live API at `https://crop-disease-api.onrender.com/docs`

### 3.3 — Verify Live Deployments

- [ ] Streamlit app loads and responds to image uploads
- [ ] Model predictions are accurate (test with a known leaf image)
- [ ] (If deployed) API returns predictions via `/predict` endpoint

> See [Cloud Deployment](Cloud-Deployment.md) for detailed deployment instructions and troubleshooting.

---

## Phase 4 — Prepare Submission Package

### 4.1 — What to Share

| Item | Format | Link/Location |
|------|--------|---------------|
| **GitHub Repository** | URL | `https://github.com/<user>/crop-disease` |
| **Live Demo** | URL | Streamlit Community Cloud link |
| **Live API** (optional) | URL | Render link |
| **README** | In repo | Project overview + setup instructions |
| **Wiki Documentation** | In repo | `wiki/` folder (12 pages) |

### 4.2 — Submission Email / Message Template

```markdown
## Crop Disease Classification — Submission

### Quick Links
- **GitHub:** https://github.com/<user>/crop-disease
- **Live Demo:** https://<streamlit-cloud-url>
- **API Docs:** https://<render-url>/docs (optional)

### Summary
Deep learning solution for identifying 12 plant diseases across 3 crops
(Tomato, Potato, Pepper) using transfer learning with PyTorch.

### Key Highlights
- 3 CNN architectures compared (ResNet-50, EfficientNet-B0, MobileNetV3-Small)
- Three-stage progressive fine-tuning (Feature Extraction → Adaptation → Refinement)
- 95%+ accuracy on test set with EfficientNet-B0 (recommended model)
- Interactive Streamlit web app with treatment recommendations
- FastAPI REST API with SOLID architecture + Docker deployment
- 78 unit tests passing, comprehensive documentation (12 wiki pages)

### How to Test
1. **Live Demo (no setup):** Click the Streamlit link → upload a leaf photo → see prediction
2. **Local Setup:** Clone repo → `pip install -r requirements.txt` → `streamlit run app/streamlit_app.py`
3. **Full Notebook:** `jupyter notebook notebooks/crop_disease_classification.ipynb` (runs all parts)

### Documentation
See the `wiki/` folder for 12 detailed pages covering architecture, training,
evaluation, deployment, and more. Start with `wiki/Home.md`.
```

### 4.3 — Evaluator Quick-Test Guide

For evaluators who want to test quickly without local setup:

| Step | Action | Time |
|------|--------|------|
| 1 | Open the **Live Demo** link | 10 sec |
| 2 | Upload a leaf photo (or use a test image from the dataset) | 10 sec |
| 3 | View the prediction, confidence score, and treatment recommendation | 30 sec |
| 4 | Try different images to test various diseases | 2 min |
| 5 | Browse the **GitHub repo** for code quality and documentation | 5 min |
| 6 | Read `wiki/Home.md` → follow links to specific topics | 10 min |
| 7 | (Optional) Check the **API docs** at the Render URL | 2 min |

---

## Sharing Checklist

### Code Quality
- [x] All 78 tests pass (`python -m pytest tests/ -v`)
- [x] No lint errors (Pylance/Ruff clean)
- [x] Type annotations on all public functions
- [x] Docstrings on all modules and classes
- [x] SOLID principles in API (`api/` package)
- [x] DRY — no duplicated constants or logic
- [x] Consistent naming conventions (snake_case files, PascalCase classes)

### Documentation
- [x] README.md — project overview, setup, performance, tech stack
- [x] Wiki Home — index with reading order
- [x] Getting Started — step-by-step setup guide
- [x] Architecture Overview — module diagram, data flow, design principles
- [x] Task Walkthrough — every requirement mapped to code
- [x] Data Pipeline — dataset, augmentation, splitting
- [x] Model Training — three-stage fine-tuning, regularization
- [x] Evaluation & Metrics — confusion matrix, profiling, business recommendation
- [x] Streamlit App — UI architecture, confidence thresholding
- [x] Deployment Guide — online API, offline mobile, hybrid strategy
- [x] Cloud Deployment — free hosting on Streamlit Cloud, HF Spaces, Render
- [x] Sharing Plan — this page
- [x] FAQ & Troubleshooting — common errors, platform notes

### Deployment
- [x] Streamlit app runs locally
- [x] FastAPI API runs locally
- [x] Docker build succeeds
- [x] (Ready) Streamlit Community Cloud deployment
- [x] (Ready) Render deployment (optional)

### Project Artifacts
- [x] Trained model checkpoints (3 architectures)
- [x] Class mapping JSON
- [x] Training configuration JSON
- [x] Generated figures (9+ PNG files)
- [x] Jupyter notebook (all sections executable)

---

## Project Metrics Summary

| Metric | Value |
|--------|-------|
| Source files | 56 |
| Unit tests | 78 passed, 1 skipped |
| Wiki pages | 12 |
| Disease classes | 12 (3 crops) |
| Model architectures | 3 |
| Training stages | 3 per model |
| Streamlit app modules | 5 |
| API endpoints | 5 (health, version, predict, docs, redoc) |
| Docker support | Multi-stage Dockerfile + compose |
| Free cloud deployment options | 3 (Streamlit, HF, Render) |

---

## Next Steps

| What | Where |
|------|-------|
| Deploy to the cloud | [Cloud Deployment](Cloud-Deployment.md) |
| Local setup guide | [Getting Started](Getting-Started.md) |
| Full requirements walkthrough | [Task Walkthrough](Task-Walkthrough.md) |
| Wiki index | [Home](Home.md) |
