# End-to-End Deployment Guide

Complete step-by-step instructions to deploy the **Crop Disease Classifier** on three free cloud platforms:

| Platform | Component | URL Pattern |
|----------|-----------|-------------|
| **Streamlit Community Cloud** | Streamlit demo app | `https://<user>-crop-disease-appstreamlit-app-<hash>.streamlit.app` |
| **Hugging Face Spaces** | Streamlit app (Docker) | `https://<user>-crop-disease-classifier.hf.space` |
| **Render** | FastAPI REST API | `https://crop-disease-api.onrender.com` |

> **Total cost: $0/month** — all three platforms offer free tiers.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Pre-Deployment Checklist](#2-pre-deployment-checklist)
3. [Step 0 — Push to GitHub with Git LFS](#step-0--push-to-github-with-git-lfs)
4. [Option A — Streamlit Community Cloud](#option-a--streamlit-community-cloud)
5. [Option B — Hugging Face Spaces](#option-b--hugging-face-spaces)
6. [Option C — Render (FastAPI API)](#option-c--render-fastapi-api)
7. [Post-Deployment Verification](#7-post-deployment-verification)
8. [Architecture Diagram](#8-architecture-diagram)
9. [Environment Variables Reference](#9-environment-variables-reference)
10. [Troubleshooting](#10-troubleshooting)
11. [Platform Limitations](#11-platform-limitations)

---

## 1. Prerequisites

### Software

| Tool | Version | Install |
|------|---------|---------|
| Python | ≥ 3.10 | [python.org](https://www.python.org/downloads/) |
| Git | ≥ 2.30 | [git-scm.com](https://git-scm.com/) |
| Git LFS | ≥ 3.0 | `brew install git-lfs` (macOS) or [git-lfs.com](https://git-lfs.com/) |
| Docker | ≥ 20 (optional) | [docker.com](https://www.docker.com/products/docker-desktop/) |

### Accounts

| Platform | Sign Up |
|----------|---------|
| GitHub | [github.com/join](https://github.com/join) |
| Streamlit Community Cloud | [share.streamlit.io](https://share.streamlit.io) (use GitHub login) |
| Hugging Face | [huggingface.co/join](https://huggingface.co/join) |
| Render | [dashboard.render.com/register](https://dashboard.render.com/register) |

### Trained Model

You need a trained model checkpoint before deploying. If you haven't trained yet:

```bash
cd crop-disease
python -m pip install -r requirements.txt
# Run the training notebook or training script
# This produces: models/efficientnet_b0_best.pth + models/class_mapping.json
```

---

## 2. Pre-Deployment Checklist

Run these commands from the project root (`crop-disease/`) to verify everything works:

```bash
# 1. Run all tests
python -m pytest tests/ -v --tb=short
# Expected: 78 passed, 1 skipped

# 2. Verify Streamlit app runs locally
streamlit run app/streamlit_app.py
# Open http://localhost:8501, upload a leaf image, verify prediction

# 3. Verify FastAPI runs locally
uvicorn api.app:app --reload
# Open http://localhost:8000/docs, test POST /predict

# 4. Verify required files exist
ls -la models/efficientnet_b0_best.pth   # trained checkpoint
ls -la models/class_mapping.json          # class index mapping
ls -la requirements.txt                   # full dependencies
ls -la requirements-api.txt               # API-only dependencies
ls -la Dockerfile                         # API container
ls -la Dockerfile.hf                      # HF Spaces container
ls -la render.yaml                        # Render IaC
ls -la .streamlit/config.toml             # Streamlit config
```

> **All 4 checks must pass before proceeding.**

---

## Step 0 — Push to GitHub with Git LFS

All three platforms deploy from a GitHub repository. This step is **required once** and shared across all deployments.

### 0.1 — Initialize Git & Git LFS

```bash
cd crop-disease

# Initialize Git (skip if already a repo)
git init
git remote add origin https://github.com/<YOUR-USERNAME>/crop-disease.git
```

### 0.2 — Set Up Git LFS for Model Files

Model checkpoints (`.pth` files) are too large for regular Git. Use Git LFS:

```bash
# Install Git LFS (one-time per machine)
git lfs install

# Track model files
git lfs track "models/*.pth"

# Verify .gitattributes was created/updated
cat .gitattributes
# Should show: models/*.pth filter=lfs diff=lfs merge=lfs -text
```

### 0.3 — Commit & Push

```bash
# Stage all project files
git add .gitattributes
git add .

# Verify model files are tracked by LFS (not regular Git)
git lfs ls-files --all
# Should list: models/efficientnet_b0_best.pth

# Commit and push
git commit -m "Initial commit: crop disease classifier with trained model"
git push -u origin main
```

### 0.4 — Verify on GitHub

1. Go to `https://github.com/<YOUR-USERNAME>/crop-disease`
2. Navigate to `models/` folder
3. Click on `efficientnet_b0_best.pth` — should show **"Stored with Git LFS"** badge
4. Verify `class_mapping.json` is visible and readable

> **GitHub free tier**: 1 GB LFS storage, 1 GB bandwidth/month. EfficientNet-B0 checkpoint is ~20 MB — well within limits.

---

## Option A — Streamlit Community Cloud

**Best for**: Quick demo, evaluators, non-technical audience.

### What gets deployed

```
Files used by Streamlit Cloud:
├── requirements.txt          ← dependencies (auto-installed)
├── .streamlit/config.toml    ← headless mode, theme
├── app/
│   ├── streamlit_app.py      ← entry point
│   ├── config.py
│   ├── disease_info.py
│   ├── model_service.py
│   └── ui_components.py
├── src/                      ← ML library (imported by app)
└── models/
    ├── efficientnet_b0_best.pth  ← via Git LFS
    └── class_mapping.json
```

### A.1 — Go to Streamlit Cloud

1. Open [share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign in with GitHub"**

### A.2 — Create App

1. Click **"Create app"** → **"Deploy a public app from GitHub"**
2. Fill in the deployment form:

   | Field | Value |
   |-------|-------|
   | **Repository** | `<YOUR-USERNAME>/crop-disease` |
   | **Branch** | `main` |
   | **Main file path** | `app/streamlit_app.py` |

3. (Optional) Click **"Advanced settings"** → set **Python version** to `3.11`
4. Click **"Deploy!"**

### A.3 — Wait for Build

The build process takes 3–8 minutes on first deploy:
1. Streamlit Cloud clones your repo (including LFS files)
2. Installs `requirements.txt` via pip
3. Runs `streamlit run app/streamlit_app.py`

Watch the build log for errors. Common issues:
- **"No module named 'src'"** → see [Troubleshooting](#module-import-errors)
- **Out of memory** → see [Platform Limitations](#11-platform-limitations)

### A.4 — Verify

Once deployed, your app is live at:
```
https://<YOUR-USERNAME>-crop-disease-appstreamlit-app-<HASH>.streamlit.app
```

**Test the deployment:**
1. Open the URL in a browser
2. Upload a test leaf image (any image from the PlantVillage dataset works)
3. Verify:
   - ✅ Prediction class name appears (e.g., "Tomato Bacterial Spot")
   - ✅ Confidence percentage is shown
   - ✅ Treatment recommendations appear
   - ✅ Confidence bar chart renders

### A.5 — Auto-Updates

Every `git push` to `main` automatically rebuilds and redeploys the app. No manual intervention needed.

---

## Option B — Hugging Face Spaces

**Best for**: ML community visibility, more resources (2 vCPU / 16 GB RAM), Docker-based.

### What gets deployed

```
Files used by HF Spaces:
├── Dockerfile.hf             ← renamed to "Dockerfile" in the Space
├── requirements.txt          ← installed inside Docker
├── .streamlit/config.toml    ← Streamlit config
├── app/                      ← Streamlit application
├── src/                      ← ML library
└── models/
    ├── efficientnet_b0_best.pth  ← via HF Git LFS
    └── class_mapping.json
```

### B.1 — Create a Hugging Face Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in the form:

   | Field | Value |
   |-------|-------|
   | **Owner** | `<YOUR-USERNAME>` |
   | **Space name** | `crop-disease-classifier` |
   | **License** | MIT |
   | **SDK** | **Docker** |
   | **Hardware** | **CPU Basic (Free)** |
   | **Visibility** | Public |

3. Click **"Create Space"**

### B.2 — Clone the Space Locally

```bash
# Clone the empty HF Space
git clone https://huggingface.co/spaces/<YOUR-USERNAME>/crop-disease-classifier
cd crop-disease-classifier
```

### B.3 — Copy Project Files

```bash
# Copy from your project (adjust the source path)
cp -r /path/to/crop-disease/src .
cp -r /path/to/crop-disease/app .
cp -r /path/to/crop-disease/models .
cp /path/to/crop-disease/requirements.txt .
cp /path/to/crop-disease/.streamlit/config.toml .streamlit/config.toml

# Use our HF-specific Dockerfile (rename it to Dockerfile)
cp /path/to/crop-disease/Dockerfile.hf ./Dockerfile
```

### B.4 — Create HF README

Create a `README.md` with Hugging Face metadata (this is **required** for HF Spaces):

```bash
cat > README.md << 'EOF'
---
title: Crop Disease Classifier
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
license: mit
---

# Crop Disease Classifier

Upload a leaf image to detect plant diseases across 12 classes from Tomato, Potato, and Pepper crops.

Built with PyTorch (EfficientNet-B0) and Streamlit.
EOF
```

### B.5 — Push to Hugging Face

```bash
# Initialize Git LFS in the Space
git lfs install
git lfs track "models/*.pth"
git add .gitattributes

# Stage all files
git add .

# Verify LFS tracks the model
git lfs ls-files
# Should show: models/efficientnet_b0_best.pth

# Commit and push
git commit -m "Add crop disease classifier"
git push
```

> **HF provides unlimited Git LFS storage** — no size limits on model files.

### B.6 — Monitor Build

1. Go to `https://huggingface.co/spaces/<YOUR-USERNAME>/crop-disease-classifier`
2. Click the **"Logs"** tab to watch the Docker build
3. Build takes ~5–10 minutes on first push
4. Status changes: `Building` → `Running` → ✅ `Live`

### B.7 — Verify

Once live, your app is at:
```
https://<YOUR-USERNAME>-crop-disease-classifier.hf.space
```

**Test the deployment:**
1. Open the URL
2. Upload a leaf image
3. Verify predictions, confidence, and treatment recommendations appear

---

## Option C — Render (FastAPI API)

**Best for**: REST API endpoint for programmatic access (curl, Postman, mobile apps).

### What gets deployed

```
Files used by Render:
├── Dockerfile                ← multi-stage API container
├── render.yaml               ← IaC Blueprint (optional one-click)
├── requirements-api.txt      ← lean API dependencies
├── api/                      ← FastAPI application
├── src/                      ← ML library
└── models/
    ├── efficientnet_b0_best.pth  ← via Git LFS
    └── class_mapping.json
```

### C.1 — Deploy via Render Blueprint (Recommended)

The `render.yaml` file enables one-click deployment:

1. Go to [dashboard.render.com/blueprints/new](https://dashboard.render.com/blueprints/new)
2. Connect your GitHub account (if not already connected)
3. Select the `crop-disease` repository
4. Render reads `render.yaml` and auto-configures:
   - Service name: `crop-disease-api`
   - Runtime: Docker
   - Free plan
   - Health check: `/health`
   - Port: 10000
5. Click **"Apply"**

### C.2 — Alternative: Manual Setup

If you prefer manual configuration:

1. Go to **[dashboard.render.com](https://dashboard.render.com)** → **"New +"** → **"Web Service"**
2. Connect your GitHub repo
3. Configure:

   | Setting | Value |
   |---------|-------|
   | **Name** | `crop-disease-api` |
   | **Region** | Oregon (US West) |
   | **Branch** | `main` |
   | **Runtime** | **Docker** |
   | **Dockerfile Path** | `./Dockerfile` |
   | **Instance Type** | **Free** |

4. Add environment variables:

   | Key | Value |
   |-----|-------|
   | `PORT` | `10000` |
   | `PYTHONUNBUFFERED` | `1` |
   | `LOG_LEVEL` | `info` |
   | `CORS_ORIGINS` | `*` (or your frontend URL) |

5. Set **Health Check Path**: `/health`
6. Click **"Deploy Web Service"**

### C.3 — Monitor Build

1. The Docker build starts automatically (~8–15 min for first build with PyTorch)
2. Watch the build log in the Render Dashboard
3. Status changes: `Building` → `Deploying` → `Live` ✅

### C.4 — Verify

Once deployed, your API is at:
```
https://crop-disease-api.onrender.com
```

**Test with curl:**

```bash
# Health check
curl https://crop-disease-api.onrender.com/health
# Expected: {"status":"healthy","model_loaded":true,...}

# Model version
curl https://crop-disease-api.onrender.com/model/version
# Expected: {"model":"efficientnet_b0","num_classes":12,...}

# Predict (replace with a real image path)
curl -X POST "https://crop-disease-api.onrender.com/predict" \
  -F "file=@test_leaf.jpg"
# Expected: {"predicted_class":"Tomato_Bacterial_spot","confidence":0.95,...}
```

**Test with Swagger UI:**
Open `https://crop-disease-api.onrender.com/docs` in a browser to access the interactive API documentation.

### C.5 — Auto-Deploys

Render automatically rebuilds on every `git push` to `main`. Configure this in:
**Dashboard → Settings → Build & Deploy → Auto-Deploy → Yes**

---

## 7. Post-Deployment Verification

After deploying to any/all platforms, run this checklist:

### Streamlit (Community Cloud or HF Spaces)

| # | Check | How |
|---|-------|-----|
| 1 | App loads without errors | Open URL in browser |
| 2 | File uploader is visible | Check sidebar/main area |
| 3 | Prediction works | Upload a leaf image |
| 4 | Confidence percentage shown | Check result section |
| 5 | Treatment info displayed | Check below prediction |
| 6 | Chart renders | Check bar chart |
| 7 | Error handling works | Upload a non-image file |

### FastAPI (Render)

| # | Check | How |
|---|-------|-----|
| 1 | Health endpoint | `curl <URL>/health` → `{"status":"healthy"}` |
| 2 | Model version | `curl <URL>/model/version` |
| 3 | Swagger UI | Open `<URL>/docs` in browser |
| 4 | Prediction | `curl -X POST <URL>/predict -F file=@leaf.jpg` |
| 5 | Error handling | `curl -X POST <URL>/predict -F file=@not_image.txt` |
| 6 | CORS headers | Check `Access-Control-Allow-Origin` in response headers |

---

## 8. Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    GitHub Repository                      │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Source Code + Models (Git LFS) + Deployment Configs │ │
│  └───────────┬──────────────┬──────────────┬───────────┘ │
└──────────────┼──────────────┼──────────────┼─────────────┘
               │              │              │
    ┌──────────▼──┐   ┌──────▼──────┐  ┌────▼────────┐
    │  Streamlit  │   │ Hugging Face│  │   Render    │
    │  Community  │   │   Spaces    │  │             │
    │   Cloud     │   │             │  │             │
    ├─────────────┤   ├─────────────┤  ├─────────────┤
    │ requirements│   │ Dockerfile  │  │ Dockerfile  │
    │   .txt      │   │ (from .hf)  │  │ render.yaml │
    │ .streamlit/ │   │ Port 7860   │  │ Port 10000  │
    │  config.toml│   │             │  │             │
    ├─────────────┤   ├─────────────┤  ├─────────────┤
    │  Streamlit  │   │  Streamlit  │  │   FastAPI   │
    │    App      │   │    App      │  │  REST API   │
    └──────┬──────┘   └──────┬──────┘  └──────┬──────┘
           │                 │                 │
    ┌──────▼──────┐   ┌──────▼──────┐  ┌──────▼──────┐
    │  *.streamlit│   │   *.hf.space│  │ *.onrender  │
    │    .app     │   │             │  │    .com     │
    │             │   │  2 vCPU     │  │  REST API   │
    │  Web Demo   │   │  16GB RAM   │  │  Swagger UI │
    └─────────────┘   └─────────────┘  └─────────────┘
         👤                 👤                💻
     Evaluators         Community        Developers
    (Web Browser)      (Web Browser)    (curl/Postman)
```

---

## 9. Environment Variables Reference

| Variable | Used By | Default | Description |
|----------|---------|---------|-------------|
| `PORT` | Dockerfile, Render | `8000` | API server port (Render overrides to `10000`) |
| `CORS_ORIGINS` | api/middleware.py | `*` | Comma-separated allowed origins |
| `PYTHONUNBUFFERED` | Docker | `1` | Disable Python output buffering |
| `LOG_LEVEL` | Docker, Render | `info` | Logging level |

---

## 10. Troubleshooting

### Module Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'src'`

**Cause:** The deployment platform runs from the repo root, but Python can't find `src/` as a package.

**Fix:** Our `app/streamlit_app.py` and `api/` already add the project root to `sys.path`. If you still see this, add to the top of the entry-point file:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

### Model File Not Found

**Symptom:** `FileNotFoundError: models/efficientnet_b0_best.pth`

**Cause:** Git LFS files weren't pulled, or the model wasn't included in the deployment.

**Fix:**
1. Verify Git LFS is set up: `git lfs ls-files` should list the `.pth` file
2. On Streamlit Cloud: ensure the repo is **public** (LFS works with public repos)
3. On HF Spaces: run `git lfs install && git lfs pull` in the Space repo
4. On Render: Git LFS is pulled automatically during Docker build

### Streamlit Cloud: App Crashes on Load

**Symptom:** App shows "This app has gone over its resource limits"

**Cause:** RAM limit (~1 GB) exceeded. Full PyTorch + model exceeds the limit.

**Fix:**
1. Use EfficientNet-B0 (~20 MB) — already the default in our app
2. Ensure `st.cache_resource` is used for model loading (already implemented)
3. Remove unnecessary imports (pandas, matplotlib not needed at runtime)

### Hugging Face Spaces: Docker Build Fails

**Symptom:** Build log shows errors during `pip install` or `COPY` step

**Fix:**
1. Check **Logs** tab on the Space page
2. Common issues:
   - Missing `Dockerfile` → ensure `Dockerfile.hf` was renamed to `Dockerfile`
   - Port mismatch → must be `7860` (HF requirement)
   - Missing files → ensure all `COPY` source paths exist

### Render: Out of Memory (512 MB)

**Symptom:** Service crashes with OOMKilled

**Cause:** Render free tier has only 512 MB RAM. PyTorch alone is ~400 MB.

**Fix (choose one):**
1. **Use `requirements-api.txt`** (already configured) — installs only torch + fastapi, not streamlit/jupyter/matplotlib
2. **Switch to ONNX Runtime** (~50 MB vs ~400 MB for PyTorch):
   ```bash
   python scripts/export_model.py  # exports to models/model.onnx
   pip install onnxruntime          # add to requirements-api.txt
   ```
3. **Upgrade to Starter plan** ($7/month, guaranteed 512 MB)
4. **Use HF Spaces** for the Streamlit demo and skip Render

### Render: Slow Cold Start

**Symptom:** First request after 15 min of inactivity takes 30–60 seconds

**Cause:** Render free tier spins down after inactivity.

**Fix:**
- This is expected behavior on the free tier
- Set up a free uptime monitor (e.g., [UptimeRobot](https://uptimerobot.com/)) to ping `/health` every 14 minutes
- Or upgrade to a paid plan for always-on instances

### Docker Build: "COPY models/*.pth failed"

**Symptom:** Docker build fails at the `COPY models/*.pth` step

**Fix:** Our Dockerfile handles this gracefully with `|| true`. If you still see issues:
1. Create the models directory: `mkdir -p models`
2. Place your trained checkpoint in `models/`
3. Or comment out the COPY line and download the model at runtime

### General: Wrong Python Version

**Symptom:** `SyntaxError` or `ImportError` on modern Python features

**Fix:**
- Streamlit Cloud: Set Python 3.11 in Advanced Settings
- HF Spaces: Our `Dockerfile.hf` uses `python:3.11-slim`
- Render: Our `Dockerfile` uses `python:3.11-slim`

---

## 11. Platform Limitations

| Resource | Streamlit Cloud | HF Spaces (Free) | Render (Free) |
|----------|----------------|-------------------|----------------|
| **CPU** | Shared | 2 vCPU | 0.1 vCPU |
| **RAM** | ~1 GB | 16 GB | 512 MB |
| **Disk** | Ephemeral | 50 GB (ephemeral) | Ephemeral |
| **Sleep** | After inactivity | After inactivity | 15 min idle |
| **Cold Start** | ~30s | ~30s | ~60s |
| **Auto-Deploy** | ✅ on `git push` | ✅ on `git push` | ✅ on `git push` |
| **Custom Domain** | ❌ | ✅ (subdomain) | ✅ (free HTTPS) |
| **GPU** | ❌ | Paid upgrade | ❌ |
| **Bandwidth** | Unlimited | Unlimited | 100 GB/month |
| **Best For** | Quick demo | ML community | REST API |

---

## Quick Reference — Deployment Files

| File | Purpose | Used By |
|------|---------|---------|
| `requirements.txt` | Full dependency list | Streamlit Cloud, HF Spaces |
| `requirements-api.txt` | Lean API-only deps | Render (via Dockerfile) |
| `Dockerfile` | Multi-stage API container | Render, docker-compose |
| `Dockerfile.hf` | Streamlit container | Hugging Face Spaces |
| `docker-compose.yml` | Local development | Local Docker |
| `render.yaml` | Render IaC Blueprint | Render (one-click deploy) |
| `.streamlit/config.toml` | Streamlit settings | Streamlit Cloud, HF Spaces |
| `models/.gitkeep` | Keeps models/ in Git | All platforms |
| `.gitignore` | Excludes `.pth` from Git (use LFS) | GitHub |

---

## Recommended Deployment Order

1. **Start with Streamlit Community Cloud** (5 min, easiest, purpose-built for Streamlit)
2. **Add Hugging Face Spaces** (10 min, more resources, ML community exposure)
3. **Optionally add Render** (10 min, if you need a REST API endpoint)

All three can run simultaneously from the same GitHub repository at zero cost.
