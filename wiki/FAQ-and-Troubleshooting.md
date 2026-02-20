# ❓ FAQ & Troubleshooting

[← Back to Home](Home.md)

Common questions, errors, and solutions for the Crop Disease Classification project.

---

## Frequently Asked Questions

### General

<details>
<summary><strong>Q: What Python version do I need?</strong></summary>

**A:** Python **3.10 or higher**. We recommend **3.11** for best compatibility with PyTorch and all dependencies. Python 3.12+ may have issues with some packages.

Check your version:
```bash
python --version
```
</details>

<details>
<summary><strong>Q: Do I need a GPU?</strong></summary>

**A:** No — everything works on CPU. A GPU speeds up training significantly:

| Hardware | Training Time (all 3 models) |
|----------|------------------------------|
| NVIDIA GPU (CUDA) | ~1-2 hours |
| Apple Silicon (MPS) | ~2-4 hours |
| CPU only | ~6-10 hours |

**Tip:** To train faster on CPU, train only EfficientNet-B0 by changing `model_names` in notebook §3:
```python
model_names = ['efficientnet_b0']  # instead of all three
```
</details>

<details>
<summary><strong>Q: Can I use Google Colab?</strong></summary>

**A:** Yes. Upload the project to Google Drive or clone from a repository. Key changes:
1. Install dependencies: `!pip install -r requirements.txt`
2. Update `DATASET_ROOT` in the notebook to point to your Colab dataset path
3. Set `runtime ▸ Change runtime type ▸ T4 GPU` for faster training
</details>

<details>
<summary><strong>Q: What if my dataset has different classes?</strong></summary>

**A:** Update `config.data.selected_classes` in `src/config.py` to list your class folder names. Also update `config.model.num_classes` to match. The rest of the pipeline adapts automatically.
</details>

<details>
<summary><strong>Q: How do I add more crops/diseases?</strong></summary>

**A:** 
1. Add image folders to the dataset directory (one folder per class)
2. Add the folder names to `config.data.selected_classes`
3. Update `config.model.num_classes`
4. Add entries to `DISEASE_INFO` in `app/disease_info.py`
5. Retrain by running the notebook
</details>

---

## Installation Issues

### `pip install` fails with build errors

**Symptom:**
```
error: subprocess-exited-with-error
× Building wheel for torch failed
```

**Solution:** Install PyTorch separately using the official selector:
1. Go to [pytorch.org/get-started](https://pytorch.org/get-started/locally/)
2. Select your OS, package manager, and CUDA version
3. Copy and run the generated command
4. Then: `pip install -r requirements.txt` (it will skip torch since it's already installed)

---

### `ModuleNotFoundError: No module named 'torch'`

**Solution:** Make sure your virtual environment is activated:
```bash
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

---

### `ImportError: cannot import name 'get_model' from 'src.models.factory'`

**Solution:** Run the notebook or scripts from the project root:
```bash
cd crop-disease
jupyter notebook notebooks/crop_disease_classification.ipynb
```

The notebook adds the project root to `sys.path` in its first code cell.

---

## Dataset Issues

### `FileNotFoundError: Data directory not found`

**Symptom:** The notebook can't find the PlantVillage dataset.

**Solution:** Make sure the dataset is in the correct location relative to the project:
```
your-workspace/
├── PlantVillage Dataset/
│   └── PlantVillage/          ← This folder must contain the class subfolders
│       ├── Tomato_Bacterial_spot/
│       ├── Tomato_Early_blight/
│       └── ...
└── crop-disease/     ← The project
```

If your dataset is elsewhere, update `DATASET_ROOT` in the notebook's setup cell:
```python
DATASET_ROOT = Path("/your/path/to/PlantVillage")
```

---

### `ValueError: Classes not found in ...: {'SomeClass'}`

**Symptom:** The selected class names don't match the folder names on disk.

**Solution:** Check the exact folder names in your dataset (they're case-sensitive and include underscores/double-underscores):
```bash
ls "PlantVillage Dataset/PlantVillage/"
```

Make sure `config.data.selected_classes` matches these names exactly.

---

### Dataset has duplicate nested folders

**Symptom:** The PlantVillage download has a nested structure:
```
PlantVillage Dataset/
├── PlantVillage/
│   ├── Tomato_Bacterial_spot/     ← actual images here
│   └── PlantVillage/              ← duplicate nested copy
│       └── Tomato_Bacterial_spot/
```

**Solution:** Point `DATASET_ROOT` to the first-level `PlantVillage/` folder (not the nested one). Our code automatically scans only direct subdirectories.

---

## Training Issues

### `CUDA out of memory`

**Solution:** Reduce batch size in `src/config.py`:
```python
batch_size: int = 16  # was 32
```

Or reduce image size (less recommended as it affects accuracy):
```python
image_size: int = 192  # was 224
```

---

### Training is extremely slow

**Possible causes and fixes:**

| Cause | Fix |
|-------|-----|
| Running on CPU | Use a GPU or train fewer models |
| `num_workers` too high | Set to 0 or 2 on Windows |
| Large batch size on low RAM | Reduce `batch_size` to 16 |
| Antivirus scanning data files | Add project folder to antivirus exclusions |

**Quick fix — train only one model:**
```python
model_names = ['efficientnet_b0']  # In notebook §3
```

---

### `RuntimeError: MPS backend out of memory`

**Symptom:** On Apple Silicon (M1/M2/M3), training crashes with MPS memory errors.

**Solution:**
```python
# Option 1: Reduce batch size
batch_size: int = 16

# Option 2: Fall back to CPU
device = torch.device("cpu")  # Instead of "mps"
```

---

### Model accuracy is very low (<50%)

**Possible causes:**

1. **Not enough training epochs** — Make sure all 3 stages run (5 + 10 + 10 epochs)
2. **Wrong learning rate** — Stage 1 should be 1e-3, not 1e-5
3. **Data loading issue** — Check that images are loading correctly:
   ```python
   img, label = dataset[0]
   print(img.shape, label)  # Should be (3, 224, 224) and an int
   ```
4. **Class mapping mismatch** — Verify `class_to_idx` has 12 entries

---

## Streamlit App Issues

### `No trained checkpoint found`

**Symptom:** The app shows a warning about using an untrained model.

**Solution:** Train the models first by running the notebook (§3). After training, checkpoint files appear in the `models/` directory (created at runtime):
```
models/                               ← created at runtime
├── efficientnet_b0_best.pth
├── resnet50_best.pth
└── mobilenetv3_best.pth
```

---

### `ModuleNotFoundError` when running Streamlit

**Solution:** Run from the project root (not from `app/`):
```bash
cd crop-disease
streamlit run app/streamlit_app.py
```

---

### App is slow on first load

**Normal behavior.** The first prediction takes ~5 seconds because the model is being loaded. Subsequent predictions are fast (~50ms) thanks to `@st.cache_resource`.

---

### Port 8501 already in use

**Solution:**
```bash
# Option 1: Use a different port
streamlit run app/streamlit_app.py --server.port 8502

# Option 2: Kill the existing process
lsof -ti :8501 | xargs kill -9
```

---

## Notebook Issues

### Cells must be run in order

The notebook has dependencies between sections. If you get `NameError: name 'dataset' is not defined`, you likely skipped a cell.

**Solution:** Run all cells from the top: `Kernel → Restart & Run All` (or `Cell → Run All` in JupyterLab).

---

### Notebook kernel keeps dying

**Cause:** Usually out of memory.

**Solution:**
1. Close other applications to free RAM
2. Reduce `batch_size` to 16
3. Train only one model instead of three
4. Restart the kernel and try again

---

## Platform-Specific Notes

### macOS Apple Silicon (M1/M2/M3/M4)

- PyTorch uses **MPS** (Metal Performance Shaders) for GPU acceleration
- AMP (mixed precision) is **automatically disabled** — it's not fully supported on MPS
- Performance is ~3-5x faster than CPU, but slower than NVIDIA CUDA
- If you encounter MPS errors, fall back to CPU:
  ```python
  device = torch.device("cpu")
  ```

### Windows

- Use `num_workers=0` if you get multiprocessing errors:
  ```python
  num_workers: int = 0  # In config.py
  ```
- Use PowerShell or Git Bash (not Command Prompt) for best experience
- Paths use backslashes but Python handles this automatically

### Linux

- Most straightforward platform. Everything should work out of the box
- For NVIDIA GPU: install CUDA toolkit and cuDNN
- For AMD GPU: consider ROCm (limited PyTorch support)

---

## Deployment Issues

### FastAPI server won't start

**Symptom:** `ModuleNotFoundError: No module named 'fastapi'`

**Solution:** Install the API dependencies:
```bash
pip install fastapi uvicorn python-multipart
```

---

### Docker build fails

**Symptom:** `COPY models/ models/` fails because the directory doesn't exist.

**Solution:** Train the model first (notebook §3) to generate the `models/` directory. Docker needs the checkpoint files to be present at build time.

---

### ONNX export fails

**Symptom:** `ModuleNotFoundError: No module named 'onnx'`

**Solution:** Install ONNX dependencies (they're optional):
```bash
pip install onnx onnxscript onnxsim
```

---

### TFLite conversion fails

**Symptom:** Error during `onnx2tf.convert()` or `ai-edge-torch.convert()`.

**Solution:**
1. Ensure the ONNX model exports cleanly first: `python -c "import onnx; onnx.checker.check_model(onnx.load('models/efficientnet_b0.onnx'))"`
2. Install TensorFlow: `pip install tensorflow`
3. Install onnx2tf: `pip install onnx2tf`
4. Try the alternative `ai-edge-torch` path if `onnx2tf` doesn't work

---

### API returns wrong predictions

**Symptom:** API predictions don't match Streamlit or notebook results.

**Cause:** Preprocessing mismatch — the API must use the exact same `val_transforms` as training.

**Solution:** Ensure the API imports `get_val_transforms()` from the same `src/data/transforms.py` used during training. Check that:
- Input is resized to 256 then center-cropped to 224
- ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` are used
- Image is converted to RGB before processing

---

## Testing Issues

### How to run tests

```bash
cd crop-disease
python -m pytest tests/ -v --tb=short
```

Expected: **78 passed, 1 skipped** (ONNX test skipped without `onnx` package).

---

### Tests fail with import errors

**Solution:** Run from the project root and ensure your virtual environment is activated:
```bash
cd crop-disease
source .venv/bin/activate
python -m pytest tests/ -v
```

---

### How to run a single test file

```bash
python -m pytest tests/test_model_factory.py -v
```

---

## Performance Tips

| Tip | Impact |
|-----|--------|
| Use a GPU (NVIDIA CUDA) | 5-10x faster training |
| Increase `num_workers` to 8 | Faster data loading (if you have 8+ CPU cores) |
| Use `pin_memory=True` | Faster CPU→GPU transfer (already enabled) |
| Reduce `early_stopping_patience` to 3 | Saves time if model converges quickly |
| Use `persistent_workers=True` | Avoids worker respawn overhead (already enabled) |

---

## Still Stuck?

1. Check the error message carefully — most errors tell you exactly what's wrong
2. Search the error on [Stack Overflow](https://stackoverflow.com/) or [PyTorch Forums](https://discuss.pytorch.org/)
3. Review the [Getting Started](Getting-Started.md) guide for setup verification
4. Make sure you're using the correct Python environment (`which python` should show `.venv/`)

---

## Next Steps

| What | Where |
|------|-------|
| Detailed setup guide | [Getting Started](Getting-Started.md) |
| Wiki index with all pages | [Home](Home.md) |
| Deploy to production (API or mobile) | [Deployment Guide](Deployment-Guide.md) |
| Free cloud hosting | [Cloud Deployment](Cloud-Deployment.md) |
| Step-by-step sharing checklist | [Sharing Plan](Sharing-Plan.md) |
| End-to-end requirements walkthrough | [Task Walkthrough](Task-Walkthrough.md) |
| Understand the project structure | [Architecture Overview](Architecture-Overview.md) |
