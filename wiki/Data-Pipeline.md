# 🔄 Data Pipeline

[← Back to Home](Home.md)

This page explains how raw images on disk become training-ready tensors. Every step is detailed so a beginner can understand what happens and **why**.

---

## Pipeline Overview

```
📁 PlantVillage/              Raw folder-per-class images
        │
        ▼
┌──────────────────┐
│ PlantDiseaseDataset │         Scan folders, build (path, label) list
│   (dataset.py)      │         Filter to 12 selected classes
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ create_stratified │           Split list into train/val/test
│  _split()         │           70% / 15% / 15% with stratification
│  (splitter.py)    │
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ create_dataloaders│           Wrap each split with transforms + DataLoader
│   (loader.py)     │           Train: augmented, shuffled
└──────────────────┘           Val/Test: deterministic, ordered
        │
        ▼
  (images, labels)             Batched tensors ready for model
```

---

## Step 1: Dataset Scanning — `PlantDiseaseDataset`

**File:** `src/data/dataset.py`

### What it does

The `PlantDiseaseDataset` class extends PyTorch's `Dataset`. It reads images from a folder structure where each subdirectory name is a class label:

```
PlantVillage/
├── Tomato_Bacterial_spot/     ← class name
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── Tomato_Early_blight/       ← class name
│   └── ...
└── Pepper__bell___healthy/    ← class name
    └── ...
```

### How it works (step by step)

1. **List all subdirectories** in the root folder (skip hidden folders starting with `.`)
2. **Filter** to only the 12 classes listed in `config.data.selected_classes`
3. **Sort alphabetically** — this creates a deterministic `class_to_idx` mapping:
   ```python
   {
     "Pepper__bell___Bacterial_spot": 0,
     "Pepper__bell___healthy": 1,
     "Potato___Early_blight": 2,
     ...
     "Tomato_healthy": 8,
     ...
   }
   ```
4. **Scan each class folder** for valid image files (`.jpg`, `.jpeg`, `.png`, `.bmp`)
5. **Build a list of tuples**: `[(path1, label0), (path2, label0), ..., (pathN, label11)]`

### Key design decisions

| Decision | Rationale |
|----------|-----------|
| Sort classes alphabetically | Deterministic indexing — same order every run regardless of filesystem |
| Filter by `selected_classes` | The full PlantVillage dataset has 38 classes; we select 12 across 3 crops for a focused study |
| `.convert("RGB")` in `__getitem__` | Handles edge cases: grayscale images or RGBA PNGs are converted to 3-channel RGB |
| Validate `VALID_EXTENSIONS` | Skips hidden files, thumbnails, or non-image files in the dataset |

---

## Step 2: Stratified Splitting — `create_stratified_split()`

**File:** `src/data/splitter.py`

### Why stratified?

If we split randomly, rare classes might end up with very few (or zero) examples in the validation or test set. **Stratification** ensures each split has the same proportion of every class as the original dataset.

### The two-stage split algorithm

**Why two stages instead of one?**  
Sklearn's `train_test_split` only does a two-way split. To get three sets (train/val/test), we split twice:

```
Stage 1: Split all data into (85% trainval) and (15% test)
         ↓ stratify=labels ensures proportions match

Stage 2: Split trainval into (70% train) and (15% val)
         ↓ val_ratio = 0.15 / 0.85 ≈ 0.176 to get exact 15% of total
```

### The math behind `val_ratio`

After removing 15% for the test set, we have 85% remaining. We need 15% of the **total** for validation, which is:

$$\text{val\_ratio} = \frac{0.15}{1 - 0.15} = \frac{0.15}{0.85} \approx 0.1765$$

So we pass `test_size=0.1765` to the second `train_test_split` call.

### Result

| Split | Approximate Size | Purpose |
|-------|-----------------|---------|
| Train (70%) | ~12,700 images | Model learns from these |
| Val (15%) | ~2,700 images | Tune hyperparameters, early stopping |
| Test (15%) | ~2,700 images | Final evaluation (never seen during training) |

---

## Step 3: Data Augmentation — `get_train_transforms()`

**File:** `src/data/transforms.py`

### What is data augmentation?

Data augmentation artificially creates variations of training images. This helps the model:
- **Generalize** — learn the disease pattern, not the specific photo angle/lighting
- **Handle variability** — real-world photos differ from controlled lab images
- **Act as regularization** — reduces overfitting by making each epoch see slightly different data

### The 9-step training pipeline (explained)

| # | Transform | Parameters | What It Simulates |
|---|-----------|-----------|-------------------|
| 1 | `RandomResizedCrop` | size=224, scale=(0.8, 1.0) | Farmer holds phone at varying distances from the leaf. Crops a random 80-100% region of the image and resizes to 224×224 |
| 2 | `RandomHorizontalFlip` | p=0.5 | Leaves can face left or right. Half the time, the image is mirrored horizontally |
| 3 | `RandomVerticalFlip` | p=0.2 | Less common, but adds orientation diversity. Only applied 20% of the time |
| 4 | `RandomRotation` | degrees=20 | Camera tilt. Rotates the image randomly between -20° and +20° |
| 5 | `ColorJitter` | brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05 | **Most important for domain gap.** Simulates sun/shade/overcast variation. Hue is kept small (0.05) because leaf color is diagnostically important |
| 6 | `GaussianBlur` | kernel=3, sigma=(0.1, 2.0), p=0.2 | Out-of-focus mobile phone photos. Applied only 20% of the time |
| 7 | `ToTensor` | — | Converts PIL Image (H×W×C, uint8 0-255) to PyTorch Tensor (C×H×W, float32 0-1) |
| 8 | `Normalize` | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] | ImageNet statistics. **Required for transfer learning** — the pretrained backbone expects inputs normalized this way |
| 9 | `RandomErasing` | p=0.1, scale=(0.02, 0.1) | Simulates partial occlusion (insect on leaf, finger in frame). Replaces a small random rectangle with noise |

### Why this order matters

```
Spatial transforms (1-4)  →  Color transforms (5-6)  →  Tensor conversion (7)  →  Normalization (8)  →  Tensor-space transforms (9)
```

- Spatial and color transforms work on **PIL images** (before `ToTensor`)
- `Normalize` must come after `ToTensor` (operates on tensors)
- `RandomErasing` must come after `Normalize` (operates on normalized tensors)

### Validation pipeline (no augmentation)

```python
Resize(256) → CenterCrop(224) → ToTensor() → Normalize(ImageNet)
```

- **Deterministic**: same image always produces the same tensor
- **Resize to 256 then crop to 224**: preserves aspect ratio better than direct resize to 224
- **Same normalization**: must match training normalization exactly

---

## Step 4: DataLoaders — `create_dataloaders()`

**File:** `src/data/loader.py`

### What is a DataLoader?

A PyTorch `DataLoader` wraps a dataset and provides:
- **Batching**: groups images into batches of 32
- **Shuffling**: randomize order each epoch (training only)
- **Parallel loading**: uses multiple CPU workers to load images while the GPU trains
- **Memory pinning**: pre-copies tensors to GPU-ready memory

### Configuration per split

| Setting | Train | Validation | Test |
|---------|-------|-----------|------|
| `batch_size` | 32 | 32 | 32 |
| `shuffle` | ✅ True | ❌ False | ❌ False |
| `drop_last` | ✅ True | ❌ False | ❌ False |
| `num_workers` | 4 | 4 | 4 |
| `pin_memory` | ✅ True | ✅ True | ✅ True |
| `persistent_workers` | ✅ True | ✅ True | ✅ True |
| `transform` | Train (augmented) | Val (deterministic) | Val (deterministic) |

### Why these settings?

| Setting | Explanation |
|---------|------------|
| `shuffle=True` (train only) | Each epoch sees images in a different order, reducing memorization |
| `drop_last=True` (train only) | If the last batch is smaller than 32, drop it. Small batches cause unstable gradients in batch-dependent layers |
| `pin_memory=True` | Pre-allocates GPU-compatible memory. Speeds up CPU→GPU transfer |
| `persistent_workers=True` | Workers stay alive between epochs. Avoids the overhead of spawning new worker processes each epoch |
| `num_workers=4` | 4 parallel data-loading processes. Good default for most machines |

### The `SplitDataset` wrapper

The `SplitDataset` class wraps a list of `(path, label)` tuples with a transform:

```python
class SplitDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except (OSError, IOError, SyntaxError):
            image = Image.new("RGB", (224, 224), (0, 0, 0))  # blank fallback
        if self.transform:
            image = self.transform(image)
        return image, label
```

This lets us apply **different transforms** to the same underlying data:
- Training split → augmentation pipeline
- Validation/test split → deterministic pipeline

---

## Putting It All Together

Here's the complete flow as called in the notebook:

```python
# 1. Load dataset (just paths + labels)
dataset = PlantDiseaseDataset(root_dir=DATASET_ROOT, selected_classes=config.data.selected_classes)

# 2. Split into train/val/test
splits = create_stratified_split(dataset.samples, config.data.split_ratios, seed=42)

# 3. Create DataLoaders with appropriate transforms
dataloaders = create_dataloaders(
    splits=splits,
    train_transform=get_train_transforms(224),
    val_transform=get_val_transforms(224),
    batch_size=32,
    num_workers=4,
)

# 4. Use in training
for images, labels in dataloaders["train"]:
    # images: tensor of shape (32, 3, 224, 224)
    # labels: tensor of shape (32,) with values 0-11
    outputs = model(images)
    loss = criterion(outputs, labels)
    ...
```

---

## Common Questions

**Q: Why not use `ImageFolder` from torchvision?**  
A: `ImageFolder` doesn't support filtering by class name. Our `PlantDiseaseDataset` lets us select exactly which 12 classes to use and provides utility methods like `get_class_counts()`.

**Q: Why is hue jitter so small (0.05)?**  
A: Leaf color is diagnostically meaningful — yellowing, browning, and spotting are disease symptoms. Large hue changes could turn a "healthy green" leaf into a "diseased yellow" one, confusing the model.

**Q: Why RandomErasing after Normalize?**  
A: `RandomErasing` replaces pixels with random values. If applied before normalization, the random values would be in [0, 1] range. After normalization, the random values match the normalized distribution, which is more natural for the model.

---

## Best Practices Applied

| Practice | How We Apply It | Why It Matters |
|----------|----------------|----------------|
| **Stratified splitting** | Same class proportions in every split | Prevents rare classes from being under-represented in val/test |
| **Fixed random seed** | `random_state=42` in all splits | Full reproducibility — same split every time |
| **Separate transforms** | Augmentation on train only; deterministic on val/test | Val/test metrics reflect real-world performance, not augmented data |
| **ImageNet normalization** | Mean/std from ImageNet used in all transforms | Required for transfer learning — pretrained weights expect this distribution |
| **Error-safe loading** | `try/except` in `SplitDataset.__getitem__` returns blank fallback | One corrupt image doesn't crash the entire training run |
| **No data leakage** | Split before augmentation; test set never seen during training | Evaluation results are trustworthy |
| **Transform ordering** | Spatial → Color → ToTensor → Normalize → Tensor-space | Each transform operates on the expected input type |
| **Persistent workers** | `persistent_workers=True` in DataLoader | Avoids subprocess respawn overhead between epochs |

---

## Next Steps

| What | Where |
|------|-------|
| How the loaded data trains three models | [Model Training](Model-Training.md) |
| What happens after training | [Evaluation & Metrics](Evaluation-and-Metrics.md) |
| How augmentation maps to assignment requirements | [Task Walkthrough — Part 2](Task-Walkthrough.md) |
| How the project is structured end-to-end | [Architecture Overview](Architecture-Overview.md) |
| How models are deployed after training | [Deployment Guide](Deployment-Guide.md) |
