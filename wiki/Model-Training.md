# 🧠 Model Training

[← Back to Home](Home.md)

This page explains the training strategy in detail: why we use three stages, how layer freezing works, and what each regularization technique does.

---

## Why Transfer Learning?

Training a CNN from scratch requires:
- Millions of images (we have ~18,000)
- Days of GPU time
- Risk of poor performance on small datasets

**Transfer learning** solves this by starting with a model that has already learned to recognize visual patterns from **ImageNet** (1.2 million images, 1000 classes). The lower layers already know edges, textures, and shapes. We only need to teach the model **what plant diseases look like**.

```
ImageNet pretrained layers:           Our custom layers:
┌─────────────┐                      ┌─────────────────────┐
│ Layer 1:    │   Already            │ Classifier Head:    │
│ Edges       │   learned            │ Dropout(0.3)        │
│ Layer 2:    │   from               │ Linear(in, 512)     │
│ Textures    │   1.2M               │ Activation          │
│ Layer 3:    │   images             │ Dropout(0.15)       │
│ Parts       │                      │ Linear(512, 12)     │
│ Layer 4:    │                      │ → 12 disease classes│
│ Objects     │                      └─────────────────────┘
└─────────────┘
    BACKBONE               CUSTOM HEAD (we add this)
```

---

## The Three-Stage Strategy

Instead of training everything at once (which can destroy pretrained knowledge), we **progressively unfreeze** layers:

```
Stage 1                    Stage 2                    Stage 3
FROZEN                     FROZEN                     TRAINABLE (low LR)
┌───────────┐              ┌───────────┐              ┌───────────┐
│ Layer 1   │■■■■■■■■■     │ Layer 1   │■■■■■■■■■     │ Layer 1   │░░░░░░░░░
│ Layer 2   │■■■■■■■■■     │ Layer 2   │■■■■■■■■■     │ Layer 2   │░░░░░░░░░
│ Layer 3   │■■■■■■■■■     │ Layer 3   │░░░░░░░░░     │ Layer 3   │░░░░░░░░░
│ Layer 4   │■■■■■■■■■     │ Layer 4   │░░░░░░░░░     │ Layer 4   │░░░░░░░░░
├───────────┤              ├───────────┤              ├───────────┤
│ Head      │░░░░░░░░░     │ Head      │░░░░░░░░░     │ Head      │░░░░░░░░░
└───────────┘              └───────────┘              └───────────┘
■ = Frozen (not learning)     ░ = Trainable (learning)
LR = 1e-3                 LR = 1e-4                 LR = 1e-5 / 5e-5
5 epochs                   10 epochs                  10 epochs
```

### Stage 1 — Feature Extraction (5 epochs, LR = 1e-3)

**What:** Only the classifier head is trainable. The entire backbone is frozen.

**Why:** The classifier head has random weights (newly created). We need to quickly train it to make reasonable predictions using the fixed pretrained features. A high learning rate (1e-3) is safe because we're only training 2 linear layers.

**Code:**
```python
from src.models.freeze import freeze_backbone

freeze_backbone(model, model_name)  # Freeze all backbone parameters
trainer = Trainer(model, ..., learning_rate=1e-3)
trainer.fit(train_loader, val_loader, num_epochs=5)
```

**Analogy:** Imagine you hired an expert photographer (pretrained backbone). In Stage 1, you're training a new assistant (classifier head) to sort the photos the expert took. The expert doesn't change what they do.

---

### Stage 2 — Adaptation (10 epochs, LR = 1e-4)

**What:** The top backbone layers are unfrozen alongside the classifier head. Lower layers remain frozen.

**Why:** The top backbone layers contain high-level features (object parts, complex patterns). These need to adapt from ImageNet objects to plant disease patterns. A lower learning rate (1e-4) prevents destroying the useful patterns.

**Which layers are unfrozen?**

| Architecture | Unfrozen in Stage 2 |
|-------------|-------------------|
| ResNet-50 | `layer3`, `layer4`, and `fc` (classifier) |
| EfficientNet-B0 | Last 30% of `features` blocks + `classifier` |
| MobileNetV3-Small | Last 30% of `features` blocks + `classifier` |

**Code:**
```python
from src.models.freeze import partial_unfreeze

partial_unfreeze(model, model_name)  # Unfreeze top layers
trainer = Trainer(model, ..., learning_rate=1e-4)
trainer.fit(train_loader, val_loader, num_epochs=10)
```

**Analogy:** Now the expert photographer starts adjusting their technique slightly based on feedback. They still use their core skills, but adapt their style to work better with plant photos.

---

### Stage 3 — Full Refinement (10 epochs, LR = 1e-5 backbone / 5e-5 head)

**What:** All parameters are trainable, but with **differential learning rates**:
- Backbone: very low LR (1e-5) — small, careful adjustments
- Head: higher LR (5e-5) — continue adapting the classifier

**Why:** End-to-end fine-tuning lets the entire model co-adapt. Differential LR ensures the backbone changes slowly (preserving ImageNet knowledge) while the head can still learn more aggressively.

**Code:**
```python
from src.models.freeze import full_unfreeze
from src.models.factory import get_differential_lr_params

full_unfreeze(model)  # All parameters trainable
param_groups = get_differential_lr_params(model, model_name, backbone_lr=1e-5, head_lr=5e-5)
trainer = Trainer(model, ..., param_groups=param_groups)
trainer.fit(train_loader, val_loader, num_epochs=10)
```

**Analogy:** The expert and assistant now work together, both fine-tuning their approach. The expert makes very small adjustments (they're already good), while the assistant continues to improve more rapidly.

---

## Layer Freezing Details

### What does "freezing" mean?

When a parameter is **frozen**, its `requires_grad` is set to `False`. This means:
- No gradient is computed for it during backpropagation
- Its values don't change during training
- It uses less GPU memory (no gradient storage needed)

### Architecture-specific freezing

**ResNet-50:**
```
conv1 + bn1          ← Always frozen in Stage 1 & 2
layer1               ← Always frozen in Stage 1 & 2
layer2               ← Always frozen in Stage 1 & 2
layer3               ← Unfrozen in Stage 2
layer4               ← Unfrozen in Stage 2
fc (classifier head) ← Always trainable
```

**EfficientNet-B0 / MobileNetV3-Small:**
```
features[0] ... features[N*0.7]   ← Frozen in Stage 1 & 2
features[N*0.7] ... features[N]   ← Unfrozen in Stage 2
classifier                         ← Always trainable
```

Where N = total number of feature blocks. "Last 30%" means the highest-level feature extractors.

---

## Regularization Stack

We use **6 complementary regularization techniques** to prevent overfitting:

| Technique | Where Applied | What It Does | Hyperparameter |
|-----------|--------------|-------------|----------------|
| **Data Augmentation** | `transforms.py` | Each epoch sees slightly different versions of images | 9-step pipeline |
| **Weight Decay** | AdamW optimizer | Penalizes large weights, encouraging simpler models | 1e-4 |
| **Label Smoothing** | CrossEntropyLoss | Distributes a small probability (10%) to non-target classes, preventing overconfidence | 0.1 |
| **Dropout** | Classifier head | Randomly zeroes neurons during training, forcing redundant representations | 0.3 (first), 0.15 (second) |
| **Gradient Clipping** | After backprop | Caps gradient magnitude to prevent exploding gradients | max_norm = 1.0 |
| **Early Stopping** | After each epoch | Stops training when validation F1 stops improving | patience = 5 epochs |

### Why so many? Don't they conflict?

No — they complement each other:
- **Augmentation** adds diversity at the input level
- **Dropout** adds diversity at the representation level
- **Weight decay** prevents any single weight from becoming too large
- **Label smoothing** prevents the model from being "100% sure" about anything
- **Gradient clipping** prevents training instability (different purpose than regularization)
- **Early stopping** is the last line of defense against overfitting

---

## Optimizer & Scheduler

### AdamW Optimizer

**What:** Adam with **decoupled weight decay**. Unlike standard Adam + L2, AdamW applies weight decay separately from the gradient update. This is mathematically more correct.

**Parameters:**
- `lr`: varies by stage (1e-3 → 1e-4 → 1e-5)
- `weight_decay`: 1e-4 (constant across stages)
- `betas`: (0.9, 0.999) — default momentum and RMS scaling

### Cosine Annealing Scheduler

**What:** Gradually decreases the learning rate following a cosine curve. Created via the `create_scheduler()` factory function in `src/training/scheduler.py`:

```
LR
 ↑
 │  ╲
 │    ╲
 │      ╲
 │        ╲
 │          ╲
 │            ╲___________
 └───────────────────────→ Epoch
  1    2    3    4    5
```

**Why cosine?** It starts high for fast learning, then smoothly decays for fine-grained convergence. No abrupt drops (unlike StepLR), so training is more stable.

**`eta_min`**: The minimum LR is 1e-7 (essentially zero), reached at the end of each stage.

---

## Mixed Precision Training (AMP)

### What is it?

Automatic Mixed Precision (AMP) uses float16 for most computations and float32 for sensitive operations (like loss computation):

```
float32 (full precision):    32 bits per number
float16 (half precision):    16 bits per number  ← 2x less memory, 2x faster
```

### How it works in our code

```python
# Forward pass in float16 (fast)
with torch.amp.autocast(device_type="cuda", enabled=True):
    outputs = model(images)
    loss = criterion(outputs, labels)

# Backward pass with gradient scaling (prevents underflow)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(...)
scaler.step(optimizer)
scaler.update()
```

### Platform behavior

| Platform | AMP Enabled? | Notes |
|----------|-------------|-------|
| NVIDIA CUDA GPU | ✅ Yes | Full support, ~2x speedup |
| Apple MPS (M1/M2/M3) | ❌ No | Not fully supported; auto-disabled |
| CPU | ❌ No | No benefit on CPU |

---

## Early Stopping

### The `EarlyStopping` class

The `EarlyStopping` class (`src/training/early_stopping.py`) encapsulates patience-based stopping logic:
- **`step(metric)`** — call after each epoch with the current val_f1; returns `True` when patience is exhausted
- **`improved`** property — `True` if the last `step()` improved on the best score (tells the trainer to save a checkpoint)
- **`best_score`** — tracks the highest val_f1 seen so far

### How it works

```
Epoch 1: val_f1 = 0.85  → best! Save checkpoint. patience_counter = 0
Epoch 2: val_f1 = 0.87  → best! Save checkpoint. patience_counter = 0
Epoch 3: val_f1 = 0.86  → no improvement. patience_counter = 1
Epoch 4: val_f1 = 0.84  → no improvement. patience_counter = 2
Epoch 5: val_f1 = 0.85  → no improvement. patience_counter = 3
Epoch 6: val_f1 = 0.83  → no improvement. patience_counter = 4
Epoch 7: val_f1 = 0.84  → no improvement. patience_counter = 5 → STOP!
```

**Key details:**
- Monitors **val_f1** (macro F1 on validation set), NOT accuracy
- Why F1 over accuracy? With imbalanced classes, a model could get 90% accuracy by being good at majority classes while failing on minority ones. F1 penalizes this
- **Patience = 5**: allows temporary dips without premature stopping
- **Checkpoint saved**: only the **best** model is saved, so even after early stopping, we have the peak-performance weights

---

## Checkpointing

### What's saved

```python
checkpoint = {
    "epoch": epoch,                      # When this was the best
    "model_state_dict": model.state_dict(),  # All model weights
    "optimizer_state_dict": optimizer.state_dict(),  # Optimizer state (for resume)
    "val_f1": val_f1,                    # The best F1 score
    "model_name": model_name,            # Architecture name (for reconstruction)
    "num_classes": num_classes,           # Number of output classes
}
torch.save(checkpoint, "models/resnet50_best.pth")
```

### Checkpoint files

| File | Size | Architecture |
|------|------|-------------|
| `resnet50_best.pth` | ~98 MB | ResNet-50 |
| `efficientnet_b0_best.pth` | ~20 MB | EfficientNet-B0 |
| `mobilenetv3_best.pth` | ~10 MB | MobileNetV3-Small |

---

## Training Timeline

For each of the 3 models, the full training consists of:

```
┌────────────────┬──────────────────┬──────────────────┐
│  Stage 1 (5ep) │  Stage 2 (10ep)  │  Stage 3 (10ep)  │
│  Frozen backbone│  Partial unfreeze│  Full fine-tune   │
│  LR = 1e-3     │  LR = 1e-4       │  LR = 1e-5/5e-5  │
└────────────────┴──────────────────┴──────────────────┘
          ← up to 25 epochs per model (with early stopping) →
```

**Total training time estimates:**
- GPU (NVIDIA): ~20-40 min per model
- macOS MPS: ~40-80 min per model
- CPU only: ~2-3 hours per model

---

## Training Curves Interpretation

The notebook generates a 2×2 training curves plot with 4 subplots:

| Subplot | What to Look For |
|---------|-----------------|
| **Train/Val Loss** | Both should decrease. If train loss decreases but val loss increases → overfitting |
| **Train/Val Accuracy** | Both should increase. Val accuracy plateauing is normal |
| **Validation F1** | Should increase and stabilize. This is the metric we checkpoint on |
| **Learning Rate** | Should show 3 distinct segments (one per stage) with cosine decay in each |

Vertical dashed lines in the plot mark **stage boundaries** — you can see the LR jumps when a new stage starts.

---

## Best Practices Applied

| Practice | How We Apply It | Why It Matters |
|----------|----------------|----------------|
| **Progressive unfreezing** | 3 stages: head → top layers → full | Prevents catastrophic forgetting of pretrained knowledge |
| **Differential learning rates** | Backbone LR < Head LR in Stage 3 | Pretrained features need smaller updates than random head weights |
| **Cosine annealing** | LR decays smoothly each stage | No abrupt drops; more stable convergence than StepLR |
| **Mixed precision (AMP)** | float16 forward + float32 loss | 2x memory savings, faster training on CUDA GPUs |
| **F1-based checkpointing** | Save best model on `val_f1`, not accuracy | F1 accounts for class imbalance; accuracy can be misleading |
| **Early stopping on F1** | Patience=5 on `val_f1` | Stops training at peak performance, not after overfitting |
| **Label smoothing** | 0.1 smoothing in CrossEntropyLoss | Model learns calibrated confidence, not overconfident predictions |
| **Gradient clipping** | `max_norm=1.0` | Prevents exploding gradients during full fine-tuning |
| **Weight decay** | AdamW with `weight_decay=1e-4` | Decoupled regularization; avoids interference with Adam's momentum |
| **Reproducibility** | Fixed seed + deterministic mode | Same results every run regardless of hardware |

---

## Next Steps

| What | Where |
|------|-------|
| What happens after training completes | [Evaluation & Metrics](Evaluation-and-Metrics.md) |
| How the training module fits in the full system | [Architecture Overview](Architecture-Overview.md) |
| How trained models are deployed | [Deployment Guide](Deployment-Guide.md) |
| How data is loaded and augmented before training | [Data Pipeline](Data-Pipeline.md) |
| Walk through every requirement | [Task Walkthrough](Task-Walkthrough.md) |
