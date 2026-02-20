# 📊 Evaluation & Metrics

[← Back to Home](Home.md)

This page explains every evaluation technique used in the project — what each metric means, how to read the confusion matrix, and how the business recommendation was derived.

---

## Evaluation Pipeline Overview

After training completes for all three models, we evaluate them on the **test set** (15% of data, never seen during training):

```
Trained model (.pth checkpoint)
        │
        ▼
Load best checkpoint (highest val_f1)
        │
        ▼
compute_predictions()  ← run model on entire test set
        │
        ├── Classification Report (per-class precision/recall/F1)
        ├── Summary Metrics (accuracy, F1 macro, F1 weighted)
        ├── Confusion Matrix (heatmap)
        ├── Correct/Incorrect Prediction Grids
        └── Model Profiling (latency, size)
```

---

## Metrics Explained

### Accuracy

$$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total predictions}}$$

**What it tells you:** Overall, how often is the model right?

**Limitation:** With imbalanced classes, a model could get 85% accuracy by always predicting "Tomato_healthy" (the largest class). That's why we also use F1.

---

### Precision, Recall, and F1 Score

For each class:

$$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$

> "Of all images the model predicted as Early Blight, how many actually were Early Blight?"

$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

> "Of all actual Early Blight images, how many did the model correctly identify?"

$$F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

> The harmonic mean — balances precision and recall. Only high when BOTH are high.

### F1 Averaging Methods

| Method | Calculation | When to Use |
|--------|------------|-------------|
| **Macro F1** | Average F1 across all classes (each class weighted equally) | Our primary metric — treats minority classes as important as majority ones |
| **Weighted F1** | Average F1 weighted by class support (sample count) | Reflects overall performance accounting for class sizes |

**Why we monitor Macro F1:**  
A rare disease with only 152 images (Potato Healthy) matters just as much as a common one with 2,127 images (Tomato Bacterial Spot). Macro F1 ensures we don't ignore the small classes.

---

### Classification Report

The `generate_classification_report()` function produces a per-class table:

```
                              precision    recall  f1-score   support

Pepper bell Bacterial spot      0.9850    0.9900    0.9875       200
Pepper bell healthy             0.9950    0.9900    0.9925       200
Potato Early blight             0.9750    0.9800    0.9775       200
...
                                                              
                   accuracy                         0.9850      2700
                  macro avg      0.9810    0.9830    0.9820      2700
               weighted avg      0.9850    0.9850    0.9850      2700
```

**How to read each column:**
- **Precision**: How trustworthy is a positive prediction? High = few false alarms
- **Recall**: How thorough is detection? High = few missed cases
- **F1-score**: Balanced view of precision and recall
- **Support**: Number of test images in this class

---

## Confusion Matrix

### What it shows

The confusion matrix is a 12×12 grid where:
- **Rows** = true classes (what the image actually is)
- **Columns** = predicted classes (what the model said)
- **Diagonal** = correct predictions
- **Off-diagonal** = mistakes

### Normalization

Our confusion matrices are **row-normalized** (divide each row by its sum):

$$\text{CM}_{ij}^{norm} = \frac{\text{CM}_{ij}}{\sum_k \text{CM}_{ik}}$$

This means each cell shows the **recall** for that true→predicted pair:
- Diagonal values = per-class recall (how many of class X were correctly identified)
- Off-diagonal values = confusion rate (how often class X was confused with class Y)

### How to read it

```
                     Predicted
              │ Early  │ Late   │ Healthy │
     ─────────┼────────┼────────┼─────────│
True Early    │  0.95  │  0.03  │  0.02   │  ← 95% correct, 3% confused with Late
     Late     │  0.02  │  0.96  │  0.02   │  ← 96% correct
     Healthy  │  0.01  │  0.01  │  0.98   │  ← 98% correct
```

**What to look for:**
1. **High diagonal values** (>0.90) = good per-class performance
2. **Clusters of off-diagonal values** = common confusions
3. **Asymmetric confusion** = model mistakes one direction more than the other

### Expected confusion patterns

| Confused Pair | Why |
|--------------|-----|
| Tomato Early Blight ↔ Potato Early Blight | Same disease on different crops — similar lesion patterns |
| Tomato Septoria Leaf Spot ↔ Tomato Target Spot | Both show spotting patterns on tomato leaves |
| Any Healthy ↔ Early-stage Disease | Mild infections closely resemble healthy leaves |

### Class name shortening

Full names like `Tomato_Spider_mites_Two_spotted_spider_mite` are too long for axis labels. The `shorten_class_name()` function (in `src/utils/text_helpers.py`) extracts the disease part:
- `Tomato___Early_blight` → `Early blight`
- `Pepper__bell___Bacterial_spot` → `Bacterial spot`
- `Tomato__Tomato_YellowLeaf__Curl_Virus` → `Tomato YellowLeaf`

---

## Prediction Visualization

### Correct Predictions Grid

Shows 5 test images the model got right, each with:
- The raw leaf image
- **Green title**: `✓ True: Early blight | Pred: Early blight (97.3%)`
- High confidence scores confirm the model is confident when correct

### Incorrect Predictions Grid

Shows 5 test images the model got wrong, each with:
- The raw leaf image
- **Red title**: `✗ True: Septoria | Pred: Target Spot (62.1%)`
- Lower confidence scores on incorrect predictions (supporting confidence thresholding)

### What to observe

1. **Confidence on errors** — incorrect predictions tend to have lower confidence (60-80%) vs. correct ones (90-99%). This validates the 70% confidence threshold strategy
2. **Visual similarity** — looking at the incorrect predictions, you'll see the true and predicted classes often look genuinely similar
3. **Edge cases** — images with unusual angles, lighting, or early-stage disease are most error-prone

---

## Model Profiling

### What we measure

The `profile_model()` function measures three things:

| Metric | How It's Measured | Why It Matters |
|--------|------------------|----------------|
| **Model Size (MB)** | Save `state_dict` to disk, check file size | Determines if model fits on mobile device |
| **CPU Latency (ms)** | 100 forward passes with warm-up, measure mean + P95 | On-device inference speed without GPU |
| **GPU Latency (ms)** | CUDA Events for precise GPU timing | Data center inference speed |

### Why P95 (95th percentile)?

Mean latency hides worst-case scenarios. P95 shows "in 95% of cases, inference takes less than X ms." This is critical for user experience — a mobile app that's fast on average but occasionally hangs for 2 seconds feels broken.

### Warm-up iterations

The first ~10 inferences are slower due to:
- Model loading into GPU cache
- JIT compilation / optimization
- Memory allocation

We run 10 warm-up iterations (discarded) before timing starts.

---

## Model Comparison

### Comparison Table

After profiling all three models, we create a comparison table:

| Metric | ResNet-50 | EfficientNet-B0 | MobileNetV3-Small |
|--------|-----------|-----------------|-------------------|
| Parameters | ~25.6M | ~5.3M | ~2.5M |
| Model Size | ~98 MB | ~20 MB | ~10 MB |
| Test Accuracy | Highest | Competitive | Lowest |
| F1 Macro | Highest | Competitive | Lowest |
| CPU Latency | Slowest | Mid | Fastest |

### Comparison Charts

Two scatter plots help visualize the tradeoffs:

1. **Accuracy vs. Latency** — shows the speed-accuracy tradeoff
   - Top-left is ideal (high accuracy, low latency)
   - ResNet-50 is top-right (accurate but slow)
   - MobileNetV3 is bottom-left (fast but less accurate)
   - EfficientNet-B0 is the compromise

2. **Accuracy vs. Model Size** — shows the size-accuracy tradeoff
   - Top-left is ideal (high accuracy, small model)
   - EfficientNet-B0 achieves near-ResNet accuracy at 5× smaller size

---

## Business Recommendation

### Decision Framework

For a mobile crop disease detection app, the deployment model must balance:

| Requirement | Importance | Why |
|------------|-----------|-----|
| **Accuracy** | Critical | Missed disease = crop loss, farmer trust erosion |
| **Model Size** | High | Must work on low-end smartphones in rural areas |
| **Latency** | High | Farmers expect instant results |
| **Offline Support** | High | No reliable internet in many agricultural regions |

### Recommendation: EfficientNet-B0

$$\text{Score} = \frac{\text{Accuracy} \times \text{F1}}{\text{Size (MB)} \times \text{Latency (ms)}}$$

EfficientNet-B0 maximizes this score by being "good enough" on accuracy while being 5× smaller than ResNet-50.

### Deployment Path

```
PyTorch (.pth)  →  ONNX (.onnx)  →  TFLite (.tflite, INT8)
    ~20 MB            ~20 MB              ~5 MB
```

INT8 quantization further reduces size from ~20 MB to ~5 MB with minimal accuracy loss (<0.5%).

### Confidence Thresholding

| Confidence | Action |
|-----------|--------|
| ≥ 70% | Show prediction with treatment recommendation |
| < 70% | Show: "Low confidence. Retake photo with better lighting." |

This prevents the worst UX failure: a confident wrong diagnosis leading to incorrect (and potentially harmful) crop treatment.

### Known Limitations

1. **Lab-to-field domain gap**: Model trained on clean lab images; real field photos are messier
2. **Single disease**: Cannot detect co-infections (multiple diseases on one leaf)
3. **3 crops only**: Tomato, Potato, Pepper. Same architecture scales to more
4. **No severity grading**: Detects what disease it is, not how advanced it is

---

## Output Files

After evaluation, these files are generated (in the runtime `models/` and `outputs/` directories):

| File | Content |
|------|---------|
| `outputs/confusion_matrix_resnet50.png` | 12×12 confusion matrix for ResNet-50 |
| `outputs/confusion_matrix_efficientnet_b0.png` | 12×12 confusion matrix for EfficientNet-B0 |
| `outputs/confusion_matrix_mobilenetv3.png` | 12×12 confusion matrix for MobileNetV3 |
| `outputs/correct_predictions.png` | Grid of correct test predictions |
| `outputs/incorrect_predictions.png` | Grid of incorrect test predictions |
| `outputs/model_comparison.png` | Scatter plots comparing all 3 models |
| `models/class_mapping.json` | Class index ↔ name mapping |
| `models/training_config.json` | Full config + training results |

---

## Best Practices Applied

| Practice | How We Apply It | Why It Matters |
|----------|----------------|----------------|
| **Macro F1 as primary metric** | Equal weight to all classes | Fair evaluation even with 14:1 class imbalance |
| **Row-normalized confusion matrix** | Each row sums to 1.0 | Shows recall per class, not biased by class size |
| **P95 latency measurement** | 95th percentile inference time | Captures worst-case user experience, not just average |
| **Warm-up iterations** | 10 discarded runs before timing | Excludes JIT/cache effects that don't reflect real usage |
| **Confidence thresholding** | 70% cutoff for predictions | Prevents misdiagnosis; incorrect predictions have lower confidence |
| **Hold-out test set** | 15% of data never seen during training | Unbiased final evaluation |
| **Multi-model comparison** | 3 architectures profiled on same test set | Data-driven model selection (accuracy vs. size vs. speed) |
| **Per-class analysis** | Classification report per class | Identifies weak spots and confused pairs |

---

## Next Steps

| What | Where |
|------|-------|
| See the evaluated model as a web app | [Streamlit App](Streamlit-App.md) |
| Map evaluation back to assignment requirements | [Task Walkthrough](Task-Walkthrough.md) |
| Deploy the recommended model to production | [Deployment Guide](Deployment-Guide.md) |
| Understand how training produced these results | [Model Training](Model-Training.md) |
| See the overall project structure | [Architecture Overview](Architecture-Overview.md) |
