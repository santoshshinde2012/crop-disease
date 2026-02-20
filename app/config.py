"""
App-level constants for the Streamlit crop disease detector.

All magic numbers and configurable values live here so the rest
of the app modules remain free of hardcoded literals.
"""

from pathlib import Path

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CLASS_MAPPING_PATH = MODELS_DIR / "class_mapping.json"

# ---- Image ----
IMAGE_SIZE = 224
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]

# ---- Inference ----
CONFIDENCE_THRESHOLD = 0.70
TOP_K = 3

# ---- Model preferences (tried in order) ----
MODEL_PREFERENCES = [
    ("efficientnet_b0", "efficientnet_b0_best.pth"),
    ("resnet50", "resnet50_best.pth"),
    ("mobilenetv3", "mobilenetv3_best.pth"),
]
DEFAULT_MODEL_NAME = "efficientnet_b0"
DEFAULT_DROPOUT = 0.3

# ---- UI ----
PAGE_TITLE = "Crop Disease Detector"
PAGE_ICON = "🌿"

# Colors for the prediction chart
COLOR_HIGH_CONFIDENCE = "#4ECDC4"
COLOR_LOW_CONFIDENCE = "#FF6B6B"
