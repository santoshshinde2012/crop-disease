"""
Streamlit entry point for Crop Disease Detection.

This file is intentionally slim -- it only wires together the
page configuration and delegates all logic to dedicated modules:

- ``config.py``          -- app-level constants
- ``disease_info.py``    -- disease database and lookup
- ``model_service.py``   -- model loading and inference
- ``ui_components.py``   -- reusable UI rendering functions

Run with: streamlit run app/streamlit_app.py
"""

import logging

import streamlit as st
from PIL import Image

from app.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, PAGE_ICON, PAGE_TITLE
from app.model_service import get_transform, load_model, predict
from app.ui_components import (
    render_analysis_results,
    render_predictions_chart,
    render_sidebar,
    render_upload_placeholder,
)

# ---- Logging (visible in the terminal running Streamlit) ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# Page Configuration (MUST be the first Streamlit call)
# ============================================================
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Model Loading (cached across reruns)
# ============================================================
@st.cache_resource
def _cached_load_model():
    """Wrapper so Streamlit caches the heavy model load."""
    return load_model()


# ---- Initialize model and transform once ----
model, class_mapping, loaded_model_name = _cached_load_model()
transform = get_transform()

if "untrained" in loaded_model_name:
    st.warning(
        "No trained checkpoint found. Using an untrained model for demo purposes."
    )

# ============================================================
# Sidebar
# ============================================================
render_sidebar(model_name=loaded_model_name, num_classes=len(class_mapping))

# ============================================================
# Main Content
# ============================================================
st.title("🌿 Crop Disease Detector")
st.markdown(
    "Upload a clear photo of a plant leaf to identify potential diseases "
    "and get treatment recommendations."
)

uploaded_file = st.file_uploader(
    "Choose a leaf image...",
    type=ALLOWED_EXTENSIONS,
    help="Upload a clear, well-lit photo of a single leaf.",
)

if uploaded_file is not None:
    # ---- File-size guard ----
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(
            f"File too large ({file_size_mb:.1f} MB). "
            f"Please upload an image smaller than {MAX_FILE_SIZE_MB} MB."
        )
        st.stop()

    image = Image.open(uploaded_file).convert("RGB")

    col_image, col_results = st.columns([1, 1])

    with col_image:
        st.subheader("📷 Uploaded Image")
        st.image(image, use_container_width=True)

    with col_results:
        st.subheader("🔍 Analysis Results")
        try:
            with st.spinner("Analyzing leaf..."):
                predictions = predict(image, model, class_mapping, transform)
            render_analysis_results(predictions)
        except Exception as exc:
            logger.exception("Prediction failed")
            st.error(
                f"An error occurred during analysis: {exc}. "
                "Please try again with a different image."
            )

    render_predictions_chart(predictions)

else:
    render_upload_placeholder()
