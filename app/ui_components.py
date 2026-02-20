"""
Reusable Streamlit UI components for the crop disease detector.

Single Responsibility: rendering logic only. All data fetching
and inference is handled by ``model_service`` and ``disease_info``.
"""

from typing import List

import plotly.graph_objects as go
import streamlit as st

from .config import (
    COLOR_HIGH_CONFIDENCE,
    COLOR_LOW_CONFIDENCE,
    CONFIDENCE_THRESHOLD,
)
from .disease_info import DiseaseRecord, get_disease_info, get_supported_crops
from .model_service import Prediction


# ---- Sidebar ----

def render_sidebar(model_name: str, num_classes: int) -> None:
    """Render the sidebar with app info and model metadata.

    Args:
        model_name: Display name of the loaded model.
        num_classes: Number of classes the model supports.
    """
    with st.sidebar:
        st.title("🌾 About")
        st.markdown(
            "**Crop Disease Detector** for digital agriculture.\n\n"
            "Upload a photo of a plant leaf to identify potential diseases "
            "and receive treatment recommendations."
        )

        st.divider()

        st.markdown(f"**Model:** {model_name}")
        st.markdown(f"**Classes:** {num_classes}")

        st.divider()

        st.markdown("**Supported Crops:**")
        for crop in get_supported_crops():
            st.markdown(f"- {crop}")

        st.divider()
        st.markdown(
            "⚠️ *This tool is for guidance only. "
            "Always consult a local agronomist for definitive diagnosis.*"
        )


# ---- Analysis Results ----

def render_high_confidence_result(
    info: DiseaseRecord,
    confidence: float,
) -> None:
    """Show detailed treatment card when confidence exceeds the threshold.

    Args:
        info: Disease record with crop, disease, severity, action, product.
        confidence: Top-1 prediction probability (0-1).
    """
    st.success(
        f"**Prediction: {info['disease']}** ({confidence:.1%} confidence)"
    )
    st.markdown(f"**🌱 Crop:** {info['crop']}")
    st.markdown(f"**🦠 Disease:** {info['disease']}")
    st.markdown(f"**⚡ Severity:** {info['severity']}")
    st.markdown("**💊 Recommended Action:**")
    st.info(info["action"])
    st.markdown("**🧪 Product Recommendation:**")
    st.success(info["product"])


def render_low_confidence_result(
    info: DiseaseRecord,
    confidence: float,
) -> None:
    """Show a warning when confidence is below the threshold.

    Args:
        info: Disease record for the best-guess class.
        confidence: Top-1 prediction probability (0-1).
    """
    st.warning(
        f"⚠️ **Low confidence prediction** ({confidence:.1%})\n\n"
        "Please retake the photo with better lighting "
        "and a clear view of the leaf."
    )
    st.markdown(f"Best guess: **{info['disease']}** on **{info['crop']}**")


def render_analysis_results(predictions: List[Prediction]) -> None:
    """Render the analysis results section (right column).

    Delegates to high- or low-confidence renderer based on the threshold.

    Args:
        predictions: Ordered list of ``Prediction`` objects.
    """
    top = predictions[0]
    info = get_disease_info(top.class_name)

    if top.probability >= CONFIDENCE_THRESHOLD:
        render_high_confidence_result(info, top.probability)
    else:
        render_low_confidence_result(info, top.probability)


# ---- Top-K Predictions Chart ----

def _format_display_name(class_name: str) -> str:
    """Convert raw class name to human-readable form.

    ``Tomato__Target_Spot`` -> ``Tomato -- Target Spot``
    """
    return (
        class_name
        .replace("___", " -- ")
        .replace("__", " -- ")
        .replace("_", " ")
    )


def render_predictions_chart(predictions: List[Prediction]) -> None:
    """Render a horizontal bar chart of top-k predictions.

    Args:
        predictions: Ordered list of ``Prediction`` objects.
    """
    st.subheader("📊 Top-3 Predictions")

    fig = go.Figure()

    for pred in reversed(predictions):
        fig.add_trace(
            go.Bar(
                x=[pred.probability],
                y=[_format_display_name(pred.class_name)],
                orientation="h",
                text=[f"{pred.probability:.1%}"],
                textposition="auto",
                marker_color=(
                    COLOR_HIGH_CONFIDENCE
                    if pred.probability >= CONFIDENCE_THRESHOLD
                    else COLOR_LOW_CONFIDENCE
                ),
            )
        )

    fig.update_layout(
        showlegend=False,
        xaxis_title="Confidence",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        height=200,
        margin=dict(l=0, r=0, t=10, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---- Placeholder ----

def render_upload_placeholder() -> None:
    """Show instructions when no image has been uploaded yet."""
    st.markdown("---")
    st.markdown(
        "### 👆 Upload an image to get started\n\n"
        "**Tips for best results:**\n"
        "- Use a clear, well-lit photo\n"
        "- Focus on a single leaf\n"
        "- Include both healthy and affected areas\n"
        "- Avoid shadows and glare"
    )
