"""Evaluation metrics, confusion matrices, and model profiling."""

from .metrics import (
    compute_predictions,
    generate_classification_report,
    compute_summary_metrics,
)
from .confusion import plot_confusion_matrix
from .predictions import get_prediction_examples, plot_prediction_grid
from .profiler import profile_model

__all__ = [
    "compute_predictions",
    "generate_classification_report",
    "compute_summary_metrics",
    "plot_confusion_matrix",
    "get_prediction_examples",
    "plot_prediction_grid",
    "profile_model",
]
