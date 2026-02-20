"""
Disease information database for crop disease detection.

Maps model class names to human-readable disease details,
severity levels, and product recommendations.

Single Responsibility: only disease metadata lives here.
"""

from typing import Dict


# Type alias for a single disease record.
DiseaseRecord = Dict[str, str]

# ---- Disease Database ----
DISEASE_INFO: Dict[str, DiseaseRecord] = {
    "Pepper__bell___Bacterial_spot": {
        "crop": "Pepper (Bell)",
        "disease": "Bacterial Spot",
        "severity": "Moderate to High",
        "action": (
            "Remove infected leaves. Apply copper-based bactericide. "
            "Ensure proper spacing for airflow."
        ),
        "product": "Ridomil Gold or copper hydroxide spray",
    },
    "Pepper__bell___healthy": {
        "crop": "Pepper (Bell)",
        "disease": "Healthy",
        "severity": "None",
        "action": "No treatment needed. Continue regular monitoring and preventive care.",
        "product": "N/A -- Preventive: Revus for protection",
    },
    "Potato___Early_blight": {
        "crop": "Potato",
        "disease": "Early Blight",
        "severity": "Moderate",
        "action": (
            "Apply fungicide at first sign. Remove lower infected leaves. "
            "Ensure proper irrigation."
        ),
        "product": "Score (Difenoconazole) or Mancozeb",
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "disease": "Late Blight",
        "severity": "High -- Urgent",
        "action": (
            "Apply systemic fungicide immediately. Destroy severely infected plants. "
            "Monitor neighboring fields."
        ),
        "product": "Ridomil Gold MZ (Metalaxyl-M + Mancozeb)",
    },
    "Potato___healthy": {
        "crop": "Potato",
        "disease": "Healthy",
        "severity": "None",
        "action": "No treatment needed. Continue regular monitoring.",
        "product": "N/A -- Preventive: Revus for protection",
    },
    "Tomato_Bacterial_spot": {
        "crop": "Tomato",
        "disease": "Bacterial Spot",
        "severity": "Moderate",
        "action": (
            "Remove infected plant material. Apply copper fungicide. "
            "Avoid overhead irrigation."
        ),
        "product": "Copper hydroxide + Actigard (plant defense activator)",
    },
    "Tomato_Early_blight": {
        "crop": "Tomato",
        "disease": "Early Blight",
        "severity": "Moderate",
        "action": (
            "Apply fungicide preventively and curatively. Mulch to prevent splash. "
            "Rotate crops."
        ),
        "product": "Score (Difenoconazole)",
    },
    "Tomato_Late_blight": {
        "crop": "Tomato",
        "disease": "Late Blight",
        "severity": "High -- Urgent",
        "action": (
            "Treat immediately with systemic fungicide. Remove infected foliage. "
            "Alert nearby farms."
        ),
        "product": "Ridomil Gold MZ or Revus",
    },
    "Tomato_Leaf_Mold": {
        "crop": "Tomato",
        "disease": "Leaf Mold",
        "severity": "Low to Moderate",
        "action": (
            "Improve air circulation. Reduce humidity in greenhouse. "
            "Apply fungicide if severe."
        ),
        "product": "Bravo (Chlorothalonil) or Score",
    },
    "Tomato__Target_Spot": {
        "crop": "Tomato",
        "disease": "Target Spot",
        "severity": "Moderate",
        "action": (
            "Apply fungicide at early onset. Prune lower canopy for airflow. "
            "Avoid leaf wetness."
        ),
        "product": "Quadris (Azoxystrobin) or Score",
    },
    "Tomato_Septoria_leaf_spot": {
        "crop": "Tomato",
        "disease": "Septoria Leaf Spot",
        "severity": "Moderate",
        "action": (
            "Remove infected lower leaves. Apply fungicide. "
            "Mulch around plants. Rotate crops."
        ),
        "product": "Bravo + Score tank mix",
    },
    "Tomato_healthy": {
        "crop": "Tomato",
        "disease": "Healthy",
        "severity": "None",
        "action": (
            "No treatment needed. Continue regular monitoring "
            "and preventive spraying schedule."
        ),
        "product": "N/A -- Preventive: Revus Top for protection",
    },
}

# Fallback for class names not in the database.
DEFAULT_INFO: DiseaseRecord = {
    "crop": "Unknown",
    "disease": "Unknown Disease",
    "severity": "Unknown",
    "action": "Consult a local agronomist for diagnosis and treatment plan.",
    "product": "Contact agricultural supplier for recommendation",
}


def get_disease_info(class_name: str) -> DiseaseRecord:
    """Look up disease info for a predicted class name.

    Args:
        class_name: Raw model class name (e.g. ``Tomato_Early_blight``).

    Returns:
        Disease record dict with keys: crop, disease, severity, action, product.
    """
    return DISEASE_INFO.get(class_name, DEFAULT_INFO)


def get_supported_crops() -> list[str]:
    """Return a sorted list of unique crop names in the database."""
    return sorted({info["crop"] for info in DISEASE_INFO.values()})
