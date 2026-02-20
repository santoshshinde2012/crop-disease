"""
Text helpers for shortening and parsing class names.

Used throughout the project by confusion matrix, predictions,
and distribution plots.
"""


def shorten_class_name(name: str) -> str:
    """Shorten a class name for display.

    ``'Tomato___Early_blight'`` becomes ``'Early blight'``.
    ``'Pepper__bell___Bacterial_spot'`` becomes ``'Bacterial spot'``.

    Args:
        name: Full class directory name.

    Returns:
        Human-readable short label (max 25 chars).
    """
    parts = name.replace("___", "__").split("__")
    short = parts[-1].replace("_", " ")
    return short[:25]


def get_crop_name(class_name: str) -> str:
    """Extract the crop name from a class directory name.

    ``'Tomato_Early_blight'``  -> ``'Tomato'``
    ``'Potato___healthy'``     -> ``'Potato'``
    ``'Pepper__bell___healthy'`` -> ``'Pepper'``

    Args:
        class_name: Full class directory name.

    Returns:
        Crop name string, or ``'Other'`` if unrecognised.
    """
    for crop in ("Tomato", "Potato", "Pepper", "Corn", "Apple"):
        if class_name.startswith(crop):
            return crop
    return "Other"
