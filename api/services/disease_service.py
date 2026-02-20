"""Disease info enrichment - maps class names to details and treatments."""

from typing import Dict

from app.disease_info import DEFAULT_INFO, DISEASE_INFO


class DiseaseInfoLookupService:
    """In-memory disease info lookup."""

    def enrich(self, class_name: str) -> Dict[str, str]:
        """Look up disease metadata for a predicted class name.

        Args:
            class_name: Raw model class name (e.g. ``Tomato_Early_blight``).

        Returns:
            Dict with keys: crop, disease, severity, action, product.
        """
        return DISEASE_INFO.get(class_name, DEFAULT_INFO)
