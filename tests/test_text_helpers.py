"""Tests for text helper functions."""

import pytest

from src.utils.text_helpers import get_crop_name, shorten_class_name


class TestTextHelpers:
    """Test suite for text formatting utilities."""

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("Tomato___Early_blight", "Early blight"),
            ("Pepper__bell___Bacterial_spot", "Bacterial spot"),
            ("Potato___healthy", "healthy"),
            ("Tomato_Leaf_Mold", "Tomato Leaf Mold"),
        ],
    )
    def test_shorten_class_name(self, name, expected):
        """shorten_class_name should produce the expected short label."""
        assert shorten_class_name(name) == expected

    def test_shorten_truncates_long_names(self):
        """Names longer than 25 chars should be truncated."""
        long_name = "Category__" + "a" * 30
        result = shorten_class_name(long_name)
        assert len(result) <= 25

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("Tomato_Early_blight", "Tomato"),
            ("Potato___healthy", "Potato"),
            ("Pepper__bell___healthy", "Pepper"),
            ("Unknown_class", "Other"),
        ],
    )
    def test_get_crop_name(self, name, expected):
        """get_crop_name should extract the correct crop."""
        assert get_crop_name(name) == expected
