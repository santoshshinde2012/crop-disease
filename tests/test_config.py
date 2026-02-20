"""Tests for configuration validation."""

import pytest
from src.config import Config, DataConfig, ModelConfig, TrainConfig


class TestConfig:
    """Test suite for Config dataclass validation."""

    def test_default_config_valid(self):
        """Default Config should pass validation."""
        cfg = Config()
        assert cfg.model.num_classes == 12
        assert len(cfg.data.selected_classes) == 12

    def test_num_classes_mismatch_raises(self):
        """Mismatched num_classes and selected_classes should raise ValueError."""
        with pytest.raises(ValueError, match="num_classes"):
            Config(model=ModelConfig(num_classes=5))

    def test_split_ratios_invalid_sum_raises(self):
        """Split ratios not summing to 1.0 should raise ValueError."""
        bad_ratios = {"train": 0.5, "val": 0.3, "test": 0.3}
        with pytest.raises(ValueError, match="sum to 1.0"):
            Config(data=DataConfig(split_ratios=bad_ratios))

    def test_split_ratios_valid(self):
        """Custom split ratios summing to 1.0 should pass."""
        ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
        cfg = Config(data=DataConfig(split_ratios=ratios))
        assert abs(sum(cfg.data.split_ratios.values()) - 1.0) < 1e-6

    def test_default_image_size(self):
        """Default image size should be 224."""
        cfg = Config()
        assert cfg.data.image_size == 224

    def test_default_training_stages(self):
        """Three training stages should have correct LR ordering."""
        cfg = Config()
        assert cfg.train.stage1_lr > cfg.train.stage2_lr
        assert cfg.train.stage2_lr > cfg.train.stage3_lr

    def test_checkpoint_dir_is_path(self):
        """Checkpoint dir should be a Path object."""
        from pathlib import Path
        cfg = Config()
        assert isinstance(cfg.train.checkpoint_dir, Path)
