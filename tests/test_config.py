"""
Tests for the configuration module.
"""

import os
import pytest
from runpy import run_path

from src.config import load_config, AppConfig, ModelConfig

def test_load_defaults():
    """Test loading configuration without any file."""
    config = load_config(None)
    assert isinstance(config, AppConfig)
    assert config.model.backend == "cpu"
    assert config.detection.confidence_threshold == 0.5


def test_validation_failure():
    """Test fail-fast validation."""
    # Create an invalid config state manually to test validation logic
    # (In real usage, _validate is called inside load_config)
    from src.config import _validate, AppConfig, ModelConfig, DetectionConfig
    
    # Invalid confidence
    bad_config = AppConfig(
        detection=DetectionConfig(confidence_threshold=1.5)
    )
    with pytest.raises(ValueError, match="confidence_threshold"):
        _validate(bad_config)

    # Invalid backend
    bad_config = AppConfig(
        model=ModelConfig(backend="invalid")
    )
    with pytest.raises(ValueError, match="backend"):
        _validate(bad_config)


def test_env_override(monkeypatch):
    """Test environment variable overrides."""
    monkeypatch.setenv("FACE_DETECT_DETECTION_CONFIDENCE_THRESHOLD", "0.9")
    monkeypatch.setenv("FACE_DETECT_MODEL_BACKEND", "cuda")
    
    config = load_config(None)
    
    assert config.detection.confidence_threshold == 0.9
    assert config.model.backend == "cuda"
