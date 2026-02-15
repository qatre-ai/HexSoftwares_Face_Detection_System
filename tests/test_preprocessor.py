"""
Tests for the preprocessing module.
"""

import numpy as np
import pytest
import cv2

from src.config import ModelConfig
from src.preprocessor import preprocess


def test_preprocess_valid_input():
    """Test standard preprocessing on a valid frame."""
    config = ModelConfig(
        input_size=(300, 300),
        scale_factor=1.0,
        mean_values=(104.0, 177.0, 123.0),
    )
    
    # Create a dummy BGR frame (solid green)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 1] = 255
    
    blob = preprocess(frame, config)
    
    # Assertions
    assert isinstance(blob, np.ndarray)
    assert blob.shape == (1, 3, 300, 300)
    assert blob.dtype == np.float32


def test_preprocess_empty_frame():
    """Test that preprocessing rejects empty frames."""
    config = ModelConfig()
    frame = np.array([])
    
    with pytest.raises(ValueError):
        preprocess(frame, config)


def test_preprocess_none_frame():
    """Test that preprocessing rejects None."""
    config = ModelConfig()
    
    with pytest.raises(ValueError):
        preprocess(None, config)


def test_preprocess_scaling_consistency():
    """Test consistent output scaling."""
    config = ModelConfig(input_size=(100, 100))
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    
    blob = preprocess(frame, config)
    assert blob.shape == (1, 3, 100, 100)
