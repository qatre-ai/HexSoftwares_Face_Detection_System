"""
Tests for the postprocessing module.
"""

import numpy as np
import pytest

from src.postprocessor import postprocess
from src.detection import Detection


def test_postprocess_valid_detection():
    """Test parsing a valid detection tensor."""
    # Synthetic SSD output: [1, 1, 1, 7]
    # [batch, class, conf, x1, y1, x2, y2]
    # High confidence detection covering top-left quarter
    tensor = np.array([[[[0, 1, 0.95, 0.0, 0.0, 0.5, 0.5]]]], dtype=np.float32)
    
    detections = postprocess(
        network_output=tensor,
        frame_width=640,
        frame_height=480,
        confidence_threshold=0.5,
    )
    
    assert len(detections) == 1
    det = detections[0]
    assert det.confidence == pytest.approx(0.95, abs=1e-5)
    assert det.x1 == 0
    assert det.y1 == 0
    assert det.x2 == 320  # 0.5 * 640
    assert det.y2 == 240  # 0.5 * 480


def test_postprocess_confidence_filtering():
    """Test that low-confidence detections are ignored."""
    # Confidence is 0.4, threshold is 0.5
    tensor = np.array([[[[0, 1, 0.4, 0.0, 0.0, 0.5, 0.5]]]], dtype=np.float32)
    
    detections = postprocess(
        network_output=tensor,
        frame_width=640,
        frame_height=480,
        confidence_threshold=0.5,
    )
    assert len(detections) == 0


def test_postprocess_clamping():
    """Test coordinate clamping to frame boundaries."""
    # Coordinates outside [0, 1] range: e.g. -0.1 to 1.2
    tensor = np.array([[[[0, 1, 0.9, -0.1, -0.1, 1.2, 1.2]]]], dtype=np.float32)
    
    detections = postprocess(
        network_output=tensor,
        frame_width=100,
        frame_height=100,
        confidence_threshold=0.5,
    )
    
    assert len(detections) == 1
    det = detections[0]
    assert det.x1 == 0
    assert det.y1 == 0
    assert det.x2 == 99
    assert det.y2 == 99


def test_postprocess_degenerate_box():
    """Test that zero-area or inverted boxes are skipped."""
    # x2 < x1 case
    tensor = np.array([[[[0, 1, 0.9, 0.5, 0.5, 0.4, 0.4]]]], dtype=np.float32)
    
    detections = postprocess(
        network_output=tensor,
        frame_width=100,
        frame_height=100,
        confidence_threshold=0.5,
    )
    assert len(detections) == 0
