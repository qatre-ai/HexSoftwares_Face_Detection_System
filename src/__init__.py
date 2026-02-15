"""
Face Detection — Production face detection library using OpenCV DNN.

Public API:
    - Detector: The single entry point for face detection.
    - Detection: Data transfer object representing a detected face.

All other modules in this package are internal implementation details
and should not be imported directly by consumers.

Usage:
    from src import Detector, Detection

    detector = Detector()
    detections = detector.detect(frame)
"""

from src.detection import Detection
from src.detector import Detector

__all__ = ["Detector", "Detection"]
