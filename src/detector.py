"""
Detector — the single public API for face detection.

This module is the ONLY intended programmatic entry point for consumers
of the face detection library. All other modules are internal.

Public contract:
    Detector.detect(frame: np.ndarray) -> list[Detection]

Constraints:
    - Input must be a BGR numpy array (as returned by OpenCV).
    - The method is stateless per call and deterministic.
    - Thread-safety is not guaranteed (single-threaded design).

Non-goals:
    - No file reading, camera access, or I/O of any kind.
    - No visualization or output writing.
    - No tracking or temporal state.
"""

import logging
from typing import List, Optional

import numpy as np

from src.config import AppConfig, load_config
from src.detection import Detection
from src.model_loader import load_model
from src.preprocessor import preprocess
from src.postprocessor import postprocess

logger = logging.getLogger(__name__)


class Detector:
    """Face detector using SSD-ResNet10 via OpenCV DNN.

    This is the single public API for the face detection system.
    All internal modules (preprocessor, postprocessor, model_loader)
    are wired together here and should not be used directly.

    Usage:
        detector = Detector()                      # Uses safe defaults
        detector = Detector(config=my_config)       # Custom config
        detections = detector.detect(frame)         # BGR numpy array

    The constructor loads the model once. Subsequent detect() calls
    reuse the loaded network — there is no per-frame setup cost
    beyond preprocessing and inference.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """Initialize the detector and load the model.

        Args:
            config: Application configuration. If None, safe defaults
                    are used (no config file required).

        Raises:
            FileNotFoundError: If model files are missing.
            RuntimeError: If the requested backend is unavailable.
            ValueError: If configuration values are invalid.
        """
        if config is None:
            config = load_config()

        self._config = config
        self._net = load_model(config.model)

        logger.info(
            "Detector initialized (backend=%s, confidence_threshold=%.2f)",
            config.model.backend,
            config.detection.confidence_threshold,
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect faces in a single BGR frame.

        Args:
            frame: A BGR image as a numpy array with shape (H, W, 3)
                   and dtype uint8. This is the standard format returned
                   by cv2.imread() and cv2.VideoCapture.read().

        Returns:
            A list of Detection objects, sorted by confidence (descending).
            Returns an empty list if no faces are detected.

        Raises:
            TypeError: If frame is not a numpy ndarray.
            ValueError: If frame has incorrect shape or is empty.
        """
        self._validate_frame(frame)

        # Preprocess: frame → blob
        blob = preprocess(frame, self._config.model)

        # Inference
        self._net.setInput(blob)
        output = self._net.forward()

        # Postprocess: raw output → Detection list
        h, w = frame.shape[:2]
        detections = postprocess(
            network_output=output,
            frame_width=w,
            frame_height=h,
            confidence_threshold=self._config.detection.confidence_threshold,
        )

        return detections

    @property
    def config(self) -> AppConfig:
        """Return the active configuration (read-only)."""
        return self._config

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        """Validate that the input frame meets the API contract.

        Raises:
            TypeError: If frame is not a numpy ndarray.
            ValueError: If frame is empty or has wrong dimensions.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError(
                f"Expected frame to be a numpy ndarray, "
                f"got {type(frame).__name__}. "
                f"Use cv2.imread() or VideoCapture.read() to obtain frames."
            )

        if frame.size == 0:
            raise ValueError(
                "Frame is empty (zero size). "
                "Ensure the input source is providing valid frames."
            )

        if frame.ndim != 3:
            raise ValueError(
                f"Expected a 3-dimensional frame (H, W, C), "
                f"got {frame.ndim} dimensions with shape {frame.shape}. "
                f"Grayscale images must be converted to BGR first."
            )

        if frame.shape[2] != 3:
            raise ValueError(
                f"Expected 3 channels (BGR), got {frame.shape[2]} channels. "
                f"Input must be a BGR image as returned by OpenCV."
            )
