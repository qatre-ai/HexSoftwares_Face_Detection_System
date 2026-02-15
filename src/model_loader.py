"""
Model loading for the face detection system.

Responsibility:
    Load the DNN model from disk, configure the compute backend,
    and return a ready-to-infer cv2.dnn.Net object.

Non-goals:
    - No preprocessing, inference, or frame-level logic.
    - No automatic model downloading.
    - No fallback to alternative models.

Failure behavior:
    - Missing model files raise FileNotFoundError with the exact
      missing path and expected location.
    - Incompatible backend raises RuntimeError.
"""

import logging
from pathlib import Path

import cv2

from src.config import ModelConfig, get_project_root

logger = logging.getLogger(__name__)


def load_model(config: ModelConfig) -> cv2.dnn.Net:
    """Load and configure the SSD face detection model.

    Args:
        config: ModelConfig containing file paths and backend preference.

    Returns:
        A configured cv2.dnn.Net ready for inference.

    Raises:
        FileNotFoundError: If prototxt or weights file does not exist.
        RuntimeError: If the requested backend is unavailable.
    """
    project_root = get_project_root()

    prototxt = Path(config.prototxt_path)
    weights = Path(config.weights_path)

    # Resolve relative paths against project root
    if not prototxt.is_absolute():
        prototxt = project_root / prototxt
    if not weights.is_absolute():
        weights = project_root / weights

    # Validate file existence — fail fast with actionable messages
    if not prototxt.is_file():
        raise FileNotFoundError(
            f"Model prototxt not found.\n"
            f"  Expected: {prototxt}\n"
            f"  Provide the file or update 'model.prototxt_path' in your config."
        )

    if not weights.is_file():
        raise FileNotFoundError(
            f"Model weights not found.\n"
            f"  Expected: {weights}\n"
            f"  Download the weights file and place it at the path above,\n"
            f"  or update 'model.weights_path' in your config."
        )

    logger.info("Loading model: prototxt=%s, weights=%s", prototxt, weights)
    net = cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))

    # Configure backend and target
    if config.backend == "cuda":
        logger.info("Setting CUDA backend and target.")
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except cv2.error as e:
            raise RuntimeError(
                f"Failed to set CUDA backend. Ensure OpenCV was built with "
                f"CUDA support (opencv-contrib-python or custom build).\n"
                f"  OpenCV error: {e}"
            ) from e
    else:
        logger.info("Using CPU backend.")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    logger.info("Model loaded successfully.")
    return net
