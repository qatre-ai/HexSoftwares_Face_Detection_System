"""
Preprocessing for the face detection pipeline.

Responsibility:
    Convert a raw BGR frame (numpy array) into a 4D DNN-compatible
    input blob using cv2.dnn.blobFromImage.

Non-goals:
    - No frame acquisition or I/O.
    - No inference or coordinate mapping.
    - No model-awareness beyond the blob parameters.

Hard-coded:
    - Channel order is BGR (mandated by the Caffe model).
    - swapRB is False (input is already BGR from OpenCV).
"""

import numpy as np
import cv2

from src.config import ModelConfig


def preprocess(frame: np.ndarray, config: ModelConfig) -> np.ndarray:
    """Convert a raw BGR frame into a DNN input blob.

    Args:
        frame: Input image as a BGR numpy array (H, W, 3).
        config: ModelConfig providing input_size, scale_factor, and mean_values.

    Returns:
        A 4D numpy array of shape (1, 3, H, W) with dtype float32,
        ready to be passed to net.setInput().

    Raises:
        ValueError: If the frame is empty or has unexpected dimensions.
    """
    if frame is None or frame.size == 0:
        raise ValueError(
            "Cannot preprocess an empty frame. "
            "Ensure the input source is providing valid frames."
        )

    blob = cv2.dnn.blobFromImage(
        image=frame,
        scalefactor=config.scale_factor,
        size=config.input_size,
        mean=config.mean_values,
        swapRB=False,   # Hard-coded: input is BGR, model expects BGR
        crop=False,
    )

    return blob
