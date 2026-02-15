"""
Postprocessing for the face detection pipeline.

Responsibility:
    Parse the raw SSD network output tensor into a list of Detection
    objects. Apply confidence thresholding, coordinate un-normalization,
    and boundary clamping.

Non-goals:
    - No drawing, saving, or display logic.
    - No model loading or inference.

Hard-coded:
    - SSD output tensor layout: [1, 1, N, 7] where each row is
      [batch_id, class_id, confidence, x1, y1, x2, y2] with
      coordinates normalized to [0, 1].
"""

from typing import List

import numpy as np

from src.detection import Detection


def postprocess(
    network_output: np.ndarray,
    frame_width: int,
    frame_height: int,
    confidence_threshold: float,
) -> List[Detection]:
    """Parse raw SSD output into a list of Detection objects.

    Args:
        network_output: Raw output from net.forward(), expected shape
                        (1, 1, N, 7).
        frame_width: Original frame width in pixels (for coordinate mapping).
        frame_height: Original frame height in pixels (for coordinate mapping).
        confidence_threshold: Minimum confidence to accept a detection.

    Returns:
        List of Detection objects, sorted by confidence (descending).
        Empty list if no detections meet the threshold.
    """
    detections: List[Detection] = []

    # SSD output shape: (1, 1, num_detections, 7)
    # Each detection: [batch_id, class_id, confidence, x1, y1, x2, y2]
    raw = network_output[0, 0]  # Shape: (N, 7)

    for i in range(raw.shape[0]):
        confidence = float(raw[i, 2])

        if confidence < confidence_threshold:
            continue

        # Un-normalize coordinates from [0, 1] to absolute pixels
        x1 = int(raw[i, 3] * frame_width)
        y1 = int(raw[i, 4] * frame_height)
        x2 = int(raw[i, 5] * frame_width)
        y2 = int(raw[i, 6] * frame_height)

        # Clamp to frame boundaries
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width - 1))
        y2 = max(0, min(y2, frame_height - 1))

        # Skip degenerate boxes
        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(Detection(
            x1=x1, y1=y1, x2=x2, y2=y2,
            confidence=confidence,
        ))

    # Sort by confidence descending for consistent output ordering
    detections.sort(key=lambda d: d.confidence, reverse=True)

    return detections
