"""
Visualization for the face detection pipeline.

Responsibility:
    Draw bounding boxes and optional confidence labels onto a frame.
    This is a pure rendering module — it produces an annotated copy
    of the frame and performs no I/O.

Non-goals:
    - No file writing, window management, or display logic.
    - No detection or model logic.
"""

from typing import List

import cv2
import numpy as np

from src.config import VisualizationConfig
from src.detection import Detection

# Hard-coded rendering constants (cosmetic internals, not user-facing)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1
_LABEL_PADDING = 4


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    config: VisualizationConfig,
) -> np.ndarray:
    """Draw bounding boxes and confidence labels onto a frame.

    Args:
        frame: Input BGR image (not modified — a copy is returned).
        detections: List of Detection objects to render.
        config: Visualization parameters (color, thickness, labels).

    Returns:
        A new BGR numpy array with detections drawn. The original
        frame is not modified.
    """
    annotated = frame.copy()

    for det in detections:
        # Bounding box
        cv2.rectangle(
            annotated,
            (det.x1, det.y1),
            (det.x2, det.y2),
            color=config.box_color,
            thickness=config.thickness,
        )

        # Confidence label
        if config.show_confidence:
            label = f"{det.confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, _FONT, _FONT_SCALE, _FONT_THICKNESS
            )

            # Label background (above the box, or below if too close to top)
            label_y = det.y1 - _LABEL_PADDING
            if label_y - text_h - _LABEL_PADDING < 0:
                label_y = det.y2 + text_h + _LABEL_PADDING

            cv2.rectangle(
                annotated,
                (det.x1, label_y - text_h - _LABEL_PADDING),
                (det.x1 + text_w + _LABEL_PADDING, label_y + _LABEL_PADDING),
                color=config.box_color,
                thickness=cv2.FILLED,
            )

            cv2.putText(
                annotated,
                label,
                (det.x1 + _LABEL_PADDING // 2, label_y),
                _FONT,
                _FONT_SCALE,
                (0, 0, 0),  # Black text on colored background
                _FONT_THICKNESS,
                cv2.LINE_AA,
            )

    return annotated


def show_frame(
    frame: np.ndarray,
    detections: List[Detection],
    config: VisualizationConfig,
) -> int:
    """Show annotated frame in a window and return key press.

    Args:
        frame: Input BGR image (not modified).
        detections: List of Detection objects to render.
        config: Visualization parameters.

    Returns:
        The key code (int) pressed during waitKey, or -1 if no key.
    """
    annotated = draw_detections(frame, detections, config)
    cv2.imshow("Face Detection", annotated)
    return cv2.waitKey(1) & 0xFF

