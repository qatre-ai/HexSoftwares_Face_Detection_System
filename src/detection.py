"""
Detection data transfer object.

This module defines the Detection dataclass — the single output type
returned by Detector.detect(). It is intentionally minimal: a frozen,
serializable container with no behavior beyond data access.

Non-goals:
    - No rendering logic.
    - No file I/O.
    - No coordinate transformation methods (that belongs in postprocessor).
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Detection:
    """A single detected face with bounding box and confidence score.

    Attributes:
        x1: Top-left x coordinate (absolute pixels).
        y1: Top-left y coordinate (absolute pixels).
        x2: Bottom-right x coordinate (absolute pixels).
        y2: Bottom-right y coordinate (absolute pixels).
        confidence: Detection confidence score in [0.0, 1.0].

    All coordinates are in absolute pixel values relative to the
    original input frame dimensions.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialization."""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": round(self.confidence, 4),
        }

    @property
    def width(self) -> int:
        """Bounding box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Bounding box height in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Bounding box area in pixels."""
        return self.width * self.height
