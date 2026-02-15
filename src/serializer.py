"""
Serialization for the face detection pipeline.

Responsibility:
    Export detection results to structured file formats (JSON, CSV)
    for downstream consumption or offline analysis.

Non-goals:
    - No rendering, display, or detection logic.
    - No streaming output — writes complete files on finalize.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

from src.detection import Detection

logger = logging.getLogger(__name__)


def save_json(
    detections_by_frame: Dict[int, List[Detection]],
    output_path: str,
) -> None:
    """Export all detections to a JSON file.

    Output schema:
        {
            "frames": [
                {
                    "frame_id": 0,
                    "detections": [
                        {"x1": ..., "y1": ..., "x2": ..., "y2": ..., "confidence": ...}
                    ]
                }
            ],
            "total_frames": N,
            "total_detections": M
        }

    Args:
        detections_by_frame: Mapping of frame_id → list of Detection objects.
        output_path: Path to the output JSON file.

    Raises:
        OSError: If the output path is not writable.
    """
    _ensure_parent_dir(output_path)

    frames = []
    total_detections = 0

    for frame_id in sorted(detections_by_frame.keys()):
        dets = detections_by_frame[frame_id]
        total_detections += len(dets)
        frames.append({
            "frame_id": frame_id,
            "detections": [d.to_dict() for d in dets],
        })

    payload = {
        "frames": frames,
        "total_frames": len(frames),
        "total_detections": total_detections,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        "JSON output saved: %s (%d frames, %d detections)",
        output_path, len(frames), total_detections,
    )


def save_csv(
    detections_by_frame: Dict[int, List[Detection]],
    output_path: str,
) -> None:
    """Export all detections to a CSV file.

    Columns: frame_id, x1, y1, x2, y2, confidence

    Args:
        detections_by_frame: Mapping of frame_id → list of Detection objects.
        output_path: Path to the output CSV file.

    Raises:
        OSError: If the output path is not writable.
    """
    _ensure_parent_dir(output_path)

    fieldnames = ["frame_id", "x1", "y1", "x2", "y2", "confidence"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total = 0
        for frame_id in sorted(detections_by_frame.keys()):
            for det in detections_by_frame[frame_id]:
                writer.writerow({
                    "frame_id": frame_id,
                    **det.to_dict(),
                })
                total += 1

    logger.info("CSV output saved: %s (%d rows)", output_path, total)


def _ensure_parent_dir(path: str) -> None:
    """Create parent directories if they don't exist."""
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
