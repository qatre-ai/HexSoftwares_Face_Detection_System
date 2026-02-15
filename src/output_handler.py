"""
Output handling for the face detection pipeline.

Responsibility:
    Route detection results to configured output sinks:
    display window, saved images, video files, JSON, or CSV.
    Supports multiple orthogonal outputs simultaneously.

Non-goals:
    - No detection logic.
    - No input acquisition.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import numpy as np

from src.config import AppConfig, get_project_root
from src.detection import Detection
from src.serializer import save_csv, save_json
from src.visualizer import draw_detections, show_frame

logger = logging.getLogger(__name__)


class OutputHandler:
    """Routes detection results to configured output sinks.

    Supports orthogonal outputs - multiple modes can be active simultaneously:
        - 'display': Show annotated frames in an OpenCV window.
        - 'save_image': Write annotated image/frames to files.
        - 'save_video': Write annotated frames to a video file.
        - 'save_json': Accumulate detections, write JSON on finalize.
        - 'save_csv': Accumulate detections, write CSV on finalize.

    Usage:
        handler = OutputHandler(config)
        handler.process_frame(frame_id, frame, detections)
        ...
        handler.finalize()  # Flush any buffered output
    """

    def __init__(self, config: AppConfig) -> None:
        """Initialize the output handler.

        Args:
            config: Application configuration (output mode, paths, vis params).
        """
        self._config = config
        self._video_writer: Optional[cv2.VideoWriter] = None
        
        # Parse output modes (comma-separated for multiple outputs)
        mode_str = config.output.mode
        self._modes: Set[str] = set(m.strip() for m in mode_str.split(','))
        
        # Buffer for serialization modes
        self._detections_buffer: Dict[int, List[Detection]] = {}

        # Resolve output path
        save_path = Path(config.output.save_path)
        if not save_path.is_absolute():
            save_path = get_project_root() / save_path
        self._save_path = save_path
        
        # Create output directory if saving files
        if self._modes & {'save_image', 'save_video', 'save_json', 'save_csv'}:
            self._save_path.mkdir(parents=True, exist_ok=True)

        logger.info("OutputHandler initialized: modes=%s, save_path=%s", 
                   self._modes, self._save_path)

    def process_frame(
        self,
        frame_id: int,
        frame: np.ndarray,
        detections: List[Detection],
    ) -> bool:
        """Process a single frame's detections through the output pipeline.

        Args:
            frame_id: Frame index.
            frame: Original BGR frame.
            detections: List of Detection objects for this frame.

        Returns:
            True to continue processing, False to signal the caller
            should stop (e.g., user pressed 'q' in display mode).
        """
        should_continue = True
        
        # Display mode
        if 'display' in self._modes:
            if not self._handle_display(frame, detections):
                should_continue = False
        
        # Save annotated image
        if 'save_image' in self._modes:
            self._handle_save_image(frame_id, frame, detections)
        
        # Save to video
        if 'save_video' in self._modes:
            self._handle_save_video(frame, detections)
        
        # Buffer for JSON/CSV
        if 'save_json' in self._modes or 'save_csv' in self._modes:
            self._detections_buffer[frame_id] = detections
        
        return should_continue

    def _handle_display(
        self,
        frame: np.ndarray,
        detections: List[Detection],
    ) -> bool:
        """Show annotated frame in a window. Returns False on quit key."""
        key = show_frame(frame, detections, self._config.visualization)
        
        if key == ord("q") or key == 27:  # 'q' or ESC
            logger.info("Quit signal received (key press).")
            return False

        return True
    
    def _handle_save_image(
        self,
        frame_id: int,
        frame: np.ndarray,
        detections: List[Detection],
    ) -> None:
        """Save annotated frame as an image file."""
        annotated = draw_detections(frame, detections, self._config.visualization)
        output_file = self._save_path / f"frame_{frame_id:06d}.jpg"
        cv2.imwrite(str(output_file), annotated)
        logger.debug("Saved frame %d to %s", frame_id, output_file)

    def _handle_save_video(
        self,
        frame: np.ndarray,
        detections: List[Detection],
    ) -> None:
        """Write annotated frame to the video writer."""
        annotated = draw_detections(frame, detections, self._config.visualization)

        if self._video_writer is None:
            output_file = str(self._save_path / "output.avi")
            h, w = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self._video_writer = cv2.VideoWriter(output_file, fourcc, 20.0, (w, h))
            logger.info("Video writer opened: %s (%dx%d)", output_file, w, h)

        self._video_writer.write(annotated)

    def finalize(self) -> None:
        """Flush buffered output and release resources.

        Must be called after all frames have been processed.
        """
        if 'save_json' in self._modes and self._detections_buffer:
            output_file = str(self._save_path / "detections.json")
            save_json(self._detections_buffer, output_file)

        if 'save_csv' in self._modes and self._detections_buffer:
            output_file = str(self._save_path / "detections.csv")
            save_csv(self._detections_buffer, output_file)

        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            logger.info("Video writer released.")

        cv2.destroyAllWindows()

        self._detections_buffer.clear()
        logger.info("OutputHandler finalized.")
