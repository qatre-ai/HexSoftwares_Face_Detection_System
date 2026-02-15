"""
Input handling for the face detection pipeline.

Responsibility:
    Abstract away frame acquisition from images, video files, image
    directories, and webcam streams. Provides a uniform iterator
    interface yielding (frame_id, frame) tuples.

Non-goals:
    - No detection, drawing, or output writing.
    - No infinite retry on bad sources.
    - No implicit fallback between source types.

Robustness:
    - Validates the source at initialization time.
    - Logs and skips unreadable frames (never crashes the pipeline).
    - Releases resources on cleanup.
"""

import logging
import os
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Image extensions recognized by this handler
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Video extensions recognized by this handler
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


class InputHandler:
    """Uniform frame iterator for images, videos, and webcam streams.

    The source type is auto-detected at initialization:
        - Integer or digit string  → webcam device index
        - File with image extension → single image
        - File with video extension → video file
        - Directory path → all images in directory (sorted)

    Usage:
        handler = InputHandler(source="path/to/video.mp4")
        for frame_id, frame in handler:
            # process frame
        handler.release()

    Invalid frames are logged and skipped. The iterator never raises
    on a single bad frame.
    """

    def __init__(
        self,
        source: Union[str, int],
        resize_width: Optional[int] = None,
    ) -> None:
        """Initialize the input handler and validate the source.

        Args:
            source: Input source — file path, directory path, video path,
                    or integer device index (or digit string like "0").
            resize_width: Optional width to downscale frames. Aspect ratio
                          is preserved. None means no resizing.

        Raises:
            FileNotFoundError: If a file/directory source does not exist.
            ValueError: If the source type cannot be determined.
            RuntimeError: If a video/webcam source cannot be opened.
        """
        self._resize_width = resize_width
        self._source_raw = source
        self._cap: Optional[cv2.VideoCapture] = None

        # Determine source type
        source_str = str(source).strip()

        if source_str.isdigit():
            # Webcam device index
            self._mode = "webcam"
            self._device_index = int(source_str)
            self._open_video_capture(self._device_index)
        elif os.path.isfile(source_str):
            ext = Path(source_str).suffix.lower()
            if ext in _IMAGE_EXTENSIONS:
                self._mode = "image"
                self._image_paths = [source_str]
            elif ext in _VIDEO_EXTENSIONS:
                self._mode = "video"
                self._open_video_capture(source_str)
            else:
                raise ValueError(
                    f"Unrecognized file extension: '{ext}' for source '{source_str}'. "
                    f"Supported images: {_IMAGE_EXTENSIONS}. "
                    f"Supported videos: {_VIDEO_EXTENSIONS}."
                )
        elif os.path.isdir(source_str):
            self._mode = "directory"
            self._image_paths = sorted(
                str(p)
                for p in Path(source_str).iterdir()
                if p.suffix.lower() in _IMAGE_EXTENSIONS
            )
            if not self._image_paths:
                raise ValueError(
                    f"No image files found in directory: '{source_str}'. "
                    f"Supported extensions: {_IMAGE_EXTENSIONS}."
                )
            logger.info("Found %d images in directory: %s", len(self._image_paths), source_str)
        else:
            raise FileNotFoundError(
                f"Input source not found: '{source_str}'. "
                f"Provide a valid file path, directory, or device index."
            )

        logger.info("InputHandler initialized: mode=%s, source=%s", self._mode, source_str)

    def _open_video_capture(self, source: Union[str, int]) -> None:
        """Open a VideoCapture and validate it.

        Raises:
            RuntimeError: If the source cannot be opened.
        """
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            source_desc = (
                f"webcam device {source}" if isinstance(source, int)
                else f"video file '{source}'"
            )
            raise RuntimeError(
                f"Failed to open {source_desc}. "
                f"Ensure the source exists and is accessible."
            )

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Iterate over frames from the configured source.

        Yields:
            Tuples of (frame_id, frame) where frame_id is a 0-based
            index and frame is a BGR numpy array.

        Invalid frames are logged and skipped (never raises mid-iteration).
        """
        if self._mode in ("image", "directory"):
            yield from self._iterate_images()
        elif self._mode in ("video", "webcam"):
            yield from self._iterate_video()

    def _iterate_images(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield frames from a list of image file paths."""
        for idx, path in enumerate(self._image_paths):
            frame = cv2.imread(path)
            if frame is None:
                logger.warning(
                    "Skipping unreadable image (frame_id=%d): %s", idx, path
                )
                continue

            frame = self._maybe_resize(frame)
            yield idx, frame

    def _iterate_video(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield frames from a video file or webcam stream."""
        frame_id = 0
        consecutive_failures = 0
        max_consecutive_failures = 30  # Safety valve for dead streams

        while True:
            ret, frame = self._cap.read()

            if not ret or frame is None:
                consecutive_failures += 1
                if self._mode == "video":
                    # End of video file
                    logger.info("End of video reached at frame %d.", frame_id)
                    break
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(
                        "Webcam produced %d consecutive failed reads. "
                        "Stopping to avoid infinite loop.",
                        max_consecutive_failures,
                    )
                    break
                logger.warning(
                    "Failed to read frame %d from webcam, skipping.", frame_id
                )
                frame_id += 1
                continue

            consecutive_failures = 0
            frame = self._maybe_resize(frame)
            yield frame_id, frame
            frame_id += 1

    def _maybe_resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame if resize_width is configured, preserving aspect ratio."""
        if self._resize_width is None:
            return frame

        h, w = frame.shape[:2]
        if w <= self._resize_width:
            return frame

        scale = self._resize_width / w
        new_w = self._resize_width
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def release(self) -> None:
        """Release any held resources (video capture handles)."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.debug("VideoCapture released.")

    def __del__(self) -> None:
        """Safety net: release resources if not explicitly released."""
        self.release()
