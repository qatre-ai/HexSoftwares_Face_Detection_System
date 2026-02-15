"""
Face Detection CLI Entrypoint.

Responsibility:
    Parse command-line arguments, configure the application, wire together
    the detector and I/O handlers, and run the main processing loop.

Usage:
    python main.py --source 0                      # Webcam
    python main.py --source images/                # Directory of images
    python main.py --source video.mp4 --output-mode save_video
    python main.py --config my_config.yaml

This module is the executable entry point. It should not be imported
by other modules.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Configure logging before importing local modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

from src.config import load_config
from src.detector import Detector
from src.input_handler import InputHandler
from src.output_handler import OutputHandler


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Face Detection System — Production CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--source",
        type=str,
        help="Input source: '0' for webcam, path to image/video file, or directory.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        help="Detection confidence threshold (0.0 - 1.0). Overrides config.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["cpu", "cuda"],
        help="Compute backend preference. Overrides config.",
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        help="Output mode(s). Use comma-separated values for multiple outputs: "
             "display, save_image, save_video, save_json, save_csv. "
             "Example: 'display,save_image,save_json'. Overrides config.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path/directory for output artifacts. Overrides config.",
    )

    return parser.parse_args()


def main() -> int:
    """Main execution loop."""
    args = parse_args()

    # 1. Load Configuration (CLI args > ENV > YAML > Defaults)
    try:
        config = load_config(args.config)
        
        # Apply CLI overrides
        if args.source is not None:
            # We must use object.__setattr__ because the dataclass is frozen
            object.__setattr__(config.input, "source", args.source)
        
        if args.confidence is not None:
            object.__setattr__(config.detection, "confidence_threshold", args.confidence)
        
        if args.backend is not None:
            object.__setattr__(config.model, "backend", args.backend)
        
        if args.output_mode is not None:
            object.__setattr__(config.output, "mode", args.output_mode)
        
        if args.output_path is not None:
            object.__setattr__(config.output, "save_path", args.output_path)

        logger.info("Configuration active for this run.")

    except Exception as e:
        logger.error("Configuration error: %s", e)
        return 1

    # 2. Initialize Components
    try:
        detector = Detector(config)
        input_handler = InputHandler(
            source=config.input.source,
            resize_width=config.input.resize_width,
        )
        output_handler = OutputHandler(config)

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error("Initialization failed: %s", e)
        return 1
    except Exception as e:
        logger.exception("Unexpected initialization error: %s", e)
        return 1

    # 3. Processing Loop
    logger.info("Starting processing loop. Press 'q' or ESC to quit in display mode.")
    
    frame_count = 0
    start_time = time.perf_counter()
    
    try:
        for frame_id, frame in input_handler:
            frame_count += 1
            
            # Detect
            detections = detector.detect(frame)
            
            # log occasional progress
            if frame_count % 30 == 0:
                logger.info("Processed %d frames...", frame_count)

            # Output
            # process_frame returns False on exit request (e.g. 'q' key)
            should_continue = output_handler.process_frame(frame_id, frame, detections)
            if not should_continue:
                logger.info("Stopping loop per user request.")
                break
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.exception("Runtime error during processing: %s", e)
        return 1
    finally:
        # 4. Cleanup
        elapsed = time.perf_counter() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0.0
        
        input_handler.release()
        output_handler.finalize()
        
        logger.info(
            "Processing finished. Total frames: %d. Avg FPS: %.2f.",
            frame_count, fps
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
