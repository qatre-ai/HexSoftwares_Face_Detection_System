"""
Configuration management for the face detection system.

Provides a layered configuration system with the following precedence
(highest to lowest):

    CLI arguments > Environment variables > YAML config file > Defaults

Design constraints:
    - The system MUST run with zero configuration (safe defaults only).
    - Missing or invalid values fail early and loudly.
    - No detection logic, I/O, or model loading belongs here.

Non-goals:
    - No dynamic reloading.
    - No database-backed or remote configuration.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
# Resolved relative to this file's location: src/config.py → face_detection/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    """Return the resolved project root directory."""
    return _PROJECT_ROOT


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Model-related configuration.

    Attributes:
        prototxt_path: Path to the .prototxt network definition (relative to project root).
        weights_path: Path to the .caffemodel weights file (relative to project root).
        backend: Compute backend — 'cpu' or 'cuda'.
        input_size: Spatial dimensions (width, height) for the DNN input blob.
        mean_values: Per-channel mean subtraction values (BGR order).
        scale_factor: Pixel value scale factor applied during blob creation.
    """

    prototxt_path: str = "models/deploy.prototxt"
    weights_path: str = "models/res10_300x300_ssd_iter_140000.caffemodel"
    backend: str = "cpu"
    input_size: Tuple[int, int] = (300, 300)
    mean_values: Tuple[float, float, float] = (104.0, 177.0, 123.0)
    scale_factor: float = 1.0


@dataclass(frozen=True)
class DetectionConfig:
    """Detection thresholds.

    Attributes:
        confidence_threshold: Minimum confidence to accept a detection.
        nms_threshold: IoU threshold for non-maximum suppression.
    """

    confidence_threshold: float = 0.5
    nms_threshold: float = 0.3


@dataclass(frozen=True)
class InputConfig:
    """Input source configuration.

    Attributes:
        source: Input source — file path, directory path, video path,
                or integer device index (as string or int).
        resize_width: Optional width to downscale input frames before detection.
                      None means no resizing.
    """

    source: str = "0"
    resize_width: Optional[int] = None


@dataclass(frozen=True)
class OutputConfig:
    """Output behavior configuration.

    Attributes:
        mode: Output mode(s). Supports multiple comma-separated values:
              'display', 'save_image', 'save_video', 'save_json', 'save_csv'.
              Example: "display,save_image,save_json"
        save_path: Directory where output artifacts are written.
    """

    mode: str = "display"
    save_path: str = "output/"


@dataclass(frozen=True)
class VisualizationConfig:
    """Visualization rendering parameters.

    Attributes:
        box_color: BGR color tuple for bounding boxes.
        thickness: Line thickness in pixels.
        show_confidence: Whether to render the confidence score label.
    """

    box_color: Tuple[int, int, int] = (0, 255, 0)
    thickness: int = 2
    show_confidence: bool = True


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration.

    Aggregates all sub-configurations into a single, frozen object.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_VALID_BACKENDS = {"cpu", "cuda"}
_VALID_OUTPUT_MODES = {"display", "save_video", "save_json", "save_csv"}


def _validate(config: AppConfig) -> None:
    """Validate configuration values. Raises ValueError on invalid state."""

    if config.model.backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Invalid model.backend: '{config.model.backend}'. "
            f"Must be one of {_VALID_BACKENDS}."
        )

    # Validate each mode in comma-separated list
    modes = set(m.strip() for m in config.output.mode.split(','))
    invalid_modes = modes - _VALID_OUTPUT_MODES - {'save_image'}
    if invalid_modes:
        raise ValueError(
            f"Invalid output.mode(s): {invalid_modes}. "
            f"Valid modes: {_VALID_OUTPUT_MODES | {'save_image'}}. "
            f"Use comma-separated values for multiple outputs."
        )


    if not (0.0 <= config.detection.confidence_threshold <= 1.0):
        raise ValueError(
            f"detection.confidence_threshold must be in [0.0, 1.0], "
            f"got {config.detection.confidence_threshold}."
        )

    if not (0.0 <= config.detection.nms_threshold <= 1.0):
        raise ValueError(
            f"detection.nms_threshold must be in [0.0, 1.0], "
            f"got {config.detection.nms_threshold}."
        )

    if len(config.model.input_size) != 2:
        raise ValueError(
            f"model.input_size must be a (width, height) tuple, "
            f"got {config.model.input_size}."
        )

    if any(d <= 0 for d in config.model.input_size):
        raise ValueError(
            f"model.input_size dimensions must be positive, "
            f"got {config.model.input_size}."
        )

    if config.model.scale_factor <= 0:
        raise ValueError(
            f"model.scale_factor must be positive, "
            f"got {config.model.scale_factor}."
        )

    if config.input.resize_width is not None and config.input.resize_width <= 0:
        raise ValueError(
            f"input.resize_width must be positive or None, "
            f"got {config.input.resize_width}."
        )


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def _parse_tuple(value, expected_len: int, cast_type=float):
    """Convert a list from YAML into a tuple of the expected type and length."""
    if isinstance(value, (list, tuple)):
        if len(value) != expected_len:
            raise ValueError(
                f"Expected {expected_len} values, got {len(value)}: {value}"
            )
        return tuple(cast_type(v) for v in value)
    return value


def _build_model_config(raw: dict) -> ModelConfig:
    """Build ModelConfig from a raw YAML dict."""
    kwargs = {}
    if "prototxt_path" in raw:
        kwargs["prototxt_path"] = str(raw["prototxt_path"])
    if "weights_path" in raw:
        kwargs["weights_path"] = str(raw["weights_path"])
    if "backend" in raw:
        kwargs["backend"] = str(raw["backend"]).lower()
    if "input_size" in raw:
        kwargs["input_size"] = _parse_tuple(raw["input_size"], 2, int)
    if "mean_values" in raw:
        kwargs["mean_values"] = _parse_tuple(raw["mean_values"], 3, float)
    if "scale_factor" in raw:
        kwargs["scale_factor"] = float(raw["scale_factor"])
    return ModelConfig(**kwargs)


def _build_detection_config(raw: dict) -> DetectionConfig:
    """Build DetectionConfig from a raw YAML dict."""
    kwargs = {}
    if "confidence_threshold" in raw:
        kwargs["confidence_threshold"] = float(raw["confidence_threshold"])
    if "nms_threshold" in raw:
        kwargs["nms_threshold"] = float(raw["nms_threshold"])
    return DetectionConfig(**kwargs)


def _build_input_config(raw: dict) -> InputConfig:
    """Build InputConfig from a raw YAML dict."""
    kwargs = {}
    if "source" in raw:
        kwargs["source"] = str(raw["source"])
    if "resize_width" in raw:
        val = raw["resize_width"]
        kwargs["resize_width"] = int(val) if val is not None else None
    return InputConfig(**kwargs)


def _build_output_config(raw: dict) -> OutputConfig:
    """Build OutputConfig from a raw YAML dict."""
    kwargs = {}
    if "mode" in raw:
        kwargs["mode"] = str(raw["mode"]).lower()
    if "save_path" in raw:
        kwargs["save_path"] = str(raw["save_path"])
    return OutputConfig(**kwargs)


def _build_visualization_config(raw: dict) -> VisualizationConfig:
    """Build VisualizationConfig from a raw YAML dict."""
    kwargs = {}
    if "box_color" in raw:
        kwargs["box_color"] = _parse_tuple(raw["box_color"], 3, int)
    if "thickness" in raw:
        kwargs["thickness"] = int(raw["thickness"])
    if "show_confidence" in raw:
        kwargs["show_confidence"] = bool(raw["show_confidence"])
    return VisualizationConfig(**kwargs)


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------

_ENV_PREFIX = "FACE_DETECT_"


def _apply_env_overrides(raw: dict) -> dict:
    """Apply environment variable overrides to the raw config dict.

    Environment variables follow the pattern:
        FACE_DETECT_MODEL_BACKEND=cuda
        FACE_DETECT_DETECTION_CONFIDENCE_THRESHOLD=0.7

    The variable name maps to the nested config key by replacing
    underscores after the section name with dots.
    """
    env_map = {
        f"{_ENV_PREFIX}MODEL_BACKEND": ("model", "backend"),
        f"{_ENV_PREFIX}MODEL_SCALE_FACTOR": ("model", "scale_factor"),
        f"{_ENV_PREFIX}DETECTION_CONFIDENCE_THRESHOLD": ("detection", "confidence_threshold"),
        f"{_ENV_PREFIX}DETECTION_NMS_THRESHOLD": ("detection", "nms_threshold"),
        f"{_ENV_PREFIX}INPUT_SOURCE": ("input", "source"),
        f"{_ENV_PREFIX}INPUT_RESIZE_WIDTH": ("input", "resize_width"),
        f"{_ENV_PREFIX}OUTPUT_MODE": ("output", "mode"),
        f"{_ENV_PREFIX}OUTPUT_SAVE_PATH": ("output", "save_path"),
    }

    for env_var, (section, key) in env_map.items():
        value = os.environ.get(env_var)
        if value is not None:
            raw.setdefault(section, {})[key] = value
            logger.debug("Config override from env: %s=%s", env_var, value)

    return raw


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load and validate application configuration.

    Precedence (highest → lowest):
        Environment variables > YAML file > Hard-coded defaults

    Args:
        config_path: Path to a YAML configuration file. If None,
                     the system runs entirely on defaults (safe for
                     programmatic usage).

    Returns:
        A validated, frozen AppConfig instance.

    Raises:
        FileNotFoundError: If config_path is provided but does not exist.
        ValueError: If any configuration value is invalid.
        yaml.YAMLError: If the YAML file is malformed.
    """
    raw: dict = {}

    # --- Layer 1: YAML file ---
    if config_path is not None:
        resolved = Path(config_path)
        if not resolved.is_absolute():
            resolved = _PROJECT_ROOT / resolved

        if not resolved.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {resolved}. "
                f"Provide a valid path or omit to use defaults."
            )

        logger.info("Loading config from: %s", resolved)
        with open(resolved, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    # --- Layer 2: Environment variable overrides ---
    raw = _apply_env_overrides(raw)

    # --- Build typed configs ---
    config = AppConfig(
        model=_build_model_config(raw.get("model", {})),
        detection=_build_detection_config(raw.get("detection", {})),
        input=_build_input_config(raw.get("input", {})),
        output=_build_output_config(raw.get("output", {})),
        visualization=_build_visualization_config(raw.get("visualization", {})),
    )

    # --- Validate ---
    _validate(config)

    logger.debug("Configuration loaded: %s", config)
    return config
