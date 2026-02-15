![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/qatre-ai/face-detection-system)
![GitHub issues](https://img.shields.io/github/issues/qatre-ai/face-detection-system)
![GitHub license](https://img.shields.io/github/license/qatre-ai/face-detection-system)
![Python](https://img.shields.io/badge/python-%3E%3D3.7-blue)

# Face Detection System

A production-grade face detection library using OpenCV's DNN module with SSD-ResNet10. Built for reliability, clean integration, and deployment flexibility.

## 🚀 Quick Start

1. **Clone & Install**
   ```bash
   git clone https://github.com/qatre-ai/face-detection-system
   cd face_detection
   pip install -r requirements.txt
   ```

2. **Download Models** (Required)
   - Download [deploy.prototxt](https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt) to `models/`
   - Download [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel) to `models/`

3. **Run Webcam Demo**
   ```bash
   python main.py --source 0
   ```

## 🎯 Who This Is For

- **Engineers**: A reference implementation of a clean, modular computer vision pipeline.
- **Students**: A production-grade example of project structure, configuration, and testing.
- **Portfolio**: A demonstration of "resume-ready" Python code quality.

## What This Does

**Face Detection Only.** This system:
- Detects faces in images, videos, and webcam streams
- Returns bounding boxes with confidence scores
- Supports batch processing and real-time display
- Exports results to multiple formats (JSON, CSV, annotated images/videos)

**What This Does NOT Do:**
- Face recognition / identity labeling
- Face tracking across frames
- Emotion, age, or gender classification
- Model training or fine-tuning

This is strictly aligned with OpenCV DNN face detection capabilities.

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/qatre-ai/face-detection-system
cd face_detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies:**
- `opencv-python >= 4.5.0`
- `numpy >= 1.21.0`
- `pyyaml >= 6.0`

### 3. Download Model Files

**Required:** Place these two files in `face_detection/models/`:

1. **deploy.prototxt**  
   Download: [deploy.prototxt](https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt)

2. **res10_300x300_ssd_iter_140000.caffemodel**  
   Download: [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)

**Verification:**
```
face_detection/
├── models/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── src/
├── main.py
└── requirements.txt
```

The system will fail immediately with a clear error if these files are missing.

## Usage

### Webcam Demo
```bash
python main.py --source 0
```
Press **'q'** or **'ESC'** to exit.

### Single Image
```bash
python main.py --source path/to/image.jpg --output-mode save_image
```
Output: `output/frame_000000.jpg` (annotated image)

### Video Processing
```bash
python main.py --source path/to/video.mp4 --output-mode save_video
```
Output: `output/output.avi` (annotated video)

### Batch Processing (Directory)
```bash
python main.py --source path/to/images/ --output-mode save_image,save_json
```
Outputs:
- `output/frame_000000.jpg`, `frame_000001.jpg`, ...
- `output/detections.json`

### Multiple Outputs (Orthogonal)
```bash
python main.py --source 0 --output-mode display,save_image,save_json
```
This simultaneously:
1. Shows live video with detections
2. Saves each annotated frame as image
3. Accumulates detections in JSON

### Configuration Override
```bash
python main.py --source 0 --confidence 0.7 --backend cuda
```

## Output Modes

| Mode | Description | Output Location |
|------|-------------|-----------------|
| `display` | Show annotated frames in window | _(none, interactive only)_ |
| `save_image` | Save each frame as `.jpg` | `output/frame_NNNNNN.jpg` |
| `save_video` | Save annotated video | `output/output.avi` |
| `save_json` | Export detection data | `output/detections.json` |
| `save_csv` | Export detection data | `output/detections.csv` |

**Combine modes with commas:** `display,save_image,save_json`

## Programmatic API

```python
import cv2
from src import Detector

# Initialize detector (uses safe defaults)
detector = Detector()

# Read image
frame = cv2.imread("image.jpg")

# Detect faces
detections = detector.detect(frame)

# Process results
for det in detections:
    print(f"Face at ({det.x1}, {det.y1}) - ({det.x2}, {det.y2})")
    print(f"Confidence: {det.confidence:.2f}")
    print(f"Area: {det.area} pixels")
```

**API Contract:**
- Input: BGR numpy array (standard OpenCV format)
- Output: List of `Detection` objects
- Stateless: No side effects, no file I/O
- Deterministic: Same input → same output

## Configuration

Configuration precedence: **CLI args > ENV vars > config.yaml > Defaults**

**Example `config.yaml`:**
```yaml
model:
  backend: "cpu"  # or "cuda"
  
detection:
  confidence_threshold: 0.5
  nms_threshold: 0.3

output:
  mode: "display,save_image"
  save_path: "output/"
```

**Environment Variables:**
```bash
export FACE_DETECT_MODEL_BACKEND=cuda
export FACE_DETECT_DETECTION_CONFIDENCE_THRESHOLD=0.7
python main.py --source 0
```

## Architecture

See [`../architecture.md`](../architecture.md) for the full architectural specification.

**Module Boundaries:**
- **Public API:** `src.Detector` and `src.Detection` only
- **Internal:** All other modules (`src.config`, `src.preprocessor`, etc.)
- **CLI:** `main.py` (not importable)

```
src/
├── __init__.py          # Public re-exports
├── detector.py          # ← ONLY public programmatic API
├── detection.py         # ← Data transfer object
├── config.py            # Configuration management
├── model_loader.py      # DNN model loading
├── preprocessor.py      # Frame → blob
├── postprocessor.py     # Tensor → Detection list
├── input_handler.py     # Unified frame iterator
├── output_handler.py    # Output routing
├── visualizer.py        # Pure rendering
└── serializer.py        # JSON/CSV export
```

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

**Test Coverage:**
- `tests/test_preprocessor.py` — Blob creation
- `tests/test_postprocessor.py` — Tensor parsing
- `tests/test_config.py` — Configuration loading
- `tests/test_detector.py` — API contract

**Note:** The `tests/sample_data/` directory contains assets for manual testing and verification. These are not required for the automated test suite.

Integration tests requiring model files will skip automatically if models are not present.

## Scope & Design Decisions

**In Scope:**
- Face detection with bounding boxes
- Batch and real-time processing
- Multiple input/output formats
- CPU and CUDA backends

**Out of Scope (by design):**
- Face recognition / embeddings
- Temporal tracking (SORT, DeepSORT)
- REST API / web service
- Model training or fine-tuning
- Async / multiprocessing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

This project follows strict architectural boundaries defined in `architecture.md`. All contributions must:
1. Preserve the public API contract
2. Maintain fail-fast error handling
3. Avoid scope creep (no recognition, tracking, etc.)
4. Include tests for new functionality
