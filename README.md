# Parking Spot Detection & Counter

A real-time parking spot detection system that processes video footage to identify empty and occupied parking spaces using computer vision and deep learning. Works with any parking lot — just create a mask and run.

<img width="935" height="503" alt="Parking Detection Demo" src="https://github.com/user-attachments/assets/0cf74a67-3e76-4cba-b740-b227512ad0d5" />

---

## Key Features

- **CNN-based classification** — Custom 3-layer convolutional neural network built with PyTorch
- **Real-time processing** — Frame differencing optimization, only re-classifies spots that change
- **Web dashboard** — Flask-based live dashboard with MJPEG streaming and stat cards
- **Works on any parking lot** — Interactive mask creation tool to define spots on any video
- **Dual model support** — CNN (default) and SVM (legacy), auto-detected by file extension
- **Multiple input sources** — Video files, webcam, or live camera feeds

---

## How It Works

### 1. Parking Spot Extraction
A binary **mask image** defines where each parking spot is located. OpenCV's connected components analysis extracts bounding boxes for every spot automatically — no manual coordinate mapping needed. Use the included [mask creation tool](#creating-a-mask-for-a-new-parking-lot) to generate masks for any parking lot.

### 2. Frame Differencing (Optimization)
Instead of classifying every spot on every frame, the system compares the current frame against the previous one. Only spots where significant visual change is detected (above a configurable threshold) are sent to the classifier. This keeps processing efficient.

### 3. ML Classification
Each spot crop is classified as empty or occupied. Two model options are supported:

- **CNN (default)** — A 3-layer convolutional neural network (`ParkingCNN`) that takes 64x64 RGB input with BatchNorm and dropout. Trained with data augmentation for better generalization to new lighting and camera angles.
- **SVM (legacy)** — Resizes to 15x15, flattens pixels, and classifies with a Support Vector Machine. Fast but brittle to visual changes.

The system auto-detects the model type (`.pth` for CNN, `.p` for SVM) — just drop in the model file and it works.

### 4. Live Display
The processed video is displayed with:
- Color-coded bounding boxes (green = empty, red = occupied)
- A real-time counter showing **available / total** spots
- FPS counter for processing speed

---

## Project Structure

```
├── main.py                  # CLI entry point
├── run_web.py               # Web dashboard entry point
├── config.py                # All constants, thresholds, default paths
│
├── src/
│   ├── detector.py          # ParkingDetector class — main detection loop
│   ├── classifier.py        # Dual model loading & prediction (SVM + CNN)
│   ├── cnn_model.py         # ParkingCNN architecture (PyTorch)
│   ├── spot_extractor.py    # Mask → bounding boxes via connected components
│   └── visualizer.py        # Drawing rectangles, counter & FPS overlay
│
├── web/
│   ├── app.py               # Flask app with MJPEG streaming & stats API
│   └── templates/
│       └── dashboard.html   # Responsive web dashboard UI
│
├── tools/
│   ├── create_mask.py       # Interactive mask creation tool
│   ├── crop_cars.py         # Crop parking spots from video for training data
│   ├── train_cnn.py         # Train CNN classifier (PyTorch)
│   └── train_model.py       # Train SVM classifier (scikit-learn)
│
├── model/
│   ├── model.pth            # Trained CNN model (default)
│   └── model.p              # Trained SVM model (legacy)
│
├── data/
│   ├── masks/               # Binary mask images
│   └── videos/              # Input video files
│
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.8+

### Installation

```bash
git clone https://github.com/Anoop-2752/Parking-spot-detection-and-counter.git
cd Parking-spot-detection-and-counter
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Run detection with default video and CNN model
python main.py

# Or launch the web dashboard
python run_web.py
# Then open http://localhost:5000
```

---

## Creating a Mask for a New Parking Lot

To use this system on any parking lot video, you need a binary mask that defines where each spot is. The interactive mask tool makes this easy:

```bash
python -m tools.create_mask --video path/to/your_parking_lot.mp4
```

| Control | Action |
|---------|--------|
| **Left click** | Place a corner point of a parking spot |
| **Right click** | Finish current spot (connects the points, min 3) |
| **U** | Undo last spot |
| **R** | Reset all spots |
| **S** | Save mask and exit |
| **Q / ESC** | Quit without saving |
| **+/-** | Navigate frames to find a clear view |

**Full workflow for a new parking lot:**
```bash
# 1. Create a mask by clicking on parking spots
python -m tools.create_mask --video new_lot.mp4 --output data/masks/new_lot_mask.png

# 2. Run detection with your mask
python main.py --video new_lot.mp4 --mask data/masks/new_lot_mask.png

# 3. Or launch the web dashboard
python run_web.py --video new_lot.mp4 --mask data/masks/new_lot_mask.png
```

---

## Usage

### CLI Detection

```bash
# Default (uses CNN model automatically)
python main.py

# Custom video and mask
python main.py --video path/to/video.mp4 --mask path/to/mask.png

# Use webcam
python main.py --source 0 --mask path/to/mask.png

# Loop video and adjust processing frequency
python main.py --loop --step 15

# Force SVM model
python main.py --model model/model.p
```

| Flag | Description | Default |
|------|-------------|---------|
| `--video` | Path to input video file | `data/videos/parking_1920_1080_loop.mp4` |
| `--source` | Camera index (0, 1) or video path (overrides `--video`) | — |
| `--mask` | Path to binary mask image | `data/masks/mask_1920_1080.png` |
| `--model` | Path to model file (`.pth` for CNN, `.p` for SVM) | Auto-detect |
| `--step` | Process every Nth frame | `30` |
| `--loop` | Restart video when it ends | off |

Press **`q`** to quit the video window.

### Web Dashboard

```bash
python run_web.py
# Open http://localhost:5000
```

| Flag | Description | Default |
|------|-------------|---------|
| `--video` | Path to input video file | `data/videos/parking_1920_1080_loop.mp4` |
| `--mask` | Path to binary mask image | `data/masks/mask_1920_1080.png` |
| `--model` | Path to model file | Auto-detect |
| `--step` | Process every Nth frame | `30` |
| `--port` | Server port | `5000` |
| `--debug` | Enable Flask debug mode | off |

Dashboard features:
- Live MJPEG video stream with bounding box overlays
- Real-time stat cards (available, occupied, total, FPS)
- Occupancy progress bar with color-coded thresholds (green → yellow → red)
- Responsive layout for desktop and mobile

---

## Training

The training data directory should have this structure:
```
clf-data/
├── empty/          # Images of empty parking spots
└── not_empty/      # Images of occupied parking spots
```

Use the data preparation tool to generate training images from any video:
```bash
python -m tools.crop_cars --video path/to/video.mp4 --mask path/to/mask.png --output ./clf-data/all_
```

### CNN (recommended)

```bash
python -m tools.train_cnn
python -m tools.train_cnn --epochs 30 --batch-size 32 --lr 0.001
```

| Flag | Description | Default |
|------|-------------|---------|
| `--data` | Path to training data directory | `./clf-data` |
| `--output` | Path to save trained model | `./model/model.pth` |
| `--epochs` | Number of training epochs | `30` |
| `--batch-size` | Batch size | `32` |
| `--lr` | Learning rate | `0.001` |
| `--test-size` | Fraction of data for testing | `0.2` |

The CNN uses data augmentation (random crops, flips, rotation, color jitter) for better generalization.

**CNN Architecture:**
```
Input: 3x64x64 (RGB)
Conv2d(3→16) → BatchNorm → ReLU → MaxPool    → 16x32x32
Conv2d(16→32) → BatchNorm → ReLU → MaxPool   → 32x16x16
Conv2d(32→64) → BatchNorm → ReLU → MaxPool   → 64x8x8
Flatten → Linear(4096→128) → ReLU → Dropout(0.5) → Linear(128→2)
Parameters: 548,482
```

### SVM (legacy)

```bash
python -m tools.train_model
python -m tools.train_model --C 10 --gamma 0.01
```

| Flag | Description | Default |
|------|-------------|---------|
| `--data` | Path to training data directory | `./clf-data` |
| `--output` | Path to save trained model | `./model/model.p` |
| `--test-size` | Fraction of data for testing | `0.2` |
| `--C` | SVM regularization parameter | `10` |
| `--gamma` | SVM kernel coefficient | `0.01` |

---

## Tech Stack

| Technology | Purpose |
|-----------|---------|
| **PyTorch** | CNN model architecture, training, and inference |
| **OpenCV** | Video processing, connected components, frame rendering |
| **Flask** | Web dashboard with live MJPEG streaming |
| **scikit-learn** | SVM classifier (legacy model) |
| **scikit-image** | Image resizing for SVM input |
| **NumPy** | Numerical operations and frame differencing |

---

## Configuration

All tunable parameters are centralized in [`config.py`](config.py):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `FRAME_STEP` | `30` | Process every Nth frame |
| `DIFF_THRESHOLD` | `0.4` | Minimum relative change to re-classify a spot |
| `CNN_INPUT_SIZE` | `64` | CNN input image size |
| `CLASSIFIER_RESIZE_DIM` | `(15, 15, 3)` | SVM input dimensions |
| `JPEG_QUALITY` | `80` | MJPEG stream quality for web dashboard |
| `COLOR_EMPTY` | Green | Bounding box color for empty spots |
| `COLOR_OCCUPIED` | Red | Bounding box color for occupied spots |

---

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
