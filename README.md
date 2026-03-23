# Parking Spot Detection & Counter

A real-time parking spot detection system that processes video footage to identify empty and occupied parking spaces using computer vision and machine learning.

<img width="935" height="503" alt="Parking Detection Demo" src="https://github.com/user-attachments/assets/0cf74a67-3e76-4cba-b740-b227512ad0d5" />

---

## How It Works

### 1. Parking Spot Extraction
A binary **mask image** defines where each parking spot is located. OpenCV's connected components analysis extracts bounding boxes for every spot automatically — no manual coordinate mapping needed.

### 2. Frame Differencing (Optimization)
Instead of classifying every spot on every frame, the system compares the current frame against the previous one. Only spots where significant visual change is detected (above a configurable threshold) are sent to the classifier. This keeps processing efficient.

### 3. ML Classification
Each spot crop is classified as empty or occupied. Two model options are supported:

- **CNN (default)** — A 3-layer convolutional neural network (ParkingCNN) that takes 64x64 RGB input and learns spatial features. Better generalization to new lighting/angles.
- **SVM (legacy)** — Resizes to 15x15, flattens pixels, and classifies with an SVM. Fast but brittle to visual changes.

The system auto-detects the model type (`.pth` for CNN, `.p` for SVM) — just drop in the model file and it works.

### 4. Live Display
The processed video is displayed with:
- Color-coded rectangles on each parking spot
- A real-time counter showing **available / total** spots

---

## Project Structure

```
├── main.py                  # Entry point (CLI)
├── config.py                # All constants, thresholds, default paths
├── src/
│   ├── detector.py          # ParkingDetector class — main detection loop
│   ├── classifier.py        # Model loading & prediction (SVM + CNN)
│   ├── cnn_model.py         # ParkingCNN architecture (PyTorch)
│   ├── spot_extractor.py    # Mask → bounding boxes via connected components
│   └── visualizer.py        # Drawing rectangles & counter overlay
├── web/
│   ├── app.py               # Flask app with routes and MJPEG streaming
│   └── templates/
│       └── dashboard.html   # Web dashboard UI
├── run_web.py               # Web dashboard entry point
├── tools/
│   ├── crop_cars.py         # Data preparation — crop spots from video
│   ├── train_model.py       # Train the SVM classifier
│   └── train_cnn.py         # Train the CNN classifier (PyTorch)
├── data/
│   ├── masks/               # Binary mask images
│   └── videos/              # Input video files
├── model/
│   └── model.p              # Pre-trained SVM classifier
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.8+

### Installation

```bash
git clone https://github.com/your-username/Parking-spot-detection-and-counter.git
cd Parking-spot-detection-and-counter
pip install -r requirements.txt
```

---

## Usage

### Run with defaults
```bash
python main.py
```

### Run with custom video and mask
```bash
python main.py --video path/to/video.mp4 --mask path/to/mask.png
```

### Use webcam as source
```bash
python main.py --source 0 --mask path/to/mask.png
```

### Loop video and adjust frame step
```bash
python main.py --loop --step 15
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--video` | Path to input video file | `data/videos/parking_1920_1080_loop.mp4` |
| `--source` | Camera index (0, 1) or video path (overrides `--video`) | — |
| `--mask` | Path to binary mask image | `data/masks/mask_1920_1080.png` |
| `--step` | Process every Nth frame | `30` |
| `--loop` | Restart video when it ends | off |

Press **`q`** to quit the video window.

---

## Web Dashboard

A Flask-based web dashboard that streams the processed video feed with live statistics.

### Launch the dashboard
```bash
python run_web.py
```
Then open **http://localhost:5000** in your browser.

### Dashboard Options

| Flag | Description | Default |
|------|-------------|---------|
| `--video` | Path to input video file | `data/videos/parking_1920_1080_loop.mp4` |
| `--mask` | Path to binary mask image | `data/masks/mask_1920_1080.png` |
| `--step` | Process every Nth frame | `30` |
| `--port` | Server port | `5000` |
| `--debug` | Enable Flask debug mode | off |

The dashboard features:
- Live MJPEG video stream with bounding box overlays
- Real-time stat cards (available, occupied, total, FPS)
- Occupancy progress bar with color-coded thresholds
- Responsive layout for desktop and mobile

---

## Data Preparation

To generate training data (cropped spot images) from a video:

```bash
python -m tools.crop_cars --video path/to/video.mp4 --mask path/to/mask.png --output ./clf-data/crops/
```

---

## Training the Model

The training data directory should have this structure:
```
clf-data/
├── empty/          # Images of empty parking spots
└── not_empty/      # Images of occupied parking spots
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

- **Python** — core language
- **PyTorch** — CNN model training and inference
- **OpenCV** — video processing, connected components, display
- **Flask** — web dashboard with live MJPEG streaming
- **scikit-learn** — SVM classifier (legacy)
- **NumPy** — numerical operations and frame differencing

---

## Configuration

All tunable parameters are centralized in [`config.py`](config.py):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `FRAME_STEP` | `30` | Process every Nth frame |
| `DIFF_THRESHOLD` | `0.4` | Minimum relative change to re-classify a spot |
| `CLASSIFIER_RESIZE_DIM` | `(15, 15, 3)` | Input dimensions for the ML model |
| `COLOR_EMPTY` | Green | Bounding box color for empty spots |
| `COLOR_OCCUPIED` | Red | Bounding box color for occupied spots |

---

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
