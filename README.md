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
Each spot crop is resized to 15x15 pixels, flattened, and fed into a pre-trained **SVM (Support Vector Machine)** classifier that predicts:
- **Empty** (green bounding box)
- **Occupied** (red bounding box)

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
│   ├── classifier.py        # Model loading & empty/occupied prediction
│   ├── spot_extractor.py    # Mask → bounding boxes via connected components
│   └── visualizer.py        # Drawing rectangles & counter overlay
├── tools/
│   └── crop_cars.py         # Data preparation — crop spots from video
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

### Adjust processing frequency
```bash
python main.py --step 15
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--video` | Path to input video file | `data/videos/parking_1920_1080_loop.mp4` |
| `--mask` | Path to binary mask image | `data/masks/mask_1920_1080.png` |
| `--step` | Process every Nth frame | `30` |

Press **`q`** to quit the video window.

---

## Data Preparation

To generate training data (cropped spot images) from a video:

```bash
python -m tools.crop_cars --video path/to/video.mp4 --mask path/to/mask.png --output ./clf-data/crops/
```

---

## Tech Stack

- **Python** — core language
- **OpenCV** — video processing, connected components, display
- **scikit-learn** — SVM classifier for spot classification
- **scikit-image** — image resizing for model input
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
