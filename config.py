"""
Central configuration for the parking spot detection system.
All tunable constants and default paths live here.
"""

import os

# ── Paths (defaults, overridable via CLI) ─────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MASK_PATH = os.path.join(BASE_DIR, "data", "masks", "mask_1920_1080.png")
DEFAULT_VIDEO_PATH = os.path.join(BASE_DIR, "data", "videos", "parking_1920_1080_loop.mp4")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.p")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pth")

# ── Detection parameters ──────────────────────────────────────
FRAME_STEP = 30                # Process every Nth frame
DIFF_THRESHOLD = 0.4           # Minimum relative diff to re-classify a spot
CLASSIFIER_RESIZE_DIM = (15, 15, 3)   # Input size for SVM model

# ── CNN configuration ─────────────────────────────────────────
CNN_INPUT_SIZE = 64
CNN_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
CNN_NORMALIZE_STD = [0.229, 0.224, 0.225]

# ── Display ───────────────────────────────────────────────────
COLOR_EMPTY = (0, 255, 0)      # Green
COLOR_OCCUPIED = (0, 0, 255)   # Red
RECT_THICKNESS = 2
COUNTER_BG_RECT = ((80, 20), (550, 80))
COUNTER_FONT_SCALE = 1
COUNTER_COLOR = (255, 255, 255)

# ── Web streaming ─────────────────────────────────────────────
JPEG_QUALITY = 80
