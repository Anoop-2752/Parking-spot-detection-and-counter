"""
Data preparation tool — crops parking spots from video frames
and saves them as individual images for training data.

Usage:
    python -m tools.crop_cars
    python -m tools.crop_cars --video path/to/video.mp4 --mask path/to/mask.png --output ./clf-data/all_
"""

import argparse
import os
import sys

import cv2

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_MASK_PATH
from src.spot_extractor import extract_spots, load_mask

SELECTED_SLOTS = {
    4, 12, 26, 29, 32, 61, 72, 89, 91, 106, 129, 131, 132,
    147, 157, 161, 164, 179, 180, 185, 201, 223, 224, 271,
    280, 281, 303, 319, 335, 341, 344, 351, 360, 377, 385, 389,
}

DEFAULT_VIDEO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "videos", "parking_1920_1080.mp4",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Crop parking spots from video frames")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Path to input video")
    parser.add_argument("--mask", default=DEFAULT_MASK_PATH, help="Path to binary mask")
    parser.add_argument("--output", default="./clf-data/all_", help="Output directory")
    parser.add_argument("--frame-skip", type=int, default=10, help="Frames to skip between crops")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    mask = load_mask(args.mask)
    spots = extract_spots(mask)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {args.video}")
        sys.exit(1)

    frame_num = 0
    saved = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        for slot_idx, (x, y, w, h) in enumerate(spots):
            if slot_idx in SELECTED_SLOTS:
                crop = frame[y:y + h, x:x + w]
                filename = f"{frame_num:08d}_{slot_idx:08d}.jpg"
                cv2.imwrite(os.path.join(args.output, filename), crop)
                saved += 1

        frame_num += args.frame_skip

    cap.release()
    print(f"Done. Saved {saved} crops to {args.output}")


if __name__ == "__main__":
    main()
