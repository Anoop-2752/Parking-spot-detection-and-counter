"""
Entry point for the parking spot detection system.
Usage:
    python main.py
    python main.py --video path/to/video.mp4 --mask path/to/mask.png
"""

import argparse

from config import DEFAULT_MASK_PATH, DEFAULT_VIDEO_PATH
from src.detector import ParkingDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Parking Spot Detection & Counter")
    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH, help="Path to input video")
    parser.add_argument("--mask", default=DEFAULT_MASK_PATH, help="Path to binary mask image")
    parser.add_argument("--step", type=int, default=None, help="Process every Nth frame")
    return parser.parse_args()


def main():
    args = parse_args()
    detector = ParkingDetector(
        video_path=args.video,
        mask_path=args.mask,
        step=args.step,
    )
    detector.run()


if __name__ == "__main__":
    main()
