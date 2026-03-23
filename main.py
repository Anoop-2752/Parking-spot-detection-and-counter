"""
Entry point for the parking spot detection system.
Usage:
    python main.py
    python main.py --video path/to/video.mp4 --mask path/to/mask.png
    python main.py --source 0          # webcam
    python main.py --loop              # restart video when it ends
"""

import argparse

from config import DEFAULT_MASK_PATH, DEFAULT_VIDEO_PATH
from src.detector import ParkingDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Parking Spot Detection & Counter")
    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH, help="Path to input video")
    parser.add_argument("--source", default=None, help="Video source — camera index (0, 1) or video path (overrides --video)")
    parser.add_argument("--mask", default=DEFAULT_MASK_PATH, help="Path to binary mask image")
    parser.add_argument("--model", default=None, help="Path to model file (.p for SVM, .pth for CNN)")
    parser.add_argument("--step", type=int, default=None, help="Process every Nth frame")
    parser.add_argument("--loop", action="store_true", help="Loop video when it ends")
    return parser.parse_args()


def main():
    args = parse_args()
    video = args.source if args.source is not None else args.video

    detector = ParkingDetector(
        video_path=video,
        mask_path=args.mask,
        model_path=args.model,
        step=args.step,
        loop=args.loop,
    )
    detector.run()


if __name__ == "__main__":
    main()
