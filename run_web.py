"""
Web dashboard entry point.
Usage:
    python run_web.py
    python run_web.py --port 5000 --video path/to/video.mp4
"""

import argparse

from config import DEFAULT_MASK_PATH, DEFAULT_VIDEO_PATH
from web.app import app, init_detector


def parse_args():
    parser = argparse.ArgumentParser(description="Parking Detection Web Dashboard")
    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH, help="Path to input video")
    parser.add_argument("--mask", default=DEFAULT_MASK_PATH, help="Path to binary mask image")
    parser.add_argument("--step", type=int, default=None, help="Process every Nth frame")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    return parser.parse_args()


def main():
    args = parse_args()
    init_detector(video_path=args.video, mask_path=args.mask, step=args.step)
    print(f"\n  Parking Detection Dashboard running at: http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
