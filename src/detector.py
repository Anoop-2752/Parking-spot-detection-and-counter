"""
ParkingDetector — the main detection loop as a class.
Holds state (spot statuses, previous frame, diffs) and
orchestrates classification, visualization, and display.
"""

import cv2
import numpy as np

from config import DIFF_THRESHOLD, FRAME_STEP
from src.classifier import is_empty, load_model
from src.spot_extractor import extract_spots, load_mask
from src.visualizer import draw_counter, draw_spots


class ParkingDetector:
    def __init__(self, video_path, mask_path, model_path=None, step=None):
        self.video_path = video_path
        self.mask_path = mask_path
        self.step = step or FRAME_STEP

        # Load resources
        self.mask = load_mask(mask_path)
        self.spots = extract_spots(self.mask)
        self.model = load_model(model_path)

        # State
        self.num_spots = len(self.spots)
        self.statuses = [True] * self.num_spots  # assume empty at start
        self.diffs = [0.0] * self.num_spots
        self.previous_frame = None
        self.frame_num = 0

    def _calc_diff(self, crop, prev_crop):
        """Mean absolute difference between two image crops."""
        return np.abs(np.mean(crop) - np.mean(prev_crop))

    def _update_diffs(self, frame):
        """Calculate per-spot diffs against the previous frame."""
        for i, (x, y, w, h) in enumerate(self.spots):
            crop = frame[y:y + h, x:x + w]
            prev_crop = self.previous_frame[y:y + h, x:x + w]
            self.diffs[i] = self._calc_diff(crop, prev_crop)

    def _classify_spots(self, frame):
        """Decide which spots to re-classify and update their statuses."""
        if self.previous_frame is None:
            # First frame — classify all spots
            indices = range(self.num_spots)
        else:
            self._update_diffs(frame)
            max_diff = np.amax(self.diffs)
            if max_diff == 0:
                return
            indices = [
                i for i in range(self.num_spots)
                if self.diffs[i] / max_diff > DIFF_THRESHOLD
            ]

        for i in indices:
            x, y, w, h = self.spots[i]
            crop = frame[y:y + h, x:x + w]
            self.statuses[i] = is_empty(self.model, crop)

    def run(self):
        """Open the video and run the detection loop."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (self.mask.shape[1], self.mask.shape[0]))

                if self.frame_num % self.step == 0:
                    self._classify_spots(frame)
                    self.previous_frame = frame.copy()

                draw_spots(frame, self.spots, self.statuses)
                draw_counter(frame, self.statuses)

                cv2.namedWindow("Parking Detector", cv2.WINDOW_NORMAL)
                cv2.imshow("Parking Detector", frame)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break

                self.frame_num += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()
