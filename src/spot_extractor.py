"""
Extract parking spot bounding boxes from a binary mask image
using OpenCV connected components analysis.
"""

import cv2


def load_mask(mask_path):
    """Load a binary mask image in grayscale. Raises if file is invalid."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask image: {mask_path}")
    return mask


def extract_spots(mask):
    """
    Run connected components on the mask and return a list of
    bounding boxes [(x, y, w, h), ...] for each parking spot.
    """
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

    spots = []
    for i in range(1, num_labels):  # skip background (label 0)
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        spots.append((x, y, w, h))

    return spots
