"""
Drawing utilities — overlays parking spot rectangles,
the availability counter, and FPS onto video frames.
"""

import cv2

from config import (
    COLOR_EMPTY,
    COLOR_OCCUPIED,
    COUNTER_BG_RECT,
    COUNTER_COLOR,
    COUNTER_FONT_SCALE,
    RECT_THICKNESS,
)


def draw_spots(frame, spots, statuses):
    """Draw a colored rectangle on each parking spot."""
    for (x, y, w, h), is_empty in zip(spots, statuses):
        color = COLOR_EMPTY if is_empty else COLOR_OCCUPIED
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, RECT_THICKNESS)


def draw_counter(frame, statuses):
    """Draw the available / total counter overlay."""
    available = sum(statuses)
    total = len(statuses)

    (pt1, pt2) = COUNTER_BG_RECT
    cv2.rectangle(frame, pt1, pt2, (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Available spots: {available} / {total}",
        (pt1[0] + 20, pt2[1] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        COUNTER_FONT_SCALE,
        COUNTER_COLOR,
        2,
    )


def draw_fps(frame, fps):
    """Draw FPS counter in the top-right corner."""
    h, w = frame.shape[:2]
    text = f"FPS: {fps:.0f}"
    cv2.rectangle(frame, (w - 200, 20), (w - 20, 70), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (w - 190, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )
