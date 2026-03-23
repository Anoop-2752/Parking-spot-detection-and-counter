"""
Mask creation tool — interactively draw parking spot regions on a video
frame and generate a binary mask image.

Usage:
    python -m tools.create_mask --video path/to/video.mp4
    python -m tools.create_mask --video path/to/video.mp4 --output mask.png

Controls:
    Left click     — place a corner point of a parking spot
    Right click    — finish current spot (connects the points)
    U              — undo last spot
    R              — reset all spots
    S              — save mask and exit
    Q / ESC        — quit without saving
    +/-            — navigate forward/backward in the video to find a good frame
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# State
current_points = []
all_spots = []
frame = None
display = None
frame_num = 0


def redraw():
    """Redraw the display with all spots and current points."""
    global display
    display = frame.copy()

    # Draw completed spots (filled semi-transparent)
    overlay = display.copy()
    for i, spot in enumerate(all_spots):
        pts = np.array(spot, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (255, 255, 255))
        cv2.polylines(display, [pts], True, (0, 255, 0), 2)

        # Spot number label
        cx = int(np.mean([p[0] for p in spot]))
        cy = int(np.mean([p[1] for p in spot]))
        cv2.putText(display, str(i + 1), (cx - 8, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)

    # Draw current points being placed
    for pt in current_points:
        cv2.circle(display, pt, 5, (0, 200, 255), -1)
    if len(current_points) > 1:
        for i in range(len(current_points) - 1):
            cv2.line(display, current_points[i], current_points[i + 1], (0, 200, 255), 2)

    # HUD
    h, w = display.shape[:2]
    cv2.rectangle(display, (0, h - 40), (w, h), (0, 0, 0), -1)
    hud = (f"Spots: {len(all_spots)} | "
           f"Points: {len(current_points)} | "
           f"Frame: {frame_num} | "
           f"LClick: add point | RClick: finish spot | "
           f"U: undo | R: reset | S: save | +/-: frame")
    cv2.putText(display, hud, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks for placing points and completing spots."""
    global current_points

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        redraw()
        cv2.imshow("Mask Creator", display)

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_points) >= 3:
            all_spots.append(current_points.copy())
            current_points = []
            redraw()
            cv2.imshow("Mask Creator", display)


def save_mask(output_path, shape):
    """Generate and save the binary mask from all spots."""
    mask = np.zeros(shape[:2], dtype=np.uint8)

    for spot in all_spots:
        pts = np.array(spot, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    cv2.imwrite(output_path, mask)
    print(f"Mask saved to {output_path} ({len(all_spots)} spots, {shape[1]}x{shape[0]})")


def parse_args():
    parser = argparse.ArgumentParser(description="Create parking spot mask from video")
    parser.add_argument("--video", required=True, help="Path to video file or camera index")
    parser.add_argument("--output", default=None, help="Output mask path (default: mask_<width>_<height>.png)")
    parser.add_argument("--frame", type=int, default=0, help="Starting frame number")
    return parser.parse_args()


def main():
    global frame, frame_num
    args = parse_args()

    # Open video
    try:
        source = int(args.video)
    except ValueError:
        source = args.video

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {args.video}")
        sys.exit(1)

    frame_num = args.frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read video frame")
        sys.exit(1)

    h, w = frame.shape[:2]
    output_path = args.output or f"data/masks/mask_{w}_{h}.png"

    print(f"Video: {args.video} ({w}x{h}, {total_frames} frames)")
    print(f"Output: {output_path}")
    print()
    print("Controls:")
    print("  Left click  — place a corner point")
    print("  Right click — finish current spot (min 3 points)")
    print("  U           — undo last spot")
    print("  R           — reset all spots")
    print("  S           — save mask and exit")
    print("  Q / ESC     — quit without saving")
    print("  +/-         — navigate frames")

    cv2.namedWindow("Mask Creator", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Mask Creator", mouse_callback)

    redraw()

    while True:
        cv2.imshow("Mask Creator", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == 27:  # Q or ESC
            print("Quit without saving.")
            break

        elif key == ord('s'):
            if len(all_spots) == 0:
                print("No spots drawn. Nothing to save.")
            else:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                save_mask(output_path, frame.shape)
            break

        elif key == ord('u'):
            if all_spots:
                all_spots.pop()
                redraw()
            elif current_points:
                current_points.pop()
                redraw()

        elif key == ord('r'):
            all_spots.clear()
            current_points.clear()
            redraw()

        elif key in (ord('+'), ord('='), ord('.')):
            frame_num = min(frame_num + 30, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, new_frame = cap.read()
            if ret:
                frame = new_frame
                redraw()

        elif key in (ord('-'), ord('_'), ord(',')):
            frame_num = max(frame_num - 30, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, new_frame = cap.read()
            if ret:
                frame = new_frame
                redraw()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
