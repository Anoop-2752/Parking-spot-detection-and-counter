import cv2
import numpy as np
from util import get_parking_spots_bboxes, empty_or_not

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

mask_path = "mask_1920_1080.png"
video_path = "data/parking_1920_1080_loop.mp4"

# Load mask
mask = cv2.imread(mask_path, 0)

# Video capture
cap = cv2.VideoCapture(video_path)

# Get connected components
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

# Initialize status and differences
spots_status = [None for _ in spots]
diffs = [None for _ in spots]

previous_frame = None
frame_nmr = 0
step = 30

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to match mask
    frame = cv2.resize(frame, (mask.shape[1], mask.shape[0]))

    # Calculate differences if previous frame exists
    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1+h, x1:x1+w, :]
            diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1+h, x1:x1+w, :])

        # Optional: print sorted diffs for debugging
        print([diffs[j] for j in np.argsort(diffs)][::-1])

    # Determine which spots to process
    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]

        for spot_indx in arr_:
            x1, y1, w, h = spots[spot_indx]
            spot_crop = frame[y1:y1+h, x1:x1+w, :]
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_indx] = spot_status

    # Update previous frame
    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    # Draw rectangles on all spots
    for spot_indx, spot in enumerate(spots):
        x1, y1, w, h = spot
        status = spots_status[spot_indx]
        color = (0, 255, 0) if status else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), color, 2)

    # Display available spots counter
    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, f'Available spots: {sum(spots_status)} / {len(spots_status)}', 
                (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()
