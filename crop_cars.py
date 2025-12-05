import os
import cv2

output_dir = "./clf-data/all_"
os.makedirs(output_dir, exist_ok=True)

mask_path = "./mask_1920_1080.png"
mask = cv2.imread(mask_path, 0)

# Connected components
analysis = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
(totalLabels, label_ids, values, centroid) = analysis

# Extract slots
slots = []
for i in range(1, totalLabels):
    x1 = values[i, cv2.CC_STAT_LEFT]
    y1 = values[i, cv2.CC_STAT_TOP]
    w  = values[i, cv2.CC_STAT_WIDTH]
    h  = values[i, cv2.CC_STAT_HEIGHT]
    slots.append([x1, y1, w, h])

# Open video ONCE
video_path = "./data/parking_1920_1080.mp4"
cap = cv2.VideoCapture(video_path)

frame_nmr = 0

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr)
    ret, frame = cap.read()

    if not ret:
        break

    # Uncomment if mask shape differs
    # frame = cv2.resize(frame, (mask.shape[1], mask.shape[0]))

    selected_slots = [132,147,164,180,344,360,377,385,341,360,179,
                      131,106,91,61,4,89,129,161,185,201,224,271,
                      303,319,335,351,389,29,12,32,72,281,280,157,
                      223,26]

    for slot_nmr, slot in enumerate(slots):
        if slot_nmr in selected_slots:

            x1, y1, w, h = slot
            crop = frame[y1:y1+h, x1:x1+w]

            filename = f"{str(frame_nmr).zfill(8)}_{str(slot_nmr).zfill(8)}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), crop)

    frame_nmr += 10  # Jump 10 frames

cap.release()
