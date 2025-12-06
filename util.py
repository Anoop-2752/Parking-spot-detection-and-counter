import pickle
from skimage.transform import resize
import numpy as np
import cv2

# Labels used for prediction outcome
EMPTY = True
NOT_EMPTY = False

# Load trained ML model for parking spot classification
MODEL = pickle.load(open("model/model.p", "rb"))


def empty_or_not(spot_bgr):
    """
    Determine whether the given parking spot crop is empty or not.
    Takes a BGR image of the parking spot, resizes it, flattens it,
    and feeds it to the trained model.
    """

    flat_data = []

    # Resize the spot image to 15x15 with 3 channels (RGB/BGR)
    img_resized = resize(spot_bgr, (15, 15, 3))

    # Flatten the resized image and store in list
    flat_data.append(img_resized.flatten())

    # Convert to numpy array (shape: 1 x 675)
    flat_data = np.array(flat_data)

    # Predict using the trained model (0 = empty, 1 = not empty)
    y_output = MODEL.predict(flat_data)

    # Return boolean result based on prediction
    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY


def get_parking_spots_bboxes(connected_components):
    """
    Extract bounding boxes of parking spots from connected component outputs.
    connected_components is the output of cv2.connectedComponentsWithStats,
    which includes label stats (x, y, width, height).
    """

    # Unpack connected components result
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1     # Scaling factor (kept as 1 for original size)

    # Loop from label 1 (skip background which is label 0)
    for i in range(1, totalLabels):

        # Extract bounding box values for each component
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w  = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h  = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        # Store bounding box in list
        slots.append([x1, y1, w, h])

    # Return list of all parking spot bounding boxes
    return slots
