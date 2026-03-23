"""
Parking spot classifier — loads a trained sklearn model and
predicts whether a cropped spot image is empty or occupied.
"""

import pickle

import numpy as np
from skimage.transform import resize

from config import MODEL_PATH, CLASSIFIER_RESIZE_DIM


def load_model(path=None):
    """Load the pickled sklearn model. Returns the model object."""
    path = path or MODEL_PATH
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {path}")
    return model


def is_empty(model, spot_bgr):
    """
    Predict whether a single parking spot crop (BGR) is empty.
    Returns True if empty, False if occupied.
    """
    img_resized = resize(spot_bgr, CLASSIFIER_RESIZE_DIM)
    features = img_resized.flatten().reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0] == 0
