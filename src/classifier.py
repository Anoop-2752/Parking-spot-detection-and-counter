"""
Parking spot classifier — supports both SVM (.p) and CNN (.pth) models.
Automatically detects model type based on file extension.
"""

import os
import pickle

import cv2
import numpy as np
from skimage.transform import resize

from config import (
    CLASSIFIER_RESIZE_DIM,
    CNN_INPUT_SIZE,
    CNN_MODEL_PATH,
    CNN_NORMALIZE_MEAN,
    CNN_NORMALIZE_STD,
    MODEL_PATH,
)


def _resolve_model_path():
    """Pick the best available model: prefer CNN (.pth) over SVM (.p)."""
    if os.path.exists(CNN_MODEL_PATH):
        return CNN_MODEL_PATH
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    raise FileNotFoundError(f"No model found. Looked for:\n  {CNN_MODEL_PATH}\n  {MODEL_PATH}")


def _load_svm(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return {"type": "svm", "model": model}


def _load_cnn(path):
    import torch
    from src.cnn_model import ParkingCNN

    model = ParkingCNN()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return {"type": "cnn", "model": model}


def load_model(path=None):
    """Load an SVM (.p) or CNN (.pth) model. Auto-detects type from extension."""
    path = path or _resolve_model_path()

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    if path.endswith(".pth"):
        return _load_cnn(path)
    return _load_svm(path)


def _predict_svm(model, spot_bgr):
    img_resized = resize(spot_bgr, CLASSIFIER_RESIZE_DIM)
    features = img_resized.flatten().reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0] == 0


def _predict_cnn(model, spot_bgr):
    import torch

    # BGR -> RGB, resize to 64x64
    img_rgb = cv2.cvtColor(spot_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))

    # Normalize to [0, 1], then apply ImageNet normalization
    img = img_resized.astype(np.float32) / 255.0
    mean = np.array(CNN_NORMALIZE_MEAN, dtype=np.float32)
    std = np.array(CNN_NORMALIZE_STD, dtype=np.float32)
    img = (img - mean) / std

    # HWC -> CHW, add batch dimension
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        prediction = output.argmax(dim=1).item()

    return prediction == 0


def is_empty(model_dict, spot_bgr):
    """
    Predict whether a single parking spot crop (BGR) is empty.
    Returns True if empty, False if occupied.
    """
    if model_dict["type"] == "cnn":
        return _predict_cnn(model_dict["model"], spot_bgr)
    return _predict_svm(model_dict["model"], spot_bgr)
