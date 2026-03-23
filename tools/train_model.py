"""
Training script for the parking spot classifier.

Loads cropped parking spot images from clf-data/, trains an SVM classifier,
evaluates it, and saves the model to model/model.p.

Usage:
    python -m tools.train_model
    python -m tools.train_model --data ./clf-data --output ./model/model.p
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CLASSIFIER_RESIZE_DIM

CATEGORIES = ["empty", "not_empty"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train parking spot classifier")
    parser.add_argument("--data", default="./clf-data", help="Path to training data directory")
    parser.add_argument("--output", default="./model/model.p", help="Path to save trained model")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for testing")
    parser.add_argument("--C", type=float, default=10, help="SVM regularization parameter")
    parser.add_argument("--gamma", type=float, default=0.01, help="SVM kernel coefficient")
    return parser.parse_args()


def load_dataset(data_dir):
    """Load images from empty/ and not_empty/ subdirectories."""
    flat_data = []
    labels = []
    skipped = 0

    for label, category in enumerate(CATEGORIES):
        category_dir = os.path.join(data_dir, category)
        if not os.path.isdir(category_dir):
            print(f"Warning: directory not found — {category_dir}")
            continue

        files = os.listdir(category_dir)
        print(f"Loading {len(files)} images from '{category}'...")

        for filename in files:
            filepath = os.path.join(category_dir, filename)
            try:
                img = imread(filepath)
                img_resized = resize(img, CLASSIFIER_RESIZE_DIM)
                flat_data.append(img_resized.flatten())
                labels.append(label)
            except Exception as e:
                skipped += 1
                continue

    if skipped > 0:
        print(f"Skipped {skipped} unreadable files")

    return np.array(flat_data), np.array(labels)


def main():
    args = parse_args()

    # Load data
    print(f"\n--- Loading dataset from {args.data} ---")
    X, y = load_dataset(args.data)

    if len(X) == 0:
        print("Error: No images loaded. Check your data directory.")
        sys.exit(1)

    print(f"Total samples: {len(X)}")
    for label, category in enumerate(CATEGORIES):
        print(f"  {category}: {np.sum(y == label)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # Train
    print(f"\n--- Training SVM (C={args.C}, gamma={args.gamma}) ---")
    start = time.time()
    model = SVC(C=args.C, gamma=args.gamma)
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Training completed in {elapsed:.1f}s")

    # Evaluate
    print("\n--- Evaluation ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)\n")
    print(classification_report(y_test, y_pred, target_names=CATEGORIES))

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
