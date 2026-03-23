"""
Training script for the CNN parking spot classifier.

Loads images from clf-data/, trains a ParkingCNN model with data
augmentation, evaluates it, and saves the weights to model/model.pth.

Usage:
    python -m tools.train_cnn
    python -m tools.train_cnn --data ./clf-data --epochs 30 --batch-size 32
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CNN_INPUT_SIZE, CNN_NORMALIZE_MEAN, CNN_NORMALIZE_STD
from src.cnn_model import ParkingCNN

CATEGORIES = ["empty", "not_empty"]


class ParkingDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.file_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            from PIL import Image
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            img = cv2.resize(img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img.transpose(2, 0, 1))

        return img, self.labels[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN parking spot classifier")
    parser.add_argument("--data", default="./clf-data", help="Path to training data directory")
    parser.add_argument("--output", default="./model/model.pth", help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for testing")
    return parser.parse_args()


def load_file_paths(data_dir):
    """Collect file paths and labels from empty/ and not_empty/ subdirectories."""
    paths = []
    labels = []

    for label, category in enumerate(CATEGORIES):
        category_dir = os.path.join(data_dir, category)
        if not os.path.isdir(category_dir):
            print(f"Warning: directory not found — {category_dir}")
            continue

        files = os.listdir(category_dir)
        print(f"Found {len(files)} images in '{category}'")

        for filename in files:
            paths.append(os.path.join(category_dir, filename))
            labels.append(label)

    return paths, labels


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load file paths
    print(f"\n--- Loading dataset from {args.data} ---")
    paths, labels = load_file_paths(args.data)

    if len(paths) == 0:
        print("Error: No images found. Check your data directory.")
        sys.exit(1)

    print(f"Total samples: {len(paths)}")

    # Split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=args.test_size, random_state=42, stratify=labels
    )
    print(f"Train: {len(train_paths)} | Val: {len(val_paths)}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(72),
        transforms.RandomCrop(CNN_INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=CNN_NORMALIZE_MEAN, std=CNN_NORMALIZE_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CNN_NORMALIZE_MEAN, std=CNN_NORMALIZE_STD),
    ])

    # Datasets and loaders
    train_dataset = ParkingDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = ParkingDataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = ParkingCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Training loop
    print(f"\n--- Training for {args.epochs} epochs ---")
    best_accuracy = 0.0
    start = time.time()

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for images, labels_batch in train_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels_batch in val_loader:
                images, labels_batch = images.to(device), labels_batch.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item() * images.size(0)

                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels_batch.cpu().numpy())

        val_loss /= len(val_dataset)
        accuracy = accuracy_score(all_labels, all_preds)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save(model.state_dict(), args.output)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train loss: {train_loss:.4f} | "
                  f"Val loss: {val_loss:.4f} | "
                  f"Val acc: {accuracy:.4f}")

    elapsed = time.time() - start
    print(f"\nTraining completed in {elapsed:.1f}s")
    print(f"Best validation accuracy: {best_accuracy:.4f} ({best_accuracy * 100:.1f}%)")

    # Final evaluation with best model
    print("\n--- Final Evaluation ---")
    model.load_state_dict(torch.load(args.output, map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels_batch in val_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_batch.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=CATEGORIES))
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
