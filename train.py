"""
Training Pipeline for Waste Segregation Image Classifier
Trains Deep CNN (WasteCNN or MobileNetV2 Transfer) on Municipal Waste Image Datasets.

Usage:
    python train.py --data-dir ./data/sample_images --epochs 25 --batch-size 32 --lr 0.0003

Author: AGGIBADALA VENKATA DURGA NAGA PRASAD
Guide: Ms. G. Hema Sri
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.model import WasteCNN, WasteMobileNetV2
from src.dataset import MunicipalWasteDataset, get_waste_transforms
from src.metrics import compute_metrics


def train_model(args):
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch is required to run the training pipeline.")
        print("Please install torch and torchvision via: pip install torch torchvision")
        sys.exit(1)

    print("=" * 70)
    print("  MUNICIPAL WASTE SEGREGATION - CNN TRAINING PIPELINE")
    print(f"  Author: AGGIBADALA VENKATA DURGA NAGA PRASAD")
    print(f"  Supervisor: Ms. G. Hema Sri")
    print(f"  Model Architecture: {args.model}")
    print(f"  Epochs: {args.epochs} | Batch Size: {args.batch_size} | Learning Rate: {args.lr}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device} (CUDA available: {torch.cuda.is_available()})")

    if not os.path.exists(args.data_dir):
        print(f"[!] Data directory not found: {args.data_dir}")
        print("Generating synthetic samples to demonstrate pipeline...")
        from data.generate_samples import generate_samples
        generate_samples(args.data_dir)

    train_transform = get_waste_transforms(img_size=args.img_size, is_train=True)
    val_transform = get_waste_transforms(img_size=args.img_size, is_train=False)

    full_dataset = MunicipalWasteDataset(root_dir=args.data_dir, transform=train_transform)
    num_classes = len(full_dataset.classes)
    print(f"[*] Found {len(full_dataset)} images across {num_classes} classes: {full_dataset.classes}")

    if len(full_dataset) < 10:
        print("[!] Small sample dataset detected. Training will run in rapid verification mode.")
        val_size = max(1, int(len(full_dataset) * 0.2))
        train_size = len(full_dataset) - val_size
    else:
        val_size = int(len(full_dataset) * 0.2)
        train_size = len(full_dataset) - val_size

    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=min(args.batch_size, len(train_set)), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=min(args.batch_size, len(val_set)), shuffle=False)

    # Initialize model
    if args.model.lower() == "mobilenetv2":
        model = WasteMobileNetV2(num_classes=num_classes, pretrained=True)
    else:
        model = WasteCNN(num_classes=num_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print("\n[*] Starting Training Loop...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

        scheduler.step()

        epoch_train_loss = running_loss / max(total_train, 1)
        epoch_train_acc = correct_train / max(total_train, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        epoch_val_loss = val_loss / max(len(val_set), 1)
        val_metrics = compute_metrics(all_labels, all_preds, full_dataset.classes)
        epoch_val_acc = val_metrics["accuracy"]

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.1f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.1f}%")

        if epoch_val_acc >= best_acc:
            best_acc = epoch_val_acc
            save_path = os.path.join(args.save_dir, "best_waste_cnn.pth")
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "accuracy": best_acc,
                "classes": full_dataset.classes,
                "args": vars(args)
            }, save_path)

    elapsed = time.time() - start_time
    print(f"\n[✓] Training completed in {elapsed:.1f}s.")
    print(f"[✓] Best Validation Accuracy: {best_acc*100:.2f}%")
    print(f"[✓] Checkpoint saved at: {os.path.join(args.save_dir, 'best_waste_cnn.pth')}")

    # Save training log
    log_path = os.path.join(args.save_dir, "training_history.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[✓] Training history saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waste Segregation CNN Trainer")
    parser.add_argument("--data-dir", type=str, default="./data/sample_images", help="Path to waste dataset")
    parser.add_argument("--save-dir", type=str, default="./models", help="Directory to save model weights")
    parser.add_argument("--model", type=str, default="wastecnn", choices=["wastecnn", "mobilenetv2"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--img-size", type=int, default=224)

    parsed_args = parser.parse_args()
    train_model(parsed_args)
