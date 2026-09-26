
"""
Train a ResNet-18 classifier on the VisionGuard NEU-DET dataset.

Expected dataset structure:

data/VisionGuard-Dataset/
    train/
        crazing/
        inclusion/
        patches/
        pitted_surface/
        rolled-in_scale/
        scratches/
    val/
        (same six classes)
    test/
        (same six classes)

Run from the VisionGuard project root:

python ml/training/train.py --dataset "data/VisionGuard-Dataset" --epochs 15 --batch-size 32 --learning-rate 0.001 --output "ml/saved_models/visionguard_resnet18.pt"
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2


# ============================================================
# PROJECT PATH AND IMPORT FIX
# ============================================================

# train.py is located at:
# VisionGuard/ml/training/train.py
#
# parents[0] = ml/training
# parents[1] = ml
# parents[2] = VisionGuard project root

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ml.model import (
    CLASS_NAMES,
    create_model,
)


# ============================================================
# DATA TRANSFORMS
# ============================================================

def get_transforms(training: bool = False):
    """
    Resize and normalize images for ResNet-18.
    Apply horizontal flipping only during training.
    """

    steps = [
        v2.Resize((224, 224)),
    ]

    if training:
        steps.append(
            v2.RandomHorizontalFlip(p=0.5)
        )

    steps.extend([
        v2.ToImage(),
        v2.ToDtype(
            torch.float32,
            scale=True,
        ),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    return v2.Compose(steps)


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset_classes(dataset, split_name):
    """
    Ensure the dataset folders match the model's class order.
    """

    expected = list(CLASS_NAMES)
    actual = list(dataset.classes)

    if set(actual) != set(expected):
        raise ValueError(
            f"\nClass folders in '{split_name}' do not match "
            f"the model classes.\n"
            f"Expected: {expected}\n"
            f"Found:    {actual}\n"
            f"Check the dataset folder names."
        )

    if actual != expected:
        raise ValueError(
            f"\nClass order mismatch in '{split_name}'.\n"
            f"Expected: {expected}\n"
            f"Found:    {actual}\n"
            f"ImageFolder sorts folder names alphabetically. "
            f"Make sure CLASS_NAMES in model.py uses that order."
        )

    if len(dataset) == 0:
        raise ValueError(
            f"No images found in the '{split_name}' dataset."
        )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, device):
    """
    Evaluate the model and calculate weighted metrics.
    """

    model.eval()

    actual = []
    predicted = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            actual.extend(
                labels.tolist()
            )

            predicted.extend(
                predictions.cpu().tolist()
            )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            actual,
            predicted,
            labels=list(range(len(CLASS_NAMES))),
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# ============================================================
# TRAINING
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train VisionGuard ResNet-18"
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to VisionGuard-Dataset",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "ml/saved_models/visionguard_resnet18.pt"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero")

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than zero")

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --------------------------------------------------------
    # Resolve paths
    # --------------------------------------------------------

    dataset_path = args.dataset

    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    output_path = args.output

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    train_path = dataset_path / "train"
    val_path = dataset_path / "val"

    print("\n" + "=" * 60)
    print("VISIONGUARD RESNET-18 TRAINING")
    print("=" * 60)

    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Dataset:      {dataset_path}")
    print(f"Output:       {output_path}")

    # --------------------------------------------------------
    # Check dataset directories
    # --------------------------------------------------------

    if not train_path.is_dir():
        raise FileNotFoundError(
            f"Training directory not found:\n{train_path}"
        )

    if not val_path.is_dir():
        raise FileNotFoundError(
            f"Validation directory not found:\n{val_path}"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
    else:
        print(
            "Training on CPU. Training may take longer."
        )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading training images...")

    train_dataset = ImageFolder(
        root=str(train_path),
        transform=get_transforms(training=True),
    )

    print("Loading validation images...")

    val_dataset = ImageFolder(
        root=str(val_path),
        transform=get_transforms(training=False),
    )

    # --------------------------------------------------------
    # Validate classes
    # --------------------------------------------------------

    validate_dataset_classes(
        train_dataset,
        "train",
    )

    validate_dataset_classes(
        val_dataset,
        "val",
    )

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            "Training and validation class mappings differ."
        )

    print("\nClasses:")

    for index, class_name in enumerate(CLASS_NAMES):
        print(f"  {index}: {class_name}")

    print(f"\nTraining images:   {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    num_workers = 0 if sys.platform == "win32" else 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print("\nCreating ResNet-18 model...")

    model = create_model(
        pretrained=True
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    best_f1 = -1.0
    best_metrics = None
    best_epoch = 0

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nStarting training...")
    print("-" * 60)

    for epoch in range(args.epochs):

        model.train()

        running_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            batch_size = labels.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

        train_loss = (
            running_loss / total_samples
            if total_samples > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        metrics = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"\nEpoch [{epoch + 1}/{args.epochs}]"
        )

        print(
            f"Training loss: {train_loss:.4f}"
        )

        print(
            f"Validation accuracy:  "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Validation precision: "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Validation recall:    "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"Validation F1:        "
            f"{metrics['f1']:.4f}"
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]
            best_metrics = metrics.copy()
            best_epoch = epoch + 1

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "metrics": best_metrics,
                "class_names": CLASS_NAMES,
                "training_config": {
                    "dataset": str(dataset_path),
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "seed": args.seed,
                    "best_epoch": best_epoch,
                },
            }

            torch.save(
                checkpoint,
                output_path,
            )

            print(
                f"✓ Best model saved: {output_path}"
            )

    # --------------------------------------------------------
    # Save best validation metrics
    # --------------------------------------------------------

    metrics_path = output_path.with_suffix(
        ".metrics.json"
    )

    summary = {
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "class_names": CLASS_NAMES,
        "checkpoint": str(output_path),
    }

    metrics_path.write_text(
        json.dumps(
            summary,
            indent=4,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation F1: {best_f1:.4f}")

    print(f"\nModel saved at:\n{output_path}")

    print(
        f"\nMetrics saved at:\n{metrics_path}"
    )

    print("\nNext step: evaluate the checkpoint on the")
    print("held-out test dataset before reporting final results.")


if __name__ == "__main__":
    main()