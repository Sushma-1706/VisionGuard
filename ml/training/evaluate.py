from pathlib import Path
import sys
import json

# ------------------------------------------------------------
# Add project root to Python import path
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from backend.app.ml.model import CLASS_NAMES, create_model
from backend.app.ml.preprocessing import image_transform
import json

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from backend.app.ml.model import CLASS_NAMES, create_model
from backend.app.ml.preprocessing import image_transform


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "data" / "VisionGuard-Dataset"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "saved_models"
    / "visionguard_resnet18.pt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "saved_models"
    / "visionguard_resnet18.test.metrics.json"
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

BATCH_SIZE = 32


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate_model(model, loader, device):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("VISIONGUARD TEST SET EVALUATION")
    print("=" * 60)

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print(f"\nDataset:")
    print(DATASET_PATH)

    print(f"\nCheckpoint:")
    print(CHECKPOINT_PATH)

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    test_path = DATASET_PATH / "test"

    if not test_path.exists():

        raise FileNotFoundError(
            f"Test dataset not found: {test_path}"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    print("\nLoading test images...")

    test_dataset = ImageFolder(
        test_path,
        transform=image_transform(224),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    print(
        f"Classes: {test_dataset.classes}"
    )

    # --------------------------------------------------------
    # Verify class ordering
    # --------------------------------------------------------

    if test_dataset.classes != CLASS_NAMES:

        raise RuntimeError(
            "\nClass order mismatch!\n"
            f"Dataset classes: {test_dataset.classes}\n"
            f"Model classes:   {CLASS_NAMES}"
        )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print("\nCreating ResNet-18 model...")

    model = create_model(
        pretrained=False
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print("\nLoading trained checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    print("Checkpoint loaded successfully.")

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print("\nEvaluating on held-out test dataset...")
    print("-" * 60)

    metrics = evaluate_model(
        model,
        test_loader,
        device,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nTEST RESULTS")
    print("=" * 60)

    print(
        f"Accuracy:  {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall:    {metrics['recall']:.4f}"
    )

    print(
        f"F1 Score:  {metrics['f1']:.4f}"
    )

    # --------------------------------------------------------
    # Per-class results
    # --------------------------------------------------------

    print("\nPER-CLASS RESULTS")
    print("=" * 60)

    for class_name in CLASS_NAMES:

        class_metrics = metrics[
            "classification_report"
        ][class_name]

        print(
            f"\n{class_name}"
        )

        print(
            f"  Precision: "
            f"{class_metrics['precision']:.4f}"
        )

        print(
            f"  Recall:    "
            f"{class_metrics['recall']:.4f}"
        )

        print(
            f"  F1:        "
            f"{class_metrics['f1-score']:.4f}"
        )

        print(
            f"  Samples:   "
            f"{int(class_metrics['support'])}"
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("\nCONFUSION MATRIX")
    print("=" * 60)

    print(
        "Rows = actual class"
    )

    print(
        "Columns = predicted class"
    )

    print(
        f"\nClasses: {CLASS_NAMES}\n"
    )

    for row in metrics["confusion_matrix"]:

        print(row)

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    output = {
        "checkpoint": str(CHECKPOINT_PATH),
        "dataset": str(test_path),
        "split": "held-out test",
        "class_names": CLASS_NAMES,
        "metrics": metrics,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        f"\nTest metrics saved at:"
        f"\n{OUTPUT_PATH}"
    )

    print("\n" + "=" * 60)
    print("TEST EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()