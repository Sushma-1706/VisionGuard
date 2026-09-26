from pathlib import Path
import random
import shutil


# ============================================================
# Configuration
# ============================================================

SEED = 42

# We will take 80% of the original training images
# for training and 20% for validation.
TRAIN_RATIO = 0.80
VAL_RATIO = 0.20


# Find the VisionGuard project root automatically.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Original NEU dataset
SOURCE_TRAIN = (
    PROJECT_ROOT
    / "data"
    / "NEU-DET"
    / "train"
    / "images"
)

SOURCE_TEST = (
    PROJECT_ROOT
    / "data"
    / "NEU-DET"
    / "validation"
    / "images"
)


# New dataset used by VisionGuard
OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "VisionGuard-Dataset"
)


# These MUST exactly match the folders
# in your NEU dataset.
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================
# Helper functions
# ============================================================

def get_images(folder: Path):
    """Return all image files inside a folder."""

    return sorted(
        [
            file
            for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower()
            in IMAGE_EXTENSIONS
        ]
    )


def create_output_directories():
    """Create train/val/test class directories."""

    for split in ["train", "val", "test"]:

        for class_name in CLASSES:

            directory = (
                OUTPUT
                / split
                / class_name
            )

            directory.mkdir(
                parents=True,
                exist_ok=True,
            )


def copy_images(
    images,
    destination,
):
    """Copy images to destination directory."""

    for image in images:

        destination_file = (
            destination
            / image.name
        )

        shutil.copy2(
            image,
            destination_file,
        )


# ============================================================
# Main
# ============================================================

def main():

    random.seed(SEED)

    print()
    print("=" * 60)
    print("VisionGuard Dataset Preparation")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Check source folders
    # --------------------------------------------------------

    print("Checking dataset...")

    if not SOURCE_TRAIN.exists():

        raise FileNotFoundError(
            f"\nTraining images not found:\n"
            f"{SOURCE_TRAIN}\n"
        )

    if not SOURCE_TEST.exists():

        raise FileNotFoundError(
            f"\nValidation images not found:\n"
            f"{SOURCE_TEST}\n"
        )

    print("✓ Original training folder found")
    print("✓ Original validation folder found")
    print()

    # --------------------------------------------------------
    # Create output folders
    # --------------------------------------------------------

    print("Creating VisionGuard-Dataset folders...")

    create_output_directories()

    print("✓ Output folders created")
    print()

    # --------------------------------------------------------
    # Process original training dataset
    # --------------------------------------------------------

    print("=" * 60)
    print("PROCESSING ORIGINAL TRAINING DATA")
    print("=" * 60)

    total_train = 0
    total_val = 0

    for class_name in CLASSES:

        source_class_folder = (
            SOURCE_TRAIN
            / class_name
        )

        if not source_class_folder.exists():

            raise FileNotFoundError(
                f"\nClass folder not found:\n"
                f"{source_class_folder}\n"
            )

        images = get_images(
            source_class_folder
        )

        if len(images) == 0:

            raise ValueError(
                f"\nNo images found in:\n"
                f"{source_class_folder}\n"
            )

        # Shuffle deterministically
        random.shuffle(images)

        total = len(images)

        train_count = int(
            total * TRAIN_RATIO
        )

        train_images = images[
            :train_count
        ]

        val_images = images[
            train_count:
        ]

        # Destination folders
        train_destination = (
            OUTPUT
            / "train"
            / class_name
        )

        val_destination = (
            OUTPUT
            / "val"
            / class_name
        )

        # Copy
        copy_images(
            train_images,
            train_destination,
        )

        copy_images(
            val_images,
            val_destination,
        )

        total_train += len(
            train_images
        )

        total_val += len(
            val_images
        )

        print(
            f"{class_name:18} "
            f"original={total:3} | "
            f"train={len(train_images):3} | "
            f"val={len(val_images):3}"
        )

    # --------------------------------------------------------
    # Process original validation dataset
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PROCESSING ORIGINAL VALIDATION DATA")
    print("=" * 60)

    total_test = 0

    for class_name in CLASSES:

        source_class_folder = (
            SOURCE_TEST
            / class_name
        )

        if not source_class_folder.exists():

            raise FileNotFoundError(
                f"\nClass folder not found:\n"
                f"{source_class_folder}\n"
            )

        images = get_images(
            source_class_folder
        )

        if len(images) == 0:

            raise ValueError(
                f"\nNo images found in:\n"
                f"{source_class_folder}\n"
            )

        test_destination = (
            OUTPUT
            / "test"
            / class_name
        )

        copy_images(
            images,
            test_destination,
        )

        total_test += len(images)

        print(
            f"{class_name:18} "
            f"test={len(images):3}"
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Training images   : {total_train}"
    )

    print(
        f"Validation images : {total_val}"
    )

    print(
        f"Test images       : {total_test}"
    )

    print(
        f"Total images      : "
        f"{total_train + total_val + total_test}"
    )

    print()
    print(
        f"Dataset location:"
    )

    print(OUTPUT)

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()