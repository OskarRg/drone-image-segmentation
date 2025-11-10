"""
Data Preparation Script.

This script is run ONLY ONCE to convert raw data
(e.g., 'data/raw') into a ready-to-use folder structure
(e.g., 'data/processed') with 'train', 'val', and 'test' splits.

Tasks:
1. Finds all image-mask pairs in 'data/raw'.
2. Splits these pairs into training, validation, and test sets.
3. Copies the files into the new 'data/processed' folder structure.

Usage Example:
`python scripts/prepare_dataset.py --raw-data data/raw --processed-data data/processed`

# TODO Create specified exit codes
"""

import argparse
import glob
import logging
import os
import shutil
import sys

from sklearn.model_selection import train_test_split

try:
    from source.logging_config import setup_logging
except ImportError:
    print("Cannot import 'source.logging_config'. Using basic configuration.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def setup_logging(level: int = logging.INFO) -> None:
        pass


logger: logging.Logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    :return: A namespace containing the parsed arguments.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Prepares and splits the dataset."
    )
    parser.add_argument(
        "--raw-data",
        type=str,
        default="data/raw",
        help="Path to the raw data folder (containing 'images' and 'masks').",
    )
    parser.add_argument(
        "--processed-data",
        type=str,
        default="data/processed",
        help="Destination path for the split data (will create 'train', 'val', 'test' folders).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Percentage of data for the test set (e.g., 0.2 for 20%).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.25,
        help="Percentage of *remaining* data (after test split) for the validation set.",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducible splits."
    )
    return parser.parse_args()


def copy_files(file_pairs: list[tuple[str, str]], dest_dir: str) -> None:
    """
    Copies a list of file pairs (image, mask) to a destination structure.

    :param file_pairs: List of tuples, e.g., [('path/to/img1.jpg', 'path/to/mask1.png'), ...]
    :param dest_dir: Destination folder, e.g., 'data/processed/train'
    """
    img_dest: str = os.path.join(dest_dir, "original_images")
    mask_dest: str = os.path.join(dest_dir, "masks")

    os.makedirs(img_dest, exist_ok=True)
    os.makedirs(mask_dest, exist_ok=True)

    for img_path, mask_path in file_pairs:
        try:
            shutil.copy(img_path, img_dest)
            shutil.copy(mask_path, mask_dest)
        except Exception as e:
            logger.error(f"Failed to copy pair {img_path}, {mask_path}: {e}")

    logger.info(f"Finished copying {len(file_pairs)} pairs to {dest_dir}")


def main() -> int:
    """Main script function."""
    setup_logging(logging.INFO)
    args: argparse.Namespace = parse_args()

    logger.info(f"Starting data preparation from: {args.raw_data}")
    logger.info(f"Output data will be in: {args.processed_data}")

    if os.path.exists(args.processed_data):
        logger.warning(
            f"Target folder '{args.processed_data}' already exists. "
            f"It might contain old data. Consider removing it and running the script again."
        )

    img_files: list[str] = sorted(
        glob.glob(os.path.join(args.raw_data, "original_images", "*.png"))
    )
    mask_files: list[str] = sorted(glob.glob(os.path.join(args.raw_data, "masks", "*.png")))

    if not img_files or len(img_files) != len(mask_files):
        logger.error(
            f"Error: Found {len(img_files)} images and {len(mask_files)} masks. "
            "Counts must match and cannot be zero. Aborting."
        )
        return 1

    all_pairs: list[tuple[str, str]] = list(zip(img_files, mask_files))
    logger.info(f"Found {len(all_pairs)} total file pairs.")

    train_val_pairs: list[tuple[str, str]]
    test_pairs: list[tuple[str, str]]
    train_val_pairs, test_pairs = train_test_split(
        all_pairs, test_size=args.test_size, random_state=args.random_seed
    )

    train_pairs: list[tuple[str, str]]
    val_pairs: list[tuple[str, str]]
    train_pairs, val_pairs = train_test_split(
        train_val_pairs, test_size=args.val_size, random_state=args.random_seed
    )

    logger.info("Split complete:")
    logger.info(f"  Training set:     {len(train_pairs)} files")
    logger.info(f"  Validation set:   {len(val_pairs)} files")
    logger.info(f"  Test set:         {len(test_pairs)} files")

    copy_files(train_pairs, os.path.join(args.processed_data, "train"))
    copy_files(val_pairs, os.path.join(args.processed_data, "val"))
    copy_files(test_pairs, os.path.join(args.processed_data, "test"))

    logger.info("Data preparation finished successfully!")


if __name__ == "__main__":
    sys.exit(main())
