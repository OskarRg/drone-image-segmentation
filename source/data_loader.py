"""
DataLoader is responsible for:
1. Finding all image and mask files.
2. Loading image-mask pairs.
3. Converting color-coded RGB masks to 2D index masks.
"""

import glob
import logging
import os

import cv2
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from source.arguments_schema import ColorMapItem, PipelineParams

logger: logging.Logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and preprocessing of the segmentation dataset.
    """

    def __init__(self, data_path: str, params: PipelineParams) -> None:
        """
        Initializes the DataLoader.

        :param data_path: Path to the processed data folder.
                          Expects 'images' and 'masks' folders inside.
        :param params: The validated Pydantic object containing all
                       pipeline parameters from 'params.yaml'.
        :type params: PipelineParams
        """
        logger.info(f"Initializing DataLoader with data_path: {data_path}")

        self.params: PipelineParams = params

        color_map_config: list[ColorMapItem] = params.preprocessing.color_map

        self.colors_rgb: NDArray[np.uint8] = np.array(
            [item.color for item in color_map_config], dtype=np.uint8
        )
        self.class_ids: NDArray[np.uint8] = np.array(
            [item.id for item in color_map_config], dtype=np.uint8
        )
        self.class_names: dict[int, str] = {item.id: item.name for item in color_map_config}
        logger.debug(f"Loaded color map for {len(self.class_ids)} classes.")

        self.image_path: str = os.path.join(data_path, "original_images")
        self.mask_path: str = os.path.join(data_path, "masks")

        self.image_files: list[str] = sorted(glob.glob(os.path.join(self.image_path, "*.png")))
        self.mask_files: list[str] = sorted(glob.glob(os.path.join(self.mask_path, "*.png")))

        self._validate_files()

    def _validate_files(self) -> None:
        """
        Helper function to check if image and mask file counts match.

        :raises FileNotFoundError: If no images or masks are found.
        """
        if not self.image_files:
            raise FileNotFoundError(f"No images found in {self.image_path}")
        if not self.mask_files:
            raise FileNotFoundError(f"No masks found in {self.mask_path}")

        if len(self.image_files) != len(self.mask_files):
            logger.warning(
                f"File count mismatch! Found {len(self.image_files)} images "
                f"but {len(self.mask_files)} masks."
            )

        logger.debug(f"Found {len(self.image_files)} image-mask pairs.")

    def _convert_mask_to_index(self, mask_rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Converts an RGB color mask to a 2D index (class) mask.

        This method iterates over the predefined color map and assigns
        a class ID to all pixels matching a specific RGB color.

        :param mask_rgb: The (H, W, 3) color mask (RGB).
        :return: The (H, W) index mask with class IDs (0-N).
        """
        height: int
        width: int
        height, width, _ = mask_rgb.shape
        mask_index: NDArray[np.uint8] = np.zeros((height, width), dtype=np.uint8)

        color: NDArray[np.uint8]
        for i, color in enumerate(self.colors_rgb):
            matches: NDArray[np.bool_] = (mask_rgb == color).all(axis=2)
            mask_index[matches] = self.class_ids[i]

        return mask_index

    def load_data(self) -> tuple[list[NDArray[np.uint8]], list[NDArray[np.uint8]]]:
        """
        Loads all data, preprocesses it, and returns it as lists of NumPy arrays.

        :return: A tuple (X, y) where:
                 X is a list of (H, W, 3) images (np.uint8).
                 y is a list of (H, W) index masks (np.uint8).
        """
        logger.debug(f"Loading and processing {len(self.image_files)} files...")

        processed_images: list[NDArray[np.uint8]] = []
        processed_masks: list[NDArray[np.uint8]] = []

        img_file: str
        mask_file: str
        for img_file, mask_file in tqdm(
            zip(self.image_files, self.mask_files), total=len(self.image_files), desc="Loading data"
        ):
            img_bgr: NDArray[np.uint8] | None = cv2.imread(img_file)
            mask_bgr: NDArray[np.uint8] | None = cv2.imread(mask_file)

            if img_bgr is None or mask_bgr is None:
                logger.warning(f"Failed to load pair: {img_file}, {mask_file}")
                continue

            img_rgb: NDArray[np.uint8] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mask_rgb: NDArray[np.uint8] = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

            mask_index: NDArray[np.uint8] = self._convert_mask_to_index(mask_rgb)

            processed_images.append(img_rgb)
            processed_masks.append(mask_index)

        logger.debug(f"Data loading complete. Loaded {len(processed_images)} pairs.")

        return processed_images, processed_masks
