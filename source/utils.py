import logging

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

logger: logging.Logger = logging.getLogger(__name__)


def flatten_data(
    X_features_list: list[NDArray[np.float32]], y_labels_list: list[NDArray[np.uint8]]
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """
    Converts lists of 2D/3D arrays into 1D/2D flat arrays (pixel tables).
    This is the "flattening" step, preparing data for scikit-learn.

    :param X_features_list: A list of (H, W, C_features) feature maps.
    :param y_labels_list: A list of (H, W) label masks.
    :return: A tuple (X_flat, y_flat) where:
             X_flat is (N_total_pixels, C_features)
             y_flat is (N_total_pixels,)
    """
    logger.debug("Flattening data (stacking pixels)...")

    X_flat_list: list[NDArray[np.float32]] = [
        img.reshape(-1, img.shape[-1])
        for img in tqdm(X_features_list, desc="Flattening X")
        if img.size > 0
    ]

    y_flat_list: list[NDArray[np.uint8]] = [
        mask.reshape(-1) for mask in tqdm(y_labels_list, desc="Flattening y")
    ]

    if not X_flat_list or not y_flat_list:
        logger.error("No data to flatten. Lists are empty.")
        raise ValueError("Cannot flatten empty data lists.")

    X_flat: NDArray[np.float32] = np.concatenate(X_flat_list, axis=0)
    y_flat: NDArray[np.uint8] = np.concatenate(y_flat_list, axis=0)

    logger.debug(f"Flattening complete. X_flat shape: {X_flat.shape}, y_flat shape: {y_flat.shape}")
    return X_flat, y_flat
