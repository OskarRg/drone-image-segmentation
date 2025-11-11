"""
Feature Extractor Module.

This module is responsible for converting raw image data (pixels)
into a high-dimensional feature vector that a classical machine learning
model can consume.

It extracts features based on color, texture, and edges as configured
in the 'params.yaml' file.
"""

import logging

import numpy as np
from numpy.typing import NDArray
from skimage import color, feature, filters
from tqdm import tqdm

from source.arguments_schema import FeatureParams

logger: logging.Logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts a set of features from a list of raw images.

    This class is initialized with configuration from 'params.yaml'
    which specifies which features (e.g., HSV, L*a*b*, LBP) to enable.
    """

    def __init__(self, params: FeatureParams) -> None:
        """
        Initializes the FeatureExtractor with pipeline parameters.

        :param params: The 'features' section of the Pydantic
                       PipelineParams object.
        :type params: FeatureParams
        """
        self.params: FeatureParams = params
        logger.debug("FeatureExtractor initialized.")
        logger.debug(f"Feature config: {self.params}")

    def _extract_color_features(self, img_rgb: NDArray[np.uint8]) -> list[NDArray[np.float32]]:
        """
        Extracts all configured color-based features (HSV, L*a*b*).

        :param img_rgb: A single (H, W, 3) RGB image.
        :return: A list of 2D feature maps.
        """

        features: list[NDArray[np.float32]] = []

        if self.params.color_space.hsv:
            try:
                img_hsv: NDArray[np.float64] = color.rgb2hsv(img_rgb)
                features.extend(
                    [
                        img_hsv[..., 0].astype(np.float32),
                        img_hsv[..., 1].astype(np.float32),
                        img_hsv[..., 2].astype(np.float32),
                    ]
                )
            except Exception as e:
                logger.warning(f"Failed to extract HSV features: {e}", exc_info=True)

        if self.params.color_space.lab:
            try:
                img_lab: NDArray[np.float64] = color.rgb2lab(img_rgb)
                features.extend(
                    [
                        img_lab[..., 0].astype(np.float32),
                        img_lab[..., 1].astype(np.float32),
                        img_lab[..., 2].astype(np.float32),
                    ]
                )
            except Exception as e:
                logger.warning(f"Failed to extract L*a*b* features: {e}", exc_info=True)

        return features

    def _extract_texture_features(self, img_gray: NDArray[np.uint8]) -> list[NDArray[np.float32]]:
        """
        Extracts all configured texture-based features (LBP).

        :param img_gray: A single (H, W) grayscale image.
        :return: A list of 2D feature maps.
        """

        features: list[NDArray[np.float32]] = []

        if self.params.texture_lbp.enabled:
            try:
                radius: int = self.params.texture_lbp.radius
                n_points: int = self.params.texture_lbp.n_points

                lbp: NDArray[np.float64] = feature.local_binary_pattern(
                    img_gray, n_points, radius, method="uniform"
                )
                features.append(lbp.astype(np.float32))
            except Exception as e:
                logger.warning(f"Failed to extract LBP features: {e}", exc_info=True)

        return features

    def _extract_edge_features(self, img_gray: NDArray[np.uint8]) -> list[NDArray[np.float32]]:
        """
        Extracts all configured edge-based features (Sobel).

        :param img_gray: A single (H, W) grayscale image.
        :return: A list of 2D feature maps.
        """

        features: list[NDArray[np.float32]] = []

        if self.params.edge_sobel.enabled:
            try:
                # skimage.filters.sobel returns the edge magnitude
                sobel_mag: NDArray[np.float64] = filters.sobel(img_gray)
                features.append(sobel_mag.astype(np.float32))
            except Exception as e:
                logger.warning(f"Failed to extract Sobel features: {e}", exc_info=True)

        return features

    def process_batch(self, images: list[NDArray[np.uint8]]) -> list[NDArray[np.float32]]:
        """
        Processes a batch of images and extracts features for each one.

        This is the main public method of the class. It iterates over
        a list of images and returns a list of corresponding feature maps.

        :param images: A list of (H, W, 3) RGB images.
        :return: A list of (H, W, N_features) feature maps.
        """
        logger.debug(f"Starting feature extraction for {len(images)} images...")

        processed_features_list: list[NDArray[np.float32]] = []

        img_rgb: NDArray[np.uint8]
        for img_rgb in tqdm(images, desc="Extracting features"):
            all_features: list[NDArray[np.float32]] = []

            # 1. Base RGB features
            if self.params.base_rgb:
                img_float: NDArray[np.float32] = img_rgb.astype(np.float32) / 255.0
                all_features.extend([img_float[..., 0], img_float[..., 1], img_float[..., 2]])

            # 2. Color features (HSV, L*a*b*)
            color_feats: list[NDArray[np.float32]] = self._extract_color_features(img_rgb)
            all_features.extend(color_feats)

            # 3. Grayscale conversion (needed for texture/edge)
            # Note: skimage.color.rgb2gray returns float64 in [0, 1]
            img_gray_float: NDArray[np.float64] = color.rgb2gray(img_rgb)
            img_gray_u8: NDArray[np.uint8] = (img_gray_float * 255).astype(np.uint8)

            # 4. Texture features (LBP)
            texture_feats: list[NDArray[np.float32]] = self._extract_texture_features(img_gray_u8)
            all_features.extend(texture_feats)

            # 5. Edge features (Sobel)
            edge_feats: list[NDArray[np.float32]] = self._extract_edge_features(img_gray_u8)
            all_features.extend(edge_feats)

            # 6. Stack all 2D feature maps into a single 3D array (H, W, N_features)
            if not all_features:
                logger.warning(
                    f"No features enabled for image of shape {img_rgb.shape}. "
                    "Skipping feature extraction for this image."
                )
                # TODO Handle appropriately
                processed_features_list.append(
                    np.empty((img_rgb.shape[0], img_rgb.shape[1], 0), dtype=np.float32)
                )
                continue

            try:
                final_features: NDArray[np.float32] = np.stack(all_features, axis=-1).astype(
                    np.float32
                )
                processed_features_list.append(final_features)

            except ValueError as e:
                logger.error(
                    "Failed to stack features for image. "
                    f"All feature maps must have the same shape. Error: {e}",
                    exc_info=True,
                )
                # TODO Create custom exceptions
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred during feature stacking: {e}", exc_info=True
                )

        feature_count: int = processed_features_list[0].shape[-1] if processed_features_list else 0
        logger.info(
            f"Feature extraction complete. "
            f"Extracted {feature_count} features for {len(processed_features_list)} images."
        )
        return processed_features_list
