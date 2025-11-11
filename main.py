"""
Main entrypoint for the terrain segmentation pipeline.

This script handles command-line argument parsing, logging setup,
and orchestrates the main processing steps (data loading,
feature extraction, training, evaluation).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.discriminant_analysis import StandardScaler

from source.arguments_schema import PipelineParams
from source.config import ConfigParser
from source.data_loader import DataLoader
from source.exit_codes import ExitCode
from source.feature_extractor import FeatureExtractor
from source.logging_config import setup_logging
from source.utils import flatten_data

logger: logging.Logger = logging.getLogger(__name__)


def _load_features_from_cache(
    params: PipelineParams,
) -> tuple[
    NDArray[np.float32], NDArray[np.uint8], NDArray[np.float32], NDArray[np.uint8], StandardScaler
]:
    """
    Loads pre-processed data (X, y) and the scaler from artifact files.

    :param params: The validated pipeline parameters object.
    :raises FileNotFoundError: If any of the required artifact files are missing.
    :return: A tuple containing (X_train_scaled, y_train_flat,
             X_val_scaled, y_val_flat, scaler).
    """
    logger.info("Loading features from cache ...")

    paths_to_check: list[str] = [
        params.artifacts.x_train_path,
        params.artifacts.y_train_path,
        params.artifacts.x_val_path,
        params.artifacts.y_val_path,
        params.artifacts.scaler_path,
    ]
    for path in paths_to_check:
        if not os.path.exists(path):
            logger.error(f"Cache file not found at: {path}")
            logger.error(
                "Run the pipeline without '--load-features' and with '--save-features' first."
            )
            raise FileNotFoundError(f"Missing cache file: {path}")

    X_train_scaled = np.load(params.artifacts.x_train_path)
    y_train_flat = np.load(params.artifacts.y_train_path)
    X_val_scaled = np.load(params.artifacts.x_val_path)
    y_val_flat = np.load(params.artifacts.y_val_path)

    scaler: StandardScaler = joblib.load(params.artifacts.scaler_path)

    logger.info("Successfully loaded all features and scaler from cache.")

    return X_train_scaled, y_train_flat, X_val_scaled, y_val_flat, scaler


def _load_and_process_data(
    args: argparse.Namespace, params: PipelineParams
) -> tuple[
    NDArray[np.float32], NDArray[np.uint8], NDArray[np.float32], NDArray[np.uint8], StandardScaler
]:
    """
    Performs the full, slow data processing pipeline (Steps 1 and 2).

    :param args: The command-line arguments.
    :param params: The validated pipeline parameters object.
    :return: A tuple containing (X_train_scaled, y_train_flat,
             X_val_scaled, y_val_flat, scaler).
    """
    logger.debug("Processing data from scratch...")

    train_path: Path = Path(args.data_path) / "train"
    val_path: Path = Path(args.data_path) / "val"

    try:
        train_loader: DataLoader = DataLoader(
            data_path=str(train_path), params=params.preprocessing
        )
        X_train, y_train = train_loader.load_data()
        logger.debug(f"Successfully loaded {len(X_train)} training pairs.")
    except FileNotFoundError:
        logger.error(f"Training data not found at {train_path}.")
        logger.error("Did you run 'scripts/prepare_dataset.py' first?")
        raise  # TODO Add custom exceptions

    try:
        val_loader: DataLoader = DataLoader(data_path=str(val_path), params=params.preprocessing)
        X_val, y_val = val_loader.load_data()
        logger.debug(f"Successfully loaded {len(X_val)} validation pairs.")
    except FileNotFoundError:
        logger.error(f"Validation data not found at {val_path}.")
        logger.error("Did you run 'scripts/prepare_dataset.py' first?")
        raise  # TODO Add custom exceptions

    logger.info("Step 2: Extracting features...")
    feature_extractor = FeatureExtractor(params=params.features)

    X_train_features = feature_extractor.process_batch(X_train)
    X_val_features = feature_extractor.process_batch(X_val)

    X_train_flat, y_train_flat = flatten_data(X_train_features, y_train)
    X_val_flat, y_val_flat = flatten_data(X_val_features, y_val)

    logger.debug("Starting normalization...")
    scaler = StandardScaler()

    logger.debug("Fitting StandardScaler on training data...")
    scaler.fit(X_train_flat)

    logger.debug("Transforming training data...")
    X_train_scaled = scaler.transform(X_train_flat)

    logger.debug("Transforming validation data...")
    X_val_scaled = scaler.transform(X_val_flat)
    logger.debug("Normalization complete.")

    if args.save_features:
        artifacts_dir: str = params.artifacts.output_dir
        logger.info(f"Saving processed artifacts to {artifacts_dir}")
        os.makedirs(artifacts_dir, exist_ok=True)

        try:
            np.save(params.artifacts.x_train_path, X_train_scaled)
            np.save(params.artifacts.y_train_path, y_train_flat)
            np.save(params.artifacts.x_val_path, X_val_scaled)
            np.save(params.artifacts.y_val_path, y_val_flat)
            joblib.dump(scaler, params.artifacts.scaler_path)
            logger.debug(f"Successfully saved all artifacts to {artifacts_dir}")
        except Exception as e:
            logger.error(f"Failed to save artifacts: {e}", exc_info=True)

    return X_train_scaled, y_train_flat, X_val_scaled, y_val_flat, scaler


def run_pipeline(args: argparse.Namespace, params: PipelineParams) -> None:
    """
    Main pipeline orchestration function.

    :param args: Parsed command-line arguments (e.g., mode, data_path).
    :type args: argparse.Namespace
    :param params: The loaded pipeline parameters (e.g., model, features).
    :type params: dict
    """
    logger.info(f"Starting pipeline in mode: {args.mode}")
    logger.info(f"Using parameters from: {args.params}")
    logger.debug(f"Parameters: {args}")
    logger.debug(f"Data path: {args.data_path}")
    logger.debug("Verbose logging enabled.")

    if args.load_features:
        X_train, y_train, X_val, y_val, scaler = _load_features_from_cache(params)
    else:
        X_train, y_train, X_val, y_val, scaler = _load_and_process_data(args, params)

    if args.mode in ["full", "train"]:
        logger.info("Step 3: Training model...")
        logger.warning("Step 3: Model training (NOT IMPLEMENTED)")

    if args.mode in ["full", "evaluate"]:
        logger.info("Step 4: Evaluating model...")
        logger.warning("Step 4: Model evaluation (NOT IMPLEMENTED)")

    logger.info("Pipeline finished successfully.")


def main() -> None:
    """
    Main script entrypoint.
    """
    setup_logging(level=logging.INFO)

    try:
        config_parser: ConfigParser = ConfigParser()
        args: argparse.Namespace
        params: PipelineParams
        args, params = config_parser.parse_config()

        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Logging level set to DEBUG.")

        if args.load_features and args.save_features:
            logger.warning("Both '--load-features' and '--save-features' were specified.")
            logger.warning("Ignoring '--save-features' and loading from cache.")
            args.save_features = False

        run_pipeline(args, params)

    except FileNotFoundError as e:
        logger.error(f"Exiting. Config file not found: {e}", exc_info=False)
        sys.exit(ExitCode.CONFIG_FILE_NOT_FOUND)
    except Exception as e:
        logger.error(f"Pipeline failed with a critical error: {e}", exc_info=True)
        sys.exit(ExitCode.GENERAL_ERROR)


if __name__ == "__main__":
    sys.exit(main())
