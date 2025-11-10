"""
Main entrypoint for the terrain segmentation pipeline.

This script handles command-line argument parsing, logging setup,
and orchestrates the main processing steps (data loading,
feature extraction, training, evaluation).
"""

import argparse
import logging
import sys
from pathlib import Path

from source.arguments_schema import PipelineParams
from source.config import ConfigParser
from source.data_loader import DataLoader
from source.exit_codes import ExitCode
from source.logging_config import setup_logging

logger: logging.Logger = logging.getLogger(__name__)


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

    train_path: Path = Path(args.data_path) / "train"
    val_path: Path = Path(args.data_path) / "val"

    try:
        train_loader: DataLoader = DataLoader(data_path=train_path, params=params)
        X_train, y_train = train_loader.load_data()
        logger.debug(f"Successfully loaded {len(X_train)} training pairs.")

    except FileNotFoundError:
        logger.error(f"Training data not found at {train_path}.")
        logger.error("Did you run `scripts/prepare_dataset.py` first?")
        sys.exit(ExitCode.TRAINING_DATA_NOT_FOUND)
    except Exception as e:
        logger.error(f"Failed during training data loading: {e}", exc_info=True)
        sys.exit(ExitCode.GENERAL_ERROR)
    try:
        val_loader = DataLoader(data_path=val_path, params=params)
        X_val, y_val = val_loader.load_data()
        logger.debug(f"Successfully loaded {len(X_val)} validation pairs.")

    except FileNotFoundError:
        logger.error(f"Validation data not found at {val_path}.")
        logger.error("Did you run 'scripts/prepare_dataset.py' first?")
        sys.exit(ExitCode.VALIDATION_DATA_NOT_FOUND)
    except Exception as e:
        logger.error(f"Failed during validation data loading: {e}", exc_info=True)
        sys.exit(ExitCode.GENERAL_ERROR)

    logger.warning("Step 2: Feature extraction (NOT IMPLEMENTED)")

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

        run_pipeline(args, params)

    except FileNotFoundError as e:
        logger.error(f"Exiting. Config file not found: {e}", exc_info=False)
        sys.exit(ExitCode.CONFIG_FILE_NOT_FOUND)
    except Exception as e:
        logger.error(f"Pipeline failed with a critical error: {e}", exc_info=True)
        sys.exit(ExitCode.GENERAL_ERROR)


if __name__ == "__main__":
    sys.exit(main())
