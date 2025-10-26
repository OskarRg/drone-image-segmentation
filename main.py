"""
Main entrypoint for the terrain segmentation pipeline.

This script handles command-line argument parsing, logging setup,
and orchestrates the main processing steps (data loading,
feature extraction, training, evaluation).
"""

import argparse
import logging
import sys

from source.config import ConfigParser, PipelineArguments
from source.logging_config import setup_logging

logger: logging.Logger = logging.getLogger(__name__)


def run_pipeline(args: argparse.Namespace, params: PipelineArguments) -> None:
    """
    Main pipeline orchestration function.

    :param args: Parsed command-line arguments (e.g., mode, data_path).
    :type args: argparse.Namespace
    :param params: The loaded pipeline parameters (e.g., model, features).
    :type params: dict
    """
    logger.info(f"Starting pipeline in mode: {args.mode}")
    logger.info(f"Using parameters from: {args.params}")
    logger.info(f"Data path: {args.data_path}")
    logger.debug("Verbose logging enabled.")

    logger.info(f"Model config: {params.get('model', 'Not specified')}")

    logger.warning("Step 1: Data loading (NOT IMPLEMENTED)")
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
        params: PipelineArguments
        args, params = config_parser.parse_config()

        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Logging level set to DEBUG.")

        run_pipeline(args, params)

    except FileNotFoundError as e:
        logger.error(f"Exiting. Config file not found: {e}", exc_info=False)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with a critical error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
