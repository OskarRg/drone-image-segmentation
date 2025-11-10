"""
Handles argument parsing and loading the pipeline parameters file.

This class encapsulates all logic related to defining, parsing,
and using command-line arguments to load the correct params.yaml.
It also validates the loaded params.yaml against the Pydantic schema.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from source.arguments_schema import PipelineParams
from source.exit_codes import ExitCode

logger: logging.Logger = logging.getLogger(__name__)

DATA_PATH: Path = Path("data/raw/")
BINARY_DATASET_PATH: str = DATA_PATH / "binary_dataset"
CLASSES_DATASET_PATH: str = DATA_PATH / "classes_dataset"


class ConfigParser:
    """
    Handles argument parsing and loading the pipeline parameters file.
    """

    def __init__(self) -> None:
        """
        Initializes the `ArgumentParser` and defines all expected arguments.
        """
        self.parser: argparse.ArgumentParser = argparse.ArgumentParser(
            description="Main pipeline for drone image terrain segmentation."
        )

        self.parser.add_argument(
            "--params",
            type=str,
            default="config/params.yaml",
            help="Path to the *pipeline parameters* file (.yaml).",
        )

        self.parser.add_argument(
            "--data-path",
            type=str,
            default="data/processed",
            help="Path to the processed data folder (train/val/test).",
        )

        self.parser.add_argument(
            "--mode",
            type=str,
            default="full",
            choices=["full", "train", "evaluate"],
            help="Pipeline run mode: 'full' (train/eval), 'train', 'evaluate'.",
        )

        self.parser.add_argument(
            "--verbose",
            action="store_true",
            default=False,
            help="Enable verbose logging (sets level to DEBUG).",
        )

    def _load_params(self, params_path: str) -> PipelineParams:
        """
        Private helper to load the params.yaml file and validate it.

        :param params_path: Path to the YAML parameters file.
        :raises FileNotFoundError: If the config file does not exist.
        :raises ValidationError: If the config file is invalid.
        :return: A validated Pydantic object with the loaded parameters.
        """
        if not os.path.exists(params_path):
            logger.error(f"Parameters file not found at: {params_path}")
            raise FileNotFoundError(f"Parameters file not found at: {params_path}")

        try:
            with open(params_path, "r") as f:
                params_dict: PipelineParams = yaml.safe_load(f)
                if params_dict is None:
                    params_dict = {}
            params_obj: PipelineParams = PipelineParams(**params_dict)

            logger.info(f"Successfully loaded and validated parameters from {params_path}")
            return params_obj

        except ValidationError as e:
            logger.error("--- Invalid 'params.yaml' configuration! ---")
            logger.error(f"File: {params_path}")
            logger.error(f"{e}")
            sys.exit(ExitCode.CONFIG_VALIDATION_ERROR)

        except Exception as e:
            logger.error(f"Failed to read and parse params file: {e}", exc_info=True)
            raise e

    def parse_config(self) -> tuple[argparse.Namespace, PipelineParams]:
        """
        Public method to parse arguments and load the parameters file.

        :return: A tuple containing:
                 1. The parsed arguments (Namespace).
                 2. The loaded and validated parameters (Pydantic object).
        """
        args: argparse.Namespace = self.parser.parse_args()
        params: PipelineParams = self._load_params(args.params)

        return args, params
