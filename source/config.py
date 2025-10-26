import argparse
import logging
import os
from typing import Any

import yaml

logger: logging.Logger = logging.getLogger(__name__)

PipelineArguments = dict[str, Any]


class ConfigParser:
    """
    Handles argument parsing and loading the pipeline parameters file.

    This class encapsulates all logic related to defining, parsing,
    and using command-line arguments to load the correct params.yaml.
    """

    def __init__(self) -> None:
        """
        Initializes the ArgumentParser and defines all expected arguments.
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

    def _load_params(self, params_path: str) -> PipelineArguments:
        """
        Private helper to load the params.yaml file.

        :param params_path: Path to the YAML parameters file.
        :type params_path: str
        :raises FileNotFoundError: If the config file does not exist.
        :return: A dictionary with the loaded parameters.
        :rtype: dict
        """
        if not os.path.exists(params_path):
            logger.error(f"Parameters file not found at: {params_path}")
            raise FileNotFoundError(f"Parameters file not found at: {params_path}")

        try:
            with open(params_path, "r") as f:
                params: PipelineArguments = yaml.safe_load(f)
            logger.info(f"Successfully loaded parameters from {params_path}")
            return params if params is not None else {}
        except Exception as e:
            logger.error(f"Failed to read/parse params file: {e}", exc_info=True)
            raise e

    def parse_config(self) -> tuple[argparse.Namespace, PipelineArguments]:
        """
        Public method to parse arguments and load the parameters file.

        :return: A tuple containing:
                 1. The parsed arguments (Namespace).
                 2. The loaded parameters dictionary (dict).
        :rtype: tuple[argparse.Namespace, dict[str, Any]]
        """
        args: argparse.Namespace = self.parser.parse_args()
        params: PipelineArguments = self._load_params(args.params)

        return args, params
