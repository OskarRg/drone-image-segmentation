"""
Pydantic Schema for config/params.yaml

This module defines the expected structure, types, and constraints
for the `params.yaml` configuration file. It ensures that the pipeline
configuration is valid upon loading.
"""

from pydantic import BaseModel


class ColorMapItem(BaseModel):
    """Defines one item in the color map list."""

    color: list[int]
    id: int
    name: str


class PreprocessingConfig(BaseModel):
    """Schema for the 'preprocessing' section."""

    color_map: list[ColorMapItem]


class PipelineParams(BaseModel):
    """
    Top-level schema for the entire 'params.yaml' file.
    """

    preprocessing: PreprocessingConfig
