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


class PreprocessingParams(BaseModel):
    """Schema for the 'preprocessing' section."""

    color_map: list[ColorMapItem]


class ColorSpaceConfig(BaseModel):
    """Schema for 'features.color_space'."""

    hsv: bool = True
    lab: bool = True


class LBPConfig(BaseModel):
    """Schema for 'features.texture_lbp'."""

    enabled: bool = True
    radius: int
    n_points: int


class SobelConfig(BaseModel):
    """Schema for 'features.edge_sobel'."""

    enabled: bool = True


class FeatureParams(BaseModel):
    """
    Defines the schema for the main 'features' section.
    This is the class expected by the `FeatureExtractor`.
    """

    base_rgb: bool = True
    color_space: ColorSpaceConfig
    texture_lbp: LBPConfig
    edge_sobel: SobelConfig


class ArtifactsConfig(BaseModel):
    """Schema for the extracted features section."""

    output_dir: str
    x_train_path: str
    y_train_path: str
    x_val_path: str
    y_val_path: str
    scaler_path: str


class PipelineParams(BaseModel):
    """
    Main schema class representing the entire `params.yaml` file.

    The `ConfigParser` will load the YAML file and validate it
    against this class, returning an instance of it.
    """

    preprocessing: PreprocessingParams
    features: FeatureParams
    artifacts: ArtifactsConfig | None = None
