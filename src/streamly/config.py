"""Typed, fail-fast configuration contracts for the Streamly pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictConfigModel(BaseModel):
    """Base model that rejects misspelled or unexpected configuration keys."""

    model_config = ConfigDict(extra="forbid")


class PrepareConfig(StrictConfigModel):
    """Data preparation settings."""

    test_size: float = Field(gt=0.0, lt=1.0)
    random_state: int


class TrainConfig(StrictConfigModel):
    """Model selection and training settings."""

    model_type: Literal["logistic_regression", "gradient_boosting", "random_forest"]
    random_state: int
    max_iter: int = Field(gt=0)
    solver: str = Field(min_length=1)
    learning_rate: float = Field(gt=0.0)
    max_depth: int = Field(gt=0)
    n_estimators: int = Field(gt=0)
    min_samples_leaf: int = Field(gt=0)


class EvaluateConfig(StrictConfigModel):
    """Held-out evaluation settings."""

    threshold: float = Field(ge=0.0, le=1.0)
    target_recall: float = Field(gt=0.0, le=1.0)


class PipelineConfig(StrictConfigModel):
    """Complete contract for params.yaml."""

    prepare: PrepareConfig
    train: TrainConfig
    evaluate: EvaluateConfig


class PromotionThresholdsConfig(StrictConfigModel):
    """Required model quality thresholds."""

    min_roc_auc: float = Field(ge=0.0, le=1.0)
    min_pr_auc: float = Field(ge=0.0, le=1.0)
    min_precision_at_recall_60: float = Field(ge=0.0, le=1.0)
    max_brier_score: float = Field(ge=0.0, le=1.0)


class RegistryConfig(StrictConfigModel):
    """Required MLflow registry target settings."""

    model_name: str = Field(min_length=1)
    target_alias: str = Field(min_length=1)


class PromotionConfig(StrictConfigModel):
    """Complete contract for configs/thresholds.yaml."""

    thresholds: PromotionThresholdsConfig
    registry: RegistryConfig


def _read_yaml_mapping(config_path: Path) -> dict[str, Any]:
    """Read a YAML mapping or fail with a clear configuration error."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, encoding="utf-8") as config_file:
        raw_config: Any = yaml.safe_load(config_file)

    if not isinstance(raw_config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")

    return cast(dict[str, Any], raw_config)


def load_pipeline_config(config_path: Path = Path("params.yaml")) -> PipelineConfig:
    """Load and validate the complete pipeline configuration."""
    return PipelineConfig.model_validate(_read_yaml_mapping(config_path))


def load_promotion_settings(
    config_path: Path = Path("configs/thresholds.yaml"),
) -> PromotionConfig:
    """Load and validate model quality-gate and registry configuration."""
    return PromotionConfig.model_validate(_read_yaml_mapping(config_path))
