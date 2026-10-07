"""Typed, fail-fast configuration contracts for the Streamly pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class EnvironmentConfig(StrictConfigModel):
    """Runtime settings and safety rules for one deployment environment."""

    environment: Literal["dev", "ci", "prod"]
    mlflow_tracking_uri: str | None
    mlflow_experiment_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_registry_alias: str = Field(min_length=1)
    model_artifact_path: Path
    allow_local_model_fallback: bool
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    @model_validator(mode="after")
    def enforce_environment_safety(self) -> EnvironmentConfig:
        """Reject configurations that weaken CI or production controls."""
        alias = self.model_registry_alias.lstrip("@").lower()
        self.model_registry_alias = alias

        if self.environment == "prod":
            if not self.mlflow_tracking_uri:
                raise ValueError("prod requires MLFLOW_TRACKING_URI")
            if self.allow_local_model_fallback:
                raise ValueError("prod must not allow local model fallback")

        if self.environment == "ci" and alias in {"champion", "production"}:
            raise ValueError("ci must not target a protected model alias")

        return self


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


ENVIRONMENT_VARIABLES: dict[str, str] = {
    "MLFLOW_TRACKING_URI": "mlflow_tracking_uri",
    "MLFLOW_EXPERIMENT_NAME": "mlflow_experiment_name",
    "MODEL_NAME": "model_name",
    "MODEL_REGISTRY_ALIAS": "model_registry_alias",
    "MODEL_ARTIFACT_PATH": "model_artifact_path",
    "ALLOW_LOCAL_MODEL_FALLBACK": "allow_local_model_fallback",
    "LOG_LEVEL": "log_level",
}


def load_environment_config(
    environment: str | None = None,
    config_dir: Path = Path("configs/environments"),
) -> EnvironmentConfig:
    """Load a named runtime profile, apply environment overrides, and validate it."""
    selected_environment = environment or os.getenv("STREAMLY_ENV", "dev")
    if selected_environment not in {"dev", "ci", "prod"}:
        raise ValueError(
            f"Unsupported STREAMLY_ENV '{selected_environment}'. Expected dev, ci, or prod."
        )

    raw_config = _read_yaml_mapping(config_dir / f"{selected_environment}.yaml")
    for variable_name, config_key in ENVIRONMENT_VARIABLES.items():
        if variable_name in os.environ:
            raw_config[config_key] = os.environ[variable_name]

    if raw_config.get("environment") != selected_environment:
        raise ValueError(
            f"Environment profile mismatch: requested '{selected_environment}' but file declares "
            f"'{raw_config.get('environment')}'."
        )

    return EnvironmentConfig.model_validate(raw_config)
