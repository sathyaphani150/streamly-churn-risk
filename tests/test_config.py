"""Tests for fail-fast YAML configuration contracts."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from streamly.config import (
    ENVIRONMENT_VARIABLES,
    load_environment_config,
    load_pipeline_config,
    load_promotion_settings,
)


@pytest.fixture(autouse=True)
def isolate_runtime_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent developer-machine variables from changing configuration test inputs."""
    monkeypatch.delenv("STREAMLY_ENV", raising=False)
    for variable_name in ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable_name, raising=False)


def valid_pipeline_config() -> dict[str, object]:
    """Return a complete valid pipeline configuration."""
    return {
        "prepare": {"test_size": 0.2, "random_state": 42},
        "train": {
            "model_type": "logistic_regression",
            "random_state": 42,
            "max_iter": 1000,
            "solver": "lbfgs",
            "learning_rate": 0.05,
            "max_depth": 4,
            "n_estimators": 100,
            "min_samples_leaf": 10,
        },
        "evaluate": {"threshold": 0.5, "target_recall": 0.6},
    }


def valid_promotion_config() -> dict[str, object]:
    """Return a complete valid promotion configuration."""
    return {
        "thresholds": {
            "min_roc_auc": 0.75,
            "min_pr_auc": 0.60,
            "min_precision_at_recall_60": 0.55,
            "max_brier_score": 0.20,
        },
        "registry": {"model_name": "streamly_churn_model", "target_alias": "champion"},
    }


def test_pipeline_config_fails_loudly_on_missing_required_key(tmp_path: Path) -> None:
    """A missing evaluation key must block the pipeline instead of using a default."""
    config = valid_pipeline_config()
    config["evaluate"] = {"threshold": 0.5}
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError, match="target_recall"):
        load_pipeline_config(config_path)


def test_promotion_config_fails_loudly_on_missing_threshold(tmp_path: Path) -> None:
    """An incomplete quality gate must block promotion."""
    config = valid_promotion_config()
    config["thresholds"] = {
        "min_roc_auc": 0.75,
        "min_pr_auc": 0.60,
        "max_brier_score": 0.20,
    }
    config_path = tmp_path / "thresholds.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError, match="min_precision_at_recall_60"):
        load_promotion_settings(config_path)


def test_pipeline_config_rejects_invalid_percentage(tmp_path: Path) -> None:
    """Out-of-range values must fail before pipeline execution."""
    config = valid_pipeline_config()
    config["prepare"] = {"test_size": 1.5, "random_state": 42}
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValidationError, match="test_size"):
        load_pipeline_config(config_path)


def write_environment_profile(tmp_path: Path, **overrides: object) -> Path:
    """Write a minimal runtime profile and return its containing directory."""
    profile: dict[str, object] = {
        "environment": "dev",
        "mlflow_tracking_uri": "sqlite:///mlruns.db",
        "mlflow_experiment_name": "streamly-churn-risk",
        "model_name": "streamly_churn_model",
        "model_registry_alias": "champion",
        "model_artifact_path": "models/baseline_model.joblib",
        "allow_local_model_fallback": True,
        "log_level": "DEBUG",
    }
    profile.update(overrides)
    environment = str(profile["environment"])
    (tmp_path / f"{environment}.yaml").write_text(
        yaml.safe_dump(profile), encoding="utf-8"
    )
    return tmp_path


def test_ci_profile_rejects_protected_alias(tmp_path: Path) -> None:
    """CI automation must never be able to promote a champion model."""
    config_dir = write_environment_profile(tmp_path, environment="ci", model_registry_alias="champion")

    with pytest.raises(ValidationError, match="protected model alias"):
        load_environment_config("ci", config_dir)


def test_prod_requires_tracking_uri(tmp_path: Path) -> None:
    """Production must fail closed when no model registry is configured."""
    config_dir = write_environment_profile(
        tmp_path,
        environment="prod",
        mlflow_tracking_uri=None,
        allow_local_model_fallback=False,
    )

    with pytest.raises(ValidationError, match="MLFLOW_TRACKING_URI"):
        load_environment_config("prod", config_dir)


def test_prod_rejects_local_fallback(tmp_path: Path) -> None:
    """Production must not silently serve the image's bundled fallback model."""
    config_dir = write_environment_profile(
        tmp_path,
        environment="prod",
        mlflow_tracking_uri="https://mlflow.example.test",
        allow_local_model_fallback=True,
    )

    with pytest.raises(ValidationError, match="must not allow local model fallback"):
        load_environment_config("prod", config_dir)


def test_environment_variables_override_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deployment-time environment variables override safe YAML defaults."""
    config_dir = write_environment_profile(tmp_path)
    monkeypatch.setenv("MODEL_REGISTRY_ALIAS", "challenger")
    monkeypatch.setenv("ALLOW_LOCAL_MODEL_FALLBACK", "false")

    settings = load_environment_config("dev", config_dir)

    assert settings.model_registry_alias == "challenger"
    assert settings.allow_local_model_fallback is False
