"""Tests for fail-fast YAML configuration contracts."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from streamly.config import load_pipeline_config, load_promotion_settings


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
