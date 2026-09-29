"""Tests for model evaluation quality gate and registry promotion.

Verifies that:
1. Candidate models meeting all configured thresholds pass the quality gate.
2. Candidate models failing any single threshold are blocked from promotion.
3. Passed models are registered in the MLflow Model Registry and assigned the deployment alias.
"""

from pathlib import Path

import mlflow
import pytest
import yaml
from mlflow.tracking import MlflowClient

from streamly.tracking.promotion import (
    PromotionGateError,
    evaluate_quality_gate,
    gate_and_promote,
    load_promotion_config,
)


@pytest.fixture
def sample_passing_metrics() -> dict[str, float]:
    """Metrics that clear all baseline thresholds."""
    return {
        "roc_auc": 0.8038,
        "pr_auc": 0.6657,
        "precision_at_recall_60": 0.6082,
        "brier_score": 0.1653,
        "accuracy": 0.7550,
    }


@pytest.fixture
def sample_thresholds() -> dict[str, float]:
    """Standard quality gate thresholds."""
    return {
        "min_roc_auc": 0.75,
        "min_pr_auc": 0.60,
        "min_precision_at_recall_60": 0.55,
        "max_brier_score": 0.20,
    }


def test_evaluate_quality_gate_passes_when_all_thresholds_cleared(
    sample_passing_metrics: dict[str, float], sample_thresholds: dict[str, float]
) -> None:
    """Quality gate must return all_passed=True when all metrics exceed thresholds."""
    passed, checks = evaluate_quality_gate(sample_passing_metrics, sample_thresholds)
    assert passed is True
    assert all(c["passed"] for c in checks.values())


@pytest.mark.parametrize(
    "corrupted_metric,corrupted_val,failing_check",
    [
        ("roc_auc", 0.70, "roc_auc"),  # Below 0.75
        ("pr_auc", 0.50, "pr_auc"),    # Below 0.60
        ("precision_at_recall_60", 0.45, "precision_at_recall_60"),  # Below 0.55
        ("brier_score", 0.25, "brier_score"),  # Above 0.20 (worse calibration)
    ],
)
def test_evaluate_quality_gate_fails_when_any_metric_below_threshold(
    sample_passing_metrics: dict[str, float],
    sample_thresholds: dict[str, float],
    corrupted_metric: str,
    corrupted_val: float,
    failing_check: str,
) -> None:
    """Quality gate must fail if even a single metric violates its threshold."""
    bad_metrics = sample_passing_metrics.copy()
    bad_metrics[corrupted_metric] = corrupted_val

    passed, checks = evaluate_quality_gate(bad_metrics, sample_thresholds)
    assert passed is False
    assert checks[failing_check]["passed"] is False


def test_load_promotion_config_success(tmp_path: Path) -> None:
    """Correctly loads thresholds and registry configuration from YAML."""
    config_file = tmp_path / "thresholds.yaml"
    content = {
        "thresholds": {"min_roc_auc": 0.75},
        "registry": {"model_name": "test_model", "target_alias": "champion"},
    }
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(content, f)

    loaded = load_promotion_config(config_file)
    assert loaded["thresholds"]["min_roc_auc"] == 0.75
    assert loaded["registry"]["model_name"] == "test_model"


def test_gate_and_promote_rejects_substandard_run(tmp_path: Path) -> None:
    """gate_and_promote must raise PromotionGateError and block registration if thresholds fail."""
    import mlflow
    from mlflow.tracking import MlflowClient

    # Set up isolated test tracking
    test_db = tmp_path / "test_gate.db"
    uri = f"sqlite:///{test_db.as_posix()}"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    # Create run with weak metrics
    exp_id = client.create_experiment("test_gate_exp")
    run = client.create_run(exp_id)
    client.log_metric(run.info.run_id, "roc_auc", 0.65)  # Weak ROC-AUC
    client.log_metric(run.info.run_id, "pr_auc", 0.50)
    client.log_metric(run.info.run_id, "precision_at_recall_60", 0.40)
    client.log_metric(run.info.run_id, "brier_score", 0.28)

    # Config with strict thresholds
    config_path = tmp_path / "strict_thresholds.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "thresholds": {
                    "min_roc_auc": 0.75,
                    "min_pr_auc": 0.60,
                },
                "registry": {"model_name": "gated_model", "target_alias": "champion"},
            },
            f,
        )

    # Gate must raise PromotionGateError and set rejection tag
    with pytest.raises(PromotionGateError, match="Quality gate REJECTED promotion"):
        gate_and_promote(run_id=run.info.run_id, config_path=config_path, tracking_uri=uri)

    # Verify run was tagged as REJECTED and no model was registered
    updated_run = client.get_run(run.info.run_id)
    assert updated_run.data.tags["promotion.status"] == "REJECTED"
    assert len(client.search_registered_models()) == 0


def test_gate_and_promote_custom_challenger_alias(tmp_path: Path) -> None:
    """gate_and_promote must support custom alias assignment such as @challenger."""
    db_file = tmp_path / "test_challenger_mlruns.db"
    uri = f"sqlite:///{db_file}"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    exp_id = client.create_experiment("test-challenger-exp")
    run = client.create_run(exp_id)

    # Log passing metrics
    client.log_metric(run.info.run_id, "roc_auc", 0.78)
    client.log_metric(run.info.run_id, "pr_auc", 0.65)
    client.log_metric(run.info.run_id, "precision_at_recall_60", 0.58)
    client.log_metric(run.info.run_id, "brier_score", 0.17)

    # Save a mock artifact
    artifact_dir = tmp_path / "mock_model"
    artifact_dir.mkdir()
    (artifact_dir / "MLmodel").write_text("model_mock")
    client.log_artifacts(run.info.run_id, str(artifact_dir), artifact_path="model")

    config_path = tmp_path / "thresholds.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "thresholds": {"min_roc_auc": 0.75, "min_pr_auc": 0.60},
                "registry": {"model_name": "test_challenger_model", "target_alias": "champion"},
            },
            f,
        )

    # Promote with explicit target_alias="challenger"
    result = gate_and_promote(
        run_id=run.info.run_id,
        config_path=config_path,
        target_alias="challenger",
        tracking_uri=uri,
    )

    assert result["status"] == "PROMOTED"
    assert result["alias"] == "@challenger"
    registered = client.get_registered_model("test_challenger_model")
    assert registered.aliases["challenger"] == 1

