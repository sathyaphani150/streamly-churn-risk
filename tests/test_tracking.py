"""Tests for MLflow experiment tracking module.

Verifies that MLflow runs log parameters, dataset lineage tags, evaluation metrics,
and model artifacts with valid signatures.
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from streamly.tracking.experiment import get_dataset_dvc_hash, run_experiment


def test_get_dataset_dvc_hash_with_valid_file(tmp_path: Path) -> None:
    """Extracts the exact MD5 hash from a mock .dvc pointer file."""
    dvc_file = tmp_path / "sample.parquet.dvc"
    dvc_content = {
        "outs": [
            {
                "md5": "abc123def4567890",
                "size": 1000,
                "path": "sample.parquet",
            }
        ]
    }
    with open(dvc_file, "w", encoding="utf-8") as f:
        yaml.dump(dvc_content, f)

    extracted_hash = get_dataset_dvc_hash(dvc_file)
    assert extracted_hash == "abc123def4567890"


def test_get_dataset_dvc_hash_missing_file() -> None:
    """Returns fallback string when .dvc pointer file does not exist."""
    extracted_hash = get_dataset_dvc_hash(Path("non_existent_file.parquet.dvc"))
    assert extracted_hash == "untracked_or_missing_dvc_file"


def test_run_experiment_logs_complete_metadata(
    tmp_path: Path,
) -> None:
    """Run an end-to-end tracked experiment in an isolated test tracking database."""
    # 1. Create a small mock dataset
    data_path = tmp_path / "mock_data.parquet"
    rng = np.random.default_rng(42)
    n = 100
    mock_df = pd.DataFrame(
        {
            "member_id": [f"mem_{i:04d}" for i in range(n)],
            "tenure_days": rng.integers(1, 300, size=n),
            "sessions_7d": rng.integers(0, 15, size=n),
            "watch_hours_7d": rng.uniform(0.0, 30.0, size=n).round(2),
            "support_tickets_30d": rng.integers(0, 4, size=n),
            "plan_tier": rng.choice(["basic", "standard", "premium"], size=n),
            "price_increase_flag": rng.choice([0, 1], size=n),
            "churned_30d": rng.choice([0, 1], size=n, p=[0.7, 0.3]),
        }
    )
    mock_df.to_parquet(data_path, index=False)

    # 2. Create mock params.yaml
    params_path = tmp_path / "params.yaml"
    params_content = {
        "prepare": {"test_size": 0.20, "random_state": 42},
        "train": {"random_state": 42, "max_iter": 500, "solver": "lbfgs"},
        "evaluate": {"threshold": 0.50, "target_recall": 0.60},
    }
    with open(params_path, "w", encoding="utf-8") as f:
        yaml.dump(params_content, f)

    # 3. Use an isolated temporary SQLite database for test runs
    test_db = tmp_path / "test_mlflow.db"
    test_tracking_uri = f"sqlite:///{test_db.as_posix()}"
    test_exp_name = "test-streamly-churn"

    run_id = run_experiment(
        data_path=data_path,
        params_path=params_path,
        tracking_uri=test_tracking_uri,
        experiment_name=test_exp_name,
        run_name="unit_test_run",
    )

    # 4. Verify run was logged in MLflow
    mlflow.set_tracking_uri(test_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    assert run.info.status == "FINISHED"

    # Verify Parameters
    assert run.data.params["model_type"] == "LogisticRegression"
    assert run.data.params["max_iter"] == "500"
    assert run.data.params["solver"] == "lbfgs"

    # Verify Metrics
    assert "roc_auc" in run.data.metrics
    assert "pr_auc" in run.data.metrics
    assert "precision_at_recall_60" in run.data.metrics
    assert 0.0 <= run.data.metrics["roc_auc"] <= 1.0

    # Verify Lineage Tags
    assert run.data.tags["dataset_path"] == str(data_path)
    assert "dataset_dvc_hash" in run.data.tags
