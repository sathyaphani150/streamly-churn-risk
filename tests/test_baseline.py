"""Tests for baseline churn model and evaluation metrics.

Verifies pipeline training, probability boundedness, metric calculations,
and precision-at-recall behavior.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from streamly.models.baseline import (
    create_baseline_pipeline,
    split_data,
    train_and_evaluate,
)
from streamly.models.evaluation import (
    compute_precision_at_recall,
    evaluate_predictions,
)


@pytest.fixture
def mock_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Fixture providing a deterministic small dataset for testing."""
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(
        {
            "tenure_days": rng.integers(1, 500, size=n),
            "sessions_7d": rng.integers(0, 20, size=n),
            "watch_hours_7d": rng.uniform(0.0, 40.0, size=n),
            "support_tickets_30d": rng.integers(0, 5, size=n),
            "price_increase_flag": rng.choice([0, 1], size=n),
            "plan_tier_basic": rng.choice([0, 1], size=n),
            "plan_tier_standard": rng.choice([0, 1], size=n),
            "plan_tier_premium": rng.choice([0, 1], size=n),
            "watch_hours_per_session": rng.uniform(0.0, 10.0, size=n),
        }
    )
    y = pd.Series(rng.choice([0, 1], size=n, p=[0.7, 0.3]), name="churned_30d")
    return X, y


def test_split_data_preserves_stratification(
    mock_dataset: tuple[pd.DataFrame, pd.Series]
) -> None:
    """Train/test split must preserve target class balance via stratification."""
    X, y = mock_dataset
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25, random_state=42)

    assert len(X_train) == 150
    assert len(X_test) == 50
    # Stratified proportion should be approximately equal
    assert abs(y_train.mean() - y_test.mean()) < 0.05


def test_pipeline_produces_valid_probabilities(
    mock_dataset: tuple[pd.DataFrame, pd.Series]
) -> None:
    """Pipeline must fit successfully and output valid probabilities in [0.0, 1.0]."""
    X, y = mock_dataset
    pipeline = create_baseline_pipeline(random_state=42)
    pipeline.fit(X, y)

    probs = pipeline.predict_proba(X)[:, 1]
    assert len(probs) == len(X)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_compute_precision_at_recall_bounded() -> None:
    """compute_precision_at_recall must return a float bounded in [0.0, 1.0]."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    p_at_r = compute_precision_at_recall(y_true, y_probs, target_recall=0.60)
    assert 0.0 <= p_at_r <= 1.0
    assert p_at_r == 1.0  # Perfect separation on top 3


def test_evaluate_predictions_all_keys_present() -> None:
    """evaluate_predictions must return all required scalar metrics and confusion matrix."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_probs = np.array([0.1, 0.4, 0.6, 0.8, 0.3, 0.7])

    metrics = evaluate_predictions(y_true, y_probs, threshold=0.50, target_recall=0.60)

    expected_keys = {
        "roc_auc",
        "pr_auc",
        "precision_at_recall_60",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "brier_score",
        "confusion_matrix",
    }
    assert expected_keys.issubset(set(metrics.keys()))
    assert isinstance(metrics["confusion_matrix"], dict)
    assert metrics["roc_auc"] >= 0.5


def test_train_and_evaluate_missing_file_raises_error() -> None:
    """train_and_evaluate must fail loudly if dataset path does not exist."""
    with pytest.raises(FileNotFoundError, match="Training data not found"):
        train_and_evaluate(data_path=Path("non_existent_data.parquet"))


@pytest.mark.parametrize(
    "model_type",
    ["logistic_regression", "gradient_boosting", "random_forest"],
)
def test_create_model_pipeline_multiple_algorithms(
    model_type: str,
    mock_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """All supported model types must instantiate a Pipeline that trains and predicts bounded probabilities."""
    from streamly.models.baseline import create_model_pipeline

    X, y = mock_dataset
    pipeline = create_model_pipeline(model_type=model_type, random_state=42)
    pipeline.fit(X, y)
    probs = pipeline.predict_proba(X)[:, 1]

    assert len(probs) == len(X)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_create_model_pipeline_invalid_type() -> None:
    """create_model_pipeline must raise ValueError on unsupported algorithm types."""
    from streamly.models.baseline import create_model_pipeline

    with pytest.raises(ValueError, match="Unsupported model_type"):
        create_model_pipeline(model_type="quantum_deep_net")

