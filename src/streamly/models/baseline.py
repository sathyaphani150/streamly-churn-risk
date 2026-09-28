"""Baseline churn risk classification model for Streamly.

Builds, trains, and evaluates a baseline model using the shared feature builder
and standardized feature scaling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from streamly.features.builder import build_training_features
from streamly.models.evaluation import evaluate_predictions


def create_baseline_pipeline(random_state: int = 42) -> Pipeline:
    """Instantiate a standardized Scikit-Learn pipeline for baseline churn prediction.

    Combines StandardScaler (to normalize disparate engagement scales like tenure and watch hours)
    with LogisticRegression for calibrated probability estimation.

    Args:
        random_state: Random state for deterministic optimization.

    Returns:
        Pipeline: Unfitted Scikit-Learn pipeline.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split feature matrix and target series into reproducible train and test sets.

    Uses stratification on y to preserve the identical churn rate across both splits.

    Args:
        X: Feature matrix.
        y: Target series.
        test_size: Proportion of records allocated to held-out test evaluation.
        random_state: Seed for reproducibility.

    Returns:
        tuple containing (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def train_and_evaluate(
    data_path: Path,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """Execute end-to-end baseline training and evaluation on held-out test set.

    Args:
        data_path: Path to raw parquet data file.
        test_size: Proportion of data reserved for test set.
        random_state: Random seed.

    Returns:
        tuple[Pipeline, dict[str, Any]]: (fitted_pipeline, metrics_dict).
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at: {data_path}")

    # 1. Load raw data and apply shared feature builder
    raw_df = pd.read_parquet(data_path)
    X, y = build_training_features(raw_df)

    # 2. Split into train and test sets
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    # 3. Fit pipeline on train set only
    pipeline = create_baseline_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)

    # 4. Predict probabilities on held-out test set
    # predict_proba returns [P(churn=0), P(churn=1)]; column 1 is churn_risk
    y_test_probs = pipeline.predict_proba(X_test)[:, 1]

    # 5. Evaluate against retention metrics
    metrics = evaluate_predictions(y_test, y_test_probs)

    return pipeline, metrics


def main() -> None:
    """CLI entrypoint for baseline training and evaluation."""
    parser = argparse.ArgumentParser(description="Streamly Baseline Churn Model")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/streamly_churn_sample.parquet"),
        help="Path to input dataset parquet file",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of data for test set (default: 0.20)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state seed (default: 42)",
    )

    args = parser.parse_args()

    print(f"Loading data from {args.data_path}...")
    pipeline, metrics = train_and_evaluate(
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("\n" + "=" * 55)
    print("        STREAMLY BASELINE CHURN MODEL EVALUATION")
    print("=" * 55)
    for metric_name, metric_val in metrics.items():
        if metric_name != "confusion_matrix":
            print(f"  {metric_name:<28}: {metric_val}")
    print("-" * 55)
    print("Confusion Matrix Breakdown (Threshold = 0.50):")
    cm = metrics["confusion_matrix"]
    print(f"  True Negatives (Retained correctly) : {cm['true_negatives']}")
    print(f"  False Positives (Spammed retention)  : {cm['false_positives']}")
    print(f"  False Negatives (Missed churners)    : {cm['false_negatives']}")
    print(f"  True Positives  (Caught churners)    : {cm['true_positives']}")
    print("=" * 55)


if __name__ == "__main__":
    main()
