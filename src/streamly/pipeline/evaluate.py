"""Pipeline Stage 3: Evaluate Model.

Loads trained model artifact and processed test dataset, calculates retention metrics,
and writes structured evaluation results to metrics.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from streamly.config import EvaluateConfig, load_pipeline_config
from streamly.features.builder import build_training_features
from streamly.models.evaluation import evaluate_predictions


def load_params(params_path: Path = Path("params.yaml")) -> EvaluateConfig:
    """Load validated evaluation parameters from params.yaml."""
    return load_pipeline_config(params_path).evaluate


def run_evaluate(
    model_path: Path = Path("models/baseline_model.joblib"),
    test_data_path: Path = Path("data/processed/test.parquet"),
    metrics_path: Path = Path("metrics.json"),
    params_path: Path = Path("params.yaml"),
) -> None:
    """Evaluate trained model on held-out test data and save metrics.json."""
    print(f"[evaluate] Loading model artifact from: {model_path}")
    if not model_path.exists():
        print(f"[evaluate] Error: Model artifact not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[evaluate] Loading test data from: {test_data_path}")
    if not test_data_path.exists():
        print(f"[evaluate] Error: Test data not found at {test_data_path}", file=sys.stderr)
        sys.exit(1)

    pipeline = joblib.load(model_path)
    test_df = pd.read_parquet(test_data_path)

    # 1. Transform test features
    X_test, y_test = build_training_features(test_df)

    # 2. Predict probabilities on held-out test set
    y_test_probs = pipeline.predict_proba(X_test)[:, 1]

    # 3. Load parameters & evaluate
    params = load_params(params_path)
    threshold = params.threshold
    target_recall = params.target_recall

    metrics = evaluate_predictions(
        y_true=y_test,
        y_probs=y_test_probs,
        threshold=threshold,
        target_recall=target_recall,
    )

    # 4. Save evaluation metrics to metrics.json
    # Pin LF so the DVC-tracked JSON has identical bytes on Windows and Linux.
    with open(metrics_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(metrics, f, indent=2)

    print(f"[evaluate] Successfully computed metrics and saved to: {metrics_path}")
    print(f"[evaluate] ROC-AUC: {metrics['roc_auc']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}")


if __name__ == "__main__":
    run_evaluate()
