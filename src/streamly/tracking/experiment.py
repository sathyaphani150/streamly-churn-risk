"""MLflow experiment tracking module for Streamly churn risk prediction.

Logs hyperparameters, dataset lineage tags, evaluation metrics, model signatures,
input examples, and serializable model artifacts to a centralized MLflow backend.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.models
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

from streamly.features.builder import FEATURE_COLUMNS, build_training_features
from streamly.models.baseline import create_baseline_pipeline, split_data
from streamly.models.evaluation import evaluate_predictions

# Silence advisory hint on tracing
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"


def get_dataset_dvc_hash(dvc_file_path: Path = Path("data/raw/streamly_churn_sample.parquet.dvc")) -> str:
    """Extract MD5 hash from DVC tracking pointer file to capture dataset lineage."""
    if not dvc_file_path.exists():
        return "untracked_or_missing_dvc_file"

    try:
        with open(dvc_file_path, encoding="utf-8") as f:
            dvc_meta = yaml.safe_load(f)
        outs = dvc_meta.get("outs", [])
        if outs and isinstance(outs, list):
            return str(outs[0].get("md5", "unknown_hash"))
    except Exception as exc:
        print(f"Warning: Failed to parse DVC hash from {dvc_file_path}: {exc}")

    return "unknown_hash"


def run_experiment(
    data_path: Path = Path("data/raw/streamly_churn_sample.parquet"),
    params_path: Path = Path("params.yaml"),
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    run_name: str = "baseline_logistic_regression",
) -> str:
    """Execute end-to-end training and log all metadata to MLflow.

    Args:
        data_path: Path to raw dataset.
        params_path: Path to params.yaml.
        tracking_uri: MLflow tracking backend URI (defaults to env or sqlite:///mlruns.db).
        experiment_name: MLflow experiment name (defaults to env or streamly-churn-risk).
        run_name: Human-readable name for the run.

    Returns:
        str: MLflow run_id.
    """
    # 1. Configure MLflow Tracking
    uri: str = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlruns.db"
    exp_name: str = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME") or "streamly-churn-risk"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp_name)

    # 2. Load parameters
    with open(params_path, encoding="utf-8") as f:
        params: dict[str, Any] = yaml.safe_load(f)

    test_size = float(params.get("prepare", {}).get("test_size", 0.20))
    random_state = int(params.get("train", {}).get("random_state", 42))
    max_iter = int(params.get("train", {}).get("max_iter", 1000))
    solver = str(params.get("train", {}).get("solver", "lbfgs"))
    threshold = float(params.get("evaluate", {}).get("threshold", 0.50))
    target_recall = float(params.get("evaluate", {}).get("target_recall", 0.60))

    # 3. Load data and transform via shared feature builder
    raw_df = pd.read_parquet(data_path)
    X, y = build_training_features(raw_df)

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    # 4. Get dataset lineage hash from DVC
    dvc_hash = get_dataset_dvc_hash()

    # 5. Start tracked MLflow Run
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"[mlflow] Active run started: {run_id} (Experiment: {exp_name})")

        # A. Log Parameters
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "scaler": "StandardScaler",
                "max_iter": max_iter,
                "solver": solver,
                "random_state": random_state,
                "test_size": test_size,
                "decision_threshold": threshold,
                "target_recall_constraint": target_recall,
                "num_train_samples": len(X_train),
                "num_test_samples": len(X_test),
            }
        )

        # B. Log Lineage Tags
        mlflow.set_tags(
            {
                "environment": os.getenv("STREAMLY_ENV", "dev"),
                "dataset_path": str(data_path),
                "dataset_dvc_hash": dvc_hash,
                "feature_count": len(FEATURE_COLUMNS),
                "author": "Streamly ML Engineering",
            }
        )

        # C. Train Model Pipeline
        pipeline = create_baseline_pipeline(random_state=random_state)
        pipeline.fit(X_train, y_train)

        # D. Predict probabilities and evaluate
        y_test_probs = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate_predictions(
            y_true=y_test,
            y_probs=y_test_probs,
            threshold=threshold,
            target_recall=target_recall,
        )

        # E. Log Scalar Metrics
        scalar_metrics = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
        mlflow.log_metrics(scalar_metrics)

        # F. Log Input Example & Model Signature
        input_example = X_test.head(3)
        train_probs = pipeline.predict_proba(X_train)[:, 1]
        signature = mlflow.models.infer_signature(
            model_input=X_train,
            model_output=np.round(train_probs, 4),
        )

        # G. Log Model Artifact
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        # H. Log Custom Artifacts (Confusion Matrix & Feature List)
        temp_dir = Path("models/temp_artifacts")
        temp_dir.mkdir(parents=True, exist_ok=True)

        cm_path = temp_dir / "confusion_matrix.json"
        with open(cm_path, "w", encoding="utf-8") as f:
            json.dump(metrics["confusion_matrix"], f, indent=2)
        mlflow.log_artifact(str(cm_path), artifact_path="evaluation")

        feat_path = temp_dir / "feature_columns.json"
        with open(feat_path, "w", encoding="utf-8") as f:
            json.dump(FEATURE_COLUMNS, f, indent=2)
        mlflow.log_artifact(str(feat_path), artifact_path="features")

        # Cleanup temp directory
        cm_path.unlink(missing_ok=True)
        feat_path.unlink(missing_ok=True)
        temp_dir.rmdir()

        print(f"[mlflow] Successfully logged params, metrics, model, and artifacts for run {run_id}.")
        print(f"[mlflow] Evaluation: ROC-AUC={metrics['roc_auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}")

        return str(run_id)


def main() -> None:
    """CLI entrypoint for running an MLflow tracked experiment."""
    parser = argparse.ArgumentParser(description="Streamly MLflow Experiment Tracker")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/streamly_churn_sample.parquet"),
        help="Path to raw dataset",
    )
    parser.add_argument(
        "--params-path",
        type=Path,
        default=Path("params.yaml"),
        help="Path to params.yaml",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="baseline_logistic_regression",
        help="Name for MLflow run",
    )

    args = parser.parse_args()
    run_experiment(
        data_path=args.data_path,
        params_path=args.params_path,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
