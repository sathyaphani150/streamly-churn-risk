"""Pipeline Stage 2: Train Model.

Loads processed training dataset, transforms features using the shared builder,
trains the baseline pipeline, and serializes the fitted model artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from streamly.config import TrainConfig, load_pipeline_config
from streamly.features.builder import build_training_features
from streamly.models.baseline import create_model_pipeline


def load_params(params_path: Path = Path("params.yaml")) -> TrainConfig:
    """Load validated training parameters from params.yaml."""
    return load_pipeline_config(params_path).train


def run_train(
    train_data_path: Path = Path("data/processed/train.parquet"),
    model_output_path: Path = Path("models/baseline_model.joblib"),
    params_path: Path = Path("params.yaml"),
) -> None:
    """Train churn prediction model and save fitted pipeline artifact."""
    print(f"[train] Loading training data from: {train_data_path}")
    if not train_data_path.exists():
        print(f"[train] Error: Training data not found at {train_data_path}", file=sys.stderr)
        sys.exit(1)

    train_df = pd.read_parquet(train_data_path)

    # 1. Transform features using shared training builder
    print("[train] Transforming features via shared feature builder...")
    X_train, y_train = build_training_features(train_df)

    # 2. Load hyperparameters
    params = load_params(params_path)
    model_type = params.model_type
    random_state = params.random_state

    # 3. Instantiate pipeline via shared model factory
    print(f"[train] Fitting Scikit-Learn Pipeline (type={model_type}, random_state={random_state})...")
    extra_kwargs: dict[str, Any] = params.model_dump(
        exclude={"model_type", "random_state"}
    )
    pipeline = create_model_pipeline(
        model_type=model_type,
        random_state=random_state,
        **extra_kwargs,
    )

    pipeline.fit(X_train, y_train)

    # 4. Serialize model artifact
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print(f"[train] Successfully saved trained model artifact to: {model_output_path}")


if __name__ == "__main__":
    run_train()
