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
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from streamly.features.builder import build_training_features


def load_params(params_path: Path = Path("params.yaml")) -> dict[str, Any]:
    """Load training parameters from params.yaml."""
    with open(params_path, encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params.get("train", {})  # type: ignore[no-any-return]


def run_train(
    train_data_path: Path = Path("data/processed/train.parquet"),
    model_output_path: Path = Path("models/baseline_model.joblib"),
    params_path: Path = Path("params.yaml"),
) -> None:
    """Train baseline churn prediction model and save fitted pipeline artifact."""
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
    random_state = int(params.get("random_state", 42))
    max_iter = int(params.get("max_iter", 1000))
    solver = str(params.get("solver", "lbfgs"))

    # 3. Instantiate pipeline (StandardScaler + LogisticRegression)
    print(f"[train] Fitting Scikit-Learn Pipeline (solver={solver}, max_iter={max_iter})...")
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=max_iter,
                    random_state=random_state,
                    solver=solver,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    # 4. Serialize model artifact
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print(f"[train] Successfully saved trained model artifact to: {model_output_path}")


if __name__ == "__main__":
    run_train()
