"""Pipeline Stage 1: Prepare & Validate Data.

Loads raw dataset, runs strict schema validation gate, and creates reproducible
stratified train/test splits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from streamly.data.validation import validate_training_data


def load_params(params_path: Path = Path("params.yaml")) -> dict[str, float | int]:
    """Load preparation parameters from params.yaml."""
    with open(params_path, encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params.get("prepare", {})  # type: ignore[no-any-return]


def run_prepare(
    raw_data_path: Path = Path("data/raw/streamly_churn_sample.parquet"),
    processed_dir: Path = Path("data/processed"),
    params_path: Path = Path("params.yaml"),
) -> None:
    """Validate raw data and generate stratified train and test datasets."""
    print(f"[prepare] Loading raw data from: {raw_data_path}")
    if not raw_data_path.exists():
        print(f"[prepare] Error: File not found {raw_data_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(raw_data_path)

    # 1. Run schema and quality validation gate
    print(f"[prepare] Running data contract validation on {len(df):,} records...")
    validated_df = validate_training_data(df)

    # 2. Load hyperparameters
    params = load_params(params_path)
    test_size = float(params.get("test_size", 0.20))
    random_state = int(params.get("random_state", 42))

    # 3. Stratified split
    print(f"[prepare] Splitting into train/test (test_size={test_size}, seed={random_state})...")
    train_df, test_df = train_test_split(
        validated_df,
        test_size=test_size,
        random_state=random_state,
        stratify=validated_df["churned_30d"],
    )

    # 4. Save processed artifacts
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"[prepare] Saved {len(train_df):,} train records to {train_path}")
    print(f"[prepare] Saved {len(test_df):,} test records to {test_path}")


if __name__ == "__main__":
    run_prepare()
