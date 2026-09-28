"""Data contract and schema quality validation layer for Streamly.

Enforces strict data contracts on both training datasets and serving inference payloads
to prevent silent data corruption, schema drift, and target leakage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final, Literal

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

ALLOWED_PLAN_TIERS: Final[list[str]] = ["basic", "standard", "premium"]

# Obvious leakage fields that must never enter serving inference
FORBIDDEN_SERVE_COLUMNS: Final[set[str]] = {
    "churned_30d",
    "cancel_reason",
    "cancellation_date",
    "days_until_cancel",
    "exit_survey_reason",
    "refund_amount",
}


class DataContractError(Exception):
    """Raised when incoming data violates the schema contract or contains leakage."""


# Core feature columns common to both training and serving
_BASE_FEATURE_COLUMNS = {
    "member_id": pa.Column(
        str,
        nullable=False,
        required=True,
        checks=pa.Check.str_length(min_value=1),
        description="Subscriber unique identifier (identity only)",
    ),
    "tenure_days": pa.Column(
        pa.Int64,
        nullable=False,
        required=True,
        checks=pa.Check.ge(0),
        description="Account tenure in days, non-negative",
    ),
    "sessions_7d": pa.Column(
        pa.Int64,
        nullable=False,
        required=True,
        checks=pa.Check.ge(0),
        description="Weekly sessions, non-negative",
    ),
    "watch_hours_7d": pa.Column(
        pa.Float64,
        nullable=False,
        required=True,
        checks=pa.Check.ge(0.0),
        description="Weekly streaming hours, non-negative",
    ),
    "support_tickets_30d": pa.Column(
        pa.Int64,
        nullable=False,
        required=True,
        checks=pa.Check.ge(0),
        description="Support tickets in last 30d, non-negative",
    ),
    "plan_tier": pa.Column(
        checks=pa.Check.isin(ALLOWED_PLAN_TIERS),
        nullable=False,
        required=True,
        description="Subscription plan tier (controlled vocabulary)",
    ),
    "price_increase_flag": pa.Column(
        pa.Int64,
        nullable=False,
        required=True,
        checks=pa.Check.isin([0, 1]),
        description="Binary indicator if price increased recently",
    ),
}

# Training Schema: Base features + ground truth label 'churned_30d'
TRAINING_SCHEMA: Final[pa.DataFrameSchema] = pa.DataFrameSchema(
    columns={
        **_BASE_FEATURE_COLUMNS,
        "churned_30d": pa.Column(
            pa.Int64,
            nullable=False,
            required=True,
            checks=pa.Check.isin([0, 1]),
            description="Binary ground truth label (1 = churned, 0 = retained)",
        ),
    },
    strict=True,  # Fail if unexpected columns exist
    coerce=False,  # Do not silently alter or cast types
)

# Serving Schema: Base features only; STRICTLY excludes churned_30d and any extra columns
SERVING_SCHEMA: Final[pa.DataFrameSchema] = pa.DataFrameSchema(
    columns=_BASE_FEATURE_COLUMNS,
    strict=True,  # Disallows unexpected columns
    coerce=False,  # Enforces explicit types without silent coercion
)


def assert_no_leakage_columns(columns: list[str] | pd.Index) -> None:
    """Assert that columns do not contain known label or post-churn leakage fields.

    Args:
        columns: List or Index of column names to check.

    Raises:
        DataContractError: If any prohibited leakage column is detected.
    """
    col_set = set(columns)
    leakage_found = col_set.intersection(FORBIDDEN_SERVE_COLUMNS)
    if leakage_found:
        raise DataContractError(
            f"Target leakage violation: serving payload contains forbidden columns: {sorted(leakage_found)}"
        )


def validate_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw or snapshot training dataset against the training data contract.

    Args:
        df: Pandas DataFrame containing historical training records.

    Returns:
        The validated DataFrame if all assertions hold.

    Raises:
        DataContractError: If schema, types, ranges, or nullability constraints fail.
    """
    try:
        return TRAINING_SCHEMA.validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        raise DataContractError(f"Training data contract violation: {exc}") from exc


def validate_serving_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate inference input payload against the serving data contract.

    Guarantees:
    1. Zero target leakage (churned_30d and post-event fields are strictly prohibited).
    2. No unexpected extra columns (strict schema).
    3. All required feature fields are non-null and within valid ranges.

    Args:
        df: Pandas DataFrame representing serving request(s).

    Returns:
        The validated DataFrame.

    Raises:
        DataContractError: If schema, leakage, or value checks fail.
    """
    # 1. Explicit anti-leakage check
    assert_no_leakage_columns(df.columns)

    # 2. Strict schema and boundary check
    try:
        return SERVING_SCHEMA.validate(df, lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        raise DataContractError(f"Serving data contract violation: {exc}") from exc


def validate_dataset_file(file_path: Path, mode: Literal["train", "serve"] = "train") -> None:
    """Validate a dataset file from disk and report results.

    Args:
        file_path: Path to .parquet or .csv file.
        mode: Either 'train' or 'serve'.

    Raises:
        DataContractError: If contract fails.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {file_path}")

    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    if mode == "train":
        validate_training_data(df)
        print(f"PASS: Training data contract verified for {file_path} ({len(df):,} rows).")
    else:
        validate_serving_data(df)
        print(f"PASS: Serving data contract verified for {file_path} ({len(df):,} rows).")


def main() -> None:
    """CLI entrypoint for running data validation quality gate."""
    parser = argparse.ArgumentParser(description="Streamly Data Quality & Schema Gate")
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to the dataset file (.parquet or .csv) to validate",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "serve"],
        default="train",
        help="Validation mode (train checks label; serve forbids label/leakage)",
    )

    args = parser.parse_args()

    try:
        validate_dataset_file(args.data_path, mode=args.mode)
        sys.exit(0)
    except Exception as exc:
        print(f"FAIL: Data contract validation failed!\nError: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
