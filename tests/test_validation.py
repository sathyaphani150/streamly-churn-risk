"""Tests for data contract and schema validation layer.

Verifies that valid training and serving datasets pass, and that intentionally
corrupted data (missing columns, unexpected columns, nulls, negative numbers,
invalid vocabularies, and target leakage) fails loudly and non-zero.
"""

import pandas as pd
import pytest

from streamly.data.validation import (
    DataContractError,
    validate_serving_data,
    validate_training_data,
)


@pytest.fixture
def valid_train_df() -> pd.DataFrame:
    """Fixture providing a clean single-row valid training DataFrame."""
    return pd.DataFrame(
        {
            "member_id": ["mem_000001"],
            "tenure_days": [120],
            "sessions_7d": [5],
            "watch_hours_7d": [12.5],
            "support_tickets_30d": [1],
            "plan_tier": ["standard"],
            "price_increase_flag": [0],
            "churned_30d": [0],
        }
    )


@pytest.fixture
def valid_serve_df(valid_train_df: pd.DataFrame) -> pd.DataFrame:
    """Fixture providing a clean single-row valid serving DataFrame (no label)."""
    return valid_train_df.drop(columns=["churned_30d"])


def test_valid_training_data_passes(valid_train_df: pd.DataFrame) -> None:
    """Verify that a schema-compliant training record passes validation."""
    validated = validate_training_data(valid_train_df)
    assert len(validated) == 1
    assert "churned_30d" in validated.columns


def test_valid_serving_data_passes(valid_serve_df: pd.DataFrame) -> None:
    """Verify that a schema-compliant serving payload passes validation."""
    validated = validate_serving_data(valid_serve_df)
    assert len(validated) == 1
    assert "churned_30d" not in validated.columns


# ==============================================================================
# LEAKAGE CHECKS
# ==============================================================================

def test_serving_rejects_churned_30d_target_leakage(valid_train_df: pd.DataFrame) -> None:
    """Serving must fail immediately if the future label is included in the payload."""
    with pytest.raises(DataContractError, match="Target leakage violation"):
        validate_serving_data(valid_train_df)


@pytest.mark.parametrize(
    "leakage_col",
    ["cancel_reason", "cancellation_date", "days_until_cancel", "refund_amount"],
)
def test_serving_rejects_post_churn_leakage_fields(
    valid_serve_df: pd.DataFrame, leakage_col: str
) -> None:
    """Serving must reject downstream post-churn / exit-survey fields."""
    corrupted_df = valid_serve_df.copy()
    corrupted_df[leakage_col] = "leakage_value"

    with pytest.raises(DataContractError, match="Target leakage violation"):
        validate_serving_data(corrupted_df)


# ==============================================================================
# SCHEMA STRUCTURE CHECKS (Strict columns, missingness, unexpected fields)
# ==============================================================================

def test_rejects_unexpected_extra_columns(valid_train_df: pd.DataFrame) -> None:
    """Strict schema must reject arbitrary extra columns."""
    corrupted_df = valid_train_df.copy()
    corrupted_df["device_brand"] = "Apple"

    with pytest.raises(DataContractError, match="Training data contract violation"):
        validate_training_data(corrupted_df)


def test_rejects_missing_required_column(valid_train_df: pd.DataFrame) -> None:
    """Must fail when any declared required column is missing."""
    corrupted_df = valid_train_df.drop(columns=["watch_hours_7d"])

    with pytest.raises(DataContractError, match="Training data contract violation"):
        validate_training_data(corrupted_df)


def test_rejects_null_in_required_field(valid_train_df: pd.DataFrame) -> None:
    """Must fail when a required feature contains null/None."""
    corrupted_df = valid_train_df.copy()
    corrupted_df.loc[0, "support_tickets_30d"] = None

    with pytest.raises(DataContractError):
        validate_training_data(corrupted_df)


# ==============================================================================
# VALUE BOUNDARY CHECKS (Ranges, vocabularies, binary values)
# ==============================================================================

@pytest.mark.parametrize("col,invalid_val", [
    ("tenure_days", -1),
    ("sessions_7d", -3),
    ("watch_hours_7d", -0.01),
    ("support_tickets_30d", -5),
])
def test_rejects_negative_numeric_values(
    valid_train_df: pd.DataFrame, col: str, invalid_val: int | float
) -> None:
    """All engagement metrics must be non-negative (>= 0)."""
    corrupted_df = valid_train_df.copy()
    corrupted_df[col] = invalid_val

    with pytest.raises(DataContractError, match="greater_than_or_equal_to"):
        validate_training_data(corrupted_df)


def test_rejects_invalid_plan_tier_vocabulary(valid_train_df: pd.DataFrame) -> None:
    """plan_tier must belong strictly to controlled vocabulary [basic, standard, premium]."""
    corrupted_df = valid_train_df.copy()
    corrupted_df["plan_tier"] = "ultra_vip"

    with pytest.raises(DataContractError, match="isin"):
        validate_training_data(corrupted_df)


@pytest.mark.parametrize("invalid_flag", [-1, 2, 99])
def test_rejects_non_binary_flag_values(
    valid_train_df: pd.DataFrame, invalid_flag: int
) -> None:
    """price_increase_flag and churned_30d must strictly be 0 or 1."""
    corrupted_df = valid_train_df.copy()
    corrupted_df["price_increase_flag"] = invalid_flag

    with pytest.raises(DataContractError, match="isin"):
        validate_training_data(corrupted_df)


def test_rejects_non_binary_churn_label(valid_train_df: pd.DataFrame) -> None:
    """churned_30d label must strictly be 0 or 1."""
    corrupted_df = valid_train_df.copy()
    corrupted_df["churned_30d"] = 3

    with pytest.raises(DataContractError, match="isin"):
        validate_training_data(corrupted_df)


# ==============================================================================
# DISK FILE AND CLI QUALITY GATE CHECKS
# ==============================================================================

def test_validate_dataset_file_success(tmp_path: pytest.TempPathFactory, valid_train_df: pd.DataFrame) -> None:
    """Valid parquet and csv files on disk pass validation."""
    from streamly.data.validation import validate_dataset_file

    parquet_path = tmp_path / "valid.parquet"  # type: ignore[operator]
    valid_train_df.to_parquet(parquet_path)
    validate_dataset_file(parquet_path, mode="train")


def test_validate_dataset_file_missing_file(tmp_path: pytest.TempPathFactory) -> None:
    """Non-existent files raise FileNotFoundError."""
    from streamly.data.validation import validate_dataset_file

    with pytest.raises(FileNotFoundError):
        validate_dataset_file(tmp_path / "non_existent.parquet")  # type: ignore[operator]


def test_cli_quality_gate_exit_codes(tmp_path: pytest.TempPathFactory, valid_train_df: pd.DataFrame) -> None:
    """CLI must exit 0 for valid data and 1 (non-zero) for invalid data."""
    import sys

    from streamly.data.validation import main as validation_main

    # 1. Valid data should exit 0
    valid_path = tmp_path / "valid.parquet"  # type: ignore[operator]
    valid_train_df.to_parquet(valid_path)

    sys.argv = ["validate", "--data-path", str(valid_path), "--mode", "train"]
    with pytest.raises(SystemExit) as exit_info:
        validation_main()
    assert exit_info.value.code == 0

    # 2. Corrupted data must exit 1 (non-zero) to block CI / pipeline
    invalid_path = tmp_path / "corrupt.parquet"  # type: ignore[operator]
    corrupted = valid_train_df.copy()
    corrupted["tenure_days"] = -99  # negative tenure
    corrupted.to_parquet(invalid_path)

    sys.argv = ["validate", "--data-path", str(invalid_path), "--mode", "train"]
    with pytest.raises(SystemExit) as exit_info:
        validation_main()
    assert exit_info.value.code == 1

