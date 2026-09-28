"""Tests for shared feature builder.

Proves that training and serving share an identical feature contract,
with exact column names, ordering, dtypes, zero leakage, and deterministic encoding.
"""

import math

import pandas as pd
import pytest

from streamly.features.builder import (
    FEATURE_COLUMNS,
    build_feature_matrix,
    build_serving_features,
    build_training_features,
    extract_target,
)


@pytest.fixture
def sample_raw_train_df() -> pd.DataFrame:
    """Fixture with representative member snapshots for training."""
    return pd.DataFrame(
        {
            "member_id": ["mem_000001", "mem_000002", "mem_000003"],
            "tenure_days": [365, 30, 720],
            "sessions_7d": [10, 0, 15],
            "watch_hours_7d": [25.5, 0.0, 42.0],
            "support_tickets_30d": [0, 3, 1],
            "plan_tier": ["basic", "standard", "premium"],
            "price_increase_flag": [0, 1, 0],
            "churned_30d": [0, 1, 0],
        }
    )


@pytest.fixture
def sample_raw_serve_df(sample_raw_train_df: pd.DataFrame) -> pd.DataFrame:
    """Equivalent serving DataFrame without the target label."""
    return sample_raw_train_df.drop(columns=["churned_30d"])


def test_feature_columns_and_ordering(sample_raw_train_df: pd.DataFrame) -> None:
    """Features must strictly match the declared FEATURE_COLUMNS contract and order."""
    features = build_feature_matrix(sample_raw_train_df)

    assert list(features.columns) == FEATURE_COLUMNS
    assert "member_id" not in features.columns
    assert "churned_30d" not in features.columns


def test_train_serve_feature_parity(
    sample_raw_train_df: pd.DataFrame, sample_raw_serve_df: pd.DataFrame
) -> None:
    """Prove that training and serving feature builders emit identical matrices for equivalent raw data."""
    X_train, y_train = build_training_features(sample_raw_train_df)
    X_serve = build_serving_features(sample_raw_serve_df)

    # 1. Shape and columns match
    assert X_train.shape == X_serve.shape
    assert list(X_train.columns) == list(X_serve.columns)

    # 2. Dtypes match exactly
    assert (X_train.dtypes == X_serve.dtypes).all()

    # 3. Values match exactly
    pd.testing.assert_frame_equal(X_train, X_serve)

    # 4. Target extracted properly
    assert list(y_train) == [0, 1, 0]


def test_single_row_inference_preserves_all_categorical_columns() -> None:
    """A single-row inference payload must preserve all one-hot columns without shape drift."""
    single_record = pd.DataFrame(
        {
            "member_id": ["mem_999999"],
            "tenure_days": [100],
            "sessions_7d": [7],
            "watch_hours_7d": [14.0],
            "support_tickets_30d": [0],
            "plan_tier": ["premium"],  # Only 1 plan tier present in this batch
            "price_increase_flag": [0],
        }
    )

    X = build_serving_features(single_record)

    assert X.shape == (1, len(FEATURE_COLUMNS))
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.loc[0, "plan_tier_basic"] == 0
    assert X.loc[0, "plan_tier_standard"] == 0
    assert X.loc[0, "plan_tier_premium"] == 1


def test_derived_feature_division_by_zero_safety() -> None:
    """When sessions_7d == 0, watch_hours_per_session must evaluate safely to 0.0 without inf/nan."""
    zero_session_record = pd.DataFrame(
        {
            "member_id": ["mem_000010"],
            "tenure_days": [50],
            "sessions_7d": [0],
            "watch_hours_7d": [0.0],
            "support_tickets_30d": [2],
            "plan_tier": ["basic"],
            "price_increase_flag": [0],
        }
    )

    X = build_serving_features(zero_session_record)

    ratio = float(X["watch_hours_per_session"].iloc[0])
    assert not math.isnan(ratio)
    assert not math.isinf(ratio)
    assert ratio == 0.0


def test_extract_target_missing_raises_key_error(sample_raw_serve_df: pd.DataFrame) -> None:
    """extract_target must fail loudly with KeyError if called on serving payloads."""
    with pytest.raises(KeyError, match="Target column 'churned_30d' is not present"):
        extract_target(sample_raw_serve_df)
