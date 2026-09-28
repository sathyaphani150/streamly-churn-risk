"""Shared feature engineering module for Streamly.

Provides a single source of truth for transforming raw member records into
model-ready numerical feature matrices, guaranteeing identical columns, dtypes,
and ordering across training and online serving.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from streamly.data.validation import (
    validate_serving_data,
    validate_training_data,
)

# Immutable feature column specification and fixed ordering for all downstream models
FEATURE_COLUMNS: Final[list[str]] = [
    "tenure_days",
    "sessions_7d",
    "watch_hours_7d",
    "support_tickets_30d",
    "price_increase_flag",
    "plan_tier_basic",
    "plan_tier_standard",
    "plan_tier_premium",
    "watch_hours_per_session",
]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw member snapshot records into the model-ready feature matrix X.

    This function is the SINGLE SHARED TRANSFORMER used by both training and
    real-time serving paths. It guarantees:
    1. `member_id` is excluded (identity only, never a model feature).
    2. `churned_30d` is excluded (ground truth label, strictly forbidden in X).
    3. Categorical `plan_tier` is deterministically encoded into fixed binary columns
       ([basic, standard, premium]), preventing shape mismatches even for single-row payloads.
    4. Derived engagement feature (`watch_hours_per_session`) is computed deterministically.
    5. Columns and dtypes are guaranteed to match `FEATURE_COLUMNS` exactly.

    Args:
        df: DataFrame containing member features (validated raw input).

    Returns:
        pd.DataFrame: Feature matrix X with exact columns and numeric dtypes.
    """
    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    # 1. Base numerical features
    tenure_days = df["tenure_days"].astype(np.int64)
    sessions_7d = df["sessions_7d"].astype(np.int64)
    watch_hours_7d = df["watch_hours_7d"].astype(np.float64)
    support_tickets_30d = df["support_tickets_30d"].astype(np.int64)
    price_increase_flag = df["price_increase_flag"].astype(np.int64)

    # 2. Deterministic One-Hot Encoding for plan_tier
    # Converting to str handles both pd.Categorical and string objects seamlessly
    plan_tier_str = df["plan_tier"].astype(str)
    plan_tier_basic = (plan_tier_str == "basic").astype(np.int64)
    plan_tier_standard = (plan_tier_str == "standard").astype(np.int64)
    plan_tier_premium = (plan_tier_str == "premium").astype(np.int64)

    # 3. Derived feature: average watch hours per session
    # Uses Laplace smoothing (+1.0) to deterministically prevent division by zero
    watch_hours_per_session = np.round(
        watch_hours_7d / (sessions_7d + 1.0),
        decimals=4,
    ).astype(np.float64)

    # 4. Construct final feature matrix in strict canonical column order
    feature_matrix = pd.DataFrame(
        {
            "tenure_days": tenure_days,
            "sessions_7d": sessions_7d,
            "watch_hours_7d": watch_hours_7d,
            "support_tickets_30d": support_tickets_30d,
            "price_increase_flag": price_increase_flag,
            "plan_tier_basic": plan_tier_basic,
            "plan_tier_standard": plan_tier_standard,
            "plan_tier_premium": plan_tier_premium,
            "watch_hours_per_session": watch_hours_per_session,
        },
        index=df.index,
    )

    return feature_matrix[FEATURE_COLUMNS]


def extract_target(df: pd.DataFrame) -> pd.Series:
    """Extract binary training target label (churned_30d) as a pandas Series.

    Args:
        df: Validated training DataFrame.

    Returns:
        pd.Series containing binary churn outcomes (int64).

    Raises:
        KeyError: If churned_30d is missing (e.g. called on serving data).
    """
    if "churned_30d" not in df.columns:
        raise KeyError("Target column 'churned_30d' is not present in DataFrame.")
    return df["churned_30d"].astype(np.int64)


def build_training_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate training dataset, extract target label y, and build feature matrix X.

    Args:
        df: Raw training DataFrame.

    Returns:
        tuple[pd.DataFrame, pd.Series]: (X_features, y_target).
    """
    validated_df = validate_training_data(df)
    X = build_feature_matrix(validated_df)
    y = extract_target(validated_df)
    return X, y


def build_serving_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate serving payload (enforcing zero leakage) and build feature matrix X.

    Args:
        df: Raw serving request DataFrame.

    Returns:
        pd.DataFrame: Feature matrix X for model scoring.
    """
    validated_df = validate_serving_data(df)
    return build_feature_matrix(validated_df)
