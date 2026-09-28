"""Synthetic dataset generator for Streamly churn assessment.

Generates reproducible historical member snapshots adhering strictly
to the required train/serve feature contract from the assignment specification.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_churn_data(
    num_rows: int = 10000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic, leakage-free historical member snapshots.

    Args:
        num_rows: Number of member records to generate (default: 10,000).
        random_seed: Seed for reproducible pseudorandom generation.

    Returns:
        pd.DataFrame containing the generated dataset with exact contract types.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Identity field (string, identity only)
    member_ids = [f"mem_{i:06d}" for i in range(1, num_rows + 1)]

    # 2. Tenure in days: skewed distribution representing newer vs loyal members
    # Exponential distribution shifted by 1 day
    tenure_days = np.clip(rng.exponential(scale=240, size=num_rows) + 1, 1, 1200).astype(int)

    # 3. Plan tier: categorical choice
    plan_tiers = rng.choice(
        ["basic", "standard", "premium"],
        size=num_rows,
        p=[0.45, 0.40, 0.15],
    )

    # 4. Weekly sessions: Poisson distribution influenced slightly by tenure
    base_sessions = rng.poisson(lam=8.0, size=num_rows)
    sessions_7d = np.clip(base_sessions, 0, 50).astype(int)

    # 5. Weekly watch hours: correlated with sessions, with random viewing variations
    # Average ~1.5 to 2.5 hours per session
    hours_per_session = rng.gamma(shape=2.0, scale=1.0, size=num_rows)
    watch_hours_7d = np.clip(
        np.round(sessions_7d * hours_per_session + rng.normal(0, 0.5, size=num_rows), 2),
        0.0,
        70.0,
    )
    # Ensure zero sessions equates to zero or near-zero watch hours
    watch_hours_7d = np.where(sessions_7d == 0, 0.0, watch_hours_7d)

    # 6. Support tickets in last 30 days: mostly 0-1, tail up to 8
    support_tickets_30d = rng.choice(
        [0, 1, 2, 3, 4, 5, 6],
        size=num_rows,
        p=[0.60, 0.22, 0.10, 0.04, 0.02, 0.015, 0.005],
    ).astype(int)

    # 7. Price increase flag: ~25% of members experienced a recent price change
    price_increase_flag = rng.choice([0, 1], size=num_rows, p=[0.75, 0.25]).astype(int)

    # 8. Compute latent churn risk logit based on behavioral relationships
    # Baseline log-odds corresponds to ~18% base probability (logit ~= -1.5)
    z = (
        -1.50
        - 0.002 * (tenure_days - 180)                       # Longer tenure reduces churn
        - 0.12 * (sessions_7d - 8)                          # More sessions reduces churn
        - 0.08 * (watch_hours_7d - 15)                      # More watch hours reduces churn
        + 0.55 * support_tickets_30d                        # Frequent support tickets increase churn
        + 0.70 * price_increase_flag                        # Price increase sharply increases churn
        + np.where(plan_tiers == "basic", 0.25, 0.0)        # Basic tier has higher churn propensity
        - np.where(plan_tiers == "premium", 0.35, 0.0)      # Premium tier has higher retention
        + rng.normal(loc=0.0, scale=0.4, size=num_rows)     # Latent idiosyncratic noise
    )

    # Logistic sigmoid function: P(churn) = 1 / (1 + exp(-z))
    churn_prob = 1.0 / (1.0 + np.exp(-z))

    # Binary realization via Bernoulli trial
    churned_30d = (rng.uniform(size=num_rows) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "member_id": member_ids,
            "tenure_days": tenure_days,
            "sessions_7d": sessions_7d,
            "watch_hours_7d": watch_hours_7d,
            "support_tickets_30d": support_tickets_30d,
            "plan_tier": pd.Categorical(plan_tiers, categories=["basic", "standard", "premium"]),
            "price_increase_flag": price_increase_flag,
            "churned_30d": churned_30d,
        }
    )

    return df


def main() -> None:
    """Generate and persist the raw synthetic churn dataset."""
    output_path = Path("data/raw/streamly_churn_sample.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic churn dataset (seed=42)...")
    df = generate_synthetic_churn_data(num_rows=10000, random_seed=42)

    df.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"Successfully saved {len(df):,} rows to {output_path}")


if __name__ == "__main__":
    main()
