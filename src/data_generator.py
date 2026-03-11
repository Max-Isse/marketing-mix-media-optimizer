"""
Data generation and preprocessing pipeline for Marketing Mix Model.
Produces realistic synthetic marketing data with seasonal patterns.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_marketing_data(
    n_weeks: int = 156,  # 3 years of weekly data
    seed: int = 42,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic marketing dataset with realistic patterns.

    Features:
    - 5 media channels with realistic spend distributions
    - Seasonal revenue patterns (Q4 uplift, summer dip)
    - Diminishing returns and carryover effects baked into ground truth
    - Gaussian noise on revenue

    Args:
        n_weeks: Number of weeks to simulate
        seed: Random seed for reproducibility
        output_path: If provided, save CSV to this path

    Returns:
        DataFrame with columns: week, tv_spend, digital_spend,
        social_spend, ooh_spend, radio_spend, trend, seasonality, revenue
    """
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2021-01-04", periods=n_weeks, freq="W-MON")

    # ── Media Spend ───────────────────────────────────────────────────────────
    channels = {
        "tv_spend": {"mean": 150_000, "std": 40_000, "floor": 0},
        "digital_spend": {"mean": 80_000, "std": 25_000, "floor": 10_000},
        "social_spend": {"mean": 45_000, "std": 15_000, "floor": 5_000},
        "ooh_spend": {"mean": 30_000, "std": 12_000, "floor": 0},
        "radio_spend": {"mean": 20_000, "std": 8_000, "floor": 0},
    }

    spend_data = {}
    for ch, params in channels.items():
        raw = rng.normal(params["mean"], params["std"], n_weeks)
        # TV goes dark in Jan/Feb some years
        if ch == "tv_spend":
            for w in range(n_weeks):
                if weeks[w].month in [1, 2] and rng.random() < 0.4:
                    raw[w] = 0
        spend_data[ch] = np.maximum(raw, params["floor"])

    # ── Trend & Seasonality ───────────────────────────────────────────────────
    trend = np.linspace(1.0, 1.35, n_weeks)  # 35% growth over 3 years

    week_of_year = np.array([w.isocalendar()[1] for w in weeks])
    seasonality = (
        1.0
        + 0.20 * np.sin(2 * np.pi * week_of_year / 52)          # annual cycle
        + 0.10 * np.sin(4 * np.pi * week_of_year / 52)          # bi-annual
        - 0.05 * ((week_of_year >= 6) & (week_of_year <= 10))   # Jan/Feb dip
        + 0.15 * ((week_of_year >= 45) & (week_of_year <= 52))  # Q4 uplift
    )

    # ── Revenue Generation (ground truth) ────────────────────────────────────
    # Apply adstock + saturation manually to produce realistic ground truth
    def _adstock(x, d):
        out = np.zeros_like(x, dtype=float)
        out[0] = x[0]
        for t in range(1, len(x)):
            out[t] = x[t] + d * out[t - 1]
        return out

    def _sat(x, a, g):
        x = np.maximum(x, 1e-10)
        return x**a / (x**a + g**a)

    channel_params = {
        "tv_spend":      {"decay": 0.7, "alpha": 0.8, "gamma": 120_000, "coef": 2.5},
        "digital_spend": {"decay": 0.3, "alpha": 0.9, "gamma":  60_000, "coef": 3.8},
        "social_spend":  {"decay": 0.2, "alpha": 1.0, "gamma":  35_000, "coef": 2.1},
        "ooh_spend":     {"decay": 0.5, "alpha": 0.7, "gamma":  25_000, "coef": 1.4},
        "radio_spend":   {"decay": 0.4, "alpha": 0.8, "gamma":  15_000, "coef": 1.1},
    }

    base_revenue = 500_000
    media_revenue = np.zeros(n_weeks)
    for ch, p in channel_params.items():
        transformed = _sat(_adstock(spend_data[ch], p["decay"]), p["alpha"], p["gamma"])
        media_revenue += p["coef"] * transformed * 100_000

    revenue = (
        base_revenue * trend * seasonality
        + media_revenue
        + rng.normal(0, 15_000, n_weeks)  # noise
    )

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "week": weeks,
            **spend_data,
            "trend": trend,
            "seasonality": seasonality,
            "revenue": np.maximum(revenue, 0),
        }
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Saved {len(df)} rows to {output_path}")

    return df


if __name__ == "__main__":
    df = generate_marketing_data(output_path="data/sample/marketing_data.csv")
    print(df.describe())
