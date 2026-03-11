"""
Unit tests for Marketing Mix Model
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mmm_model import MarketingMixModel, adstock_transform, hill_saturation
from src.data_generator import generate_marketing_data


# ── Transform Tests ───────────────────────────────────────────────────────────

class TestAdstock:
    def test_no_decay(self):
        """With decay=0, adstock equals input."""
        x = np.array([100, 50, 200, 0, 100], dtype=float)
        result = adstock_transform(x, decay=0.0)
        np.testing.assert_array_almost_equal(result, x)

    def test_full_decay(self):
        """With decay=1, adstock is cumulative sum."""
        x = np.array([100, 100, 100], dtype=float)
        result = adstock_transform(x, decay=1.0)
        assert result[2] == pytest.approx(300.0)

    def test_carryover(self):
        """Verify carryover: spend 100 in week 0, 0 thereafter."""
        x = np.array([100, 0, 0, 0], dtype=float)
        result = adstock_transform(x, decay=0.5)
        assert result[0] == 100
        assert result[1] == pytest.approx(50)
        assert result[2] == pytest.approx(25)
        assert result[3] == pytest.approx(12.5)


class TestHillSaturation:
    def test_output_range(self):
        """Output should be in [0, 1]."""
        x = np.linspace(0, 1_000_000, 100)
        out = hill_saturation(x, alpha=1.0, gamma=100_000)
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_half_saturation(self):
        """At x=gamma, output should be exactly 0.5."""
        gamma = 50_000
        out = hill_saturation(np.array([gamma], dtype=float), alpha=1.0, gamma=gamma)
        assert out[0] == pytest.approx(0.5, abs=1e-6)

    def test_monotonic(self):
        """Hill function should be monotonically increasing."""
        x = np.linspace(1, 500_000, 100)
        out = hill_saturation(x, alpha=0.9, gamma=80_000)
        assert np.all(np.diff(out) > 0)


# ── Model Tests ───────────────────────────────────────────────────────────────

@pytest.fixture
def training_data():
    return generate_marketing_data(n_weeks=52, seed=42)


@pytest.fixture
def fitted_model(training_data):
    channels = ["tv_spend", "digital_spend", "social_spend", "ooh_spend", "radio_spend"]
    model = MarketingMixModel(channels=channels, target="revenue")
    model.fit(training_data)
    return model


class TestMarketingMixModel:
    def test_fit_runs(self, training_data):
        channels = ["tv_spend", "digital_spend", "social_spend"]
        model = MarketingMixModel(channels=channels)
        model.fit(training_data)
        assert model.is_fitted

    def test_r2_above_threshold(self, fitted_model):
        """Model should achieve R² > 0.70 on synthetic data."""
        assert fitted_model.train_r2_ > 0.70, f"R² too low: {fitted_model.train_r2_}"

    def test_predict_shape(self, fitted_model, training_data):
        preds = fitted_model.predict(training_data)
        assert preds.shape == (len(training_data),)

    def test_predict_positive(self, fitted_model, training_data):
        """Revenue predictions should be positive for non-zero spend."""
        preds = fitted_model.predict(training_data)
        assert np.mean(preds > 0) > 0.95

    def test_contribution_sums(self, fitted_model, training_data):
        """Channel contributions should all be non-negative (positive coefficients expected)."""
        contrib = fitted_model.channel_contribution(training_data)
        assert len(contrib) == len(fitted_model.channels)
        assert "contribution_pct" in contrib.columns

    def test_optimiser_respects_budget(self, fitted_model, training_data):
        """Optimal allocation should sum to total budget (within 1%)."""
        total = 5_000_000
        result = fitted_model.optimise_budget(total, training_data)
        allocated = sum(result["optimal_allocation"].values())
        assert abs(allocated - total) / total < 0.01

    def test_insight_payload_keys(self, fitted_model, training_data):
        payload = fitted_model.generate_insight_payload(training_data, 5_000_000)
        required_keys = [
            "model_performance", "channel_attribution",
            "budget_optimisation", "top_channel", "lowest_roi_channel"
        ]
        for key in required_keys:
            assert key in payload, f"Missing key: {key}"


# ── Data Generator Tests ──────────────────────────────────────────────────────

class TestDataGenerator:
    def test_shape(self):
        df = generate_marketing_data(n_weeks=52)
        assert len(df) == 52

    def test_no_negative_revenue(self):
        df = generate_marketing_data(n_weeks=104)
        assert (df["revenue"] >= 0).all()

    def test_columns_present(self):
        df = generate_marketing_data(n_weeks=10)
        expected = ["week", "tv_spend", "digital_spend", "social_spend",
                    "ooh_spend", "radio_spend", "trend", "seasonality", "revenue"]
        for col in expected:
            assert col in df.columns

    def test_reproducibility(self):
        df1 = generate_marketing_data(n_weeks=52, seed=0)
        df2 = generate_marketing_data(n_weeks=52, seed=0)
        pd.testing.assert_frame_equal(df1, df2)
