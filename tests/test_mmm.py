"""
Unit tests for the Bayesian Marketing Mix Model.

Everything except TestPyMCFit runs without PyMC: those tests install a synthetic
posterior with set_posterior() and check the code that works on posterior draws.
TestPyMCFit samples the real model and is skipped when PyMC is not installed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generator import generate_marketing_data
from src.mmm_model import (
    MarketingMixModel,
    adstock_transform,
    gelman_rubin,
    hill_saturation,
    lag_matrix,
)

CHANNELS = ["tv_spend", "digital_spend", "social_spend", "ooh_spend", "radio_spend"]


def synthetic_posterior(n_channels, n_controls, n_draws=200, seed=0):
    """Draws from the model's priors, standing in for a fitted posterior."""
    rng = np.random.default_rng(seed)
    return {
        "intercept": rng.normal(0.5, 0.1, n_draws),
        "beta": np.abs(rng.normal(0, 0.5, (n_draws, n_channels))),
        "decay": rng.beta(3, 3, (n_draws, n_channels)),
        "alpha": rng.gamma(3, 1 / 3, (n_draws, n_channels)),
        "gamma": rng.beta(2, 2, (n_draws, n_channels)),
        "control_coef": rng.normal(0, 0.1, (n_draws, n_controls)),
        "sigma": np.abs(rng.normal(0, 0.2, n_draws)),
    }


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


class TestLagMatrix:
    def test_columns_are_shifted_copies(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        lagged = lag_matrix(x, max_lag=3)
        assert lagged.shape == (4, 3)
        np.testing.assert_array_equal(lagged[:, 0], x)
        np.testing.assert_array_equal(lagged[:, 1], [0, 1, 2, 3])
        np.testing.assert_array_equal(lagged[:, 2], [0, 0, 1, 2])

    def test_window_longer_than_series(self):
        assert lag_matrix(np.array([5.0, 6.0]), max_lag=4).shape == (2, 4)

    def test_matches_geometric_adstock_within_the_window(self):
        """Weighted lags reproduce the recursive adstock for a pulse inside the window."""
        x = np.zeros(8)
        x[0] = 100
        weights = 0.5 ** np.arange(8)
        np.testing.assert_allclose(lag_matrix(x, 8) @ weights, adstock_transform(x, 0.5))


class TestGelmanRubin:
    def test_agreeing_chains_are_close_to_one(self):
        chains = np.random.default_rng(0).normal(size=(4, 2000, 3))
        assert gelman_rubin(chains) < 1.01

    def test_disagreeing_chains_are_flagged(self):
        chains = np.random.default_rng(0).normal(size=(4, 500))
        chains[0] += 5
        assert gelman_rubin(chains) > 1.5

    def test_single_chain_gives_nan(self):
        assert np.isnan(gelman_rubin(np.zeros((1, 100))))


# ── Model Tests (synthetic posterior, no PyMC needed) ─────────────────────────

@pytest.fixture
def training_data():
    return generate_marketing_data(n_weeks=52, seed=42)


@pytest.fixture
def fitted_model(training_data):
    model = MarketingMixModel(channels=CHANNELS, target="revenue")
    return model.set_posterior(training_data, synthetic_posterior(len(CHANNELS), n_controls=3))


class TestMarketingMixModel:
    def test_set_posterior_marks_the_model_fitted(self, fitted_model):
        assert fitted_model.is_fitted
        assert fitted_model.control_names_ == ["trend", "seasonality", "trend_x_seasonality"]

    def test_set_posterior_rejects_wrong_shapes(self, training_data):
        posterior = synthetic_posterior(n_channels=2, n_controls=3)
        with pytest.raises(ValueError):
            MarketingMixModel(channels=CHANNELS).set_posterior(training_data, posterior)

    def test_predict_before_fit_fails(self, training_data):
        with pytest.raises(AssertionError):
            MarketingMixModel(channels=CHANNELS).predict(training_data)

    def test_predict_shape(self, fitted_model, training_data):
        assert fitted_model.predict(training_data).shape == (len(training_data),)
        assert fitted_model.predict(training_data, return_draws=True).shape == (200, len(training_data))

    def test_interval_brackets_the_mean(self, fitted_model, training_data):
        interval = fitted_model.predict_interval(training_data)
        assert (interval["lower"] <= interval["mean"]).all()
        assert (interval["mean"] <= interval["upper"]).all()

    def test_media_effect_matches_a_manual_calculation(self, fitted_model, training_data):
        """One channel, one draw, computed by hand with the public transforms."""
        posterior = {name: values[:1] for name, values in fitted_model.posterior_.items()}
        x_lag, _ = fitted_model._design(training_data)
        media = fitted_model._media_draws(x_lag, posterior)[0, :, 0]

        decay, alpha, gamma, beta = (posterior[k][0, 0] for k in ("decay", "alpha", "gamma", "beta"))
        weights = decay ** np.arange(fitted_model.max_lag)
        scaled = training_data[CHANNELS[0]].values / training_data[CHANNELS[0]].max()
        adstocked = lag_matrix(scaled, fitted_model.max_lag) @ (weights / weights.sum())
        expected = np.where(adstocked > 0, hill_saturation(adstocked, alpha, gamma), 0.0)
        expected = expected * beta * training_data["revenue"].mean()
        np.testing.assert_allclose(media, expected, rtol=1e-6)

    def test_channel_with_no_spend_contributes_nothing(self, training_data):
        data = training_data.copy()
        data["radio_spend"] = 0.0
        model = MarketingMixModel(channels=CHANNELS).set_posterior(data, synthetic_posterior(5, 3))
        contribution = model.channel_contribution(data).set_index("channel")
        assert contribution.loc["radio_spend", "contribution"] == 0.0

    def test_more_spend_never_lowers_predicted_revenue(self, fitted_model, training_data):
        boosted = training_data.copy()
        boosted["digital_spend"] *= 1.5
        assert (fitted_model.predict(boosted) >= fitted_model.predict(training_data) - 1e-6).all()

    def test_contributions(self, fitted_model, training_data):
        contrib = fitted_model.channel_contribution(training_data)
        assert len(contrib) == len(CHANNELS)
        assert (contrib["contribution"] >= 0).all()
        assert contrib["contribution_pct"].sum() == pytest.approx(100)
        assert (contrib["contribution_lower"] <= contrib["contribution"]).all()
        assert (contrib["contribution"] <= contrib["contribution_upper"]).all()
        assert contrib["contribution_pct"].is_monotonic_decreasing

    def test_roi_is_contribution_over_spend(self, fitted_model, training_data):
        roi = fitted_model.channel_roi(training_data).set_index("channel")
        contrib = fitted_model.channel_contribution(training_data).set_index("channel")
        for channel in CHANNELS:
            expected = contrib.loc[channel, "contribution"] / training_data[channel].sum()
            assert roi.loc[channel, "roi"] == pytest.approx(expected)
        assert roi["prob_roi_above_1"].between(0, 1).all()

    def test_optimiser_respects_budget_and_bounds(self, fitted_model, training_data):
        total = 5_000_000
        result = fitted_model.optimise_budget(total, training_data)
        allocation = np.array(list(result["optimal_allocation"].values()))
        assert allocation.sum() == pytest.approx(total)
        assert (allocation >= total * 0.01 - 1).all() and (allocation <= total * 0.8 + 1).all()

    def test_optimiser_is_at_least_as_good_as_the_current_mix(self, fitted_model, training_data):
        result = fitted_model.optimise_budget(5_000_000, training_data)
        assert result["expected_revenue"] >= result["current_mix_revenue"] - 1e-6
        assert result["expected_uplift"] >= -1e-6
        low, high = result["expected_revenue_interval"]
        assert low <= result["expected_revenue"] <= high
        assert 0 <= result["prob_better_than_current_mix"] <= 1

    def test_optimiser_roi_differs_by_channel(self, fitted_model, training_data):
        roi = fitted_model.optimise_budget(5_000_000, training_data)["roi_by_channel"]
        assert set(roi) == set(CHANNELS)
        assert len({round(value, 6) for value in roi.values()}) > 1

    def test_model_without_controls(self, training_data):
        data = training_data.drop(columns=["trend", "seasonality"])
        model = MarketingMixModel(channels=CHANNELS[:3]).set_posterior(data, synthetic_posterior(3, 0))
        assert model.control_names_ == []
        assert model.predict(data).shape == (len(data),)

    def test_insight_payload_keys_and_json(self, fitted_model, training_data):
        payload = fitted_model.generate_insight_payload(training_data, 5_000_000)
        required_keys = [
            "model_performance", "sampler_diagnostics", "channel_attribution", "channel_roi",
            "budget_optimisation", "top_channel", "lowest_roi_channel",
        ]
        for key in required_keys:
            assert key in payload, f"Missing key: {key}"
        json.dumps(payload, allow_nan=False)  # must be valid JSON for the LLM prompt and the API

    def test_insight_payload_handles_a_channel_with_no_spend(self, training_data):
        data = training_data.copy()
        data["radio_spend"] = 0.0
        model = MarketingMixModel(channels=CHANNELS).set_posterior(data, synthetic_posterior(5, 3))
        payload = model.generate_insight_payload(data, 5_000_000)
        json.dumps(payload, allow_nan=False)
        assert payload["lowest_roi_channel"] != "radio_spend"


# ── PyMC Fit (skipped when PyMC is not installed) ─────────────────────────────

@pytest.fixture(scope="module")
def data():
    return generate_marketing_data(n_weeks=104, seed=42)


@pytest.fixture(scope="module")
def sampled_model(data):
    pytest.importorskip("pymc")
    model = MarketingMixModel(channels=CHANNELS, target="revenue")
    return model.fit(data, draws=300, tune=500, chains=2, progressbar=False)


class TestPyMCFit:
    def test_posterior_shapes(self, sampled_model):
        posterior = sampled_model.posterior_
        assert posterior["intercept"].shape == (600,)
        assert posterior["beta"].shape == (600, len(CHANNELS))
        assert posterior["control_coef"].shape == (600, 3)

    def test_parameters_respect_their_supports(self, sampled_model):
        posterior = sampled_model.posterior_
        assert (posterior["beta"] >= 0).all()
        assert ((posterior["decay"] > 0) & (posterior["decay"] < 1)).all()
        assert ((posterior["gamma"] > 0) & (posterior["gamma"] < 1)).all()

    def test_fit_quality(self, sampled_model):
        assert sampled_model.train_r2_ > 0.70, f"R² too low: {sampled_model.train_r2_}"

    def test_diagnostics_are_recorded(self, sampled_model):
        diagnostics = sampled_model.diagnostics_
        assert diagnostics["chains"] == 2 and diagnostics["draws_per_chain"] == 300
        assert diagnostics["max_rhat"] is not None and diagnostics["max_rhat"] > 0.9
        assert diagnostics["divergences"] >= 0

    def test_downstream_methods_run_on_a_real_posterior(self, sampled_model, data):
        assert len(sampled_model.channel_roi(data)) == len(CHANNELS)
        result = sampled_model.optimise_budget(10_000_000, data)
        assert sum(result["optimal_allocation"].values()) == pytest.approx(10_000_000)


def test_fit_without_pymc_gives_a_clear_error(training_data):
    try:
        import pymc  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="pip install pymc"):
            MarketingMixModel(channels=CHANNELS).fit(training_data)
    else:
        pytest.skip("PyMC is installed")


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
