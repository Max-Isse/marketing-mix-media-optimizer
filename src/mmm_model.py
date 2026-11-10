"""
Marketing Mix Media Optimizer
==============================
Bayesian Marketing Mix Model (PyMC) with adstock and Hill saturation, posterior
credible intervals for attribution and ROI, and a budget optimiser that works on
the posterior.

Model, per week t and channel c (spend scaled by the channel's maximum, revenue
scaled by its mean):

    adstock[t, c]   = sum_l  w[c, l] * spend[t - l, c],   w[c, l] ∝ decay[c] ** l
    saturated[t, c] = adstock^alpha[c] / (adstock^alpha[c] + gamma[c]^alpha[c])   (zero when adstock is zero)
    mu[t]           = intercept + sum_c beta[c] * saturated[t, c] + controls[t] · control_coef
    revenue[t]      ~ Normal(mu[t], sigma)

Priors:

    decay[c]      ~ Beta(3, 3)         carryover rate
    alpha[c]      ~ Gamma(3, 3)        Hill shape (mean 1)
    gamma[c]      ~ Beta(2, 2)         half-saturation point, as a share of maximum spend
    beta[c]       ~ HalfNormal(0.5)    channel effect; positive by construction
    intercept     ~ Normal(0.5, 0.5)
    control_coef  ~ Normal(0, 0.5)
    sigma         ~ HalfNormal(0.2)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import mean_absolute_percentage_error, r2_score

MAX_LAG = 12          # weeks of carryover kept in the adstock window
ADSTOCK_FLOOR = 1e-9  # keeps log(adstock) finite; weeks with no adstocked spend contribute exactly zero
POSTERIOR_KEYS = ("intercept", "beta", "decay", "alpha", "gamma", "control_coef", "sigma")


# ── Adstock & Saturation Transforms ──────────────────────────────────────────

def adstock_transform(x: np.ndarray, decay: float) -> np.ndarray:
    """
    Apply geometric adstock transformation to capture carryover effects.

    adstock[t] = spend[t] + decay * adstock[t-1]

    Args:
        x: Array of media spend values over time
        decay: Decay rate in [0, 1]; higher = longer carryover effect

    Returns:
        Adstocked spend array
    """
    adstocked = np.zeros_like(x, dtype=float)
    adstocked[0] = x[0]
    for t in range(1, len(x)):
        adstocked[t] = x[t] + decay * adstocked[t - 1]
    return adstocked


def hill_saturation(x: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """
    Apply Hill (S-curve) saturation transformation.

    Args:
        x: Adstocked spend values
        alpha: Shape parameter (steepness of the curve)
        gamma: Half-saturation point (spend at 50% max effect)

    Returns:
        Saturated spend array in [0, 1]
    """
    x_safe = np.maximum(x, 1e-10)
    return x_safe ** alpha / (x_safe ** alpha + gamma ** alpha)


def lag_matrix(x: np.ndarray, max_lag: int = MAX_LAG) -> np.ndarray:
    """
    Stack lagged copies of a spend series: column l holds spend from l weeks earlier.

    Weeks before the start of the series count as zero spend.

    Returns:
        Array of shape (len(x), max_lag)
    """
    x = np.asarray(x, dtype=float)
    lagged = np.zeros((len(x), max_lag))
    for lag in range(min(max_lag, len(x))):
        lagged[lag:, lag] = x[: len(x) - lag]
    return lagged


def to_records(frame: pd.DataFrame) -> list[dict]:
    """DataFrame → list of dicts with NaN replaced by None, so the result is valid JSON."""
    return [
        {key: (None if isinstance(value, float) and np.isnan(value) else value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def gelman_rubin(chains: np.ndarray) -> float:
    """
    Largest R-hat across the parameters in `chains`, shape (n_chains, n_draws, ...).

    Values close to 1 mean the chains agree. Returns NaN with fewer than two chains.
    """
    chains = np.asarray(chains, dtype=float)
    n_chains, n_draws = chains.shape[:2]
    if n_chains < 2 or n_draws < 2:
        return float("nan")
    within = chains.var(axis=1, ddof=1).mean(axis=0)
    between = chains.mean(axis=1).var(axis=0, ddof=1)
    pooled = (n_draws - 1) / n_draws * within + between
    return float(np.max(np.sqrt(pooled / np.maximum(within, 1e-300))))


# ── Marketing Mix Model ───────────────────────────────────────────────────────

class MarketingMixModel:
    """
    Bayesian Marketing Mix Model (MMM) with:
    - Adstock (carryover) and Hill saturation estimated per channel
    - Trend and seasonality controls
    - Posterior credible intervals for attribution and ROI
    - Budget allocation optimiser over the posterior
    - LLM-ready insight extraction
    """

    def __init__(
        self,
        channels: list[str],
        target: str = "revenue",
        controls: list[str] | None = None,
        max_lag: int = MAX_LAG,
    ):
        self.channels = channels
        self.target = target
        self.controls = controls
        self.max_lag = max_lag
        self.posterior_: dict = {}
        self.diagnostics_: dict = {}
        self.idata_ = None
        self.is_fitted = False

    # ── Scaling & Feature Engineering ─────────────────────────────────────────

    def _raw_controls(self, df: pd.DataFrame) -> np.ndarray:
        columns = [df[name].values.astype(float) for name in self.controls_]
        if self.interaction_:
            columns.append(df["trend"].values.astype(float) * df["seasonality"].values.astype(float))
        if not columns:
            return np.zeros((len(df), 0))
        return np.column_stack(columns)

    def _prepare(self, df: pd.DataFrame) -> None:
        """Learn the scaling constants from the training data."""
        if self.controls is None:
            self.controls_ = [name for name in ("trend", "seasonality") if name in df.columns]
        else:
            self.controls_ = list(self.controls)
        # Baseline revenue often scales with trend × seasonality, so include the product.
        self.interaction_ = "trend" in self.controls_ and "seasonality" in self.controls_
        self.control_names_ = self.controls_ + (["trend_x_seasonality"] if self.interaction_ else [])

        self._spend_scale = np.array([max(float(df[ch].max()), 1.0) for ch in self.channels])
        raw = self._raw_controls(df)
        self._control_mean = raw.mean(axis=0)
        std = raw.std(axis=0)
        self._control_std = np.where(std == 0, 1.0, std)
        self._y_scale = float(df[self.target].mean())

    def _spend_lags(self, spend: np.ndarray) -> np.ndarray:
        """(T, C) spend in currency → (T, C, L) scaled lag tensor."""
        scaled = spend / self._spend_scale
        return np.stack([lag_matrix(scaled[:, c], self.max_lag) for c in range(len(self.channels))], axis=1)

    def _design(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        spend = np.column_stack([df[ch].values.astype(float) for ch in self.channels])
        controls = (self._raw_controls(df) - self._control_mean) / self._control_std
        return self._spend_lags(spend), controls

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        cores: int = 1,
        target_accept: float = 0.9,
        random_seed: int = 42,
        progressbar: bool = True,
    ) -> "MarketingMixModel":
        """
        Sample the posterior with NUTS.

        Args:
            df: DataFrame with channel spend columns, the target and any controls
            draws, tune, chains: Sampler settings
            cores: Chains run in parallel. The default of 1 runs them one after
                another, which avoids multiprocessing problems on Windows.
            target_accept: Raise towards 0.95–0.99 if the sampler reports divergences
            random_seed: Seed for reproducible draws

        Returns:
            self (fitted model)
        """
        try:
            import pymc as pm
            import pytensor.tensor as pt
        except ImportError as error:
            raise ImportError("fit() needs PyMC. Install it with: pip install pymc") from error

        self._prepare(df)
        x_lag, controls = self._design(df)
        y = df[self.target].values.astype(float) / self._y_scale
        coords = {"channel": self.channels}
        if self.control_names_:
            coords["control"] = self.control_names_
        with pm.Model(coords=coords) as model:
            lags = pt.as_tensor_variable(np.arange(self.max_lag, dtype=float))
            spend_lags = pt.as_tensor_variable(x_lag)                        # (T, C, L)
            decay = pm.Beta("decay", alpha=3.0, beta=3.0, dims="channel")
            alpha = pm.Gamma("alpha", alpha=3.0, beta=3.0, dims="channel")
            gamma = pm.Beta("gamma", alpha=2.0, beta=2.0, dims="channel")
            beta = pm.HalfNormal("beta", sigma=0.5, dims="channel")
            intercept = pm.Normal("intercept", mu=0.5, sigma=0.5)
            sigma = pm.HalfNormal("sigma", sigma=0.2)

            weights = decay[:, None] ** lags[None, :]                       # (C, L)
            weights = weights / weights.sum(axis=1, keepdims=True)
            adstocked = (spend_lags * weights[None, :, :]).sum(axis=2)       # (T, C)
            saturated = pt.switch(
                adstocked > 0,
                pm.math.sigmoid(alpha * (pt.log(pt.maximum(adstocked, ADSTOCK_FLOOR)) - pt.log(gamma))),
                0.0,
            )
            mu = intercept + (saturated * beta).sum(axis=1)
            if self.control_names_:
                control_coef = pm.Normal("control_coef", mu=0.0, sigma=0.5, dims="control")
                mu = mu + pt.dot(pt.as_tensor_variable(controls), control_coef)

            pm.Normal("y", mu=mu, sigma=sigma, observed=y)
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=progressbar,
            )

        self.model_ = model
        self.idata_ = idata

        by_chain = {}
        for name in POSTERIOR_KEYS:
            if name == "control_coef" and not self.control_names_:
                by_chain[name] = np.zeros((chains, draws, 0))
            else:
                by_chain[name] = np.asarray(idata.posterior[name].values, dtype=float)
        posterior = {name: values.reshape((-1,) + values.shape[2:]) for name, values in by_chain.items()}

        rhats = [gelman_rubin(values) for values in by_chain.values() if values.size]
        self.diagnostics_ = {
            "max_rhat": None if np.all(np.isnan(rhats)) else float(np.nanmax(rhats)),
            "divergences": int(np.asarray(idata.sample_stats["diverging"].values).sum()),
            "chains": chains,
            "draws_per_chain": draws,
        }
        return self._finalise(df, posterior)

    def set_posterior(self, df: pd.DataFrame, posterior: dict) -> "MarketingMixModel":
        """
        Install posterior draws produced elsewhere (a saved fit, or a test fixture).

        `posterior` maps each name in POSTERIOR_KEYS to an array whose first axis is
        the draw: intercept (S,), sigma (S,), beta/decay/alpha/gamma (S, n_channels),
        control_coef (S, n_controls).
        """
        self._prepare(df)
        posterior = {name: np.asarray(posterior[name], dtype=float) for name in POSTERIOR_KEYS}
        n_draws = len(posterior["intercept"])
        expected = {
            "intercept": (n_draws,),
            "sigma": (n_draws,),
            "control_coef": (n_draws, len(self.control_names_)),
            **{name: (n_draws, len(self.channels)) for name in ("beta", "decay", "alpha", "gamma")},
        }
        for name, shape in expected.items():
            if posterior[name].shape != shape:
                raise ValueError(f"posterior['{name}'] has shape {posterior[name].shape}, expected {shape}")
        return self._finalise(df, posterior)

    def _finalise(self, df: pd.DataFrame, posterior: dict) -> "MarketingMixModel":
        self.posterior_ = posterior
        self.is_fitted = True
        y = df[self.target].values.astype(float)
        y_pred = self.predict(df)
        self.train_r2_ = r2_score(y, y_pred)
        self.train_mape_ = mean_absolute_percentage_error(y, np.maximum(y_pred, 1))
        return self

    # ── Posterior Computations ────────────────────────────────────────────────

    def _thin(self, n_draws: int | None, seed: int = 0) -> dict:
        """Random subset of the posterior draws (all of them when n_draws is None)."""
        total = len(self.posterior_["intercept"])
        if n_draws is None or n_draws >= total:
            return self.posterior_
        index = np.random.default_rng(seed).choice(total, size=n_draws, replace=False)
        return {name: values[index] for name, values in self.posterior_.items()}

    def _media_draws(self, x_lag: np.ndarray, posterior: dict) -> np.ndarray:
        """Revenue from each channel, per draw and week: shape (S, T, C)."""
        lags = np.arange(self.max_lag)
        weights = posterior["decay"][:, :, None] ** lags                    # (S, C, L)
        weights = weights / weights.sum(axis=2, keepdims=True)
        adstocked = np.einsum("tcl,scl->stc", x_lag, weights)
        saturated = expit(
            posterior["alpha"][:, None, :]
            * (np.log(np.maximum(adstocked, ADSTOCK_FLOOR)) - np.log(posterior["gamma"][:, None, :]))
        )
        saturated = np.where(adstocked > 0, saturated, 0.0)  # no spend, no effect
        return saturated * posterior["beta"][:, None, :] * self._y_scale

    def _baseline_draws(self, controls: np.ndarray, posterior: dict) -> np.ndarray:
        """Revenue not attributed to media, per draw and week: shape (S, T)."""
        baseline = posterior["intercept"][:, None] + posterior["control_coef"] @ controls.T
        return baseline * self._y_scale

    def predict(self, df: pd.DataFrame, return_draws: bool = False) -> np.ndarray:
        """
        Predict expected revenue for a spend dataframe.

        Returns the posterior mean per week, or every posterior draw (S, T)
        when return_draws is True.
        """
        assert self.is_fitted, "Model must be fitted before predict()"
        x_lag, controls = self._design(df)
        draws = self._baseline_draws(controls, self.posterior_) + self._media_draws(x_lag, self.posterior_).sum(axis=2)
        return draws if return_draws else draws.mean(axis=0)

    def predict_interval(self, df: pd.DataFrame, prob: float = 0.9) -> pd.DataFrame:
        """
        Posterior mean and credible interval for expected weekly revenue.

        The interval covers uncertainty in the model's parameters. It does not
        include week-to-week noise, so actual revenue will fall outside it more often.
        """
        draws = self.predict(df, return_draws=True)
        tail = (1 - prob) / 2
        return pd.DataFrame(
            {
                "mean": draws.mean(axis=0),
                "lower": np.quantile(draws, tail, axis=0),
                "upper": np.quantile(draws, 1 - tail, axis=0),
            }
        )

    # ── Attribution ───────────────────────────────────────────────────────────

    def _contribution_draws(self, df: pd.DataFrame) -> np.ndarray:
        """Total revenue attributed to each channel over the period: shape (S, C)."""
        assert self.is_fitted
        x_lag, _ = self._design(df)
        return self._media_draws(x_lag, self.posterior_).sum(axis=1)

    def channel_contribution(self, df: pd.DataFrame, prob: float = 0.9) -> pd.DataFrame:
        """
        Decompose revenue into per-channel contributions with credible intervals.

        Returns:
            DataFrame with columns: channel, contribution, contribution_lower,
            contribution_upper, contribution_pct (share of all media-driven revenue)
        """
        draws = self._contribution_draws(df)
        tail = (1 - prob) / 2
        mean = draws.mean(axis=0)
        result = pd.DataFrame(
            {
                "channel": self.channels,
                "contribution": mean,
                "contribution_lower": np.quantile(draws, tail, axis=0),
                "contribution_upper": np.quantile(draws, 1 - tail, axis=0),
                "contribution_pct": mean / mean.sum() * 100,
            }
        )
        return result.sort_values("contribution_pct", ascending=False).reset_index(drop=True)

    def channel_roi(self, df: pd.DataFrame, prob: float = 0.9) -> pd.DataFrame:
        """
        Revenue returned per unit of spend, by channel, with credible intervals.

        Returns:
            DataFrame with columns: channel, spend, roi, roi_lower, roi_upper and
            prob_roi_above_1 (posterior probability the channel returned more than it cost)
        """
        spend = np.array([float(df[ch].sum()) for ch in self.channels])
        with np.errstate(divide="ignore", invalid="ignore"):
            draws = self._contribution_draws(df) / np.where(spend > 0, spend, np.nan)
        tail = (1 - prob) / 2
        result = pd.DataFrame(
            {
                "channel": self.channels,
                "spend": spend,
                "roi": draws.mean(axis=0),
                "roi_lower": np.quantile(draws, tail, axis=0),
                "roi_upper": np.quantile(draws, 1 - tail, axis=0),
                "prob_roi_above_1": (draws > 1).mean(axis=0),
            }
        )
        return result.sort_values("roi", ascending=False).reset_index(drop=True)

    # ── Budget Optimiser ──────────────────────────────────────────────────────

    def optimise_budget(
        self,
        total_budget: float,
        df_baseline: pd.DataFrame,
        n_weeks: int = 52,
        n_draws: int | None = 200,
        min_share: float = 0.01,
        max_share: float = 0.8,
        prob: float = 0.9,
    ) -> dict:
        """
        Find the budget allocation that maximises posterior-mean revenue.

        The budget is spread evenly over the planning horizon. The optimiser
        (SLSQP) works on each channel's share of the budget, bounded between
        min_share and max_share.

        Args:
            total_budget: Total media budget for the horizon
            df_baseline: Baseline dataframe; supplies the controls and the current channel mix
            n_weeks: Planning horizon in weeks (capped at the length of df_baseline)
            n_draws: Posterior draws used in the optimisation (None = all)

        Returns:
            dict with optimal_allocation, expected_revenue and its credible interval,
            roi_by_channel, the comparison with the current channel mix, and status
        """
        assert self.is_fitted
        posterior = self._thin(n_draws)
        horizon = df_baseline.iloc[:n_weeks].reset_index(drop=True)
        n_periods = len(horizon)
        _, controls = self._design(horizon)
        baseline = self._baseline_draws(controls, posterior).sum(axis=1)            # (S,)

        def media_draws(shares: np.ndarray) -> np.ndarray:
            weekly = np.tile(shares * total_budget / n_periods, (n_periods, 1))
            return self._media_draws(self._spend_lags(weekly), posterior).sum(axis=1)  # (S, C)

        def objective(shares: np.ndarray) -> float:
            return -media_draws(shares).sum(axis=1).mean() / (self._y_scale * n_periods)

        historical = np.array([float(df_baseline[ch].sum()) for ch in self.channels])
        current_mix = historical / historical.sum() if historical.sum() > 0 else np.full(len(self.channels), 1 / len(self.channels))
        start = np.clip(current_mix, min_share, max_share)
        start = start / start.sum()

        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=[(min_share, max_share)] * len(self.channels),
            constraints={"type": "eq", "fun": lambda shares: shares.sum() - 1.0},
            options={"maxiter": 500, "ftol": 1e-10},
        )
        shares = np.clip(result.x, min_share, max_share)
        shares = shares / shares.sum()
        if objective(shares) > objective(start):  # never return something worse than the starting mix
            shares = start

        budgets = shares * total_budget
        optimal_media = media_draws(shares)
        current_media = media_draws(current_mix)
        revenue = baseline + optimal_media.sum(axis=1)
        current_revenue = baseline + current_media.sum(axis=1)
        tail = (1 - prob) / 2

        return {
            "optimal_allocation": dict(zip(self.channels, budgets.tolist())),
            "expected_revenue": float(revenue.mean()),
            "expected_revenue_interval": [float(np.quantile(revenue, tail)), float(np.quantile(revenue, 1 - tail))],
            "roi_by_channel": dict(zip(self.channels, (optimal_media.mean(axis=0) / budgets).tolist())),
            "current_mix_revenue": float(current_revenue.mean()),
            "expected_uplift": float((revenue - current_revenue).mean()),
            "prob_better_than_current_mix": float((revenue > current_revenue).mean()),
            "status": str(result.message),
        }

    # ── Insight Extraction (LLM-Ready) ────────────────────────────────────────

    def generate_insight_payload(self, df: pd.DataFrame, total_budget: float) -> dict:
        """
        Generate a structured insight payload ready for LLM summarisation.

        Returns:
            Structured dict with model performance, sampler diagnostics, attribution
            and ROI with credible intervals, and the budget optimisation
        """
        contributions = self.channel_contribution(df)
        roi = self.channel_roi(df)
        optimisation = self.optimise_budget(total_budget, df)

        return {
            "model_performance": {
                "r2_score": round(float(self.train_r2_), 4),
                "mape_pct": round(float(self.train_mape_) * 100, 2),
            },
            "sampler_diagnostics": self.diagnostics_,
            "credible_interval_pct": 90,
            "channel_attribution": to_records(contributions.round(2)),
            "channel_roi": to_records(roi.round(3)),
            "budget_optimisation": {
                "total_budget": total_budget,
                "optimal_allocation": {k: round(v, 2) for k, v in optimisation["optimal_allocation"].items()},
                "expected_revenue": round(optimisation["expected_revenue"], 2),
                "expected_revenue_interval": [round(v, 2) for v in optimisation["expected_revenue_interval"]],
                "roi_by_channel": {k: round(v, 3) for k, v in optimisation["roi_by_channel"].items()},
                "expected_uplift_vs_current_mix": round(optimisation["expected_uplift"], 2),
                "prob_better_than_current_mix": round(optimisation["prob_better_than_current_mix"], 3),
            },
            "top_channel": contributions.iloc[0]["channel"],
            "lowest_roi_channel": roi.dropna(subset=["roi"]).iloc[-1]["channel"],
        }
