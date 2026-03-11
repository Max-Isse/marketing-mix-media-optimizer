"""
Marketing Mix Media Optimizer
==============================
Bayesian Marketing Mix Model with adstock transformations and LLM-powered
insight generation for marketing spend optimisation.

Author: Portfolio Project
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")


# ── Adstock & Saturation Transforms ──────────────────────────────────────────

def adstock_transform(x: np.ndarray, decay: float) -> np.ndarray:
    """
    Apply geometric adstock transformation to capture carryover effects.

    Adstock models the lagged/accumulated effect of advertising spend:
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

    Models diminishing returns: as spend increases, incremental
    return decreases. Half-saturation point at gamma.

    Args:
        x: Adstocked spend values
        alpha: Shape parameter (steepness of the curve)
        gamma: Half-saturation point (spend at 50% max effect)

    Returns:
        Saturated spend array in [0, 1]
    """
    x_safe = np.maximum(x, 1e-10)
    return x_safe ** alpha / (x_safe ** alpha + gamma ** alpha)


# ── Marketing Mix Model ───────────────────────────────────────────────────────

class MarketingMixModel:
    """
    Bayesian-inspired Marketing Mix Model (MMM) with:
    - Adstock (carryover) transformations per media channel
    - Hill saturation (diminishing returns) per channel
    - Trend and seasonality controls
    - Budget allocation optimiser
    - LLM-ready insight extraction
    """

    def __init__(self, channels: list[str], target: str = "revenue"):
        self.channels = channels
        self.target = target
        self.params: dict = {}
        self.scaler = StandardScaler()
        self.baseline_scaler = StandardScaler()
        self.is_fitted = False

    # ── Feature Engineering ───────────────────────────────────────────────────

    def _build_features(
        self,
        df: pd.DataFrame,
        adstock_decays: dict,
        hill_alphas: dict,
        hill_gammas: dict,
    ) -> np.ndarray:
        """Apply adstock + saturation transforms to all channels."""
        features = []
        for ch in self.channels:
            spend = df[ch].values.astype(float)
            decay = adstock_decays.get(ch, 0.5)
            alpha = hill_alphas.get(ch, 1.0)
            gamma = hill_gammas.get(ch, np.mean(spend))

            transformed = hill_saturation(adstock_transform(spend, decay), alpha, gamma)
            features.append(transformed)

        # Add trend + seasonality controls if present
        if "trend" in df.columns:
            features.append(df["trend"].values.astype(float))
        if "seasonality" in df.columns:
            features.append(df["seasonality"].values.astype(float))

        return np.column_stack(features)

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        adstock_decays: dict | None = None,
        hill_alphas: dict | None = None,
        hill_gammas: dict | None = None,
    ) -> "MarketingMixModel":
        """
        Fit the MMM using optimised least squares.

        Args:
            df: DataFrame with channel spend columns + target column
            adstock_decays: Per-channel decay rates (default 0.5)
            hill_alphas: Per-channel Hill alpha params (default 1.0)
            hill_gammas: Per-channel Hill gamma (half-sat) params

        Returns:
            self (fitted model)
        """
        adstock_decays = adstock_decays or {ch: 0.5 for ch in self.channels}
        hill_alphas = hill_alphas or {ch: 1.0 for ch in self.channels}
        hill_gammas = hill_gammas or {ch: df[ch].mean() for ch in self.channels}

        self.params = {
            "adstock_decays": adstock_decays,
            "hill_alphas": hill_alphas,
            "hill_gammas": hill_gammas,
        }

        X = self._build_features(df, adstock_decays, hill_alphas, hill_gammas)
        y = df[self.target].values.astype(float)

        X_scaled = self.scaler.fit_transform(X)
        y_mean, y_std = y.mean(), y.std()
        y_norm = (y - y_mean) / y_std

        # Ordinary least squares via closed-form solution
        X_aug = np.column_stack([np.ones(len(X_scaled)), X_scaled])
        coeffs = np.linalg.lstsq(X_aug, y_norm, rcond=None)[0]

        self.intercept_ = coeffs[0] * y_std + y_mean
        self.coef_ = coeffs[1:] * y_std
        self._y_std = y_std
        self._y_mean = y_mean
        self.is_fitted = True

        # Store training scores
        y_pred = X_aug @ coeffs * y_std + y_mean
        self.train_r2_ = r2_score(y, y_pred)
        self.train_mape_ = mean_absolute_percentage_error(y, np.maximum(y_pred, 1))

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict revenue given media spend dataframe."""
        assert self.is_fitted, "Model must be fitted before predict()"
        X = self._build_features(df, **self.params)
        X_scaled = self.scaler.transform(X)
        return X_scaled @ self.coef_ + self.intercept_

    # ── Attribution ───────────────────────────────────────────────────────────

    def channel_contribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decompose predicted revenue into per-channel contributions.

        Returns:
            DataFrame with columns: channel, contribution, contribution_pct
        """
        assert self.is_fitted
        X = self._build_features(df, **self.params)
        X_scaled = self.scaler.transform(X)

        n_channels = len(self.channels)
        channel_coefs = self.coef_[:n_channels]
        channel_features = X_scaled[:, :n_channels]

        contributions = channel_features * channel_coefs  # (T, C)
        total_contrib = contributions.sum(axis=0)

        result = pd.DataFrame(
            {
                "channel": self.channels,
                "contribution": total_contrib,
                "contribution_pct": total_contrib / total_contrib.sum() * 100,
            }
        ).sort_values("contribution_pct", ascending=False)

        return result

    # ── Budget Optimiser ──────────────────────────────────────────────────────

    def optimise_budget(
        self, total_budget: float, df_baseline: pd.DataFrame, n_weeks: int = 52
    ) -> dict:
        """
        Find optimal media budget allocation to maximise predicted revenue.

        Uses scipy L-BFGS-B constrained optimisation.

        Args:
            total_budget: Total annual media budget
            df_baseline: Baseline dataframe (used to infer saturation params)
            n_weeks: Planning horizon in weeks

        Returns:
            dict with optimal_allocation, expected_revenue, roi_by_channel
        """
        assert self.is_fitted
        n = len(self.channels)

        def negative_revenue(budgets):
            """Objective: maximise revenue = minimise negative revenue."""
            sim_df = df_baseline.copy().iloc[:n_weeks].reset_index(drop=True)
            weekly = budgets / n_weeks
            for i, ch in enumerate(self.channels):
                sim_df[ch] = weekly[i]
            return -self.predict(sim_df).sum()

        # Initial allocation: proportional to historical spend
        x0 = np.array([df_baseline[ch].sum() for ch in self.channels])
        x0 = x0 / x0.sum() * total_budget

        # Constraints: sum = total_budget; each channel >= 1% of budget
        constraints = {"type": "eq", "fun": lambda x: x.sum() - total_budget}
        bounds = [(total_budget * 0.01, total_budget * 0.8) for _ in self.channels]

        result = minimize(
            negative_revenue,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )

        optimal_budgets = result.x
        optimal_revenue = -result.fun

        return {
            "optimal_allocation": dict(zip(self.channels, optimal_budgets)),
            "expected_revenue": optimal_revenue,
            "status": result.message,
            "roi_by_channel": {
                ch: optimal_revenue * (b / total_budget) / b
                for ch, b in zip(self.channels, optimal_budgets)
            },
        }

    # ── Insight Extraction (LLM-Ready) ────────────────────────────────────────

    def generate_insight_payload(
        self, df: pd.DataFrame, total_budget: float
    ) -> dict:
        """
        Generate structured insight payload ready for LLM summarisation.

        This dict can be passed directly to an LLM (e.g. Claude/GPT-4) to
        produce natural language executive summaries for non-technical
        stakeholders.

        Returns:
            Structured dict with model performance, attribution, and optimisation
        """
        contributions = self.channel_contribution(df)
        optimisation = self.optimise_budget(total_budget, df)

        return {
            "model_performance": {
                "r2_score": round(self.train_r2_, 4),
                "mape_pct": round(self.train_mape_ * 100, 2),
            },
            "channel_attribution": contributions.to_dict(orient="records"),
            "budget_optimisation": {
                "total_budget": total_budget,
                "optimal_allocation": optimisation["optimal_allocation"],
                "expected_revenue": round(optimisation["expected_revenue"], 2),
                "roi_by_channel": {
                    k: round(v, 3)
                    for k, v in optimisation["roi_by_channel"].items()
                },
            },
            "top_channel": contributions.iloc[0]["channel"],
            "lowest_roi_channel": min(
                optimisation["roi_by_channel"],
                key=optimisation["roi_by_channel"].get,
            ),
        }
