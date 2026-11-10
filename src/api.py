"""
Marketing Mix Model – REST API
================================
FastAPI deployment of the Bayesian MMM with LLM insight generation.
Demonstrates: model serving, input validation, async endpoints.

The model is sampled with PyMC when the server starts, which takes a few minutes.
Sampler settings can be changed with the environment variables MMM_DRAWS,
MMM_TUNE and MMM_CHAINS.

Run locally:
    uvicorn src.api:app --reload --port 8000

API docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from typing import Optional
import json
import os

from src.mmm_model import MarketingMixModel, to_records
from src.data_generator import generate_marketing_data
from src.llm_insights import generate_llm_insights

app = FastAPI(
    title="Marketing Mix Media Optimizer API",
    description="Bayesian MMM with LLM-powered insights for marketing budget optimization",
    version="1.0.0",
)

# ── Global model (loaded at startup) ─────────────────────────────────────────
_model: Optional[MarketingMixModel] = None
_training_df: Optional[pd.DataFrame] = None


@app.on_event("startup")
async def startup_event():
    """Train model on startup using synthetic data."""
    global _model, _training_df
    print("🚀 Training MMM on startup...")
    _training_df = generate_marketing_data(n_weeks=156)
    channels = ["tv_spend", "digital_spend", "social_spend", "ooh_spend", "radio_spend"]
    _model = MarketingMixModel(channels=channels, target="revenue")
    _model.fit(
        _training_df,
        draws=int(os.environ.get("MMM_DRAWS", "500")),
        tune=int(os.environ.get("MMM_TUNE", "500")),
        chains=int(os.environ.get("MMM_CHAINS", "2")),
        progressbar=False,
    )
    print(f"✓ Model trained — R²={_model.train_r2_:.3f}, MAPE={_model.train_mape_*100:.1f}%")


# ── Request / Response Models ─────────────────────────────────────────────────

class SpendScenario(BaseModel):
    tv_spend: float = Field(..., ge=0, description="Weekly TV spend in £")
    digital_spend: float = Field(..., ge=0, description="Weekly digital spend in £")
    social_spend: float = Field(..., ge=0, description="Weekly social spend in £")
    ooh_spend: float = Field(..., ge=0, description="Weekly OOH spend in £")
    radio_spend: float = Field(..., ge=0, description="Weekly radio spend in £")
    n_weeks: int = Field(default=52, ge=1, le=260)


class OptimiseRequest(BaseModel):
    total_annual_budget: float = Field(..., ge=100_000, description="Total budget in £")
    generate_insights: bool = Field(default=True)


class PredictResponse(BaseModel):
    predicted_revenue: float
    weekly_breakdown: list[float]


class OptimiseResponse(BaseModel):
    optimal_allocation: dict[str, float]
    expected_annual_revenue: float
    expected_revenue_interval: list[float]
    roi_by_channel: dict[str, float]
    expected_uplift_vs_current_mix: float
    prob_better_than_current_mix: float
    llm_insights: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy", "model_trained": _model is not None}


@app.get("/model/metrics")
async def model_metrics():
    """Return model training performance metrics."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not yet trained")
    return {
        "r2_score": round(_model.train_r2_, 4),
        "mape_pct": round(_model.train_mape_ * 100, 2),
        "n_training_weeks": len(_training_df),
        "channels": _model.channels,
        "sampler_diagnostics": _model.diagnostics_,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(scenario: SpendScenario):
    """Predict revenue for a given media spend scenario."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not yet trained")

    sim_df = _training_df.iloc[: scenario.n_weeks].copy().reset_index(drop=True)
    channels = ["tv_spend", "digital_spend", "social_spend", "ooh_spend", "radio_spend"]
    for ch in channels:
        sim_df[ch] = getattr(scenario, ch)

    predictions = _model.predict(sim_df)
    return PredictResponse(
        predicted_revenue=float(predictions.sum()),
        weekly_breakdown=[round(float(p), 2) for p in predictions],
    )


@app.post("/optimise", response_model=OptimiseResponse)
async def optimise(request: OptimiseRequest):
    """Optimise media budget allocation to maximise predicted revenue."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not yet trained")

    result = _model.optimise_budget(request.total_annual_budget, _training_df)
    insights = None

    if request.generate_insights:
        payload = _model.generate_insight_payload(
            _training_df, request.total_annual_budget
        )
        insights = generate_llm_insights(payload)

    return OptimiseResponse(
        optimal_allocation={k: round(v, 2) for k, v in result["optimal_allocation"].items()},
        expected_annual_revenue=round(result["expected_revenue"], 2),
        expected_revenue_interval=[round(v, 2) for v in result["expected_revenue_interval"]],
        roi_by_channel={k: round(v, 3) for k, v in result["roi_by_channel"].items()},
        expected_uplift_vs_current_mix=round(result["expected_uplift"], 2),
        prob_better_than_current_mix=round(result["prob_better_than_current_mix"], 3),
        llm_insights=insights,
    )


@app.get("/attribution")
async def attribution():
    """Return channel revenue attribution with 90% credible intervals."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not yet trained")

    return to_records(_model.channel_contribution(_training_df))


@app.get("/roi")
async def roi():
    """Return ROI by channel with 90% credible intervals."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not yet trained")

    return to_records(_model.channel_roi(_training_df))
