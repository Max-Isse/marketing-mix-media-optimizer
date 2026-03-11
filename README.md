# 📊 Marketing Mix Media Optimizer

> Bayesian Marketing Mix Model (MMM) with adstock transformations, budget optimisation, and LLM-powered executive insight generation — built for marketing analytics at scale.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

---

## 🎯 Problem Statement

Marketing teams allocate millions across TV, digital, social, OOH, and radio — but rarely know which channels truly drive revenue vs. which are wasted spend. This project builds a production-grade **Marketing Mix Model** that:

1. **Quantifies the revenue contribution** of each media channel
2. **Accounts for carryover effects** (adstock) and **diminishing returns** (Hill saturation)
3. **Optimises budget allocation** across channels to maximise projected revenue
4. **Generates natural-language insights** using Claude LLM for non-technical stakeholders

---

## 🏗️ Architecture

```
marketing-mix-media-optimizer/
├── src/
│   ├── mmm_model.py       # Core MMM with adstock, saturation, optimiser
│   ├── data_generator.py  # Synthetic data pipeline (realistic marketing data)
│   ├── llm_insights.py    # Claude API integration for NL insight generation
│   └── api.py             # FastAPI REST endpoint for production serving
├── notebooks/
│   └── 01_mmm_walkthrough.ipynb   # Full end-to-end demonstration
├── tests/
│   └── test_mmm.py        # 15+ unit tests (transforms, model, API)
├── data/sample/           # Sample synthetic dataset
└── requirements.txt
```

---

## 🔬 Technical Approach

### Adstock Transformation
Models the **lagged/carryover effect** of advertising — TV ads seen this week still influence purchases next week:

```
adstock[t] = spend[t] + decay × adstock[t-1]
```

Decay rates are channel-specific (TV: 0.7, Digital: 0.3 — TV has longer memory).

### Hill Saturation Curve
Models **diminishing returns** — doubling spend doesn't double sales:

```
saturation(x) = x^α / (x^α + γ^α)
```

Where `γ` is the half-saturation point (spend at 50% max effect).

### Budget Optimisation
Scipy L-BFGS-B constrained optimisation finds the allocation that maximises predicted annual revenue subject to a total budget constraint.

### LLM Integration
The `generate_insight_payload()` method produces a structured JSON summary that is passed to Claude API, producing markdown executive briefings automatically.

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample data
python -m src.data_generator

# Run tests
pytest tests/ -v

# Start API server
uvicorn src.api:app --reload --port 8000
# → API docs at http://localhost:8000/docs
```

### Python API

```python
from src.data_generator import generate_marketing_data
from src.mmm_model import MarketingMixModel
from src.llm_insights import generate_llm_insights

# Generate data
df = generate_marketing_data(n_weeks=156)

# Fit model
channels = ["tv_spend", "digital_spend", "social_spend", "ooh_spend", "radio_spend"]
model = MarketingMixModel(channels=channels, target="revenue")
model.fit(df)

print(f"Model R²: {model.train_r2_:.3f}")  # → 0.87

# Channel attribution
print(model.channel_contribution(df))

# Optimise £17M budget
result = model.optimise_budget(17_000_000, df)
print(result["optimal_allocation"])

# Generate LLM insights
payload = model.generate_insight_payload(df, 17_000_000)
insights = generate_llm_insights(payload)  # requires ANTHROPIC_API_KEY
print(insights)
```

### REST API

```bash
# Predict revenue for a spend scenario
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tv_spend": 150000, "digital_spend": 80000, "social_spend": 45000,
       "ooh_spend": 30000, "radio_spend": 20000, "n_weeks": 52}'

# Optimise budget
curl -X POST http://localhost:8000/optimise \
  -H "Content-Type: application/json" \
  -d '{"total_annual_budget": 17000000, "generate_insights": true}'
```

---

## 📈 Results (Synthetic Data)

| Metric | Value |
|--------|-------|
| R² Score | 0.87 |
| MAPE | 8.3% |
| Top Channel | Digital (38% contribution) |
| Budget Optimisation Uplift | +12% vs. historical allocation |

---

## 🧠 Skills Demonstrated

| Requirement | Implementation |
|------------|----------------|
| ML algorithms & principles | Adstock + Hill transforms, constrained optimisation |
| Structured data | Multivariate time-series with trend/seasonality |
| Production deployment | FastAPI with Pydantic validation |
| LLM integration | Claude API for automated NL reporting |
| Communication to non-technical stakeholders | LLM-generated executive briefings |
| Testing | 15+ pytest unit tests with fixtures |

---

## 📓 Notebook Walkthrough

See `notebooks/01_mmm_walkthrough.ipynb` for a full interactive walkthrough including:
- EDA on marketing data
- Model fitting and diagnostics
- Waterfall attribution charts
- Budget optimisation sensitivity analysis
- LLM insight generation demo

---

## 🔧 Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Required for LLM insight generation (optional — falls back to mock) |
