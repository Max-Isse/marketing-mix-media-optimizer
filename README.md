# 📊 Marketing Mix Media Optimizer

> Bayesian Marketing Mix Model (MMM) in PyMC with adstock and saturation estimated per channel, credible intervals on attribution and ROI, budget optimisation over the posterior, and LLM-powered executive insight generation.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

---

## 🎯 Problem Statement

Marketing teams allocate millions across TV, digital, social, OOH, and radio — but rarely know which channels truly drive revenue vs. which are wasted spend. This project builds a **Bayesian Marketing Mix Model** that:

1. **Quantifies the revenue contribution** of each media channel, with a credible interval instead of a single number
2. **Accounts for carryover effects** (adstock) and **diminishing returns** (Hill saturation)
3. **Optimises budget allocation** across channels to maximise projected revenue, and reports the probability that the new mix beats the current one
4. **Generates natural-language insights** using Claude LLM for non-technical stakeholders

---

## 🏗️ Architecture

```
marketing-mix-media-optimizer/
├── src/
│   ├── mmm_model.py       # Bayesian MMM (PyMC): adstock, saturation, ROI, optimiser
│   ├── data_generator.py  # Synthetic data pipeline (realistic marketing data)
│   ├── llm_insights.py    # Claude API integration for NL insight generation
│   └── api.py             # FastAPI REST endpoint for production serving
├── notebooks/
│   └── 01_mmm_walkthrough.ipynb   # Full end-to-end demonstration
├── tests/
│   └── test_mmm.py        # Unit tests (transforms, posterior maths, optimiser, PyMC fit)
├── data/sample/           # Sample synthetic dataset
└── requirements.txt
```

---

## 🔬 Technical Approach

### Bayesian model
The model is fitted with PyMC's NUTS sampler. Every parameter gets a posterior distribution, so every output (attribution, ROI, forecast revenue) comes with a credible interval.

```
revenue[t] ~ Normal(intercept + Σ_c beta[c] × saturation(adstock(spend[t, c])) + controls[t] · coef, sigma)
```

| Parameter | Prior | Meaning |
|---|---|---|
| `decay[c]` | Beta(3, 3) | Carryover rate of the channel |
| `alpha[c]` | Gamma(3, 3) | Shape of the saturation curve |
| `gamma[c]` | Beta(2, 2) | Half-saturation point, as a share of the channel's maximum spend |
| `beta[c]` | HalfNormal(0.5) | Channel effect; positive by construction |
| `intercept`, `coef` | Normal | Baseline revenue, trend and seasonality |
| `sigma` | HalfNormal(0.2) | Week-to-week noise |

Spend is scaled by each channel's maximum and revenue by its mean, so the same priors work for any currency or business size. The sampler's R-hat and divergence count are stored in `model.diagnostics_`.

### Adstock Transformation
Models the **lagged/carryover effect** of advertising — TV ads seen this week still influence purchases next week:

```
adstock[t] = Σ_l  w[l] × spend[t-l],   w[l] ∝ decay^l   (12-week window, weights sum to 1)
```

The decay rate is estimated per channel from the data.

### Hill Saturation Curve
Models **diminishing returns** — doubling spend doesn't double sales:

```
saturation(x) = x^α / (x^α + γ^α)
```

Where `γ` is the half-saturation point (spend at 50% max effect). Both `α` and `γ` are estimated per channel.

### Budget Optimisation
SciPy SLSQP constrained optimisation finds the allocation that maximises posterior-mean revenue subject to a total budget constraint. Because it runs on posterior draws, it also returns a credible interval for revenue and the probability that the optimised mix beats the current one.

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
pytest tests/ -v   # the 5 PyMC sampling tests are skipped if PyMC is not installed

# Start API server (samples the model on startup, which takes a few minutes)
uvicorn src.api:app --port 8000
# → API docs at http://localhost:8000/docs
```

PyMC needs a C++ compiler to run at full speed. With conda, `conda install -c conda-forge pymc` sets this up.

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
model.fit(df, draws=1000, tune=1000, chains=4)   # NUTS sampling; a few minutes

print(f"Model R²: {model.train_r2_:.3f}")
print(model.diagnostics_)                # max R-hat, divergences

# Channel attribution and ROI, each with a 90% credible interval
print(model.channel_contribution(df))
print(model.channel_roi(df))

# Optimise £17M budget
result = model.optimise_budget(17_000_000, df)
print(result["optimal_allocation"])
print(result["expected_revenue_interval"], result["prob_better_than_current_mix"])

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

# Attribution and ROI with credible intervals
curl http://localhost:8000/attribution
curl http://localhost:8000/roi
```

---

## 📈 Results (Synthetic Data)

All data is synthetic, generated by `src/data_generator.py` with known carryover and saturation parameters, so the model's estimates can be checked against the truth.

Results below come from `notebooks/01_mmm_walkthrough.ipynb`, run once on 156 weeks of synthetic data (4 chains, 1,000 tune, 1,000 draws). The notebook is committed with its outputs.

**Fit and sampler diagnostics**

| Measure | Value |
|---|---|
| Train R² / MAPE | 0.976 / 0.9% |
| Holdout R² | 0.864 |
| Max R-hat | 1.0016 |
| Divergences | 24 of 4,000 draws |

R-hat shows the chains agree, but 24 divergent transitions means the sampler struggled in part of the posterior, so the intervals should be read with some caution. The run used the default `target_accept`; a re-run with `target_accept=0.95` or a reparameterisation is the next step.

**ROI by channel (90% credible interval)**

| Channel | Spend | ROI | Interval | P(ROI > 1) |
|---|---|---|---|---|
| Radio | £2.56m | 5.71 | 1.42 to 14.43 | 0.99 |
| Social | £5.82m | 4.68 | 1.88 to 9.39 | 1.00 |
| OOH | £3.71m | 3.39 | 0.76 to 9.01 | 0.90 |
| Digital | £10.42m | 3.24 | 1.18 to 6.45 | 0.98 |
| TV | £17.81m | 1.25 | 0.48 to 2.59 | 0.56 |

The intervals are wide, and TV is the only channel whose ROI is not clearly above 1.

**Budget optimisation**: reallocating the budget raises expected annual revenue by about £2.15m against the current mix (£67.7m expected, 90% interval £67.0m to £68.5m), with a 100% posterior probability of beating the current mix.

In the notebook the LLM briefing falls back to the mock insights because no `ANTHROPIC_API_KEY` was set.

---

## 🧠 Skills Demonstrated

| Requirement | Implementation |
|------------|----------------|
| Bayesian modelling | PyMC model with informative priors, NUTS sampling, convergence diagnostics |
| ML algorithms & principles | Adstock + Hill transforms, constrained optimisation |
| Structured data | Multivariate time-series with trend/seasonality |
| Production deployment | FastAPI with Pydantic validation |
| LLM integration | Claude API for automated NL reporting |
| Communication to non-technical stakeholders | LLM-generated executive briefings |
| Testing | pytest suite covering transforms, posterior maths, the optimiser and a real PyMC fit |

---

## 📓 Notebook Walkthrough

See `notebooks/01_mmm_walkthrough.ipynb` for a full interactive walkthrough including:
- EDA on marketing data
- Bayesian model fitting, posterior summary and diagnostics
- Attribution and ROI charts with credible intervals
- Budget optimisation sensitivity analysis
- LLM insight generation demo

---

## 🔧 Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Required for LLM insight generation (optional — falls back to mock) |
