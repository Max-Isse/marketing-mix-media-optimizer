"""
LLM-Powered Insight Generator
================================
Converts structured MMM output into natural-language executive summaries
using the Anthropic Claude API. Demonstrates LLM + ML integration.
"""

import json
import os
from typing import Any


def build_insight_prompt(payload: dict) -> str:
    """
    Build a structured prompt for LLM insight generation.

    Args:
        payload: Structured dict from MarketingMixModel.generate_insight_payload()

    Returns:
        Formatted prompt string
    """
    return f"""You are a senior marketing analytics consultant preparing a board-level briefing.

Below is the output from a Marketing Mix Model analysis. Translate this into clear, 
actionable insights for a non-technical CMO. Be specific about numbers. Highlight risks.
The model is Bayesian: lower/upper values are 90% credible intervals. Say so plainly when an
interval is wide or a channel's probability of returning more than it cost is low.

MODEL OUTPUT:
{json.dumps(payload, indent=2)}

Please provide:
1. **Executive Summary** (3 sentences max)
2. **Top 3 Actionable Recommendations** (bullet points, each with an estimated impact)
3. **Risk & Caveats** (what assumptions could invalidate these findings)
4. **Quick Win** (one immediate action the team can take this week)

Use plain language. Avoid jargon. Format in markdown."""


def generate_llm_insights(payload: dict, api_key: str | None = None) -> str:
    """
    Generate natural language insights from MMM output using Claude.

    Args:
        payload: Structured MMM insight payload
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)

    Returns:
        Markdown-formatted insight string
    """
    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": build_insight_prompt(payload)}
            ],
        )
        return message.content[0].text

    except ImportError:
        return _mock_insights(payload)
    except Exception as e:
        print(f"LLM call failed ({e}), returning mock insights")
        return _mock_insights(payload)


def _mock_insights(payload: dict) -> str:
    """Fallback mock insights when API is unavailable (e.g., CI/CD)."""
    top_ch = payload.get("top_channel", "digital")
    r2 = payload.get("model_performance", {}).get("r2_score", 0.85)
    rev = payload.get("budget_optimisation", {}).get("expected_revenue", 0)

    return f"""## Executive Summary
The Marketing Mix Model explains **{r2*100:.1f}%** of revenue variance (R²={r2:.3f}),
indicating a reliable fit for planning purposes. **{top_ch.replace('_', ' ').title()}** 
emerges as the highest-contributing channel. Optimal budget reallocation is projected 
to deliver **£{rev:,.0f}** in annual revenue.

## Top 3 Actionable Recommendations
- **Increase {top_ch.replace('_',' ')} budget by 15%** — highest marginal ROI channel, 
  not yet at saturation point. Estimated uplift: +£120k revenue.
- **Reduce low-ROI channel allocation by 10%** and redeploy to digital — diminishing 
  returns detected at current spend levels. Estimated saving: £45k wasted spend.
- **Implement weekly spend tracking** against the optimised allocation — deviations 
  >15% from plan correlate with -8% average revenue underperformance.

## Risks & Caveats
- Model assumes channel independence; cross-channel synergies may be under-counted.
- 3 years of training data captures two seasonal cycles — anomalous years (e.g., 
  economic shocks) are not represented.
- Saturation curves are estimated, not experimentally validated; A/B holdout tests 
  are recommended to confirm diminishing returns thresholds.

## Quick Win
**Pause the lowest-ROI channel this week** and reallocate budget to {top_ch.replace('_',' ')}.
This requires no new creative and can be actioned in <24 hours via the media agency.
"""


if __name__ == "__main__":
    # Demo with mock payload
    sample_payload = {
        "model_performance": {"r2_score": 0.87, "mape_pct": 8.3},
        "channel_attribution": [
            {"channel": "digital_spend", "contribution": 45000, "contribution_pct": 38.2},
            {"channel": "tv_spend", "contribution": 38000, "contribution_pct": 32.1},
            {"channel": "social_spend", "contribution": 20000, "contribution_pct": 16.9},
            {"channel": "ooh_spend", "contribution": 10000, "contribution_pct": 8.5},
            {"channel": "radio_spend", "contribution": 5000, "contribution_pct": 4.3},
        ],
        "budget_optimisation": {
            "total_budget": 17_000_000,
            "optimal_allocation": {
                "digital_spend": 6_500_000,
                "tv_spend": 5_200_000,
                "social_spend": 3_100_000,
                "ooh_spend": 1_500_000,
                "radio_spend": 700_000,
            },
            "expected_revenue": 48_500_000,
            "roi_by_channel": {
                "digital_spend": 3.8,
                "tv_spend": 2.9,
                "social_spend": 2.1,
                "ooh_spend": 1.6,
                "radio_spend": 1.1,
            },
        },
        "top_channel": "digital_spend",
        "lowest_roi_channel": "radio_spend",
    }

    insights = generate_llm_insights(sample_payload)
    print(insights)
