import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from utils.data import fetch_prices, ASSET_CATEGORIES

st.set_page_config(layout="wide")
st.title("⚠️ Advanced Risk Analyzer")

st.markdown(
"""
This dashboard evaluates **portfolio downside risk** using
statistical measures, simulations, and stress scenarios.
"""
)

# =====================================================
# ASSET SELECTION
# =====================================================
category = st.selectbox("Asset Category", list(ASSET_CATEGORIES.keys()))
tickers = st.multiselect(
    "Select Assets",
    ASSET_CATEGORIES[category],
    default=ASSET_CATEGORIES[category][:3]
)

prices = fetch_prices(tickers)
returns = prices.pct_change().dropna()

weights = np.ones(len(tickers)) / len(tickers)
portfolio_returns = returns @ weights

# =====================================================
# CONFIDENCE LEVEL
# =====================================================
confidence = st.slider(
    "Confidence Level for Risk Metrics",
    min_value=0.90,
    max_value=0.99,
    value=0.95,
    step=0.01
)

# =====================================================
# VaR & CVaR
# =====================================================
VaR = np.percentile(portfolio_returns, (1 - confidence) * 100)
CVaR = portfolio_returns[portfolio_returns <= VaR].mean()

col1, col2 = st.columns(2)
col1.metric("Value at Risk (VaR)", f"{VaR*100:.2f}%")
col2.metric("Conditional VaR (CVaR)", f"{CVaR*100:.2f}%")

st.markdown(
"""
### 📘 Risk Metrics Explained

**Value at Risk (VaR)**  
Worst expected daily loss at a given confidence level.

**Conditional VaR (CVaR / Expected Shortfall)**  
Average loss *when VaR is breached* — a more conservative risk measure.

📌 CVaR is preferred in **institutional risk management**.
"""
)

# =====================================================
# RETURN DISTRIBUTION
# =====================================================
st.subheader("📉 Portfolio Return Distribution")

fig_dist = px.histogram(
    portfolio_returns,
    nbins=60,
    title="Distribution of Daily Portfolio Returns"
)

fig_dist.add_vline(
    x=VaR,
    line_dash="dash",
    annotation_text="VaR",
    annotation_position="top left"
)

fig_dist.add_vline(
    x=CVaR,
    line_dash="dot",
    annotation_text="CVaR",
    annotation_position="bottom left"
)

st.plotly_chart(fig_dist, use_container_width=True)

st.info(
"""
**How to read this chart**
- Most returns cluster near zero
- Left tail = downside risk
- VaR & CVaR highlight extreme loss regions
"""
)

# =====================================================
# STRESS TESTING
# =====================================================
st.subheader("🔥 Stress Testing Scenarios")

scenario = st.selectbox(
    "Select Stress Scenario",
    ["Normal Market", "Market Crash (-20%)", "Volatility Spike (2×)", "Mild Correction (-10%)"]
)

stress_returns = portfolio_returns.copy()

if scenario == "Market Crash (-20%)":
    stress_returns -= 0.20
elif scenario == "Volatility Spike (2×)":
    stress_returns *= 2
elif scenario == "Mild Correction (-10%)":
    stress_returns -= 0.10

stress_nav = (1 + stress_returns).cumprod()

fig_stress = px.line(
    stress_nav,
    title=f"Portfolio Performance Under: {scenario}"
)
st.plotly_chart(fig_stress, use_container_width=True)

st.info(
"""
**Stress testing**
- Evaluates resilience under extreme conditions
- Helps understand tail risk beyond historical data
"""
)

# =====================================================
# MONTE CARLO SIMULATION (FAN CHART)
# =====================================================
st.subheader("🎲 Monte Carlo Risk Projection")

n_sims = st.slider("Number of Simulations", 200, 2000, 500, step=100)

mu = portfolio_returns.mean()
sigma = portfolio_returns.std()

sim_returns = np.random.normal(mu, sigma, (252, n_sims))
sim_nav = (1 + sim_returns).cumprod(axis=0)

percentiles = np.percentile(sim_nav, [5, 25, 50, 75, 95], axis=1)

fan_df = pd.DataFrame({
    "P5": percentiles[0],
    "P25": percentiles[1],
    "Median": percentiles[2],
    "P75": percentiles[3],
    "P95": percentiles[4]
})

fig_fan = px.line(
    fan_df,
    title="Monte Carlo Fan Chart (1-Year Horizon)"
)
st.plotly_chart(fig_fan, use_container_width=True)

st.info(
"""
**Monte Carlo Fan Chart**
- Shows range of possible future outcomes
- Wider band = higher uncertainty
- Median path = most likely outcome
"""
)

# =====================================================
# FINAL TAKEAWAY
# =====================================================
st.success(
"""
### ✅ Key Risk Insights

- VaR shows **threshold losses**
- CVaR captures **tail severity**
- Stress tests reveal **fragility**
- Monte Carlo shows **uncertainty envelope**

Together, these provide a **holistic view of portfolio risk**.
"""
)
