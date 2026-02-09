import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

from utils.data import fetch_prices, ASSET_CATEGORIES
from utils.metrics import sharpe_ratio, max_drawdown

st.set_page_config(page_title="AI Portfolio Optimizer", layout="wide")

st.title("📈 AI Portfolio Optimizer")
st.subheader("Smarter investing through AI, risk analytics & simulations")

# ---------------------------
# ASSET SELECTION
# ---------------------------
category = st.selectbox("Select Asset Category", list(ASSET_CATEGORIES.keys()))
tickers = st.multiselect(
    "Choose Assets",
    ASSET_CATEGORIES[category],
    default=ASSET_CATEGORIES[category][:3]
)

prices = fetch_prices(tickers)
returns = prices.pct_change().dropna()

# ---------------------------
# EDA SECTION
# ---------------------------
st.markdown("## 📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    fig_price = px.line(prices, title="Asset Price Trends")
    st.plotly_chart(fig_price, use_container_width=True)

with col2:
    fig_corr = px.imshow(
        returns.corr(),
        text_auto=True,
        title="Asset Correlation Matrix"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

st.info(
"""
**How to interpret this section:**
- 📈 Price Trends show long-term growth or volatility.
- 🔗 Correlation Matrix helps diversification:
  - Low correlation = better risk reduction
"""
)

# ---------------------------
# PORTFOLIO METRICS
# ---------------------------
portfolio_returns = returns.mean(axis=1)
cumulative = (1 + portfolio_returns).cumprod()

col1, col2, col3 = st.columns(3)

col1.metric("Sharpe Ratio", f"{sharpe_ratio(portfolio_returns):.2f}")
col2.metric("Max Drawdown", f"{max_drawdown(cumulative)*100:.2f}%")
col3.metric("Portfolio Growth", f"{cumulative.iloc[-1]*1000:,.0f}")

st.markdown(
"""
### 📘 What do these metrics mean?

**Sharpe Ratio**
- Measures *return per unit of risk*
- **>1 is good**, **>2 is excellent**
- Higher = better risk-adjusted performance

**Maximum Drawdown**
- Worst peak-to-trough loss
- Helps understand downside risk

**Portfolio Growth**
- Simulated value assuming equal-weight allocation
"""
)

fig = px.line(cumulative, title="📊 Portfolio Cumulative Growth")
st.plotly_chart(fig, use_container_width=True)
