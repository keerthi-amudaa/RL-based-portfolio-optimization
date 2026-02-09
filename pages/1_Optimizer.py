import streamlit as st
import plotly.express as px
import pandas as pd

from utils.data import fetch_prices, ASSET_CATEGORIES
from utils.optimizer import mean_variance_opt
from utils.metrics import sharpe_ratio, max_drawdown

st.title("⚙️ Mean–Variance Portfolio Optimizer")

st.markdown(
"""
This optimizer uses **Modern Portfolio Theory (MPT)** to compute
**optimal asset weights** that minimize portfolio risk.

📌 This method won the **Nobel Prize in Economics**.
"""
)

# -----------------------------
# ASSET SELECTION
# -----------------------------
category = st.selectbox("Asset Category", list(ASSET_CATEGORIES.keys()))
tickers = st.multiselect(
    "Select Assets",
    ASSET_CATEGORIES[category],
    default=ASSET_CATEGORIES[category][:3]
)

prices = fetch_prices(tickers)
returns = prices.pct_change().dropna()

# -----------------------------
# OPTIMIZATION
# -----------------------------
weights = mean_variance_opt(returns)

weights_df = pd.DataFrame({
    "Asset": tickers,
    "Weight": weights
})

st.markdown("## 📊 Optimized Portfolio Allocation")
st.dataframe(weights_df, use_container_width=True)

fig = px.pie(
    names=tickers,
    values=weights,
    title="Mean–Variance Optimized Weights"
)
st.plotly_chart(fig)

# -----------------------------
# EXPLANATION
# -----------------------------
st.markdown(
"""
## 📘 How are these weights calculated?

The optimizer solves the following problem:

**Objective**
> Minimize total portfolio variance

**Mathematically**
> wᵀ Σ w  
where Σ is the covariance matrix of asset returns.

**Constraints**
- All weights ≥ 0 (no short selling)
- Sum of weights = 1 (fully invested)

### 🧠 Intuition
- Assets with **high volatility** get lower weights
- Assets with **low correlation** are preferred
- Diversification reduces overall risk

⚠️ This method assumes:
- Returns are stationary
- Risk is fully captured by variance
- Allocation is static (no rebalancing logic)
"""
)

# -----------------------------
# PERFORMANCE METRICS
# -----------------------------
portfolio_returns = returns @ weights
cumulative = (1 + portfolio_returns).cumprod()

col1, col2, col3 = st.columns(3)
col1.metric("Sharpe Ratio", f"{sharpe_ratio(portfolio_returns):.2f}")
col2.metric("Max Drawdown", f"{max_drawdown(cumulative)*100:.2f}%")
col3.metric("Final Value", f"{cumulative.iloc[-1]*1000:,.0f}")

st.info(
"""
### 📌 Key takeaway
Mean–Variance Optimization produces a **stable, low-risk baseline**.
It does **not adapt** to market regime changes — this is where RL helps.
"""
)
