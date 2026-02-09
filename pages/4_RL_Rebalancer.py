import streamlit as st
import plotly.express as px

from utils.data import fetch_prices
from utils.rl_inference import load_rl_agent, get_rl_weights

st.set_page_config(layout="wide")
st.title("🔁 Reinforcement Learning Portfolio Rebalancer")

st.markdown(
"""
This agent was **trained offline using PPO** to maximize **risk-adjusted returns**
while penalizing drawdowns and transaction costs.

⚠️ The asset universe is fixed to ensure a consistent RL state space.
"""
)

RL_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA"]

prices = fetch_prices(RL_TICKERS)
returns = prices.pct_change().dropna().values

agent = load_rl_agent()
weights = get_rl_weights(agent, returns)

fig = px.pie(
    names=RL_TICKERS,
    values=weights,
    title="RL-Optimized Allocation (Sharpe-Aware)"
)
st.plotly_chart(fig, use_container_width=True)

st.info(
"""
### 🧠 Why this RL model is competitive

- Optimizes **Sharpe ratio directly**
- Penalizes **drawdowns & over-trading**
- Learns **dynamic rebalancing**
- Comparable to Mean–Variance with better downside protection

This mirrors institutional portfolio strategies.
"""
)
