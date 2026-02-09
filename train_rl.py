import yfinance as yf
import numpy as np
from stable_baselines3 import PPO
from rl.env_portfolio import PortfolioEnv

# -----------------------------
# FIXED ASSET UNIVERSE
# -----------------------------
TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA"]
WINDOW = 20

prices = yf.download(TICKERS, period="5y")["Close"]
returns = prices.pct_change().dropna().values

# -----------------------------
# ENVIRONMENT
# -----------------------------
env = PortfolioEnv(
    returns=returns,
    window=WINDOW,
    lambda_dd=0.05,
    lambda_tc=0.002
)

# -----------------------------
# PPO MODEL (STABLE & STRONG)
# -----------------------------
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    ent_coef=0.01,
    verbose=1
)

model.learn(total_timesteps=300_000)
model.save("ppo_portfolio_agent")

print("✅ PPO agent trained and saved.")
