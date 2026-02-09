import numpy as np
import pandas as pd
from utils.metrics import sharpe_ratio, max_drawdown

def backtest_static(weights, returns):
    portfolio_returns = returns @ weights
    cumulative = (1 + portfolio_returns).cumprod()

    return {
        "Sharpe": sharpe_ratio(portfolio_returns),
        "Max Drawdown": max_drawdown(cumulative),
        "Final Value": cumulative.iloc[-1]
    }

def backtest_rl(agent, returns, window=20):
    nav = 1.0
    nav_series = []
    prev_weights = None

    for t in range(window, len(returns)):
        window_returns = returns.iloc[t-window:t].values

        vol = window_returns.std(axis=0)
        corr = np.corrcoef(window_returns.T)

        obs = np.concatenate([
            window_returns.flatten(),
            vol,
            corr.flatten()
        ])

        action, _ = agent.predict(obs, deterministic=True)
        weights = action / np.sum(action)

        port_ret = np.dot(returns.iloc[t], weights)
        nav *= (1 + port_ret)
        nav_series.append(nav)

        prev_weights = weights

    nav_series = pd.Series(nav_series)
    returns_series = nav_series.pct_change().dropna()

    return {
        "Sharpe": sharpe_ratio(returns_series),
        "Max Drawdown": max_drawdown(nav_series),
        "Final Value": nav_series.iloc[-1]
    }
