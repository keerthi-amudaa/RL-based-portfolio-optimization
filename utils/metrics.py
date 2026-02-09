import numpy as np

def sharpe_ratio(returns, risk_free=0.02):
    excess = returns.mean() - risk_free / 252
    return np.sqrt(252) * excess / returns.std()

def max_drawdown(cumulative):
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def value_at_risk(returns, confidence=0.95):
    return np.percentile(returns, (1 - confidence) * 100)
