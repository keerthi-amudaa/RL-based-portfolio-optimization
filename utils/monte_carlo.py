import numpy as np

def monte_carlo_simulation(returns, weights, n_sims=500):
    mean = returns.mean().values
    cov = returns.cov().values

    sims = []
    for _ in range(n_sims):
        daily_returns = np.random.multivariate_normal(mean, cov, 252)
        portfolio = np.dot(daily_returns, weights)
        sims.append(portfolio.cumsum())

    return sims
