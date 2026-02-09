import numpy as np
from scipy.optimize import minimize

def mean_variance_opt(returns):
    cov = returns.cov() * 252
    mean = returns.mean() * 252
    n = len(mean)

    def objective(w):
        return np.dot(w.T, np.dot(cov, w))

    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    )

    bounds = tuple((0, 1) for _ in range(n))
    w0 = np.ones(n) / n

    result = minimize(objective, w0, bounds=bounds, constraints=constraints)
    return result.x
