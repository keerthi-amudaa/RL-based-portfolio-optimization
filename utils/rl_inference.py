import numpy as np
from stable_baselines3 import PPO

WINDOW = 20
N_ASSETS = 4  # MUST MATCH TRAINING

def load_rl_agent(path="ppo_portfolio_agent"):
    return PPO.load(path)

def get_rl_weights(agent, returns):
    if returns.shape[1] != N_ASSETS:
        raise ValueError(
            f"Expected {N_ASSETS} assets, got {returns.shape[1]}"
        )

    window_returns = returns[-WINDOW:]
    vol = window_returns.std(axis=0)
    corr = np.corrcoef(window_returns.T)

    obs = np.concatenate([
        window_returns.flatten(),
        vol,
        corr.flatten()
    ])

    obs = obs.reshape(1, -1)  # 🔑 CRITICAL FIX

    action, _ = agent.predict(obs, deterministic=True)
    weights = action[0] / np.sum(action[0])

    return weights
