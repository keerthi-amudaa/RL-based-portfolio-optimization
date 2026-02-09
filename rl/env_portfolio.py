import gymnasium as gym
import numpy as np

class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        returns,
        window=20,
        lambda_dd=0.05,
        lambda_tc=0.002
    ):
        super().__init__()

        self.returns = returns
        self.window = window
        self.lambda_dd = lambda_dd
        self.lambda_tc = lambda_tc

        self.n_assets = returns.shape[1]

        # Action: portfolio weights
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        # Observation size (FIXED)
        self.obs_dim = (
            self.n_assets * window     # rolling returns
            + self.n_assets             # volatility
            + self.n_assets ** 2        # correlation
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

    def reset(self, seed=None):
        self.t = self.window
        self.nav = 1.0
        self.max_nav = 1.0
        self.prev_weights = np.ones(self.n_assets) / self.n_assets
        return self._get_obs(), {}

    def _get_obs(self):
        window_returns = self.returns[self.t - self.window : self.t]

        vol = window_returns.std(axis=0)
        corr = np.corrcoef(window_returns.T)

        obs = np.concatenate([
            window_returns.flatten(),
            vol,
            corr.flatten()
        ])

        return obs.astype(np.float32)

    def step(self, action):
        action = np.clip(action, 0, 1)
        weights = action / (np.sum(action) + 1e-8)
    
        port_ret = np.dot(self.returns[self.t], weights)
        self.nav *= (1 + port_ret)
        self.max_nav = max(self.max_nav, self.nav)
    
        drawdown = (self.max_nav - self.nav) / self.max_nav
        turnover = np.sum((weights - self.prev_weights) ** 2)
    
        # ✅ portfolio sharpe
        recent = self.returns[self.t - self.window : self.t]
        port_recent = recent @ weights
    
        mean_ret = port_recent.mean()
        std_ret = port_recent.std() + 1e-8
        sharpe_t = mean_ret / std_ret
    
        reward = (
            sharpe_t
            - self.lambda_dd * drawdown
            - self.lambda_tc * turnover
        )
    
        self.prev_weights = weights
        self.t += 1
        done = self.t >= len(self.returns) - 1
    
        return self._get_obs(), reward, done, False, {}

