from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import pandas as pd


@dataclass
class EnvConfig:
    initial_cash: float = 10000.0
    max_position: float = 1.0
    transaction_cost: float = 0.0005
    risk_penalty: float = 0.1


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, data: pd.DataFrame, config: EnvConfig | None = None):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.config = config or EnvConfig()

        self.feature_columns = [
            "Close",
            "RSI_14",
            "MACD",
            "MACD_Signal",
            "MA_10",
            "MA_50",
        ]

        self.action_space = gym.spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        obs_size = len(self.feature_columns) + 2
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self._reset_state()

    def _reset_state(self) -> None:
        self.step_index = 0
        self.cash = self.config.initial_cash
        self.position = 0.0
        self.entry_price = 0.0
        self.net_worth = self.config.initial_cash
        self.peak_worth = self.config.initial_cash

    def _get_features(self) -> np.ndarray:
        row = self.data.loc[self.step_index, self.feature_columns].astype(float)
        features = row.values
        extra = np.array([self.cash, self.position], dtype=np.float32)
        return np.concatenate([features, extra]).astype(np.float32)

    def _update_net_worth(self, price: float) -> None:
        self.net_worth = self.cash + self.position * price
        self.peak_worth = max(self.peak_worth, self.net_worth)

    def _risk_penalty(self) -> float:
        drawdown = (self.peak_worth - self.net_worth) / max(self.peak_worth, 1.0)
        return self.config.risk_penalty * drawdown

    def _apply_action(self, action: int, price: float) -> None:
        if action == 1:
            if self.position < self.config.max_position and self.cash > 0:
                fee = price * self.config.transaction_cost
                self.cash -= price + fee
                self.position += 1.0
                self.entry_price = price
        elif action == 2:
            if self.position > 0:
                fee = price * self.config.transaction_cost
                self.cash += price - fee
                self.position -= 1.0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._reset_state()
        return self._get_features(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        price = float(self.data.loc[self.step_index, "Close"])

        prev_worth = self.net_worth
        self._apply_action(action, price)
        self._update_net_worth(price)

        reward = (self.net_worth - prev_worth) - self._risk_penalty()

        self.step_index += 1
        terminated = self.step_index >= len(self.data) - 1
        truncated = False

        obs = self._get_features()
        info = {
            "net_worth": self.net_worth,
            "cash": self.cash,
            "position": self.position,
        }

        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        print(
            f"Step: {self.step_index} | Net worth: {self.net_worth:.2f} | "
            f"Cash: {self.cash:.2f} | Position: {self.position:.2f}"
        )
