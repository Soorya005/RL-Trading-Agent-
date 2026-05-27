from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterable

from .environment import Action, State, TradingEnvironment


@dataclass
class QLearningTrader:
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon: float = 0.1
    seed: int | None = None
    q_table: Dict[State, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._random = random.Random(self.seed)

    def select_action(self, state: State, explore: bool = True) -> Action:
        if explore and self._random.random() < self.epsilon:
            return Action(self._random.randrange(len(Action)))

        values = self._values_for(state)
        best_value = max(values)
        for index, value in enumerate(values):
            if value == best_value:
                return Action(index)

    def learn(self, state: State, action: Action, reward: float, next_state: State, done: bool) -> None:
        values = self._values_for(state)
        current_q = values[action]
        next_q = 0.0 if done else max(self._values_for(next_state))
        values[action] = current_q + self.learning_rate * (reward + self.discount_factor * next_q - current_q)

    def train(self, environment: TradingEnvironment, episodes: int = 100) -> None:
        if episodes <= 0:
            raise ValueError("episodes must be positive")

        for _ in range(episodes):
            state = environment.reset()
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = environment.step(action)
                self.learn(state, action, reward, next_state, done)
                state = next_state

    def act_greedily(self, environment: TradingEnvironment) -> dict[str, float | int]:
        state = environment.reset()
        done = False
        info: dict[str, float | int] = {
            "cash": environment.cash,
            "position": environment.position,
            "portfolio_value": environment.cash + (environment.position * environment.prices[environment.index]),
            "price": environment.prices[environment.index],
        }
        while not done:
            action = self.select_action(state, explore=False)
            state, _, done, info = environment.step(action)
        return info

    def _values_for(self, state: State) -> list[float]:
        return self.q_table.setdefault(state, [0.0] * len(Action))


def train_trading_agent(prices: Iterable[float], episodes: int = 100, seed: int | None = None) -> tuple[QLearningTrader, dict[str, float | int]]:
    environment = TradingEnvironment(list(prices))
    trader = QLearningTrader(seed=seed)
    trader.train(environment, episodes=episodes)
    final_info = trader.act_greedily(environment)
    return trader, final_info
