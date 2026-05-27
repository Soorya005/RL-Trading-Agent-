from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence


class Action(IntEnum):
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass(frozen=True)
class State:
    trend: int
    holding: int


class TradingEnvironment:
    """Single-asset trading environment with discrete actions and rewards."""

    def __init__(self, prices: Sequence[float], starting_cash: float = 1_000.0) -> None:
        if len(prices) < 2:
            raise ValueError("prices must contain at least two observations")
        if starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        if any(price <= 0 for price in prices):
            raise ValueError("prices must be positive values")

        self.prices = [float(price) for price in prices]
        self.starting_cash = float(starting_cash)
        self.reset()

    def reset(self) -> State:
        self.index = 0
        self.cash = self.starting_cash
        self.position = 0
        return self._state()

    def step(self, action: int) -> tuple[State, float, bool, dict[str, float | int]]:
        action = Action(action)
        current_price = self.prices[self.index]
        portfolio_before = self.cash + (self.position * current_price)

        if action is Action.BUY and self.position == 0 and self.cash >= current_price:
            self.position = 1
            self.cash -= current_price
        elif action is Action.SELL and self.position == 1:
            self.position = 0
            self.cash += current_price

        self.index += 1
        next_price = self.prices[self.index]
        portfolio_after = self.cash + (self.position * next_price)
        reward = portfolio_after - portfolio_before
        done = self.index == len(self.prices) - 1

        info = {
            "cash": self.cash,
            "position": self.position,
            "portfolio_value": portfolio_after,
            "price": next_price,
        }
        return self._state(), reward, done, info

    def _state(self) -> State:
        if self.index == 0:
            trend = 0
        else:
            price_change = self.prices[self.index] - self.prices[self.index - 1]
            trend = 1 if price_change > 0 else -1 if price_change < 0 else 0
        return State(trend=trend, holding=self.position)
