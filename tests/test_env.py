"""Unit tests for the trading environment."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.trading_env import EnvConfig, TradingEnv


def make_sample_data(n: int = 100) -> pd.DataFrame:
    """Create synthetic OHLCV + indicator data for testing."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "Close": close,
            "RSI_14": np.random.uniform(20, 80, n),
            "MACD": np.random.randn(n) * 0.1,
            "MACD_Signal": np.random.randn(n) * 0.1,
            "MA_10": close + np.random.randn(n) * 0.5,
            "MA_50": close + np.random.randn(n) * 1.0,
        }
    )


class TestTradingEnvReset:
    def test_reset_returns_observation(self):
        data = make_sample_data()
        env = TradingEnv(data=data)
        obs, info = env.reset()

        assert obs.shape == (8,)  # 6 features + cash + position
        assert isinstance(info, dict)

    def test_reset_clears_state(self):
        data = make_sample_data()
        env = TradingEnv(data=data)

        # Take some steps
        env.reset()
        env.step(1)
        env.step(2)

        # Reset should clear
        env.reset()
        assert env.step_index == 0
        assert env.cash == env.config.initial_cash
        assert env.position == 0.0
        assert len(env.trade_history) == 0


class TestTradingEnvStep:
    def test_step_returns_tuple(self):
        data = make_sample_data()
        env = TradingEnv(data=data)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(0)

        assert obs.shape == (8,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "net_worth" in info

    def test_hold_action_no_position_change(self):
        data = make_sample_data()
        env = TradingEnv(data=data)
        env.reset()

        initial_cash = env.cash
        env.step(0)  # hold

        assert env.position == 0.0
        assert env.cash == initial_cash

    def test_buy_action_increases_position(self):
        data = make_sample_data()
        env = TradingEnv(data=data)
        env.reset()

        env.step(1)  # buy

        assert env.position == 1.0
        assert env.cash < env.config.initial_cash

    def test_sell_without_position_does_nothing(self):
        data = make_sample_data()
        env = TradingEnv(data=data)
        env.reset()

        initial_cash = env.cash
        env.step(2)  # sell with no position

        assert env.position == 0.0
        assert env.cash == initial_cash

    def test_buy_sell_roundtrip(self):
        data = make_sample_data()
        env = TradingEnv(data=data)
        env.reset()

        env.step(1)  # buy
        assert env.position == 1.0

        env.step(2)  # sell
        assert env.position == 0.0

    def test_episode_terminates(self):
        data = make_sample_data(n=5)
        env = TradingEnv(data=data)
        env.reset()

        for i in range(4):
            _, _, terminated, _, _ = env.step(0)

        assert terminated


class TestTradingEnvTradeLog:
    def test_trade_log_records_all_steps(self):
        data = make_sample_data(n=10)
        env = TradingEnv(data=data)
        env.reset()

        for _ in range(9):
            env.step(0)

        log = env.get_trade_log()
        assert len(log) == 9
        assert "action_name" in log.columns
        assert "price" in log.columns

    def test_trade_log_empty_after_reset(self):
        data = make_sample_data()
        env = TradingEnv(data=data)
        env.reset()
        env.step(1)
        env.reset()

        log = env.get_trade_log()
        assert len(log) == 0

    def test_trade_log_action_names(self):
        data = make_sample_data(n=10)
        env = TradingEnv(data=data)
        env.reset()

        env.step(0)  # hold
        env.step(1)  # buy
        env.step(2)  # sell

        log = env.get_trade_log()
        assert log.iloc[0]["action_name"] == "hold"
        assert log.iloc[1]["action_name"] == "buy"
        assert log.iloc[2]["action_name"] == "sell"


class TestEnvConfig:
    def test_default_config(self):
        cfg = EnvConfig()
        assert cfg.initial_cash == 10000.0
        assert cfg.max_position == 1.0
        assert cfg.transaction_cost == 0.0005
        assert cfg.risk_penalty == 0.1

    def test_custom_config(self):
        cfg = EnvConfig(initial_cash=50000, max_position=5.0)
        env = TradingEnv(data=make_sample_data(), config=cfg)
        env.reset()

        assert env.cash == 50000.0
