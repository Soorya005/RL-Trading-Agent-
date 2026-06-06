"""Unit tests for evaluation metrics."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import compute_returns, max_drawdown, sharpe_ratio, total_return


class TestComputeReturns:
    def test_basic_returns(self):
        equity = np.array([100.0, 110.0, 105.0])
        returns = compute_returns(equity)

        assert len(returns) == 2
        assert abs(returns[0] - 0.1) < 1e-9  # 10/100
        assert abs(returns[1] - (-5 / 110)) < 1e-9

    def test_empty_curve(self):
        returns = compute_returns(np.array([100.0]))
        assert len(returns) == 0

    def test_constant_curve(self):
        equity = np.array([100.0, 100.0, 100.0])
        returns = compute_returns(equity)
        assert all(r == 0.0 for r in returns)


class TestSharpeRatio:
    def test_zero_returns(self):
        returns = np.array([0.0, 0.0, 0.0])
        assert sharpe_ratio(returns) == 0.0

    def test_positive_returns(self):
        returns = np.array([0.01, 0.02, 0.01, 0.03, 0.01])
        sr = sharpe_ratio(returns)
        assert sr > 0  # Positive returns should give positive Sharpe

    def test_empty_returns(self):
        assert sharpe_ratio(np.array([])) == 0.0

    def test_constant_positive_returns(self):
        returns = np.array([0.01, 0.01, 0.01])
        # Constant returns → zero std → Sharpe = 0
        assert sharpe_ratio(returns) == 0.0


class TestMaxDrawdown:
    def test_no_drawdown(self):
        equity = np.array([100.0, 110.0, 120.0, 130.0])
        assert max_drawdown(equity) == 0.0

    def test_known_drawdown(self):
        equity = np.array([100.0, 120.0, 90.0, 110.0])
        # Peak is 120, trough is 90 → drawdown = 30/120 = 0.25
        dd = max_drawdown(equity)
        assert abs(dd - 0.25) < 1e-9

    def test_full_drawdown(self):
        equity = np.array([100.0, 50.0])
        dd = max_drawdown(equity)
        assert abs(dd - 0.5) < 1e-9


class TestTotalReturn:
    def test_positive_return(self):
        equity = np.array([100.0, 150.0])
        assert abs(total_return(equity) - 0.5) < 1e-9

    def test_negative_return(self):
        equity = np.array([100.0, 80.0])
        assert abs(total_return(equity) - (-0.2)) < 1e-9

    def test_no_change(self):
        equity = np.array([100.0, 100.0])
        assert total_return(equity) == 0.0

    def test_short_curve(self):
        assert total_return(np.array([100.0])) == 0.0
        assert total_return(np.array([])) == 0.0
