from __future__ import annotations

import numpy as np


def compute_returns(equity_curve: np.ndarray) -> np.ndarray:
    equity_curve = np.asarray(equity_curve, dtype=float)
    returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-9)
    return returns


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    if returns.size == 0:
        return 0.0
    excess = returns - risk_free_rate
    std = np.std(excess)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(252))


def max_drawdown(equity_curve: np.ndarray) -> float:
    equity_curve = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / np.maximum(peak, 1e-9)
    return float(np.max(drawdown))


def total_return(equity_curve: np.ndarray) -> float:
    if len(equity_curve) < 2:
        return 0.0
    return float(equity_curve[-1] / equity_curve[0] - 1.0)
