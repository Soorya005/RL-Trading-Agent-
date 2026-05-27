# RL-Trading-Agent-

A minimal reinforcement learning trading agent implemented in pure Python.

## Features

- Single-asset trading environment with `hold`, `buy`, and `sell` actions
- Long-only single-unit position management to keep the state/action space compact
- Tabular Q-learning trader for training on price sequences
- Lightweight `unittest` coverage for the core environment and training flow

## Quick start

```python
from rl_trading_agent import train_trading_agent

prices = [100, 102, 101, 105, 108]
trader, summary = train_trading_agent(prices, episodes=100, seed=42)

print(summary)
```

## Running tests

```bash
python -m unittest discover -s tests
```
