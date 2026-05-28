# Rust Backtesting Engine (Stub)

This is a minimal PyO3 module to validate Python-Rust integration and run a fast backtest.

## Build and install (dev)
From this folder:

```bash
maturin develop
```

## Test in Python
```python
import rust_engine
print(rust_engine.hello_engine())
```

## Rust backtest API
The module exposes `backtest(prices, actions, initial_cash, transaction_cost)` and returns a dict
with `equity_curve`, `total_return`, `sharpe_ratio`, and `max_drawdown`.
