# Quant Trading Basics (Beginner)

## Core concepts
- OHLCV: Open, High, Low, Close, Volume per time step.
- Candlesticks: visual representation of OHLCV.
- Volatility: variability of price; higher means more risk.
- Indicators: engineered features like RSI, MACD, moving averages.
- Backtesting: simulate a strategy on historical data.
- Risk management: control losses and exposure.
- Drawdown: peak-to-trough loss; measures risk.
- Sharpe ratio: risk-adjusted return metric.

## Why these matter for RL
The agent only sees data you provide. If features are noisy, the policy learns noise. Risk metrics guide reward shaping and evaluation.

## Common beginner mistakes
- Overfitting on a single asset or short date range
- Ignoring transaction costs and slippage
- Using future data by mistake (look-ahead bias)
- Assuming backtest results will generalize
