# Project Overview

This project builds a research-grade, risk-aware RL trading simulator. It is not a profit-seeking bot. The focus is on experimentation, evaluation, and engineering quality.

## System goals
- Modular architecture with clear boundaries
- Reproducible training runs
- Risk-aware reward functions and metrics
- Rust engine for fast backtests

## High-level architecture
- Data layer: download, clean, and store OHLCV data
- Feature layer: indicators and normalized features
- Environment layer: Gymnasium trading environment
- Agent layer: PPO training and inference
- Evaluation layer: PnL, Sharpe, drawdown, trade stats
- Rust engine: fast backtests and metrics
- Dashboard: visual reporting and diagnostics

## Reality check
- Real quant systems use robust market data, transaction cost models, slippage models, and strict risk controls.
- This project is a learning and research simulator, not a live trading system.
