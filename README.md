# Risk-Aware Reinforcement Learning Trading System

This is a research-oriented, modular RL trading project that pairs Python-based training with a high-performance Rust backtesting engine.

## Why this project exists
- Learn quant trading basics, RL fundamentals, and Rust systems design.
- Build an end-to-end trading simulator, not a promise of real profits.
- Emphasize experiment design, evaluation, and reproducibility.

## Project structure
- data/: Raw and processed datasets
- notebooks/: Learning notebooks and phase demos
- env/: Gymnasium trading environment
- agents/: RL agent training and inference code
- rust_engine/: Rust backtesting engine
- dashboard/: Visualization and reporting
- evaluation/: Metrics and evaluation utilities
- models/: Saved checkpoints
- configs/: Config files (paths, hyperparameters)
- utils/: Shared helpers
- docs/: Learning notes and design docs

## MVP phases
1. Data download, cleaning, indicators, visualization
2. Custom Gymnasium trading environment
3. PPO training, saving, evaluation
4. Rust backtesting engine
5. Python-Rust integration
6. Dashboard for results and diagnostics
7. Advanced improvements (optional)

## What makes this portfolio-ready
- Clear architecture and documentation
- Reproducible experiments with configs
- Risk-aware reward engineering
- Rust acceleration and Python integration

## Next step
Start with Phase 1 in docs/ and build the first dataset.
