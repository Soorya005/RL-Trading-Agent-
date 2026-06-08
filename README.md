# Risk-Aware Reinforcement Learning Trading System
Only meant for research purposes not yet production grade 

A research-oriented, modular RL trading simulator that pairs Python-based PPO training with a high-performance Rust backtesting engine.

## Why this project exists
- Learn quant trading basics, RL fundamentals, and Rust systems design.
- Build an end-to-end trading simulator, not a promise of real profits.
- Emphasize experiment design, evaluation, and reproducibility.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  run_pipeline.py                     │
│            (end-to-end orchestrator)                 │
├──────────┬──────────┬───────────┬───────────────────┤
│  Data    │  Env     │  Agent    │  Evaluation        │
│  Layer   │  Layer   │  Layer    │  Layer             │
├──────────┼──────────┼───────────┼───────────────────┤
│ yfinance │ Gym env  │ PPO       │ Python metrics     │
│ ta       │ reward   │ SB3       │ Rust backtest      │
│ pandas   │ trading  │ training  │ dashboard          │
└──────────┴──────────┴───────────┴───────────────────┘
```

## Project structure
```
trading-agent/
├── agents/          # RL agent training (PPO)
├── configs/         # YAML configuration files
├── dashboard/       # Streamlit visualization
├── data/            # Raw and processed datasets
├── docs/            # Learning notes and design docs
├── env/             # Gymnasium trading environment
├── evaluation/      # Metrics and evaluation scripts
├── models/          # Saved model checkpoints
├── notebooks/       # Learning notebooks
├── rust_engine/     # Rust backtesting engine (PyO3)
├── tests/           # Unit tests
├── utils/           # Shared helpers
└── run_pipeline.py  # One-command pipeline runner
```

## Quickstart

### 1. Install dependencies
```bash
cd trading-agent
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python run_pipeline.py
```
This downloads BTC data, trains PPO, evaluates on the test split, and prints metrics.

### 3. Run with options
```bash
# Skip data download (reuse existing CSV)
python run_pipeline.py --skip-data

# Override training timesteps
python run_pipeline.py --timesteps 50000

# Skip Rust backtest
python run_pipeline.py --skip-rust
```

### 4. Run individual steps
```bash
# Step 1: Download and prepare data
python utils/prepare_data.py

# Step 2: Train PPO
python agents/train_ppo.py

# Step 3: Evaluate
python evaluation/evaluate_ppo.py --save_trade_log

# Step 4: Rust backtest (requires maturin)
cd rust_engine && maturin develop && cd ..
python evaluation/evaluate_ppo_rust.py
```

### 5. Launch dashboard
```bash
streamlit run dashboard/app.py
```

### 6. Run tests
```bash
python -m pytest tests/ -v
```

## Configuration

All settings live in `configs/base.yaml`:

| Section    | Key               | Description                   |
|------------|-------------------|-------------------------------|
| `market`   | `symbol`          | Trading pair (e.g. BTC-USD)   |
| `market`   | `start` / `end`   | Date range for data           |
| `training` | `timesteps`       | Total PPO training steps      |
| `training` | `learning_rate`   | PPO learning rate             |
| `training` | `batch_size`      | Minibatch size                |
| `reward`   | `risk_penalty`    | Drawdown penalty weight       |
| `reward`   | `transaction_cost`| Fee per trade                 |

## MVP Phases
1. ✅ Data download, cleaning, indicators, visualization
2. ✅ Custom Gymnasium trading environment
3. ✅ PPO training, saving, evaluation
4. ✅ Rust backtesting engine
5. ✅ Python-Rust integration
6. ✅ Dashboard for results and diagnostics
7. 🔄 Advanced improvements (ongoing)

## What makes this portfolio-ready
- Clear architecture and documentation
- Reproducible experiments with YAML configs
- Risk-aware reward engineering with drawdown penalties
- Rust acceleration and Python integration via PyO3
- Comprehensive unit tests
- TensorBoard training logging
- End-to-end pipeline orchestrator

## Reality check
This project is a **learning and research simulator**, not a live trading system.
Real quant systems use robust market data feeds, sophisticated transaction cost models,
slippage modeling, and strict risk controls.
