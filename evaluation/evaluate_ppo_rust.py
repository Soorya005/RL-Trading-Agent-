import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.trading_env import TradingEnv
from utils.data_loader import load_processed_data

# Graceful import of the Rust engine
try:
    import rust_engine

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PPO using the Rust backtester.")
    parser.add_argument("--data", type=str, default="data/processed/btc_1d_indicators.csv")
    parser.add_argument("--model", type=str, default="models/ppo_btc_1d.zip")
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--initial_cash", type=float, default=10000.0)
    parser.add_argument("--transaction_cost", type=float, default=0.0005)
    return parser


def collect_actions(env: TradingEnv, model: PPO) -> np.ndarray:
    obs, _ = env.reset()
    done = False
    actions = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
        actions.append(int(action))

    return np.array(actions, dtype=np.int64)


def main() -> None:
    if not RUST_AVAILABLE:
        print("=" * 60)
        print("ERROR: rust_engine module not found.")
        print("")
        print("To build and install the Rust backtester:")
        print("  1. Install Rust: https://rustup.rs/")
        print("  2. Install maturin: pip install maturin")
        print("  3. Build the module:")
        print("     cd rust_engine && maturin develop && cd ..")
        print("")
        print("Then re-run this script.")
        print("=" * 60)
        sys.exit(1)

    args = build_arg_parser().parse_args()

    df = load_processed_data(args.data)
    split_index = int(len(df) * args.train_split)
    test_df = df.iloc[split_index:].reset_index(drop=True)

    env = TradingEnv(data=test_df)
    model = PPO.load(args.model)

    actions = collect_actions(env, model)
    prices = test_df["Close"].to_numpy(dtype=float)

    result = rust_engine.backtest(
        prices.tolist(),
        actions.tolist(),
        args.initial_cash,
        args.transaction_cost,
    )

    print("=" * 50)
    print("Evaluation Metrics (Rust Backtest)")
    print("=" * 50)
    print(f"  {'total_return':20s}: {result['total_return']:.4f}")
    print(f"  {'sharpe_ratio':20s}: {result['sharpe_ratio']:.4f}")
    print(f"  {'max_drawdown':20s}: {result['max_drawdown']:.4f}")
    print("=" * 50)

    equity_curve = np.array(result["equity_curve"], dtype=float)
    output_dir = Path("evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "equity_curve_rust.npy", equity_curve)
    print(f"Saved equity curve to {output_dir / 'equity_curve_rust.npy'}")

    # Compare with Python backtest if available
    py_curve_path = output_dir / "equity_curve.npy"
    if py_curve_path.exists():
        py_curve = np.load(py_curve_path)
        from evaluation.metrics import compute_returns, max_drawdown as py_mdd, sharpe_ratio as py_sr, total_return as py_tr

        print("\n" + "-" * 50)
        print("Python vs Rust Comparison")
        print("-" * 50)
        print(f"  {'Metric':20s} {'Python':>12s} {'Rust':>12s}")
        print(f"  {'-'*20} {'-'*12} {'-'*12}")
        print(f"  {'Total Return':20s} {py_tr(py_curve):>12.4f} {result['total_return']:>12.4f}")
        py_returns = compute_returns(py_curve)
        print(f"  {'Sharpe Ratio':20s} {py_sr(py_returns):>12.4f} {result['sharpe_ratio']:>12.4f}")
        print(f"  {'Max Drawdown':20s} {py_mdd(py_curve):>12.4f} {result['max_drawdown']:>12.4f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
