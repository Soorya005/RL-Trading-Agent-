import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rust_engine
from env.trading_env import TradingEnv
from utils.data_loader import load_processed_data


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

    print("Rust backtest metrics:")
    print(f"- total_return: {result['total_return']:.4f}")
    print(f"- sharpe_ratio: {result['sharpe_ratio']:.4f}")
    print(f"- max_drawdown: {result['max_drawdown']:.4f}")

    equity_curve = np.array(result["equity_curve"], dtype=float)
    np.save("evaluation/equity_curve_rust.npy", equity_curve)
    print("Saved equity curve to evaluation/equity_curve_rust.npy")


if __name__ == "__main__":
    main()
