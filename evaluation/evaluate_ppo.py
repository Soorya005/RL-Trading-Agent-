import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.trading_env import EnvConfig, TradingEnv
from evaluation.metrics import compute_returns, max_drawdown, sharpe_ratio, total_return
from utils.data_loader import load_processed_data


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a PPO model on the test split.")
    parser.add_argument("--data", type=str, default="data/processed/btc_1d_indicators.csv")
    parser.add_argument("--model", type=str, default="models/ppo_btc_1d.zip")
    parser.add_argument("--train_split", type=float, default=0.8)
    return parser


def run_episode(env: TradingEnv, model: PPO) -> np.ndarray:
    obs, _ = env.reset()
    done = False
    equity = [env.net_worth]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
        equity.append(env.net_worth)

    return np.array(equity, dtype=float)


def main() -> None:
    args = build_arg_parser().parse_args()

    df = load_processed_data(args.data)
    split_index = int(len(df) * args.train_split)
    test_df = df.iloc[split_index:].reset_index(drop=True)

    config = EnvConfig()
    env = TradingEnv(data=test_df, config=config)

    model = PPO.load(args.model)
    equity_curve = run_episode(env, model)
    returns = compute_returns(equity_curve)

    metrics = {
        "total_return": total_return(equity_curve),
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity_curve),
    }

    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value:.4f}")

    output_dir = Path("evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "equity_curve.npy", equity_curve)
    print("Saved equity curve to evaluation/equity_curve.npy")


if __name__ == "__main__":
    main()
