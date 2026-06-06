import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
    parser.add_argument("--save_trade_log", action="store_true", help="Save trade log CSV.")
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

    print("=" * 50)
    print("Evaluation Metrics (Python Backtest)")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"  {key:20s}: {value:.4f}")
    print(f"  {'test_steps':20s}: {len(test_df)}")
    print(f"  {'final_net_worth':20s}: {equity_curve[-1]:.2f}")
    print("=" * 50)

    output_dir = Path("evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "equity_curve.npy", equity_curve)
    print(f"Saved equity curve to {output_dir / 'equity_curve.npy'}")

    # Save trade log if requested
    if args.save_trade_log:
        trade_log = env.get_trade_log()
        if not trade_log.empty:
            log_path = output_dir / "trade_log.csv"
            trade_log.to_csv(log_path, index=False)
            print(f"Saved trade log to {log_path} ({len(trade_log)} entries)")

            # Print trade summary
            buys = (trade_log["action"] == 1).sum()
            sells = (trade_log["action"] == 2).sum()
            holds = (trade_log["action"] == 0).sum()
            print(f"\nTrade Summary: {buys} buys, {sells} sells, {holds} holds")


if __name__ == "__main__":
    main()
