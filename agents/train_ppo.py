import argparse
import sys
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.trading_env import EnvConfig, TradingEnv
from utils.data_loader import load_processed_data


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO on the trading environment.")
    parser.add_argument("--data", type=str, default="data/processed/btc_1d_indicators.csv")
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--model", type=str, default="models/ppo_btc_1d")
    parser.add_argument("--train_split", type=float, default=0.8)
    return parser


def make_env(data: pd.DataFrame, config: EnvConfig) -> TradingEnv:
    return TradingEnv(data=data, config=config)


def main() -> None:
    args = build_arg_parser().parse_args()

    df = load_processed_data(args.data)
    split_index = int(len(df) * args.train_split)
    train_df = df.iloc[:split_index].reset_index(drop=True)

    config = EnvConfig()
    env = DummyVecEnv([lambda: make_env(train_df, config)])

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps)

    output_path = Path(args.model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path.as_posix())

    print(f"Saved model to: {output_path}")


if __name__ == "__main__":
    main()
