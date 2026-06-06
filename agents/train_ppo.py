import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.trading_env import EnvConfig, TradingEnv
from utils.data_loader import load_processed_data

CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"


def load_config(config_path: Path | None = None) -> dict:
    path = config_path or CONFIG_PATH
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)


class TrainingSummaryCallback(BaseCallback):
    """Logs episode reward statistics during training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                self.episode_rewards.append(ep_info["r"])
        return True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO on the trading environment.")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--train_split", type=float, default=None)
    parser.add_argument("--config", type=str, default=None)
    return parser


def make_env(data: pd.DataFrame, config: EnvConfig) -> TradingEnv:
    return TradingEnv(data=data, config=config)


def main() -> None:
    args = build_arg_parser().parse_args()

    config_path = Path(args.config) if args.config else CONFIG_PATH
    config = load_config(config_path)
    training_cfg = config.get("training", {})
    reward_cfg = config.get("reward", {})

    data_path = args.data or "data/processed/btc_1d_indicators.csv"
    timesteps = args.timesteps or training_cfg.get("timesteps", 200000)
    model_path = args.model or "models/ppo_btc_1d"
    train_split = args.train_split or training_cfg.get("train_split", 0.8)

    # PPO hyperparameters from config
    learning_rate = training_cfg.get("learning_rate", 0.0003)
    batch_size = training_cfg.get("batch_size", 64)
    n_epochs = training_cfg.get("n_epochs", 10)
    gamma = training_cfg.get("gamma", 0.99)
    gae_lambda = training_cfg.get("gae_lambda", 0.95)
    clip_range = training_cfg.get("clip_range", 0.2)

    print("=" * 60)
    print("PPO Training Configuration")
    print("=" * 60)
    print(f"  Data:           {data_path}")
    print(f"  Timesteps:      {timesteps:,}")
    print(f"  Train split:    {train_split}")
    print(f"  Learning rate:  {learning_rate}")
    print(f"  Batch size:     {batch_size}")
    print(f"  N epochs:       {n_epochs}")
    print(f"  Gamma:          {gamma}")
    print(f"  GAE lambda:     {gae_lambda}")
    print(f"  Clip range:     {clip_range}")
    print("=" * 60)

    df = load_processed_data(data_path)
    split_index = int(len(df) * train_split)
    train_df = df.iloc[:split_index].reset_index(drop=True)
    print(f"Training on {len(train_df)} rows (out of {len(df)} total)")

    env_config = EnvConfig(
        risk_penalty=reward_cfg.get("risk_penalty", 0.1),
        transaction_cost=reward_cfg.get("transaction_cost", 0.0005),
    )
    env = DummyVecEnv([lambda: make_env(train_df, env_config)])

    # Set up TensorBoard logging
    tensorboard_log = PROJECT_ROOT / "runs" / "ppo_training"
    tensorboard_log.parent.mkdir(parents=True, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        verbose=1,
        tensorboard_log=str(tensorboard_log),
        seed=config.get("project", {}).get("seed", 42),
    )

    callback = TrainingSummaryCallback()

    print("\nStarting training...")
    start_time = time.time()
    model.learn(total_timesteps=timesteps, callback=callback)
    elapsed = time.time() - start_time

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path.as_posix())

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"  Model saved to:  {output_path}")
    print(f"  Elapsed time:    {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Total timesteps: {timesteps:,}")
    if callback.episode_rewards:
        import numpy as np

        rewards = np.array(callback.episode_rewards)
        print(f"  Episodes:        {len(rewards)}")
        print(f"  Mean reward:     {rewards.mean():.2f}")
        print(f"  Std reward:      {rewards.std():.2f}")
        print(f"  Min reward:      {rewards.min():.2f}")
        print(f"  Max reward:      {rewards.max():.2f}")
    print(f"  TensorBoard:     tensorboard --logdir {tensorboard_log}")
    print("=" * 60)


if __name__ == "__main__":
    main()
