import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def load_config(config_path: Path | None = None) -> dict:
    """Load YAML configuration file."""
    path = config_path or CONFIG_PATH
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)


def download_data(symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start_date, end=end_date, interval=interval, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    return df


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df.dropna().copy()


def add_indicators(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cfg = config or {}
    features_cfg = cfg.get("features", {})
    rsi_window = features_cfg.get("rsi_window", 14)
    ma_fast = features_cfg.get("ma_fast", 10)
    ma_slow = features_cfg.get("ma_slow", 50)

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
        df["Close"] = close

    df["RSI_14"] = RSIIndicator(close=close, window=rsi_window).rsi()
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MA_10"] = SMAIndicator(close=df["Close"], window=ma_fast).sma_indicator()
    df["MA_50"] = SMAIndicator(close=df["Close"], window=ma_slow).sma_indicator()
    return df.dropna().copy()


def normalize_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    scaler_path: Path | None = None,
) -> pd.DataFrame:
    """Z-score normalize indicator columns and optionally save scaler stats."""
    if columns is None:
        columns = ["Close", "RSI_14", "MACD", "MACD_Signal", "MA_10", "MA_50"]

    scaler_stats: dict[str, dict[str, float]] = {}
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        mean = float(df[col].mean())
        std = float(df[col].std())
        if std == 0:
            std = 1.0
        scaler_stats[col] = {"mean": mean, "std": std}
        df[col] = (df[col] - mean) / std

    if scaler_path is not None:
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, "w") as f:
            json.dump(scaler_stats, f, indent=2)

    return df


def save_csv(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download BTC data, add indicators, and save CSV.")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--interval", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--normalize", action="store_true", help="Apply z-score normalization to features.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else CONFIG_PATH
    config = load_config(config_path)
    market_cfg = config.get("market", {})

    symbol = args.symbol or market_cfg.get("symbol", "BTC-USD")
    start = args.start or market_cfg.get("start", "2018-01-01")
    end = args.end or market_cfg.get("end", "2024-12-31")
    interval = args.interval or market_cfg.get("interval", "1d")
    output = args.output or "data/processed/btc_1d_indicators.csv"

    print(f"Downloading {symbol} from {start} to {end} ({interval})...")
    df = download_data(symbol, start, end, interval)
    df = validate_and_clean(df)
    df = add_indicators(df, config)

    output_path = Path(output)
    if args.normalize:
        scaler_path = output_path.parent / "scaler_stats.json"
        df = normalize_features(df, scaler_path=scaler_path)
        print(f"Saved scaler stats to: {scaler_path}")

    save_csv(df, output_path)
    print(f"Saved: {output_path} ({len(df)} rows)")
    print(df.tail())


if __name__ == "__main__":
    main()
