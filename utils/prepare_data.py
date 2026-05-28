import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator


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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
        df["Close"] = close

    df["RSI_14"] = RSIIndicator(close=close, window=14).rsi()
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MA_10"] = SMAIndicator(close=df["Close"], window=10).sma_indicator()
    df["MA_50"] = SMAIndicator(close=df["Close"], window=50).sma_indicator()
    return df.dropna().copy()


def save_csv(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download BTC data, add indicators, and save CSV.")
    parser.add_argument("--symbol", type=str, default="BTC-USD")
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--output", type=str, default="data/processed/btc_1d_indicators.csv")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    df = download_data(args.symbol, args.start, args.end, args.interval)
    df = validate_and_clean(df)
    df = add_indicators(df)

    output_path = save_csv(df, Path(args.output))
    print(f"Saved: {output_path}")
    print(df.tail())


if __name__ == "__main__":
    main()
