"""Unit tests for data preparation utilities."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.prepare_data import add_indicators, normalize_features, validate_and_clean


def make_ohlcv_data(n: int = 200) -> pd.DataFrame:
    """Create synthetic OHLCV data large enough for indicator computation."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(n) * 0.2,
            "High": close + abs(np.random.randn(n) * 0.5),
            "Low": close - abs(np.random.randn(n) * 0.5),
            "Close": close,
            "Volume": np.random.randint(1000, 10000, n).astype(float),
        }
    )


class TestValidateAndClean:
    def test_valid_data(self):
        df = make_ohlcv_data()
        result = validate_and_clean(df)
        assert len(result) == len(df)

    def test_missing_column_raises(self):
        df = make_ohlcv_data().drop(columns=["Volume"])
        with pytest.raises(ValueError, match="Missing columns"):
            validate_and_clean(df)

    def test_drops_nan(self):
        df = make_ohlcv_data()
        df.loc[5, "Close"] = np.nan
        result = validate_and_clean(df)
        assert len(result) == len(df) - 1


class TestAddIndicators:
    def test_adds_indicator_columns(self):
        df = make_ohlcv_data()
        df = validate_and_clean(df)
        result = add_indicators(df)

        for col in ["RSI_14", "MACD", "MACD_Signal", "MA_10", "MA_50"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nans_in_result(self):
        df = make_ohlcv_data()
        df = validate_and_clean(df)
        result = add_indicators(df)

        assert not result.isnull().any().any(), "Result contains NaN values"

    def test_fewer_rows_due_to_warmup(self):
        df = make_ohlcv_data()
        df = validate_and_clean(df)
        result = add_indicators(df)

        # MA_50 requires 50-step warmup, so we lose some rows
        assert len(result) < len(df)

    def test_custom_config(self):
        df = make_ohlcv_data()
        df = validate_and_clean(df)
        config = {"features": {"rsi_window": 7, "ma_fast": 5, "ma_slow": 20}}
        result = add_indicators(df, config=config)

        assert "RSI_14" in result.columns  # Column name stays same (from library)
        assert len(result) > 0


class TestNormalizeFeatures:
    def test_normalize_produces_zero_mean(self):
        df = make_ohlcv_data()
        df = validate_and_clean(df)
        df = add_indicators(df)
        result = normalize_features(df, columns=["Close", "RSI_14"])

        assert abs(result["Close"].mean()) < 1e-9
        assert abs(result["RSI_14"].mean()) < 1e-9

    def test_normalize_produces_unit_std(self):
        df = make_ohlcv_data()
        df = validate_and_clean(df)
        df = add_indicators(df)
        result = normalize_features(df, columns=["Close"])

        assert abs(result["Close"].std() - 1.0) < 0.01

    def test_scaler_file_saved(self, tmp_path):
        df = make_ohlcv_data()
        df = validate_and_clean(df)
        df = add_indicators(df)
        scaler_path = tmp_path / "scaler.json"
        normalize_features(df, columns=["Close"], scaler_path=scaler_path)

        assert scaler_path.exists()

        import json

        with open(scaler_path) as f:
            stats = json.load(f)
        assert "Close" in stats
        assert "mean" in stats["Close"]
        assert "std" in stats["Close"]
