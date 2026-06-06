"""Smoke test for the Rust backtesting engine."""

import sys


def main() -> None:
    try:
        import rust_engine
    except ImportError:
        print("rust_engine not available. Build it first:")
        print("  cd rust_engine && maturin develop")
        sys.exit(1)

    # Test 1: hello_engine
    msg = rust_engine.hello_engine()
    assert msg == "rust engine is ready", f"Unexpected message: {msg}"
    print(f"[PASS] hello_engine: {msg}")

    # Test 2: backtest with synthetic data
    prices = [100.0, 105.0, 103.0, 110.0, 108.0]
    actions = [1, 0, 0, 2, 0]  # buy, hold, hold, sell, hold
    initial_cash = 10000.0
    transaction_cost = 0.001

    result = rust_engine.backtest(prices, actions, initial_cash, transaction_cost)

    assert "equity_curve" in result, "Missing equity_curve in result"
    assert "total_return" in result, "Missing total_return in result"
    assert "sharpe_ratio" in result, "Missing sharpe_ratio in result"
    assert "max_drawdown" in result, "Missing max_drawdown in result"

    assert len(result["equity_curve"]) == len(prices), (
        f"Equity curve length mismatch: {len(result['equity_curve'])} vs {len(prices)}"
    )

    print(f"[PASS] backtest: equity_curve length = {len(result['equity_curve'])}")
    print(f"       total_return  = {result['total_return']:.6f}")
    print(f"       sharpe_ratio  = {result['sharpe_ratio']:.6f}")
    print(f"       max_drawdown  = {result['max_drawdown']:.6f}")

    # Test 3: verify total_return is reasonable
    # Bought at 100 + fee, sold at 110 - fee => ~10% return on 10000 initial
    assert result["total_return"] > 0, "Expected positive return for buy-low-sell-high"
    print(f"[PASS] total_return is positive: {result['total_return']:.6f}")

    # Test 4: error handling – mismatched lengths
    try:
        rust_engine.backtest([100.0, 200.0], [1], 10000.0, 0.001)
        print("[FAIL] Should have raised ValueError for mismatched lengths")
        sys.exit(1)
    except ValueError:
        print("[PASS] ValueError raised for mismatched prices/actions")

    print("\nAll Rust engine smoke tests passed!")


if __name__ == "__main__":
    main()
