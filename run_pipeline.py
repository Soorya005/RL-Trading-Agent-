"""
run_pipeline.py — End-to-end orchestrator for the RL Trading System.

Usage:
    python run_pipeline.py                   # Full pipeline
    python run_pipeline.py --skip-data       # Skip data download (use existing CSV)
    python run_pipeline.py --skip-rust       # Skip Rust backtest
    python run_pipeline.py --timesteps 50000 # Override training timesteps
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def run_step(name: str, cmd: list[str], cwd: Path) -> bool:
    """Run a subprocess step. Returns True on success."""
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=str(cwd))
    elapsed = time.time() - start

    status = "[PASS]" if result.returncode == 0 else "[FAIL]"
    print(f"\n  {status} ({elapsed:.1f}s)")

    return result.returncode == 0


def check_rust_available() -> bool:
    """Check if rust_engine module can be imported."""
    try:
        import rust_engine
        return True
    except ImportError:
        return False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full RL trading pipeline.")
    parser.add_argument("--skip-data", action="store_true", help="Skip data download step.")
    parser.add_argument("--skip-rust", action="store_true", help="Skip Rust backtest step.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override training timesteps.")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config()
    python = sys.executable

    print("=" * 60)
    print("  Risk-Aware RL Trading Pipeline")
    print("=" * 60)

    results: dict[str, str] = {}
    total_start = time.time()

    # ── Step 1: Data Preparation ──
    data_path = PROJECT_ROOT / "data" / "processed" / "btc_1d_indicators.csv"

    if args.skip_data and data_path.exists():
        print(f"\n[SKIP] Skipping data download (file exists: {data_path})")
        results["Data Preparation"] = "[SKIP] Skipped"
    else:
        ok = run_step(
            "Data Preparation",
            [python, "utils/prepare_data.py"],
            cwd=PROJECT_ROOT,
        )
        results["Data Preparation"] = "[PASS] Pass" if ok else "[FAIL] Fail"
        if not ok:
            print("\n[FAIL] Data preparation failed. Aborting pipeline.")
            sys.exit(1)

    # ── Step 2: PPO Training ──
    train_cmd = [python, "agents/train_ppo.py"]
    if args.timesteps:
        train_cmd.extend(["--timesteps", str(args.timesteps)])

    ok = run_step("PPO Training", train_cmd, cwd=PROJECT_ROOT)
    results["PPO Training"] = "[PASS] Pass" if ok else "[FAIL] Fail"
    if not ok:
        print("\n[FAIL] Training failed. Aborting pipeline.")
        sys.exit(1)

    # ── Step 3: Python Evaluation ──
    eval_cmd = [python, "evaluation/evaluate_ppo.py", "--save_trade_log"]
    ok = run_step("Python Evaluation", eval_cmd, cwd=PROJECT_ROOT)
    results["Python Evaluation"] = "[PASS] Pass" if ok else "[FAIL] Fail"

    # ── Step 4: Rust Backtest (optional) ──
    if args.skip_rust:
        print("\n[SKIP] Skipping Rust backtest (--skip-rust)")
        results["Rust Backtest"] = "[SKIP] Skipped"
    elif check_rust_available():
        ok = run_step(
            "Rust Backtest",
            [python, "evaluation/evaluate_ppo_rust.py"],
            cwd=PROJECT_ROOT,
        )
        results["Rust Backtest"] = "[PASS] Pass" if ok else "[FAIL] Fail"
    else:
        print("\n[SKIP] Skipping Rust backtest (rust_engine not installed)")
        results["Rust Backtest"] = "[SKIP] Not installed"

    # ── Summary ──
    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print("  Pipeline Summary")
    print("=" * 60)
    for step, status in results.items():
        print(f"  {step:25s}  {status}")
    print(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    # Print evaluation metrics if available
    eq_path = PROJECT_ROOT / "evaluation" / "equity_curve.npy"
    if eq_path.exists():
        from evaluation.metrics import compute_returns, max_drawdown, sharpe_ratio, total_return

        equity = np.load(eq_path)
        returns = compute_returns(equity)
        print(f"\n  {'Metric':20s}  {'Value':>12s}")
        print(f"  {'-'*20}  {'-'*12}")
        print(f"  {'Total Return':20s}  {total_return(equity):>12.4f}")
        print(f"  {'Sharpe Ratio':20s}  {sharpe_ratio(returns):>12.4f}")
        print(f"  {'Max Drawdown':20s}  {max_drawdown(equity):>12.4f}")
        print(f"  {'Final Net Worth':20s}  ${equity[-1]:>11,.2f}")

    print("=" * 60)
    print("\n[INFO] Next: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
