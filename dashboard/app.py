import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import compute_returns, max_drawdown, sharpe_ratio, total_return

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RL Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .main { padding-top: 1rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #0f3460;
        margin-bottom: 0.5rem;
    }
    h1 { color: #e94560; }
    .stMetric { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    config_path = PROJECT_ROOT / "configs" / "base.yaml"
    if config_path.exists():
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        st.subheader("Market")
        st.text(f"Symbol:   {config.get('market', {}).get('symbol', 'N/A')}")
        st.text(f"Interval: {config.get('market', {}).get('interval', 'N/A')}")
        st.text(f"Period:   {config.get('market', {}).get('start', '?')} → {config.get('market', {}).get('end', '?')}")

        st.subheader("Training")
        training = config.get("training", {})
        st.text(f"Algorithm:     {training.get('algo', 'N/A')}")
        st.text(f"Timesteps:     {training.get('timesteps', 'N/A'):,}")
        st.text(f"Learning Rate: {training.get('learning_rate', 'N/A')}")
        st.text(f"Batch Size:    {training.get('batch_size', 'N/A')}")
        st.text(f"Train Split:   {training.get('train_split', 'N/A')}")

        st.subheader("Reward")
        reward = config.get("reward", {})
        st.text(f"Risk Penalty:      {reward.get('risk_penalty', 'N/A')}")
        st.text(f"Transaction Cost:  {reward.get('transaction_cost', 'N/A')}")
    else:
        st.warning("Config file not found.")

    st.markdown("---")
    st.caption("Risk-Aware RL Trading System")

# ─── Header ──────────────────────────────────────────────────────────────────

st.title("📈 Risk-Aware RL Trading Dashboard")
st.caption("Visualization of evaluation results from the PPO agent")

# ─── Equity Curve Selection ──────────────────────────────────────────────────

curve_options = {
    "Python backtest": PROJECT_ROOT / "evaluation" / "equity_curve.npy",
    "Rust backtest": PROJECT_ROOT / "evaluation" / "equity_curve_rust.npy",
}

choice = st.radio("Equity curve source", list(curve_options.keys()), horizontal=True)
curve_path = curve_options[choice]

if not curve_path.exists():
    if choice == "Rust backtest":
        st.warning("Run evaluation first: `python evaluation/evaluate_ppo_rust.py`")
    else:
        st.warning("Run evaluation first: `python evaluation/evaluate_ppo.py`")
    st.stop()

equity_curve = np.load(curve_path)
returns = compute_returns(equity_curve)

# ─── Key Metrics ─────────────────────────────────────────────────────────────

st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Return", f"{total_return(equity_curve):.2%}")
col2.metric("Sharpe Ratio", f"{sharpe_ratio(returns):.2f}")
col3.metric("Max Drawdown", f"{max_drawdown(equity_curve):.2%}")
col4.metric("Final Net Worth", f"${equity_curve[-1]:,.2f}")

# ─── Equity Curve ────────────────────────────────────────────────────────────

st.subheader("💰 Equity Curve")
equity_df = pd.DataFrame({"Net Worth ($)": equity_curve}, index=range(len(equity_curve)))
st.line_chart(equity_df, use_container_width=True)

# ─── Drawdown Chart ──────────────────────────────────────────────────────────

st.subheader("📉 Drawdown")
peak = np.maximum.accumulate(equity_curve)
drawdown = (peak - equity_curve) / np.maximum(peak, 1e-9)
drawdown_df = pd.DataFrame({"Drawdown": -drawdown}, index=range(len(drawdown)))
st.area_chart(drawdown_df, use_container_width=True)

# ─── Daily Returns ───────────────────────────────────────────────────────────

st.subheader("📅 Daily Returns")
returns_df = pd.DataFrame({"Return": returns}, index=range(len(returns)))
st.bar_chart(returns_df, use_container_width=True)

# ─── Rolling Sharpe Ratio ────────────────────────────────────────────────────

st.subheader("📈 Rolling Sharpe Ratio (30-step window)")
if len(returns) > 30:
    rolling_mean = pd.Series(returns).rolling(30).mean()
    rolling_std = pd.Series(returns).rolling(30).std()
    rolling_sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(252)
    rolling_sharpe_df = pd.DataFrame({"Rolling Sharpe": rolling_sharpe.values})
    st.line_chart(rolling_sharpe_df, use_container_width=True)
else:
    st.info("Not enough data points for rolling Sharpe (need > 30).")

# ─── Trade Log ───────────────────────────────────────────────────────────────

trade_log_path = PROJECT_ROOT / "evaluation" / "trade_log.csv"
with st.expander("📋 Trade Log", expanded=False):
    if trade_log_path.exists():
        trade_log = pd.read_csv(trade_log_path)
        st.dataframe(trade_log, use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        buys = (trade_log["action"] == 1).sum()
        sells = (trade_log["action"] == 2).sum()
        holds = (trade_log["action"] == 0).sum()
        col_a.metric("Buys", str(buys))
        col_b.metric("Sells", str(sells))
        col_c.metric("Holds", str(holds))
    else:
        st.info("No trade log available. Run evaluation with `--save_trade_log` flag.")

# ─── Python vs Rust Comparison ───────────────────────────────────────────────

st.subheader("⚡ Python vs Rust Comparison")
show_compare = st.checkbox("Show side-by-side comparison")
if show_compare:
    py_path = curve_options["Python backtest"]
    rs_path = curve_options["Rust backtest"]

    if not py_path.exists() or not rs_path.exists():
        st.warning("Generate both curves to compare: run both evaluation scripts.")
        st.stop()

    py_curve = np.load(py_path)
    rs_curve = np.load(rs_path)
    min_len = min(len(py_curve), len(rs_curve))
    df_compare = pd.DataFrame(
        {
            "Python": py_curve[:min_len],
            "Rust": rs_curve[:min_len],
        }
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Python Total Return", f"{total_return(py_curve):.2%}")
    col_b.metric("Rust Total Return", f"{total_return(rs_curve):.2%}")
    py_returns = compute_returns(py_curve)
    rs_returns = compute_returns(rs_curve)
    col_c.metric("Python Sharpe", f"{sharpe_ratio(py_returns):.2f}")
    col_d.metric("Rust Sharpe", f"{sharpe_ratio(rs_returns):.2f}")

    st.line_chart(df_compare, use_container_width=True)

    # Difference chart
    diff = py_curve[:min_len] - rs_curve[:min_len]
    diff_df = pd.DataFrame({"Difference (Python - Rust)": diff})
    st.caption("Difference between Python and Rust equity curves")
    st.line_chart(diff_df, use_container_width=True)
