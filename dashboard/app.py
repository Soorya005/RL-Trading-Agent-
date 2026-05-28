from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from evaluation.metrics import compute_returns, max_drawdown, sharpe_ratio, total_return


st.set_page_config(page_title="RL Trading Dashboard", layout="wide")

st.title("Risk-Aware RL Trading Dashboard")
st.write("This dashboard visualizes evaluation results from the PPO agent.")

curve_options = {
    "Python backtest": Path("evaluation/equity_curve.npy"),
    "Rust backtest": Path("evaluation/equity_curve_rust.npy"),
}

choice = st.radio("Equity curve source", list(curve_options.keys()), horizontal=True)
curve_path = curve_options[choice]

if not curve_path.exists():
    if choice == "Rust backtest":
        st.warning("Run evaluation first: python evaluation/evaluate_ppo_rust.py")
    else:
        st.warning("Run evaluation first: python evaluation/evaluate_ppo.py")
    st.stop()

equity_curve = np.load(curve_path)
returns = compute_returns(equity_curve)

col1, col2, col3 = st.columns(3)
col1.metric("Total Return", f"{total_return(equity_curve):.2%}")
col2.metric("Sharpe Ratio", f"{sharpe_ratio(returns):.2f}")
col3.metric("Max Drawdown", f"{max_drawdown(equity_curve):.2%}")

st.subheader("Equity Curve")
st.line_chart(equity_curve)

st.subheader("Daily Returns")
st.line_chart(returns)

st.subheader("Python vs Rust Comparison")
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

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Python Total Return", f"{total_return(py_curve):.2%}")
    col_b.metric("Rust Total Return", f"{total_return(rs_curve):.2%}")
    col_c.metric("Curve Length", str(min_len))

    st.line_chart(df_compare)
