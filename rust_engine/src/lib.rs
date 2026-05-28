use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
fn hello_engine() -> PyResult<String> {
    Ok("rust engine is ready".to_string())
}

fn compute_returns(equity_curve: &[f64]) -> Vec<f64> {
    if equity_curve.len() < 2 {
        return Vec::new();
    }
    let mut returns = Vec::with_capacity(equity_curve.len() - 1);
    for i in 1..equity_curve.len() {
        let prev = equity_curve[i - 1].max(1e-9);
        returns.push((equity_curve[i] - equity_curve[i - 1]) / prev);
    }
    returns
}

fn sharpe_ratio(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let var = returns
        .iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>()
        / returns.len() as f64;
    let std = var.sqrt();
    if std == 0.0 {
        return 0.0;
    }
    mean / std * 252_f64.sqrt()
}

fn max_drawdown(equity_curve: &[f64]) -> f64 {
    let mut peak = f64::MIN;
    let mut max_dd = 0.0;
    for value in equity_curve {
        if *value > peak {
            peak = *value;
        }
        let dd = (peak - value) / peak.max(1e-9);
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

fn total_return(equity_curve: &[f64]) -> f64 {
    if equity_curve.len() < 2 {
        return 0.0;
    }
    equity_curve[equity_curve.len() - 1] / equity_curve[0].max(1e-9) - 1.0
}

#[pyfunction]
fn backtest(
    py: Python,
    prices: Vec<f64>,
    actions: Vec<i64>,
    initial_cash: f64,
    transaction_cost: f64,
) -> PyResult<PyObject> {
    if prices.len() != actions.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "prices and actions must have the same length",
        ));
    }

    let mut cash = initial_cash;
    let mut position = 0.0_f64;
    let mut equity_curve: Vec<f64> = Vec::with_capacity(prices.len());

    for (price, action) in prices.iter().zip(actions.iter()) {
        let fee = price * transaction_cost;
        match *action {
            1 => {
                if position < 1.0 && cash >= price + fee {
                    cash -= price + fee;
                    position += 1.0;
                }
            }
            2 => {
                if position > 0.0 {
                    cash += price - fee;
                    position -= 1.0;
                }
            }
            _ => {}
        }

        let net_worth = cash + position * price;
        equity_curve.push(net_worth);
    }

    let returns = compute_returns(&equity_curve);
    let metrics = PyDict::new(py);
    metrics.set_item("equity_curve", equity_curve)?;
    metrics.set_item("total_return", total_return(&equity_curve))?;
    metrics.set_item("sharpe_ratio", sharpe_ratio(&returns))?;
    metrics.set_item("max_drawdown", max_drawdown(&equity_curve))?;

    Ok(metrics.to_object(py))
}

#[pymodule]
fn rust_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_engine, m)?)?;
    m.add_function(wrap_pyfunction!(backtest, m)?)?;
    Ok(())
}
