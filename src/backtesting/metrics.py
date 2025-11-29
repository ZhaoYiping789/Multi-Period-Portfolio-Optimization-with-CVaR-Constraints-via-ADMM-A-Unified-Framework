"""Performance metrics for portfolio backtesting."""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def compute_metrics(
    returns: np.ndarray,
    weights: np.ndarray,
    rf_rate: float = 0.02 / 252,
    lambda_tc: float = 0.005,
) -> Dict:
    """
    Compute comprehensive performance metrics.

    Parameters
    ----------
    returns : np.ndarray
        Asset returns, shape (T, n)
    weights : np.ndarray
        Portfolio weights, shape (T, n)
    rf_rate : float
        Daily risk-free rate
    lambda_tc : float
        Transaction cost parameter

    Returns
    -------
    Dict
        Dictionary of performance metrics
    """
    # Portfolio returns
    portfolio_returns = (returns * weights).sum(axis=1)

    # Transaction costs
    turnover = np.abs(np.diff(weights, axis=0)).sum(axis=1)
    tc_costs = (lambda_tc / 2) * (np.diff(weights, axis=0) ** 2).sum(axis=1)
    net_returns = portfolio_returns[1:] - tc_costs

    # Cumulative returns
    cum_returns = (1 + portfolio_returns).cumprod()
    cum_net_returns = np.concatenate([[1], (1 + net_returns).cumprod()])

    # Basic statistics
    mean_return = portfolio_returns.mean()
    std_return = portfolio_returns.std()
    mean_net_return = net_returns.mean()

    # Risk-adjusted metrics
    sharpe = sharpe_ratio(portfolio_returns, rf_rate)
    sortino = sortino_ratio(portfolio_returns, rf_rate)
    sharpe_net = sharpe_ratio(net_returns, rf_rate)

    # Risk metrics
    max_dd = max_drawdown(cum_returns)
    var_95 = np.percentile(-portfolio_returns, 95)
    cvar_95 = -portfolio_returns[-portfolio_returns >= var_95].mean()

    # Trading metrics
    avg_turnover = turnover.mean()
    total_tc = tc_costs.sum()

    # Annualized metrics (252 trading days)
    annual_return = mean_return * 252
    annual_net_return = mean_net_return * 252
    annual_vol = std_return * np.sqrt(252)

    return {
        # Returns
        "mean_return": mean_return,
        "annual_return": annual_return,
        "std_return": std_return,
        "annual_vol": annual_vol,
        "mean_net_return": mean_net_return,
        "annual_net_return": annual_net_return,
        "total_return": cum_returns[-1] - 1,
        "total_net_return": cum_net_returns[-1] - 1,
        # Risk-adjusted
        "sharpe_ratio": sharpe,
        "sharpe_net": sharpe_net,
        "sortino_ratio": sortino,
        # Risk
        "max_drawdown": max_dd,
        "var_95": var_95,
        "cvar_95": cvar_95,
        # Trading
        "avg_turnover": avg_turnover,
        "total_turnover": turnover.sum(),
        "total_tc": total_tc,
        # Time series
        "portfolio_returns": portfolio_returns,
        "net_returns": net_returns,
        "cumulative_returns": cum_returns,
        "cumulative_net_returns": cum_net_returns,
        "turnover": turnover,
    }


def sharpe_ratio(returns: np.ndarray, rf_rate: float = 0.0) -> float:
    """
    Compute Sharpe ratio.

    Sharpe = (E[R] - Rf) / σ(R)

    Parameters
    ----------
    returns : np.ndarray
        Returns
    rf_rate : float
        Risk-free rate

    Returns
    -------
    float
        Sharpe ratio (annualized)
    """
    excess_returns = returns - rf_rate
    if excess_returns.std() == 0:
        return 0.0
    sharpe = excess_returns.mean() / excess_returns.std()
    return sharpe * np.sqrt(252)  # Annualized


def sortino_ratio(returns: np.ndarray, rf_rate: float = 0.0) -> float:
    """
    Compute Sortino ratio (downside deviation).

    Sortino = (E[R] - Rf) / σ_downside(R)

    Parameters
    ----------
    returns : np.ndarray
        Returns
    rf_rate : float
        Risk-free rate

    Returns
    -------
    float
        Sortino ratio (annualized)
    """
    excess_returns = returns - rf_rate
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    sortino = excess_returns.mean() / downside_returns.std()
    return sortino * np.sqrt(252)  # Annualized


def max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Compute maximum drawdown.

    MDD = max_t [ (peak_t - value_t) / peak_t ]

    Parameters
    ----------
    cumulative_returns : np.ndarray
        Cumulative returns (starting from 1)

    Returns
    -------
    float
        Maximum drawdown (positive value)
    """
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (running_max - cumulative_returns) / running_max
    return drawdown.max()


def calmar_ratio(returns: np.ndarray, rf_rate: float = 0.0) -> float:
    """
    Compute Calmar ratio.

    Calmar = Annual Return / Max Drawdown

    Parameters
    ----------
    returns : np.ndarray
        Returns
    rf_rate : float
        Risk-free rate

    Returns
    -------
    float
        Calmar ratio
    """
    annual_return = returns.mean() * 252
    cum_returns = (1 + returns).cumprod()
    mdd = max_drawdown(cum_returns)

    if mdd == 0:
        return 0.0

    return annual_return / mdd


def information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """
    Compute Information ratio.

    IR = E[R - R_bench] / σ(R - R_bench)

    Parameters
    ----------
    returns : np.ndarray
        Portfolio returns
    benchmark_returns : np.ndarray
        Benchmark returns

    Returns
    -------
    float
        Information ratio (annualized)
    """
    excess_returns = returns - benchmark_returns
    if excess_returns.std() == 0:
        return 0.0

    ir = excess_returns.mean() / excess_returns.std()
    return ir * np.sqrt(252)


def downside_deviation(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Compute downside deviation.

    Parameters
    ----------
    returns : np.ndarray
        Returns
    threshold : float
        Minimum acceptable return

    Returns
    -------
    float
        Downside deviation
    """
    downside_returns = returns - threshold
    downside_returns = downside_returns[downside_returns < 0]

    if len(downside_returns) == 0:
        return 0.0

    return np.sqrt((downside_returns ** 2).mean())


def tail_ratio(returns: np.ndarray, percentile: float = 95) -> float:
    """
    Compute tail ratio (right tail / left tail).

    Parameters
    ----------
    returns : np.ndarray
        Returns
    percentile : float
        Percentile for tails

    Returns
    -------
    float
        Tail ratio
    """
    right_tail = np.percentile(returns, percentile)
    left_tail = np.abs(np.percentile(returns, 100 - percentile))

    if left_tail == 0:
        return 0.0

    return right_tail / left_tail


def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Compute Omega ratio.

    Ω = E[(R - L)₊] / E[(L - R)₊]

    Parameters
    ----------
    returns : np.ndarray
        Returns
    threshold : float
        Threshold return (usually 0 or risk-free rate)

    Returns
    -------
    float
        Omega ratio
    """
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()

    if losses == 0:
        return np.inf

    return gains / losses


def value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Compute Value-at-Risk.

    VaR_α = -quantile(R, α)

    Parameters
    ----------
    returns : np.ndarray
        Returns
    alpha : float
        Confidence level

    Returns
    -------
    float
        VaR (positive for loss)
    """
    return -np.percentile(returns, alpha * 100)


def conditional_var(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Compute Conditional Value-at-Risk (Expected Shortfall).

    CVaR_α = -E[R | R ≤ VaR_α]

    Parameters
    ----------
    returns : np.ndarray
        Returns
    alpha : float
        Confidence level

    Returns
    -------
    float
        CVaR (positive for loss)
    """
    var = value_at_risk(returns, alpha)
    tail_losses = returns[returns <= -var]

    if len(tail_losses) == 0:
        return var

    return -tail_losses.mean()


def create_summary_table(metrics_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create summary table from multiple strategies' metrics.

    Parameters
    ----------
    metrics_dict : Dict[str, Dict]
        Dictionary mapping strategy names to metrics

    Returns
    -------
    pd.DataFrame
        Summary table
    """
    rows = []

    for strategy, metrics in metrics_dict.items():
        row = {
            "Strategy": strategy,
            "Annual Return (%)": metrics["annual_return"] * 100,
            "Annual Vol (%)": metrics["annual_vol"] * 100,
            "Sharpe Ratio": metrics["sharpe_ratio"],
            "Sortino Ratio": metrics["sortino_ratio"],
            "Max Drawdown (%)": metrics["max_drawdown"] * 100,
            "CVaR 95% (%)": metrics["cvar_95"] * 100,
            "Avg Turnover": metrics["avg_turnover"],
            "Total TC (%)": metrics["total_tc"] * 100,
            "Net Return (%)": metrics["annual_net_return"] * 100,
            "Sharpe (Net)": metrics["sharpe_net"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.round(2)
