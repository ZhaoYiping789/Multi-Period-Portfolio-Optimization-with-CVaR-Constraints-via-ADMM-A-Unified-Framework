"""Visualization functions for portfolio analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


def plot_cumulative_returns(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    title: str = "Cumulative Returns",
    figsize: tuple = (14, 7),
):
    """
    Plot cumulative returns for multiple strategies.

    Parameters
    ----------
    results : Dict[str, Dict]
        Results from multiple strategies
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])

    # Plot cumulative returns
    for strategy_name, metrics in results.items():
        cum_returns = metrics["cumulative_returns"]
        cum_net_returns = metrics["cumulative_net_returns"]

        # Gross returns
        ax1.plot(cum_returns, label=f"{strategy_name} (Gross)", linewidth=2, alpha=0.7)

        # Net returns (dashed)
        ax1.plot(
            cum_net_returns,
            label=f"{strategy_name} (Net)",
            linestyle='--',
            linewidth=1.5,
            alpha=0.7,
        )

    ax1.set_ylabel("Cumulative Return")
    ax1.set_title(title)
    ax1.legend(loc="upper left", ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color='black', linestyle='-', linewidth=0.5)

    # Plot drawdowns
    for strategy_name, metrics in results.items():
        cum_returns = metrics["cumulative_returns"]
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max * 100

        ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, label=strategy_name)

    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Drawdown")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_risk_metrics(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
):
    """
    Plot risk metrics comparison.

    Parameters
    ----------
    results : Dict[str, Dict]
        Results from multiple strategies
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    strategies = list(results.keys())
    n_strategies = len(strategies)

    # Extract metrics
    sharpe_ratios = [results[s]["sharpe_ratio"] for s in strategies]
    sharpe_net = [results[s]["sharpe_net"] for s in strategies]
    sortino_ratios = [results[s]["sortino_ratio"] for s in strategies]
    max_drawdowns = [results[s]["max_drawdown"] * 100 for s in strategies]
    cvar_95 = [results[s]["cvar_95"] * 100 for s in strategies]

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Sharpe Ratio
    ax = axes[0, 0]
    x = np.arange(n_strategies)
    width = 0.35
    ax.bar(x - width/2, sharpe_ratios, width, label='Gross', alpha=0.8)
    ax.bar(x + width/2, sharpe_net, width, label='Net', alpha=0.8)
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Sortino Ratio
    ax = axes[0, 1]
    ax.bar(strategies, sortino_ratios, alpha=0.8, color='coral')
    ax.set_ylabel('Sortino Ratio')
    ax.set_title('Sortino Ratio')
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Max Drawdown
    ax = axes[0, 2]
    ax.bar(strategies, max_drawdowns, alpha=0.8, color='red')
    ax.set_ylabel('Max Drawdown (%)')
    ax.set_title('Maximum Drawdown')
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # CVaR 95%
    ax = axes[1, 0]
    ax.bar(strategies, cvar_95, alpha=0.8, color='darkred')
    ax.set_ylabel('CVaR 95% (%)')
    ax.set_title('Conditional Value-at-Risk (95%)')
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Return vs Risk scatter
    ax = axes[1, 1]
    annual_returns = [results[s]["annual_return"] * 100 for s in strategies]
    annual_vols = [results[s]["annual_vol"] * 100 for s in strategies]

    for i, s in enumerate(strategies):
        ax.scatter(annual_vols[i], annual_returns[i], marker='o', label=s, s=100)
        ax.annotate(s, (annual_vols[i], annual_returns[i]), fontsize=9)

    ax.set_xlabel('Annual Volatility (%)')
    ax.set_ylabel('Annual Return (%)')
    ax.set_title('Risk-Return Profile')
    ax.grid(True, alpha=0.3)

    # Return distribution (box plot)
    ax = axes[1, 2]
    returns_data = [results[s]["portfolio_returns"] * 100 for s in strategies]
    bp = ax.boxplot(returns_data, labels=strategies, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_ylabel('Daily Return (%)')
    ax.set_title('Return Distribution')
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_turnover(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
):
    """
    Plot turnover and transaction costs.

    Parameters
    ----------
    results : Dict[str, Dict]
        Results from multiple strategies
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    strategies = list(results.keys())

    # Average turnover
    avg_turnover = [results[s]["avg_turnover"] for s in strategies]
    total_tc = [results[s]["total_tc"] * 100 for s in strategies]

    ax1.bar(strategies, avg_turnover, alpha=0.8, color='steelblue')
    ax1.set_ylabel('Average Turnover')
    ax1.set_title('Average Daily Turnover')
    ax1.set_xticklabels(strategies, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(strategies, total_tc, alpha=0.8, color='indianred')
    ax2.set_ylabel('Total Transaction Costs (%)')
    ax2.set_title('Cumulative Transaction Costs')
    ax2.set_xticklabels(strategies, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_weights_heatmap(
    weights: np.ndarray,
    asset_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
    title: str = "Portfolio Weights Over Time",
):
    """
    Plot portfolio weights as heatmap.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights, shape (T, n)
    asset_names : List[str], optional
        Asset names
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    title : str
        Plot title
    """
    T, n = weights.shape

    if asset_names is None:
        asset_names = [f"Asset {i+1}" for i in range(n)]

    # Transpose for better visualization
    plt.figure(figsize=figsize)
    sns.heatmap(
        weights.T,
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Weight'},
        yticklabels=asset_names,
        xticklabels=False,
    )
    plt.xlabel('Time')
    plt.ylabel('Assets')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_sensitivity_analysis(
    sensitivity_results: Dict,
    parameter_name: str,
    metric_name: str = "sharpe_ratio",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    """
    Plot sensitivity analysis results.

    Parameters
    ----------
    sensitivity_results : Dict
        Dictionary mapping parameter values to results
    parameter_name : str
        Name of parameter being varied
    metric_name : str
        Metric to plot
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    param_values = list(sensitivity_results.keys())
    metric_values = [sensitivity_results[p][metric_name] for p in param_values]

    plt.figure(figsize=figsize)
    plt.plot(param_values, metric_values, marker='o', linewidth=2, markersize=8)
    plt.xlabel(parameter_name)
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(f'Sensitivity Analysis: {metric_name} vs {parameter_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_admm_convergence(
    history: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8),
):
    """
    Plot ADMM convergence history.

    Parameters
    ----------
    history : Dict
        ADMM history containing residuals and objectives
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Primal residual
    ax = axes[0, 0]
    ax.semilogy(history["primal_residuals"], linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Primal Residual')
    ax.set_title('Primal Residual (log scale)')
    ax.grid(True, alpha=0.3)

    # Dual residual
    ax = axes[0, 1]
    ax.semilogy(history["dual_residuals"], linewidth=2, color='orange')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Dual Residual')
    ax.set_title('Dual Residual (log scale)')
    ax.grid(True, alpha=0.3)

    # Objective value
    ax = axes[1, 0]
    ax.plot(history["objectives"], linewidth=2, color='green')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value')
    ax.set_title('Objective Value')
    ax.grid(True, alpha=0.3)

    # Rho values (if adaptive)
    ax = axes[1, 1]
    if "rho_values" in history and len(history["rho_values"]) > 1:
        ax.plot(history["rho_values"], linewidth=2, color='red')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('ρ (Penalty Parameter)')
        ax.set_title('Adaptive ρ Updates')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Fixed ρ', ha='center', va='center', fontsize=14)
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def generate_summary_table(results: Dict[str, Dict], save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generate summary statistics table.

    Parameters
    ----------
    results : Dict[str, Dict]
        Results from multiple strategies
    save_path : str, optional
        Path to save CSV

    Returns
    -------
    pd.DataFrame
        Summary table
    """
    from ..backtesting.metrics import create_summary_table

    df = create_summary_table(results)

    if save_path:
        import time
        # Try to save CSV with retry logic (in case file is locked)
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                df.to_csv(save_path, index=False)
                print(f"Saved table to {save_path}")
                break
            except PermissionError:
                if attempt < max_attempts - 1:
                    print(f"Warning: File locked, retrying in 2 seconds... (attempt {attempt+1}/{max_attempts})")
                    time.sleep(2)
                else:
                    print(f"Error: Could not save CSV to {save_path} (file may be open in Excel)")
                    # Save to alternative location
                    alt_path = str(save_path).replace('.csv', '_backup.csv')
                    df.to_csv(alt_path, index=False)
                    print(f"Saved to backup location: {alt_path}")

    return df


def plot_efficient_frontier(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    """
    Plot efficient frontier with strategy positions.

    Parameters
    ----------
    results : Dict[str, Dict]
        Results from multiple strategies
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    strategies = list(results.keys())
    returns = [results[s]["annual_return"] * 100 for s in strategies]
    risks = [results[s]["annual_vol"] * 100 for s in strategies]
    sharpes = [results[s]["sharpe_ratio"] for s in strategies]

    plt.figure(figsize=figsize)

    # Scatter with color by Sharpe ratio
    scatter = plt.scatter(risks, returns, c=sharpes, s=200, cmap='RdYlGn', edgecolors='black', linewidth=1.5)

    # Annotate strategies
    for i, s in enumerate(strategies):
        plt.annotate(s, (risks[i], returns[i]), fontsize=10, ha='right')

    plt.xlabel('Annual Volatility (%)')
    plt.ylabel('Annual Return (%)')
    plt.title('Efficient Frontier')
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()
