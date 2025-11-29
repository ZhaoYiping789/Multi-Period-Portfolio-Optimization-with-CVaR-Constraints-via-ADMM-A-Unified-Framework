"""
Run complete analysis with optimal parameters and generate advanced figures.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_CONFIG, MODEL_CONFIG, ADMM_CONFIG,
    FIGURES_DIR, TABLES_DIR, RF_RATE
)

from src.data.loader import load_sp500_data
from src.data.preprocessor import (
    compute_returns, generate_scenarios, estimate_expected_returns,
    estimate_covariance, split_train_test
)
from src.models.admm import ADMMOptimizer
from src.models.baselines import (
    StaticPortfolio, MyopicOptimizer, BoydVarianceOptimizer, MinVariancePortfolio
)
from src.optimization.single_period import SinglePeriodMVO
from src.backtesting.backtest import Backtester
from src.backtesting.metrics import compute_metrics, create_summary_table

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# Optimal parameters from sensitivity analysis
OPTIMAL_PARAMS = {
    'alpha': 0.05,
    'beta': 0.05,        # CVaR limit (relaxed from original 0.03)
    'lambda_tc': 0.005,  # Transaction cost penalty
    'return_weight': 2.0,
}


def load_data():
    """Load and prepare data."""
    print("Loading data...")
    prices = load_sp500_data(
        start_date=DATA_CONFIG["start_date"],
        end_date=DATA_CONFIG["end_date"],
        n_assets=DATA_CONFIG["n_assets"],
        cache_file=DATA_CONFIG["cache_file"],
    )
    returns = compute_returns(prices, method="log")
    train_returns, test_returns = split_train_test(returns, DATA_CONFIG["train_end"])

    window = MODEL_CONFIG["scenario_window"]
    full_returns = returns

    scenarios, _ = generate_scenarios(full_returns, window_size=window)
    mu_estimates = estimate_expected_returns(full_returns, window_size=window)
    cov_estimates = estimate_covariance(full_returns, window_size=window, shrinkage=0.1)

    test_start_idx = len(train_returns) - window
    test_scenarios = scenarios[test_start_idx:]
    test_mu = mu_estimates.iloc[test_start_idx:].values
    test_cov = cov_estimates[test_start_idx:]
    test_returns_aligned = test_returns.iloc[:len(test_mu)]

    print(f"Test period: {len(test_returns_aligned)} days")
    print(f"Assets: {test_returns_aligned.columns.tolist()[:10]}...")

    return test_returns_aligned, test_scenarios, test_mu, test_cov, prices.columns.tolist()


def run_all_strategies(test_returns, test_scenarios, test_mu, test_cov):
    """Run all strategies including optimal ADMM-CVaR."""
    import cvxpy as cp

    results = {}
    T = len(test_mu)
    n = test_mu.shape[1]
    w0 = np.ones(n) / n

    # 1. Equal Weight (Benchmark)
    print("\n1. Running Equal Weight Strategy...")
    ew_weights = np.tile(w0, (T, 1))
    returns_arr = test_returns.values if hasattr(test_returns, 'values') else test_returns
    T_min = min(len(returns_arr), len(ew_weights))
    results['Equal Weight'] = compute_metrics(returns_arr[:T_min], ew_weights[:T_min], RF_RATE, 0.001)
    results['Equal Weight']['weights'] = ew_weights[:T_min]
    results['Equal Weight']['strategy_name'] = 'Equal Weight'

    # 2. Minimum Variance (use average covariance)
    print("\n2. Running Minimum Variance Strategy...")
    avg_cov = np.mean(test_cov, axis=0)
    w_mv = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w_mv, avg_cov)), [cp.sum(w_mv) == 1, w_mv >= 0])
    prob.solve(solver=cp.OSQP, verbose=False)
    mv_weights = np.tile(w_mv.value, (T, 1))
    results['Min Variance'] = compute_metrics(returns_arr[:T_min], mv_weights[:T_min], RF_RATE, 0.001)
    results['Min Variance']['weights'] = mv_weights[:T_min]
    results['Min Variance']['strategy_name'] = 'Min Variance'

    # 3. Myopic CVaR
    print("\n3. Running Myopic CVaR Strategy...")
    myopic_strategy = MyopicOptimizer(alpha=0.05, beta=0.05, method="cvar")
    myopic_result = myopic_strategy.optimize(test_mu, test_scenarios, w0, test_cov)
    myopic_weights = myopic_result['weights']
    T_min = min(len(returns_arr), len(myopic_weights))
    results['Myopic CVaR'] = compute_metrics(returns_arr[:T_min], myopic_weights[:T_min], RF_RATE, 0.001)
    results['Myopic CVaR']['weights'] = myopic_weights[:T_min]
    results['Myopic CVaR']['strategy_name'] = 'Myopic CVaR'

    # 4. Boyd Variance
    print("\n4. Running Boyd Variance Strategy...")
    boyd_strategy = BoydVarianceOptimizer(lambda_tc=0.005, risk_aversion=1.0)
    boyd_result = boyd_strategy.optimize(test_mu, w0, test_scenarios, test_cov)
    boyd_weights = boyd_result['weights']
    T_min = min(len(returns_arr), len(boyd_weights))
    results['Boyd Var'] = compute_metrics(returns_arr[:T_min], boyd_weights[:T_min], RF_RATE, 0.005)
    results['Boyd Var']['weights'] = boyd_weights[:T_min]
    results['Boyd Var']['strategy_name'] = 'Boyd Var'

    # 5. ADMM-CVaR with OPTIMAL parameters
    print("\n5. Running ADMM-CVaR (Optimal) Strategy...")
    admm_config = {
        'rho': 1.0,
        'max_iter': 100,
        'tol_primal': 1e-4,
        'tol_dual': 1e-4,
        'adaptive_rho': True,
        'n_jobs': -1,
        'verbose': True,
        'return_weight': OPTIMAL_PARAMS['return_weight'],
    }
    admm_strategy = ADMMOptimizer(
        alpha=OPTIMAL_PARAMS['alpha'],
        beta=OPTIMAL_PARAMS['beta'],
        lambda_tc=OPTIMAL_PARAMS['lambda_tc'],
        **admm_config,
    )
    admm_result = admm_strategy.optimize(test_mu, test_scenarios, w0)
    admm_weights = admm_result['weights']
    T_min = min(len(returns_arr), len(admm_weights))
    results['ADMM-CVaR'] = compute_metrics(returns_arr[:T_min], admm_weights[:T_min], RF_RATE, OPTIMAL_PARAMS['lambda_tc'])
    results['ADMM-CVaR']['weights'] = admm_weights[:T_min]
    results['ADMM-CVaR']['strategy_name'] = 'ADMM-CVaR'

    return results


def plot_cumulative_returns_advanced(results, save_path):
    """Advanced cumulative returns plot with confidence bands."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

    # Color palette
    colors = {
        'Equal Weight': '#808080',
        'Min Variance': '#2196F3',
        'Myopic CVaR': '#FF9800',
        'Boyd Var': '#9C27B0',
        'ADMM-CVaR': '#E53935',
    }

    ax1 = axes[0]
    for name, metrics in results.items():
        cum_ret = metrics['cumulative_returns']
        color = colors.get(name, '#000000')
        linewidth = 3 if name == 'ADMM-CVaR' else 1.5
        alpha = 1.0 if name == 'ADMM-CVaR' else 0.7
        ax1.plot(cum_ret, label=name, color=color, linewidth=linewidth, alpha=alpha)

    ax1.axhline(y=1, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.set_ylabel('Cumulative Return', fontsize=14)
    ax1.set_title('Portfolio Performance Comparison (2023-2024)', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(cum_ret))

    # Drawdown subplot
    ax2 = axes[1]
    for name, metrics in results.items():
        cum_ret = metrics['cumulative_returns']
        running_max = np.maximum.accumulate(cum_ret)
        drawdown = (cum_ret - running_max) / running_max * 100
        color = colors.get(name, '#000000')
        ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color=color, label=name)
        ax2.plot(drawdown, color=color, linewidth=0.5)

    ax2.set_xlabel('Trading Days', fontsize=14)
    ax2.set_ylabel('Drawdown (%)', fontsize=14)
    ax2.set_title('Drawdown Analysis', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, len(drawdown))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_risk_return_scatter(results, save_path):
    """Risk-return scatter plot with annotations."""
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = {
        'Equal Weight': '#808080',
        'Min Variance': '#2196F3',
        'Myopic CVaR': '#FF9800',
        'Boyd Var': '#9C27B0',
        'ADMM-CVaR': '#E53935',
    }

    for name, metrics in results.items():
        ret = metrics['annual_return'] * 100
        vol = metrics['annual_vol'] * 100
        sharpe = metrics['sharpe_ratio']
        color = colors.get(name, '#000000')

        size = 300 if name == 'ADMM-CVaR' else 200
        marker = '*' if name == 'ADMM-CVaR' else 'o'

        ax.scatter(vol, ret, s=size, c=color, marker=marker,
                  label=f'{name} (SR={sharpe:.2f})', edgecolors='black', linewidth=1.5, zorder=5)

        # Annotation
        offset = (10, 10) if name != 'ADMM-CVaR' else (15, 15)
        ax.annotate(name, (vol, ret), xytext=offset, textcoords='offset points',
                   fontsize=11, fontweight='bold' if name == 'ADMM-CVaR' else 'normal')

    ax.set_xlabel('Annual Volatility (%)', fontsize=14)
    ax.set_ylabel('Annual Return (%)', fontsize=14)
    ax.set_title('Risk-Return Profile', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_metrics_comparison(results, save_path):
    """Bar chart comparing key metrics."""
    strategies = list(results.keys())

    metrics_data = {
        'Sharpe Ratio': [results[s]['sharpe_ratio'] for s in strategies],
        'Annual Return (%)': [results[s]['annual_return'] * 100 for s in strategies],
        'Max Drawdown (%)': [results[s]['max_drawdown'] * 100 for s in strategies],
        'CVaR 95% (%)': [results[s]['cvar_95'] * 100 for s in strategies],
        'Sortino Ratio': [results[s]['sortino_ratio'] for s in strategies],
        'Avg Turnover': [results[s]['avg_turnover'] for s in strategies],
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    colors = ['#808080', '#2196F3', '#FF9800', '#9C27B0', '#E53935']

    for idx, (metric_name, values) in enumerate(metrics_data.items()):
        ax = axes[idx]
        bars = ax.bar(strategies, values, color=colors, edgecolor='black', linewidth=1)

        # Highlight best
        if 'Drawdown' in metric_name or 'CVaR' in metric_name or 'Turnover' in metric_name:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

    plt.suptitle('Strategy Performance Comparison', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_rolling_sharpe(results, window=60, save_path=None):
    """Rolling Sharpe ratio comparison."""
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = {
        'Equal Weight': '#808080',
        'Min Variance': '#2196F3',
        'Myopic CVaR': '#FF9800',
        'Boyd Var': '#9C27B0',
        'ADMM-CVaR': '#E53935',
    }

    for name, metrics in results.items():
        returns = metrics['portfolio_returns']
        rolling_mean = pd.Series(returns).rolling(window).mean()
        rolling_std = pd.Series(returns).rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

        color = colors.get(name, '#000000')
        linewidth = 2.5 if name == 'ADMM-CVaR' else 1.2
        ax.plot(rolling_sharpe, label=name, color=color, linewidth=linewidth)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Trading Days', fontsize=14)
    ax.set_ylabel(f'Rolling Sharpe Ratio ({window}-day)', fontsize=14)
    ax.set_title(f'Rolling Sharpe Ratio Comparison', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_weight_evolution(weights, asset_names, save_path):
    """Stacked area chart of portfolio weights over time."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Select top 10 assets by average weight
    avg_weights = np.mean(weights, axis=0)
    top_indices = np.argsort(avg_weights)[-10:][::-1]

    weights_top = weights[:, top_indices]
    names_top = [asset_names[i] for i in top_indices]

    # Stack plot
    ax.stackplot(range(len(weights)), weights_top.T, labels=names_top, alpha=0.8)

    ax.set_xlabel('Trading Days', fontsize=14)
    ax.set_ylabel('Portfolio Weight', fontsize=14)
    ax.set_title('ADMM-CVaR Portfolio Weight Evolution (Top 10 Assets)', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), frameon=True)
    ax.set_xlim(0, len(weights))
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_monthly_returns_heatmap(results, save_path):
    """Monthly returns heatmap for ADMM-CVaR."""
    returns = results['ADMM-CVaR']['portfolio_returns']

    # Create date index (assume starting from 2023-01)
    dates = pd.date_range(start='2023-01-01', periods=len(returns), freq='B')
    returns_series = pd.Series(returns, index=dates)

    # Monthly returns
    monthly = returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100

    # Reshape to year x month
    monthly_df = monthly.to_frame('return')
    monthly_df['year'] = monthly_df.index.year
    monthly_df['month'] = monthly_df.index.month

    pivot = monthly_df.pivot(index='year', columns='month', values='return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(14, 4))

    # Custom colormap: red for negative, green for positive
    cmap = sns.diverging_palette(10, 130, as_cmap=True)

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap, center=0,
                linewidths=1, linecolor='white', cbar_kws={'label': 'Return (%)'}, ax=ax)

    ax.set_title('ADMM-CVaR Monthly Returns (%)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Month', fontsize=14)
    ax.set_ylabel('Year', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_return_distribution(results, save_path):
    """Return distribution comparison (violin plot)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    data = []
    names = []
    for name, metrics in results.items():
        data.append(metrics['portfolio_returns'] * 100)
        names.append(name)

    colors = ['#808080', '#2196F3', '#FF9800', '#9C27B0', '#E53935']

    parts = ax.violinplot(data, positions=range(len(names)), showmeans=True, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel('Daily Return (%)', fontsize=14)
    ax.set_title('Return Distribution Comparison', fontsize=16, fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_radar_chart(results, save_path):
    """Radar chart comparing strategies across multiple metrics."""
    categories = ['Sharpe', 'Return', 'Low Vol', 'Low DD', 'Low CVaR', 'Sortino']

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    colors = {
        'Equal Weight': '#808080',
        'Min Variance': '#2196F3',
        'Myopic CVaR': '#FF9800',
        'Boyd Var': '#9C27B0',
        'ADMM-CVaR': '#E53935',
    }

    # Normalize metrics to 0-1 scale
    all_sharpe = [results[s]['sharpe_ratio'] for s in results]
    all_return = [results[s]['annual_return'] for s in results]
    all_vol = [results[s]['annual_vol'] for s in results]
    all_dd = [results[s]['max_drawdown'] for s in results]
    all_cvar = [results[s]['cvar_95'] for s in results]
    all_sortino = [results[s]['sortino_ratio'] for s in results]

    def normalize(val, all_vals, invert=False):
        min_v, max_v = min(all_vals), max(all_vals)
        if max_v == min_v:
            return 0.5
        norm = (val - min_v) / (max_v - min_v)
        return 1 - norm if invert else norm

    for name, metrics in results.items():
        values = [
            normalize(metrics['sharpe_ratio'], all_sharpe),
            normalize(metrics['annual_return'], all_return),
            normalize(metrics['annual_vol'], all_vol, invert=True),
            normalize(metrics['max_drawdown'], all_dd, invert=True),
            normalize(metrics['cvar_95'], all_cvar, invert=True),
            normalize(metrics['sortino_ratio'], all_sortino),
        ]
        values += values[:1]

        color = colors.get(name, '#000000')
        linewidth = 3 if name == 'ADMM-CVaR' else 1.5
        ax.plot(angles, values, 'o-', linewidth=linewidth, label=name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_title('Strategy Comparison Radar', fontsize=16, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def generate_summary_table_latex(results, save_path):
    """Generate publication-quality LaTeX table."""
    df = create_summary_table(results)

    latex = df.to_latex(
        index=False,
        float_format="%.4f",
        caption="Strategy Performance Comparison (2023-2024)",
        label="tab:results",
        column_format='l' + 'r' * (len(df.columns) - 1),
        bold_rows=True,
    )

    with open(save_path, 'w') as f:
        f.write(latex)
    print(f"Saved: {save_path}")


def main():
    print("=" * 60)
    print("ELEC5470 Final Project - Optimal Strategy Analysis")
    print("=" * 60)
    print(f"\nOptimal Parameters: {OPTIMAL_PARAMS}")

    # Load data
    test_returns, test_scenarios, test_mu, test_cov, asset_names = load_data()

    # Run all strategies
    print("\n" + "=" * 60)
    print("Running All Strategies")
    print("=" * 60)
    results = run_all_strategies(test_returns, test_scenarios, test_mu, test_cov)

    # Generate figures
    print("\n" + "=" * 60)
    print("Generating Advanced Figures")
    print("=" * 60)

    # 1. Cumulative returns
    plot_cumulative_returns_advanced(
        results,
        FIGURES_DIR / "optimal_cumulative_returns.png"
    )

    # 2. Risk-return scatter
    plot_risk_return_scatter(
        results,
        FIGURES_DIR / "optimal_risk_return.png"
    )

    # 3. Metrics comparison
    plot_metrics_comparison(
        results,
        FIGURES_DIR / "optimal_metrics_comparison.png"
    )

    # 4. Rolling Sharpe
    plot_rolling_sharpe(
        results,
        window=60,
        save_path=FIGURES_DIR / "optimal_rolling_sharpe.png"
    )

    # 5. Weight evolution
    admm_weights = results['ADMM-CVaR']['weights']
    plot_weight_evolution(
        admm_weights,
        asset_names,
        FIGURES_DIR / "optimal_weight_evolution.png"
    )

    # 6. Monthly returns heatmap
    plot_monthly_returns_heatmap(
        results,
        FIGURES_DIR / "optimal_monthly_heatmap.png"
    )

    # 7. Return distribution
    plot_return_distribution(
        results,
        FIGURES_DIR / "optimal_return_distribution.png"
    )

    # 8. Radar chart
    plot_radar_chart(
        results,
        FIGURES_DIR / "optimal_radar_chart.png"
    )

    # Generate tables
    print("\n" + "=" * 60)
    print("Generating Tables")
    print("=" * 60)

    summary_df = create_summary_table(results)
    summary_df.to_csv(TABLES_DIR / "optimal_results.csv", index=False)
    print(f"Saved: {TABLES_DIR / 'optimal_results.csv'}")

    generate_summary_table_latex(results, TABLES_DIR / "optimal_results.tex")

    # Print summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
