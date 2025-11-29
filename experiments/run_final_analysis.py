"""
Final analysis script for paper preparation.
Runs comprehensive sensitivity analysis and generates publication-quality figures.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from itertools import product
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
from src.backtesting.backtest import Backtester

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
    return test_returns_aligned, test_scenarios, test_mu, test_cov


def run_single_experiment(test_returns, test_scenarios, test_mu, test_cov,
                          alpha, beta, lambda_tc, return_weight=1.0, max_iter=50):
    """Run single ADMM experiment with given parameters."""
    admm_config = {
        'rho': 1.0,
        'max_iter': max_iter,
        'tol_primal': 1e-4,
        'tol_dual': 1e-4,
        'adaptive_rho': True,
        'n_jobs': -1,
        'verbose': False,
        'return_weight': return_weight,
    }

    optimizer = ADMMOptimizer(
        alpha=alpha,
        beta=beta,
        lambda_tc=lambda_tc,
        **admm_config,
    )

    backtester = Backtester(test_returns, test_scenarios, test_mu, test_cov, RF_RATE, lambda_tc)
    result = backtester.run_strategy(optimizer, strategy_name="ADMM")
    return result


def sensitivity_beta_lambda(data, save_results=True):
    """2D sensitivity analysis: beta vs lambda_tc."""
    print("\n" + "="*60)
    print("Running 2D Sensitivity Analysis: beta vs lambda_tc")
    print("="*60)

    test_returns, test_scenarios, test_mu, test_cov = data

    # Parameter ranges
    beta_range = [0.02, 0.03, 0.05, 0.07, 0.10]
    lambda_range = [0.0005, 0.001, 0.002, 0.005, 0.01]

    # Results storage
    results_grid = {
        'sharpe': np.zeros((len(beta_range), len(lambda_range))),
        'return': np.zeros((len(beta_range), len(lambda_range))),
        'cvar': np.zeros((len(beta_range), len(lambda_range))),
        'turnover': np.zeros((len(beta_range), len(lambda_range))),
        'max_dd': np.zeros((len(beta_range), len(lambda_range))),
    }

    total = len(beta_range) * len(lambda_range)
    pbar = tqdm(total=total, desc="Grid Search")

    for i, beta in enumerate(beta_range):
        for j, lambda_tc in enumerate(lambda_range):
            try:
                result = run_single_experiment(
                    test_returns, test_scenarios, test_mu, test_cov,
                    alpha=0.05, beta=beta, lambda_tc=lambda_tc, max_iter=50
                )
                results_grid['sharpe'][i, j] = result['sharpe_ratio']
                results_grid['return'][i, j] = result['annual_return']
                results_grid['cvar'][i, j] = result['cvar_95']
                results_grid['turnover'][i, j] = result['avg_turnover']
                results_grid['max_dd'][i, j] = result['max_drawdown']
            except Exception as e:
                print(f"Error at beta={beta}, lambda={lambda_tc}: {e}")
                results_grid['sharpe'][i, j] = np.nan

            pbar.update(1)

    pbar.close()

    # Save results
    if save_results:
        df_results = []
        for i, beta in enumerate(beta_range):
            for j, lambda_tc in enumerate(lambda_range):
                df_results.append({
                    'beta': beta,
                    'lambda_tc': lambda_tc,
                    'sharpe_ratio': results_grid['sharpe'][i, j],
                    'annual_return': results_grid['return'][i, j],
                    'cvar_95': results_grid['cvar'][i, j],
                    'avg_turnover': results_grid['turnover'][i, j],
                    'max_drawdown': results_grid['max_dd'][i, j],
                })
        df = pd.DataFrame(df_results)
        df.to_csv(TABLES_DIR / 'sensitivity_2d_results.csv', index=False)
        print(f"Results saved to {TABLES_DIR / 'sensitivity_2d_results.csv'}")

    return results_grid, beta_range, lambda_range


def sensitivity_return_weight(data):
    """Sensitivity to return weight parameter."""
    print("\n" + "="*60)
    print("Running Sensitivity Analysis: Return Weight")
    print("="*60)

    test_returns, test_scenarios, test_mu, test_cov = data

    return_weights = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    results = {}

    for rw in tqdm(return_weights, desc="Return Weight"):
        try:
            result = run_single_experiment(
                test_returns, test_scenarios, test_mu, test_cov,
                alpha=0.05, beta=0.07, lambda_tc=0.002,
                return_weight=rw, max_iter=50
            )
            results[rw] = result
        except Exception as e:
            print(f"Error at return_weight={rw}: {e}")

    return results


def plot_sensitivity_heatmap(results_grid, beta_range, lambda_range, metric='sharpe'):
    """Plot 2D sensitivity heatmap."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['sharpe', 'return', 'cvar']
    titles = ['Sharpe Ratio', 'Annual Return', 'CVaR (95%)']
    cmaps = ['RdYlGn', 'RdYlGn', 'RdYlGn_r']

    for ax, metric, title, cmap in zip(axes, metrics, titles, cmaps):
        data = results_grid[metric]
        if metric == 'return':
            data = data * 100  # Convert to percentage
        elif metric == 'cvar':
            data = data * 100

        im = ax.imshow(data, cmap=cmap, aspect='auto', origin='lower')

        # Labels
        ax.set_xticks(range(len(lambda_range)))
        ax.set_xticklabels([f'{x:.4f}' for x in lambda_range], rotation=45)
        ax.set_yticks(range(len(beta_range)))
        ax.set_yticklabels([f'{x:.2f}' for x in beta_range])

        ax.set_xlabel(r'$\lambda_{tc}$ (Transaction Cost)')
        ax.set_ylabel(r'$\beta$ (CVaR Limit)')
        ax.set_title(title)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)

        # Annotate cells
        for i in range(len(beta_range)):
            for j in range(len(lambda_range)):
                val = data[i, j]
                if not np.isnan(val):
                    text_color = 'white' if abs(val - np.nanmean(data)) > np.nanstd(data) else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=9, color=text_color)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sensitivity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'sensitivity_heatmap.pdf', bbox_inches='tight')
    print(f"Saved to {FIGURES_DIR / 'sensitivity_heatmap.png'}")
    plt.close()


def plot_sensitivity_3d(results_grid, beta_range, lambda_range):
    """Plot 3D surface for sensitivity analysis."""
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 5))

    B, L = np.meshgrid(lambda_range, beta_range)

    # Sharpe Ratio surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(B, L, results_grid['sharpe'], cmap='viridis', alpha=0.8)
    ax1.set_xlabel(r'$\lambda_{tc}$')
    ax1.set_ylabel(r'$\beta$')
    ax1.set_zlabel('Sharpe Ratio')
    ax1.set_title('Sharpe Ratio Surface')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)

    # Annual Return surface
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(B, L, results_grid['return'] * 100, cmap='plasma', alpha=0.8)
    ax2.set_xlabel(r'$\lambda_{tc}$')
    ax2.set_ylabel(r'$\beta$')
    ax2.set_zlabel('Annual Return (%)')
    ax2.set_title('Annual Return Surface')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sensitivity_3d_surface.png', dpi=300, bbox_inches='tight')
    print(f"Saved to {FIGURES_DIR / 'sensitivity_3d_surface.png'}")
    plt.close()


def plot_return_weight_sensitivity(results):
    """Plot return weight sensitivity."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    rw_values = list(results.keys())
    sharpes = [results[rw]['sharpe_ratio'] for rw in rw_values]
    returns = [results[rw]['annual_return'] * 100 for rw in rw_values]
    cvars = [results[rw]['cvar_95'] * 100 for rw in rw_values]
    turnovers = [results[rw]['avg_turnover'] for rw in rw_values]

    # Sharpe ratio
    axes[0].plot(rw_values, sharpes, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    axes[0].set_xlabel('Return Weight ($\\gamma_r$)')
    axes[0].set_ylabel('Sharpe Ratio')
    axes[0].set_title('Sharpe Ratio vs Return Weight')
    axes[0].grid(True, alpha=0.3)

    # Return vs CVaR trade-off
    axes[1].plot(cvars, returns, 'o-', linewidth=2, markersize=8, color='#A23B72')
    for i, rw in enumerate(rw_values):
        axes[1].annotate(f'$\\gamma_r$={rw}', (cvars[i], returns[i]),
                        textcoords="offset points", xytext=(5,5), fontsize=9)
    axes[1].set_xlabel('CVaR 95% (%)')
    axes[1].set_ylabel('Annual Return (%)')
    axes[1].set_title('Return-Risk Trade-off')
    axes[1].grid(True, alpha=0.3)

    # Turnover
    axes[2].bar(range(len(rw_values)), turnovers, color='#F18F01', alpha=0.8)
    axes[2].set_xticks(range(len(rw_values)))
    axes[2].set_xticklabels([str(rw) for rw in rw_values])
    axes[2].set_xlabel('Return Weight ($\\gamma_r$)')
    axes[2].set_ylabel('Average Turnover')
    axes[2].set_title('Portfolio Turnover')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sensitivity_return_weight.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'sensitivity_return_weight.pdf', bbox_inches='tight')
    print(f"Saved to {FIGURES_DIR / 'sensitivity_return_weight.png'}")
    plt.close()


def plot_pareto_frontier(results_grid, beta_range, lambda_range):
    """Plot Pareto frontier of return vs risk."""
    fig, ax = plt.subplots(figsize=(10, 7))

    returns = results_grid['return'].flatten() * 100
    cvars = results_grid['cvar'].flatten() * 100
    sharpes = results_grid['sharpe'].flatten()

    # Color by Sharpe ratio
    scatter = ax.scatter(cvars, returns, c=sharpes, cmap='RdYlGn', s=100,
                        edgecolors='black', linewidth=0.5)

    # Find Pareto optimal points
    pareto_mask = np.ones(len(returns), dtype=bool)
    for i in range(len(returns)):
        for j in range(len(returns)):
            if i != j:
                # j dominates i if j has higher return and lower risk
                if returns[j] >= returns[i] and cvars[j] <= cvars[i]:
                    if returns[j] > returns[i] or cvars[j] < cvars[i]:
                        pareto_mask[i] = False
                        break

    # Highlight Pareto optimal points
    pareto_returns = returns[pareto_mask]
    pareto_cvars = cvars[pareto_mask]

    # Sort by CVaR for line plot
    sort_idx = np.argsort(pareto_cvars)
    ax.plot(pareto_cvars[sort_idx], pareto_returns[sort_idx], 'r--',
           linewidth=2, label='Pareto Frontier', alpha=0.7)
    ax.scatter(pareto_cvars, pareto_returns, c='red', s=150, marker='*',
              edgecolors='black', linewidth=1, zorder=5, label='Pareto Optimal')

    ax.set_xlabel('CVaR 95% (%)', fontsize=14)
    ax.set_ylabel('Annual Return (%)', fontsize=14)
    ax.set_title('Return-Risk Trade-off (Pareto Analysis)', fontsize=16)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sharpe Ratio', fontsize=12)

    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'pareto_frontier.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'pareto_frontier.pdf', bbox_inches='tight')
    print(f"Saved to {FIGURES_DIR / 'pareto_frontier.png'}")
    plt.close()


def generate_latex_tables(results_grid, beta_range, lambda_range, rw_results=None):
    """Generate LaTeX tables for paper."""

    # Table 1: 2D Sensitivity Results (best configurations)
    df_results = []
    for i, beta in enumerate(beta_range):
        for j, lambda_tc in enumerate(lambda_range):
            df_results.append({
                'beta': beta,
                'lambda_tc': lambda_tc,
                'sharpe': results_grid['sharpe'][i, j],
                'return': results_grid['return'][i, j] * 100,
                'cvar': results_grid['cvar'][i, j] * 100,
                'turnover': results_grid['turnover'][i, j],
                'max_dd': results_grid['max_dd'][i, j] * 100,
            })

    df = pd.DataFrame(df_results)

    # Sort by Sharpe ratio
    df_sorted = df.sort_values('sharpe', ascending=False).head(10)

    # Generate LaTeX
    latex_table = """
\\begin{table}[h]
\\centering
\\caption{Top 10 Parameter Configurations by Sharpe Ratio}
\\label{tab:sensitivity}
\\begin{tabular}{cccccc}
\\toprule
$\\beta$ & $\\lambda_{tc}$ & Sharpe & Return (\\%) & CVaR (\\%) & Turnover \\\\
\\midrule
"""
    for _, row in df_sorted.iterrows():
        latex_table += f"{row['beta']:.2f} & {row['lambda_tc']:.4f} & {row['sharpe']:.3f} & {row['return']:.2f} & {row['cvar']:.2f} & {row['turnover']:.3f} \\\\\n"

    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    with open(TABLES_DIR / 'sensitivity_table.tex', 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to {TABLES_DIR / 'sensitivity_table.tex'}")

    # Table 2: Return Weight Sensitivity
    if rw_results:
        latex_rw = """
\\begin{table}[h]
\\centering
\\caption{Sensitivity to Return Weight Parameter}
\\label{tab:return_weight}
\\begin{tabular}{ccccc}
\\toprule
$\\gamma_r$ & Sharpe & Return (\\%) & CVaR (\\%) & Turnover \\\\
\\midrule
"""
        for rw, metrics in rw_results.items():
            latex_rw += f"{rw:.1f} & {metrics['sharpe_ratio']:.3f} & {metrics['annual_return']*100:.2f} & {metrics['cvar_95']*100:.2f} & {metrics['avg_turnover']:.3f} \\\\\n"

        latex_rw += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        with open(TABLES_DIR / 'return_weight_table.tex', 'w') as f:
            f.write(latex_rw)
        print(f"LaTeX table saved to {TABLES_DIR / 'return_weight_table.tex'}")


def create_main_results_table():
    """Create main results comparison table in LaTeX format."""
    # Data from experiment_results_summary.log
    results = {
        'ADMM-CVaR (Ours)': {'return': 12.33, 'vol': 13.21, 'sharpe': 0.78, 'max_dd': 11.30, 'cvar': 1.94, 'turnover': 0.0171},
        'Static Equal-Weight': {'return': 16.46, 'vol': 13.02, 'sharpe': 1.11, 'max_dd': 11.30, 'cvar': 1.82, 'turnover': 0.00},
        'Myopic CVaR': {'return': 42.23, 'vol': 26.73, 'sharpe': 1.51, 'max_dd': 14.86, 'cvar': 3.43, 'turnover': 0.1433},
        'Myopic MVO': {'return': 58.14, 'vol': 39.68, 'sharpe': 1.41, 'max_dd': 26.20, 'cvar': 5.32, 'turnover': 0.1085},
        'Boyd Variance': {'return': 64.19, 'vol': 40.22, 'sharpe': 1.55, 'max_dd': 28.05, 'cvar': 5.37, 'turnover': 0.0357},
    }

    latex_table = """
\\begin{table}[t]
\\centering
\\caption{Performance Comparison of Portfolio Optimization Strategies}
\\label{tab:main_results}
\\begin{tabular}{lcccccc}
\\toprule
Strategy & Return & Vol & Sharpe & Max DD & CVaR & Turnover \\\\
 & (\\%) & (\\%) & & (\\%) & (\\%) & \\\\
\\midrule
"""
    for name, metrics in results.items():
        latex_table += f"{name} & {metrics['return']:.2f} & {metrics['vol']:.2f} & {metrics['sharpe']:.2f} & {metrics['max_dd']:.2f} & {metrics['cvar']:.2f} & {metrics['turnover']:.4f} \\\\\n"

    latex_table += """\\bottomrule
\\end{tabular}
\\vspace{0.5em}
\\footnotesize{Note: Return and Vol are annualized. CVaR is daily 95\\% CVaR. Turnover is average daily turnover.}
\\end{table}
"""

    with open(TABLES_DIR / 'main_results_table.tex', 'w') as f:
        f.write(latex_table)
    print(f"Main results table saved to {TABLES_DIR / 'main_results_table.tex'}")


def main():
    """Run all final analyses."""
    print("="*60)
    print("ELEC5470 Final Project - Final Analysis")
    print("="*60)

    # Load data
    data = load_data()

    # 1. Run 2D sensitivity analysis
    results_grid, beta_range, lambda_range = sensitivity_beta_lambda(data)

    # 2. Run return weight sensitivity
    rw_results = sensitivity_return_weight(data)

    # 3. Generate figures
    print("\n" + "="*60)
    print("Generating Publication Figures")
    print("="*60)

    plot_sensitivity_heatmap(results_grid, beta_range, lambda_range)
    plot_sensitivity_3d(results_grid, beta_range, lambda_range)
    plot_pareto_frontier(results_grid, beta_range, lambda_range)

    if rw_results:
        plot_return_weight_sensitivity(rw_results)

    # 4. Generate LaTeX tables
    print("\n" + "="*60)
    print("Generating LaTeX Tables")
    print("="*60)

    generate_latex_tables(results_grid, beta_range, lambda_range, rw_results)
    create_main_results_table()

    print("\n" + "="*60)
    print("Final Analysis Complete!")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
