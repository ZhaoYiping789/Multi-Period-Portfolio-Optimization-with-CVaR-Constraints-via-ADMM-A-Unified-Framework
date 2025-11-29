"""Configuration for experiments."""

import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)

# Data parameters
DATA_CONFIG = {
    "start_date": "2017-01-01",
    "end_date": "2025-01-01",
    "n_assets": 50,
    "train_end": "2022-12-31",
    "cache_file": str(DATA_DIR / "sp500_data.csv"),
}

# Model parameters
MODEL_CONFIG = {
    "alpha": 0.05,                  # CVaR confidence level (95% CVaR)
    "beta": 0.07,                   # CVaR limit (7% daily loss) - RELAXED from 0.03
    "lambda_tc": 0.002,             # Transaction cost parameter - REDUCED from 0.005
    "mu_min": 0.0002,               # Minimum expected return (2 bps daily)
    "gamma": 1.0,                   # Discount factor
    "scenario_window": 252,         # Scenario generation window
    "scenario_method": "historical", # 'historical' or 'parametric'
    "return_weight": 2.0,           # Weight for return maximization - INCREASED from 1.0
    "lambda_herf": 0.0,             # Concentration penalty (Herfindahl index) - DISABLED
    "w_max": 1.0,                   # Maximum weight per asset (100% = no constraint)
}

# ADMM parameters
ADMM_CONFIG = {
    "rho": 1.0,                     # Initial penalty parameter
    "max_iter": 100,                # Maximum iterations (back to original)
    "tol_primal": 1e-4,             # Primal residual tolerance
    "tol_dual": 1e-4,               # Dual residual tolerance
    "adaptive_rho": True,           # Adaptive rho updates
    "n_jobs": -1,                   # Parallel workers (-1 = all cores)
    "verbose": True,
}

# MPC parameters
MPC_CONFIG = {
    "horizon": 20,                  # Rolling horizon (20 days)
}

# Baseline parameters
BASELINE_CONFIG = {
    "mvo_gamma": 1.0,               # Mean-variance risk aversion
    "boyd_risk_aversion": 1.0,      # Boyd method risk aversion
}

# Sensitivity analysis ranges
SENSITIVITY_CONFIG = {
    "alpha_range": [0.01, 0.05, 0.10],
    "beta_range": [0.02, 0.03, 0.05],
    "lambda_tc_range": [0.001, 0.005, 0.01],
    "horizon_range": [5, 10, 20, 40],
    "n_assets_range": [20, 50, 100],
}

# Plotting parameters
PLOT_CONFIG = {
    "figsize_default": (12, 6),
    "figsize_large": (14, 8),
    "dpi": 300,
    "style": "whitegrid",
}

# Risk-free rate
RF_RATE = 0.02 / 252  # 2% annual -> daily

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
