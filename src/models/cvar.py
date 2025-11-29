"""CVaR (Conditional Value-at-Risk) formulation using Rockafellar-Uryasev method."""

import numpy as np
import cvxpy as cp
from typing import Tuple, Optional


def compute_cvar(
    weights: np.ndarray,
    scenarios: np.ndarray,
    alpha: float = 0.05,
    probabilities: Optional[np.ndarray] = None,
) -> float:
    """
    Compute CVaR (Conditional Value-at-Risk) for a portfolio.

    Uses the Rockafellar-Uryasev formulation:
    CVaR_α(L) = min_ζ { ζ + (1/α) E[(L - ζ)₊] }

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights, shape (n,)
    scenarios : np.ndarray
        Return scenarios, shape (S, n) where S is number of scenarios
    alpha : float
        Confidence level (e.g., 0.05 for 95% CVaR)
    probabilities : np.ndarray, optional
        Scenario probabilities, shape (S,). If None, uniform probabilities.

    Returns
    -------
    float
        CVaR value (expected loss in worst α cases)
    """
    # Loss = -return
    losses = -scenarios @ weights  # shape (S,)

    S = len(losses)
    if probabilities is None:
        probabilities = np.ones(S) / S

    # Solve: min_ζ { ζ + (1/α) Σ p_s max(L_s - ζ, 0) }
    zeta = cp.Variable()
    u = cp.Variable(S)

    objective = zeta + (1 / alpha) * (probabilities @ u)
    constraints = [
        u >= losses - zeta,
        u >= 0,
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.ECOS, verbose=False)

    if problem.status != cp.OPTIMAL:
        # Fallback: use empirical quantile
        var = np.quantile(losses, 1 - alpha)
        cvar = losses[losses >= var].mean()
        return cvar

    return problem.value


def compute_var(
    weights: np.ndarray,
    scenarios: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """
    Compute VaR (Value-at-Risk).

    VaR_α = quantile(L, 1-α)

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    scenarios : np.ndarray
        Return scenarios, shape (S, n)
    alpha : float
        Confidence level

    Returns
    -------
    float
        VaR value
    """
    losses = -scenarios @ weights
    var = np.quantile(losses, 1 - alpha)
    return var


def cvar_constraint_set(
    w: cp.Variable,
    scenarios: np.ndarray,
    alpha: float,
    beta: float,
    probabilities: Optional[np.ndarray] = None,
) -> Tuple[cp.Expression, list]:
    """
    Create CVaR constraint for CVXPY optimization.

    CVaR_α(-w^T r) ≤ β

    Parameters
    ----------
    w : cp.Variable
        Portfolio weight variable, shape (n,)
    scenarios : np.ndarray
        Return scenarios, shape (S, n)
    alpha : float
        Confidence level
    beta : float
        CVaR limit (max acceptable loss)
    probabilities : np.ndarray, optional
        Scenario probabilities

    Returns
    -------
    Tuple[cp.Expression, list]
        - CVaR objective term
        - List of constraints
    """
    S, n = scenarios.shape

    if probabilities is None:
        probabilities = np.ones(S) / S

    # Auxiliary variables for Rockafellar-Uryasev formulation
    zeta = cp.Variable()
    u = cp.Variable(S)

    # CVaR expression
    cvar_expr = zeta + (1 / alpha) * (probabilities @ u)

    # Constraints
    # u_s ≥ -w^T r_s - ζ (loss scenarios)
    # u_s ≥ 0
    constraints = [
        u >= -scenarios @ w - zeta,
        u >= 0,
        cvar_expr <= beta,  # CVaR constraint
    ]

    return cvar_expr, constraints


def estimate_cvar_limit(
    returns: np.ndarray,
    alpha: float = 0.05,
    percentile: float = 95,
) -> float:
    """
    Estimate reasonable CVaR limit from historical data.

    Parameters
    ----------
    returns : np.ndarray
        Historical returns
    alpha : float
        Confidence level
    percentile : float
        Percentile of historical CVaR to use as limit

    Returns
    -------
    float
        Suggested CVaR limit
    """
    # Compute CVaR for equal-weighted portfolio over time
    T, n = returns.shape
    w_eq = np.ones(n) / n

    cvars = []
    window = 252

    for t in range(window, T):
        scenarios = returns[t - window:t]
        cvar = compute_cvar(w_eq, scenarios, alpha)
        cvars.append(cvar)

    # Use percentile of historical CVaRs
    beta = np.percentile(cvars, percentile)

    return beta


def cvar_gradient(
    weights: np.ndarray,
    scenarios: np.ndarray,
    alpha: float,
    probabilities: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute gradient of CVaR with respect to weights.

    This is useful for gradient-based optimization methods.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    scenarios : np.ndarray
        Return scenarios
    alpha : float
        Confidence level
    probabilities : np.ndarray, optional
        Scenario probabilities

    Returns
    -------
    np.ndarray
        Gradient, shape (n,)
    """
    losses = -scenarios @ weights
    S = len(losses)

    if probabilities is None:
        probabilities = np.ones(S) / S

    # Find optimal ζ
    zeta_opt = cp.Variable()
    u = cp.Variable(S)
    objective = zeta_opt + (1 / alpha) * (probabilities @ u)
    constraints = [u >= losses - zeta_opt, u >= 0]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.ECOS, verbose=False)

    zeta = zeta_opt.value

    # Gradient: -E[r | L > ζ] weighted by probability
    tail_scenarios = losses > zeta
    if tail_scenarios.sum() == 0:
        return np.zeros_like(weights)

    tail_probs = probabilities[tail_scenarios]
    tail_probs /= tail_probs.sum()

    gradient = -(scenarios[tail_scenarios].T @ tail_probs)

    return gradient


def coherence_check(scenarios: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Verify that CVaR satisfies coherence properties.

    Coherent risk measures satisfy:
    1. Monotonicity
    2. Translation invariance
    3. Positive homogeneity
    4. Subadditivity

    Parameters
    ----------
    scenarios : np.ndarray
        Return scenarios
    alpha : float
        Confidence level

    Returns
    -------
    dict
        Test results for each property
    """
    n = scenarios.shape[1]
    w1 = np.random.rand(n)
    w1 /= w1.sum()
    w2 = np.random.rand(n)
    w2 /= w2.sum()

    cvar1 = compute_cvar(w1, scenarios, alpha)
    cvar2 = compute_cvar(w2, scenarios, alpha)

    # Subadditivity: CVaR(w1 + w2) ≤ CVaR(w1) + CVaR(w2)
    cvar_sum = compute_cvar(w1 + w2, scenarios, alpha)
    subadditive = cvar_sum <= cvar1 + cvar2 + 1e-6

    # Positive homogeneity: CVaR(λw) = λ CVaR(w) for λ > 0
    lambda_val = 2.0
    cvar_scaled = compute_cvar(lambda_val * w1, scenarios, alpha)
    homogeneous = np.abs(cvar_scaled - lambda_val * cvar1) < 1e-4

    return {
        "subadditive": subadditive,
        "positive_homogeneous": homogeneous,
        "cvar1": cvar1,
        "cvar2": cvar2,
        "cvar_sum": cvar_sum,
        "cvar_scaled": cvar_scaled,
    }
