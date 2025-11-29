"""Single-period CVaR portfolio optimization."""

import numpy as np
import cvxpy as cp
from typing import Optional, Dict
from ..models.cvar import cvar_constraint_set


class SinglePeriodCVaROptimizer:
    """
    Single-period portfolio optimizer with CVaR constraints.

    This solves:
        min  -(μ^T w)  [maximize expected return]
        s.t. 1^T w = 1                [budget]
             w ≥ 0                     [long-only]
             μ^T w ≥ μ_min             [minimum return]
             CVaR_α(-w^T r) ≤ β        [CVaR constraint]
    """

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.03,
        mu_min: float = 0.0002,
        solver: str = "ECOS",
    ):
        """
        Parameters
        ----------
        alpha : float
            CVaR confidence level (e.g., 0.05 for 95% CVaR)
        beta : float
            CVaR limit (max acceptable loss, e.g., 0.03 for 3%)
        mu_min : float
            Minimum expected return (e.g., 0.0002 for 2bps daily)
        solver : str
            CVXPY solver to use
        """
        self.alpha = alpha
        self.beta = beta
        self.mu_min = mu_min
        self.solver = solver

    def optimize(
        self,
        mu: np.ndarray,
        scenarios: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Solve single-period CVaR optimization.

        Parameters
        ----------
        mu : np.ndarray
            Expected returns, shape (n,)
        scenarios : np.ndarray
            Return scenarios, shape (S, n)
        probabilities : np.ndarray, optional
            Scenario probabilities

        Returns
        -------
        Dict
            Optimization results containing:
            - weights: optimal portfolio weights
            - cvar: realized CVaR
            - expected_return: expected return
            - status: solver status
            - solve_time: computation time
        """
        n = len(mu)
        S = len(scenarios)

        if probabilities is None:
            probabilities = np.ones(S) / S

        # Decision variable
        w = cp.Variable(n)

        # Objective: maximize expected return = minimize -μ^T w
        objective = cp.Minimize(-mu @ w)

        # CVaR constraint
        cvar_expr, cvar_constraints = cvar_constraint_set(
            w, scenarios, self.alpha, self.beta, probabilities
        )

        # All constraints
        constraints = [
            cp.sum(w) == 1,          # budget
            w >= 0,                   # long-only
            mu @ w >= self.mu_min,   # minimum return
        ] + cvar_constraints

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver, verbose=False)

        # Extract results
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"Warning: Solver status = {problem.status}")
            # Return equal weights as fallback
            return {
                "weights": np.ones(n) / n,
                "cvar": np.nan,
                "expected_return": np.nan,
                "status": problem.status,
                "solve_time": problem.solver_stats.solve_time if problem.solver_stats else None,
            }

        weights_opt = w.value
        weights_opt = np.clip(weights_opt, 0, 1)  # Numerical safety
        weights_opt /= weights_opt.sum()           # Renormalize

        return {
            "weights": weights_opt,
            "cvar": cvar_expr.value,
            "expected_return": mu @ weights_opt,
            "status": problem.status,
            "solve_time": problem.solver_stats.solve_time if problem.solver_stats else None,
        }


class SinglePeriodMVO:
    """
    Single-period Mean-Variance Optimization (Markowitz).

    This solves:
        min  w^T Σ w - γ μ^T w
        s.t. 1^T w = 1
             w ≥ 0
    """

    def __init__(self, gamma: float = 1.0, solver: str = "OSQP"):
        """
        Parameters
        ----------
        gamma : float
            Risk aversion parameter (higher = more risk-averse)
        solver : str
            CVXPY solver
        """
        self.gamma = gamma
        self.solver = solver

    def optimize(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
    ) -> Dict:
        """
        Solve mean-variance optimization.

        Parameters
        ----------
        mu : np.ndarray
            Expected returns
        cov : np.ndarray
            Covariance matrix

        Returns
        -------
        Dict
            Optimization results
        """
        n = len(mu)
        w = cp.Variable(n)

        # Objective: minimize risk - γ * return
        risk = cp.quad_form(w, cov)
        objective = cp.Minimize(risk - self.gamma * mu @ w)

        constraints = [
            cp.sum(w) == 1,
            w >= 0,
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver, verbose=False)

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return {
                "weights": np.ones(n) / n,
                "variance": np.nan,
                "expected_return": np.nan,
                "status": problem.status,
            }

        weights_opt = w.value
        weights_opt = np.clip(weights_opt, 0, 1)
        weights_opt /= weights_opt.sum()

        return {
            "weights": weights_opt,
            "variance": weights_opt @ cov @ weights_opt,
            "expected_return": mu @ weights_opt,
            "status": problem.status,
        }


def optimize_with_tc_penalty(
    mu: np.ndarray,
    scenarios: np.ndarray,
    w_prev: np.ndarray,
    alpha: float = 0.05,
    beta: float = 0.03,
    lambda_tc: float = 0.005,
    rho: float = 1.0,
    u_bar: Optional[np.ndarray] = None,
    return_weight: float = 1.0,
    lambda_herf: float = 0.0,
    w_max: float = 1.0,
) -> Dict:
    """
    Single-period optimization with transaction cost penalty (for ADMM).

    This is the w-update step in ADMM:
        min  -γ·μ^T w + (ρ/2) ||w - û||² + λ_herf ||w||²
        s.t. CVaR constraints, budget, long-only, w ≤ w_max, etc.

    where û = u - y/ρ from ADMM.

    Parameters
    ----------
    mu : np.ndarray
        Expected returns
    scenarios : np.ndarray
        Return scenarios
    w_prev : np.ndarray
        Previous weights
    alpha : float
        CVaR confidence level
    beta : float
        CVaR limit
    lambda_tc : float
        Transaction cost parameter (not used in this subproblem)
    rho : float
        ADMM penalty parameter
    u_bar : np.ndarray, optional
        ADMM consensus variable û = u - y/ρ
    return_weight : float
        Weight for return maximization term (γ in objective)
    lambda_herf : float
        Concentration penalty (Herfindahl index) - penalizes sum of squared weights
    w_max : float
        Maximum weight per asset (e.g., 0.15 for 15% max)

    Returns
    -------
    Dict
        Optimization results
    """
    n = len(mu)
    S = len(scenarios)
    probabilities = np.ones(S) / S

    if u_bar is None:
        u_bar = w_prev

    # Decision variable
    w = cp.Variable(n)

    # Objective: -γ·μ^T w + (ρ/2) ||w - û||² + λ_herf ||w||²
    # This balances return maximization with ADMM consensus and concentration penalty
    objective_terms = [
        -return_weight * (mu @ w),
        (rho / 2) * cp.sum_squares(w - u_bar)
    ]

    # Add concentration penalty (Herfindahl index)
    if lambda_herf > 0:
        objective_terms.append(lambda_herf * cp.sum_squares(w))

    objective = cp.Minimize(cp.sum(objective_terms))

    # CVaR constraint
    cvar_expr, cvar_constraints = cvar_constraint_set(
        w, scenarios, alpha, beta, probabilities
    )

    # Constraints
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= w_max,  # Maximum weight per asset
        mu @ w >= 0.0001,  # Minimum return (can be parameterized)
    ] + cvar_constraints

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS, verbose=False)

    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        # Fallback: project u_bar onto feasible set
        w_opt = np.clip(u_bar, 0, 1)
        w_opt /= w_opt.sum()
    else:
        w_opt = w.value
        w_opt = np.clip(w_opt, 0, 1)
        w_opt /= w_opt.sum()

    return {
        "weights": w_opt,
        "cvar": cvar_expr.value if problem.status == cp.OPTIMAL else np.nan,
        "status": problem.status,
    }
