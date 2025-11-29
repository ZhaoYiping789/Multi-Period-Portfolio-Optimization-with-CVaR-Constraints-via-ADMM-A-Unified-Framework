"""Baseline portfolio optimization methods for comparison."""

import numpy as np
from typing import Dict
from ..optimization.single_period import SinglePeriodCVaROptimizer, SinglePeriodMVO


class StaticPortfolio:
    """
    Buy-and-hold static portfolio (no rebalancing).
    """

    def __init__(self, method: str = "equal_weight"):
        """
        Parameters
        ----------
        method : str
            'equal_weight' or 'market_cap' or 'risk_parity'
        """
        self.method = method

    def optimize(self, mu: np.ndarray, **kwargs) -> Dict:
        """
        Generate static portfolio weights.

        Parameters
        ----------
        mu : np.ndarray
            Expected returns (can be shape (n,) or (T, n))

        Returns
        -------
        Dict
            Contains 'weights' key
        """
        if mu.ndim == 1:
            n = len(mu)
        else:
            T, n = mu.shape

        if self.method == "equal_weight":
            w = np.ones(n) / n

        elif self.method == "risk_parity":
            # Equal risk contribution (simplified: inverse volatility)
            scenarios = kwargs.get("scenarios")
            if scenarios is not None:
                if scenarios.ndim == 3:
                    scenarios = scenarios[0]  # Use first period
                vol = scenarios.std(axis=0)
                w = 1 / (vol + 1e-6)
                w /= w.sum()
            else:
                w = np.ones(n) / n

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Return same weights for all periods if T > 1
        if mu.ndim == 2:
            T = mu.shape[0]
            return {"weights": np.tile(w, (T, 1))}
        else:
            return {"weights": w}


class MyopicOptimizer:
    """
    Myopic (single-period) optimization applied sequentially.

    Solves single-period problem at each time step without considering
    future transaction costs.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.03,
        mu_min: float = 0.0002,
        method: str = "cvar",
    ):
        """
        Parameters
        ----------
        alpha : float
            CVaR confidence level
        beta : float
            CVaR limit
        mu_min : float
            Minimum expected return
        method : str
            'cvar' or 'mvo'
        """
        self.alpha = alpha
        self.beta = beta
        self.mu_min = mu_min
        self.method = method

        if method == "cvar":
            self.optimizer = SinglePeriodCVaROptimizer(alpha, beta, mu_min)
        elif method == "mvo":
            self.optimizer = SinglePeriodMVO(gamma=1.0)
        else:
            raise ValueError(f"Unknown method: {method}")

    def optimize(
        self,
        mu: np.ndarray,
        scenarios: np.ndarray,
        w0: np.ndarray,
        cov: np.ndarray = None,
    ) -> Dict:
        """
        Solve myopic optimization for each period.

        Parameters
        ----------
        mu : np.ndarray
            Expected returns, shape (T, n)
        scenarios : np.ndarray
            Scenarios, shape (T, S, n)
        w0 : np.ndarray
            Initial portfolio
        cov : np.ndarray, optional
            Covariance matrices for MVO, shape (T, n, n)

        Returns
        -------
        Dict
            Contains 'weights' array of shape (T, n)
        """
        T, n = mu.shape
        weights = []

        for t in range(T):
            if self.method == "cvar":
                result = self.optimizer.optimize(mu[t], scenarios[t])
            elif self.method == "mvo":
                if cov is None:
                    raise ValueError("MVO requires covariance matrix")
                result = self.optimizer.optimize(mu[t], cov[t])

            weights.append(result["weights"])

        return {"weights": np.array(weights)}


class BoydVarianceOptimizer:
    """
    Boyd et al. (2017) multi-period optimizer with variance (not CVaR).

    Uses same ADMM framework but with variance instead of CVaR constraints.
    """

    def __init__(
        self,
        lambda_tc: float = 0.005,
        gamma: float = 1.0,
        risk_aversion: float = 1.0,
    ):
        """
        Parameters
        ----------
        lambda_tc : float
            Transaction cost parameter
        gamma : float
            Discount factor
        risk_aversion : float
            Risk aversion coefficient
        """
        self.lambda_tc = lambda_tc
        self.gamma = gamma
        self.risk_aversion = risk_aversion

    def optimize(
        self,
        mu: np.ndarray,
        w0: np.ndarray,
        scenarios: np.ndarray = None,  # Accept but ignore for compatibility
        cov: np.ndarray = None,
    ) -> Dict:
        """
        Solve Boyd's multi-period variance problem.

        This is a simpler QP that can be solved directly.

        Parameters
        ----------
        mu : np.ndarray
            Expected returns, shape (T, n)
        cov : np.ndarray
            Covariance matrices, shape (T, n, n)
        w0 : np.ndarray
            Initial portfolio

        Returns
        -------
        Dict
            Contains 'weights'
        """
        import cvxpy as cp

        T, n = mu.shape

        # Decision variables
        w = cp.Variable((T, n))

        # Objective: transaction costs + variance penalty - expected return
        tc_cost = (self.lambda_tc / 2) * cp.sum_squares(w[0] - w0)
        for t in range(1, T):
            tc_cost += self.gamma ** (t - 1) * (self.lambda_tc / 2) * cp.sum_squares(w[t] - w[t - 1])

        # Variance and return
        variance_penalty = 0
        expected_return = 0
        for t in range(T):
            variance_penalty += self.risk_aversion * cp.quad_form(w[t], cov[t])
            expected_return += mu[t] @ w[t]

        objective = cp.Minimize(tc_cost + variance_penalty - expected_return)

        # Constraints
        constraints = []
        for t in range(T):
            constraints.append(cp.sum(w[t]) == 1)
            constraints.append(w[t] >= 0)

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, verbose=False)

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"Warning: Boyd solver status = {problem.status}")
            return {"weights": np.tile(w0, (T, 1))}

        return {"weights": w.value}


class MinVariancePortfolio:
    """
    Minimum variance portfolio (Markowitz with no expected return).
    """

    def __init__(self):
        pass

    def optimize(self, cov: np.ndarray) -> Dict:
        """
        Solve minimum variance problem.

        Parameters
        ----------
        cov : np.ndarray
            Covariance matrix, shape (n, n) or (T, n, n)

        Returns
        -------
        Dict
            Contains 'weights'
        """
        import cvxpy as cp

        if cov.ndim == 2:
            n = cov.shape[0]
            w = cp.Variable(n)

            objective = cp.Minimize(cp.quad_form(w, cov))
            constraints = [cp.sum(w) == 1, w >= 0]

            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.OSQP, verbose=False)

            return {"weights": w.value}

        else:
            # Time-varying covariance
            T, n, _ = cov.shape
            weights = []

            for t in range(T):
                result = self.optimize(cov[t])
                weights.append(result["weights"])

            return {"weights": np.array(weights)}


class NaiveDiversification:
    """
    Naive 1/N portfolio (equal weight).
    """

    def optimize(self, n: int, T: int = 1) -> Dict:
        """
        Generate equal-weight portfolio.

        Parameters
        ----------
        n : int
            Number of assets
        T : int
            Number of periods

        Returns
        -------
        Dict
            Contains 'weights'
        """
        w = np.ones(n) / n

        if T > 1:
            return {"weights": np.tile(w, (T, 1))}
        else:
            return {"weights": w}
