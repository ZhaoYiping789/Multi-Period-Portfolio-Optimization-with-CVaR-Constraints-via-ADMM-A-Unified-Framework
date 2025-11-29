"""ADMM algorithm for multi-period portfolio optimization with CVaR constraints."""

import numpy as np
from typing import Dict, Optional, Callable
from tqdm import tqdm
from joblib import Parallel, delayed


class ADMMOptimizer:
    """
    ADMM-based multi-period portfolio optimizer.

    Solves:
        min  Σ_{t=1}^T γ^{t-1} (λ/2)||w_t - w_{t-1}||²
        s.t. CVaR_α(-w_t^T r_t) ≤ β_t, ∀t
             1^T w_t = 1, w_t ≥ 0, ∀t

    Using ADMM decomposition:
        w-update: parallel CVaR subproblems
        u-update: tridiagonal system for transaction costs
        y-update: dual variable update
    """

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.03,
        lambda_tc: float = 0.005,
        gamma: float = 1.0,
        rho: float = 1.0,
        max_iter: int = 100,
        tol_primal: float = 1e-4,
        tol_dual: float = 1e-4,
        adaptive_rho: bool = True,
        n_jobs: int = 1,
        verbose: bool = True,
        return_weight: float = 1.0,
        lambda_herf: float = 0.0,
        w_max: float = 1.0,
    ):
        """
        Parameters
        ----------
        alpha : float
            CVaR confidence level
        beta : float
            CVaR limit
        lambda_tc : float
            Transaction cost parameter
        gamma : float
            Discount factor
        rho : float
            ADMM penalty parameter
        max_iter : int
            Maximum ADMM iterations
        tol_primal : float
            Primal residual tolerance
        tol_dual : float
            Dual residual tolerance
        adaptive_rho : bool
            Whether to adaptively update rho
        n_jobs : int
            Number of parallel workers (-1 for all cores)
        verbose : bool
            Print iteration info
        return_weight : float
            Weight for return maximization in objective (higher = more return focus)
        lambda_herf : float
            Concentration penalty parameter (Herfindahl index)
        w_max : float
            Maximum weight per asset
        """
        self.alpha = alpha
        self.beta = beta
        self.lambda_tc = lambda_tc
        self.gamma = gamma
        self.rho = rho
        self.max_iter = max_iter
        self.tol_primal = tol_primal
        self.tol_dual = tol_dual
        self.adaptive_rho = adaptive_rho
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.return_weight = return_weight
        self.lambda_herf = lambda_herf
        self.w_max = w_max

    def optimize(
        self,
        mu: np.ndarray,
        scenarios: np.ndarray,
        w0: np.ndarray,
        beta_t: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Run ADMM optimization.

        Parameters
        ----------
        mu : np.ndarray
            Expected returns, shape (T, n)
        scenarios : np.ndarray
            Return scenarios, shape (T, S, n)
        w0 : np.ndarray
            Initial portfolio, shape (n,)
        beta_t : np.ndarray, optional
            Time-varying CVaR limits, shape (T,)

        Returns
        -------
        Dict
            Optimization results
        """
        # Lazy import to avoid circular dependency
        from ..optimization.single_period import optimize_with_tc_penalty

        T, n = mu.shape
        S = scenarios.shape[1]

        if beta_t is None:
            beta_t = np.full(T, self.beta)

        # Initialize variables
        w = np.tile(w0, (T, 1))  # (T, n)
        u = np.tile(w0, (T, 1))  # (T, n)
        y = np.zeros((T, n))     # (T, n)

        # History
        history = {
            "primal_residuals": [],
            "dual_residuals": [],
            "objectives": [],
            "rho_values": [self.rho],
        }

        pbar = tqdm(range(self.max_iter), desc="ADMM", disable=not self.verbose)

        for iteration in pbar:
            # === Step 1: w-update (parallel over t) ===
            w_old = w.copy()

            if self.n_jobs == 1:
                # Sequential
                w_new = []
                for t in range(T):
                    u_bar = u[t] - y[t] / self.rho
                    result = optimize_with_tc_penalty(
                        mu[t],
                        scenarios[t],
                        w[t],
                        self.alpha,
                        beta_t[t],
                        self.lambda_tc,
                        self.rho,
                        u_bar,
                        self.return_weight,
                        self.lambda_herf,
                        self.w_max,
                    )
                    w_new.append(result["weights"])
                w = np.array(w_new)
            else:
                # Parallel
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(optimize_with_tc_penalty)(
                        mu[t],
                        scenarios[t],
                        w[t],
                        self.alpha,
                        beta_t[t],
                        self.lambda_tc,
                        self.rho,
                        u[t] - y[t] / self.rho,
                        self.return_weight,
                        self.lambda_herf,
                        self.w_max,
                    )
                    for t in range(T)
                )
                w = np.array([r["weights"] for r in results])

            # === Step 2: u-update (tridiagonal system) ===
            u_old = u.copy()
            u = self._update_u(w, y, w0)

            # === Step 3: y-update (dual) ===
            y = y + self.rho * (w - u)

            # === Compute residuals ===
            primal_res = np.linalg.norm(w - u)
            dual_res = self.rho * np.linalg.norm(u - u_old) if iteration > 0 else 0

            history["primal_residuals"].append(primal_res)
            history["dual_residuals"].append(dual_res)

            # Objective value
            obj = self._compute_objective(w, w0)
            history["objectives"].append(obj)

            # Update progress bar
            pbar.set_postfix({
                "primal": f"{primal_res:.2e}",
                "dual": f"{dual_res:.2e}",
                "obj": f"{obj:.4f}",
                "rho": f"{self.rho:.2e}",
            })

            # === Check convergence ===
            if primal_res < self.tol_primal and dual_res < self.tol_dual:
                if self.verbose:
                    print(f"\nConverged at iteration {iteration}")
                break

            # === Adaptive rho update ===
            if self.adaptive_rho and iteration > 0:
                if primal_res > 10 * dual_res:
                    self.rho *= 2
                    y /= 2
                elif dual_res > 10 * primal_res:
                    self.rho /= 2
                    y *= 2
                history["rho_values"].append(self.rho)

        u_old = u.copy() if iteration > 0 else np.tile(w0, (T, 1))

        return {
            "weights": w,
            "u": u,
            "y": y,
            "history": history,
            "converged": primal_res < self.tol_primal and dual_res < self.tol_dual,
            "iterations": iteration + 1,
            "objective": obj,
        }

    def _update_u(self, w: np.ndarray, y: np.ndarray, w0: np.ndarray) -> np.ndarray:
        """
        Solve u-update via tridiagonal system.

        Solves: (λ + ρ) u_t - λ u_{t+1} - λ u_{t-1} = ρ ŵ_t

        where ŵ_t = w_t + y_t / ρ
        """
        T, n = w.shape
        w_tilde = w + y / self.rho

        # Solve for each asset dimension independently
        u = np.zeros((T, n))

        for i in range(n):
            # Build tridiagonal system
            diag = np.full(T, self.lambda_tc + self.rho)
            off_diag = np.full(T - 1, -self.lambda_tc)

            # Right-hand side
            rhs = self.rho * w_tilde[:, i]
            rhs[0] += self.lambda_tc * w0[i]  # Boundary condition

            # Solve tridiagonal system using Thomas algorithm
            u[:, i] = self._solve_tridiagonal(diag, off_diag, off_diag, rhs)

        return u

    def _solve_tridiagonal(
        self,
        diag: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        rhs: np.ndarray,
    ) -> np.ndarray:
        """
        Solve tridiagonal system Ax = b using Thomas algorithm.

        Parameters
        ----------
        diag : np.ndarray
            Main diagonal, length n
        lower : np.ndarray
            Lower diagonal, length n-1
        upper : np.ndarray
            Upper diagonal, length n-1
        rhs : np.ndarray
            Right-hand side, length n

        Returns
        -------
        np.ndarray
            Solution x
        """
        n = len(diag)
        c_star = np.zeros(n - 1)
        d_star = np.zeros(n)

        # Forward sweep
        c_star[0] = upper[0] / diag[0]
        d_star[0] = rhs[0] / diag[0]

        for i in range(1, n - 1):
            denom = diag[i] - lower[i - 1] * c_star[i - 1]
            c_star[i] = upper[i] / denom
            d_star[i] = (rhs[i] - lower[i - 1] * d_star[i - 1]) / denom

        d_star[n - 1] = (rhs[n - 1] - lower[n - 2] * d_star[n - 2]) / (
            diag[n - 1] - lower[n - 2] * c_star[n - 2]
        )

        # Back substitution
        x = np.zeros(n)
        x[n - 1] = d_star[n - 1]

        for i in range(n - 2, -1, -1):
            x[i] = d_star[i] - c_star[i] * x[i + 1]

        return x

    def _compute_objective(self, w: np.ndarray, w0: np.ndarray) -> float:
        """Compute objective value (total transaction cost)."""
        T = len(w)
        obj = 0.0

        # First period cost
        obj += (self.lambda_tc / 2) * np.linalg.norm(w[0] - w0) ** 2

        # Subsequent periods
        for t in range(1, T):
            obj += self.gamma ** (t - 1) * (self.lambda_tc / 2) * np.linalg.norm(w[t] - w[t - 1]) ** 2

        return obj


class MPCOptimizer(ADMMOptimizer):
    """
    Model Predictive Control (MPC) extension of ADMM optimizer.

    Solves rolling-horizon optimization problems.
    """

    def __init__(self, horizon: int = 20, **kwargs):
        """
        Parameters
        ----------
        horizon : int
            MPC rolling horizon length
        **kwargs
            Arguments for ADMMOptimizer
        """
        super().__init__(**kwargs)
        self.horizon = horizon

    def optimize_mpc(
        self,
        mu: np.ndarray,
        scenarios: np.ndarray,
        w0: np.ndarray,
    ) -> Dict:
        """
        Run MPC optimization (rolling horizon).

        Parameters
        ----------
        mu : np.ndarray
            Expected returns, shape (T_total, n)
        scenarios : np.ndarray
            Scenarios, shape (T_total, S, n)
        w0 : np.ndarray
            Initial portfolio

        Returns
        -------
        Dict
            MPC results with full trajectory
        """
        T_total = len(mu)
        n = len(w0)

        w_trajectory = []
        w_current = w0

        for t in tqdm(range(T_total), desc="MPC", disable=not self.verbose):
            # Determine horizon end
            t_end = min(t + self.horizon, T_total)
            H = t_end - t

            if H == 0:
                break

            # Solve over horizon [t, t+H)
            result = self.optimize(
                mu[t:t_end],
                scenarios[t:t_end],
                w_current,
            )

            # Execute first action only
            w_next = result["weights"][0]
            w_trajectory.append(w_next)

            # Update current portfolio
            w_current = w_next

        w_trajectory = np.array(w_trajectory)

        return {
            "weights": w_trajectory,
            "converged": True,
        }
