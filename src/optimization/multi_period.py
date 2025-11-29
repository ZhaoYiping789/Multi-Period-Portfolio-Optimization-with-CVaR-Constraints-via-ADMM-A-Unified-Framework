"""Multi-period optimization wrapper."""

import numpy as np
from typing import Dict, Optional
from ..models.admm import ADMMOptimizer, MPCOptimizer


class MultiPeriodOptimizer:
    """
    Wrapper for multi-period portfolio optimization.

    Supports both full-horizon and MPC approaches.
    """

    def __init__(
        self,
        method: str = "admm",
        horizon: Optional[int] = None,
        **kwargs
    ):
        """
        Parameters
        ----------
        method : str
            'admm' for full-horizon or 'mpc' for model predictive control
        horizon : int, optional
            MPC horizon (required if method='mpc')
        **kwargs
            Additional arguments for optimizer
        """
        self.method = method

        if method == "admm":
            self.optimizer = ADMMOptimizer(**kwargs)
        elif method == "mpc":
            if horizon is None:
                raise ValueError("MPC requires horizon parameter")
            self.optimizer = MPCOptimizer(horizon=horizon, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def optimize(
        self,
        mu: np.ndarray,
        scenarios: np.ndarray,
        w0: np.ndarray,
        **kwargs
    ) -> Dict:
        """
        Run multi-period optimization.

        Parameters
        ----------
        mu : np.ndarray
            Expected returns, shape (T, n)
        scenarios : np.ndarray
            Return scenarios, shape (T, S, n)
        w0 : np.ndarray
            Initial portfolio
        **kwargs
            Additional optimizer arguments

        Returns
        -------
        Dict
            Optimization results
        """
        if self.method == "mpc":
            return self.optimizer.optimize_mpc(mu, scenarios, w0, **kwargs)
        else:
            return self.optimizer.optimize(mu, scenarios, w0, **kwargs)
