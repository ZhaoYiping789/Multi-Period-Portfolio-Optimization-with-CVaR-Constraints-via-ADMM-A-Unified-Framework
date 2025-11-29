"""Backtesting framework for portfolio strategies."""

import numpy as np
import pandas as pd
from typing import Dict, Callable, Optional
from tqdm import tqdm
from .metrics import compute_metrics


class Backtester:
    """
    Backtest portfolio strategies on historical data.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        scenarios: np.ndarray,
        mu: np.ndarray,
        cov: Optional[np.ndarray] = None,
        rf_rate: float = 0.02 / 252,
        lambda_tc: float = 0.005,
    ):
        """
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns, shape (T, n)
        scenarios : np.ndarray
            Return scenarios for optimization, shape (T, S, n)
        mu : np.ndarray
            Expected returns, shape (T, n)
        cov : np.ndarray, optional
            Covariance matrices, shape (T, n, n)
        rf_rate : float
            Daily risk-free rate
        lambda_tc : float
            Transaction cost parameter
        """
        self.returns = returns.values if isinstance(returns, pd.DataFrame) else returns
        self.returns_df = returns if isinstance(returns, pd.DataFrame) else None
        self.scenarios = scenarios
        self.mu = mu
        self.cov = cov
        self.rf_rate = rf_rate
        self.lambda_tc = lambda_tc

        self.T, self.n = self.returns.shape

        # Initial portfolio (equal weight)
        self.w0 = np.ones(self.n) / self.n

    def run_strategy(
        self,
        strategy,
        strategy_name: str = "Strategy",
        **kwargs
    ) -> Dict:
        """
        Run a portfolio strategy and compute metrics.

        Parameters
        ----------
        strategy : object
            Strategy object with optimize() method
        strategy_name : str
            Name for display
        **kwargs
            Additional arguments to pass to strategy

        Returns
        -------
        Dict
            Results including weights and metrics
        """
        print(f"\nRunning {strategy_name}...")

        # Optimize
        opt_kwargs = {
            "mu": self.mu,
            "scenarios": self.scenarios,
            "w0": self.w0,
        }

        if self.cov is not None and hasattr(strategy, "optimize"):
            # Check if strategy needs covariance
            import inspect
            sig = inspect.signature(strategy.optimize)
            if "cov" in sig.parameters:
                opt_kwargs["cov"] = self.cov

        # Merge with additional kwargs
        opt_kwargs.update(kwargs)

        # Run optimization
        result = strategy.optimize(**opt_kwargs)
        weights = result["weights"]

        # Ensure weights is 2D (T, n)
        if weights.ndim == 1:
            weights = np.tile(weights, (self.T, 1))

        # Align dimensions: ensure returns and weights have same length
        T_returns = len(self.returns)
        T_weights = len(weights)

        if T_returns != T_weights:
            # Use the minimum length to ensure alignment
            T_min = min(T_returns, T_weights)
            returns_aligned = self.returns[:T_min]
            weights_aligned = weights[:T_min]
        else:
            returns_aligned = self.returns
            weights_aligned = weights

        # Compute metrics
        metrics = compute_metrics(
            returns_aligned,
            weights_aligned,
            self.rf_rate,
            self.lambda_tc,
        )

        # Add strategy info
        metrics["strategy_name"] = strategy_name
        metrics["weights"] = weights
        metrics["optimization_result"] = result

        return metrics

    def run_multiple_strategies(
        self,
        strategies: Dict[str, object],
    ) -> Dict[str, Dict]:
        """
        Run multiple strategies and compare.

        Parameters
        ----------
        strategies : Dict[str, object]
            Dictionary mapping strategy names to strategy objects

        Returns
        -------
        Dict[str, Dict]
            Results for each strategy
        """
        results = {}

        for name, strategy in strategies.items():
            results[name] = self.run_strategy(strategy, name)

        return results

    def rolling_backtest(
        self,
        strategy,
        window_size: int = 252,
        rebalance_freq: int = 20,
        strategy_name: str = "Strategy",
    ) -> Dict:
        """
        Perform rolling-window backtest with periodic rebalancing.

        Parameters
        ----------
        strategy : object
            Strategy object
        window_size : int
            Training window size
        rebalance_freq : int
            Rebalancing frequency (days)
        strategy_name : str
            Strategy name

        Returns
        -------
        Dict
            Backtest results
        """
        print(f"\nRunning rolling backtest for {strategy_name}...")

        weights_trajectory = []
        w_current = self.w0

        for t in tqdm(range(window_size, self.T, rebalance_freq)):
            # Training data: [t-window_size, t)
            train_mu = self.mu[t - window_size:t]
            train_scenarios = self.scenarios[t - window_size:t]
            train_cov = self.cov[t - window_size:t] if self.cov is not None else None

            # Optimize
            try:
                if train_cov is not None:
                    result = strategy.optimize(
                        mu=train_mu,
                        scenarios=train_scenarios,
                        w0=w_current,
                        cov=train_cov,
                    )
                else:
                    result = strategy.optimize(
                        mu=train_mu,
                        scenarios=train_scenarios,
                        w0=w_current,
                    )

                # Get weights for next rebalancing period
                w_next = result["weights"][0] if result["weights"].ndim == 2 else result["weights"]

            except Exception as e:
                print(f"Error at t={t}: {e}")
                w_next = w_current

            # Hold weights until next rebalancing
            for _ in range(min(rebalance_freq, self.T - t)):
                weights_trajectory.append(w_next)

            w_current = w_next

        weights_trajectory = np.array(weights_trajectory)

        # Compute metrics on test period
        test_returns = self.returns[window_size:window_size + len(weights_trajectory)]
        test_weights = weights_trajectory[:len(test_returns)]

        metrics = compute_metrics(
            test_returns,
            test_weights,
            self.rf_rate,
            self.lambda_tc,
        )

        metrics["strategy_name"] = strategy_name
        metrics["weights"] = test_weights

        return metrics


class WalkForwardAnalysis:
    """
    Walk-forward analysis for out-of-sample testing.
    """

    def __init__(
        self,
        returns: np.ndarray,
        train_size: int = 1260,  # 5 years
        test_size: int = 252,    # 1 year
        step_size: int = 126,    # 6 months
    ):
        """
        Parameters
        ----------
        returns : np.ndarray
            Historical returns
        train_size : int
            Training window size
        test_size : int
            Test window size
        step_size : int
            Step size for rolling window
        """
        self.returns = returns
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size

    def run(
        self,
        strategy,
        strategy_name: str = "Strategy",
    ) -> Dict:
        """
        Perform walk-forward analysis.

        Parameters
        ----------
        strategy : object
            Strategy to test
        strategy_name : str
            Strategy name

        Returns
        -------
        Dict
            Walk-forward results
        """
        T = len(self.returns)
        all_weights = []
        all_returns = []

        for start in tqdm(
            range(0, T - self.train_size - self.test_size, self.step_size),
            desc=f"Walk-forward {strategy_name}",
        ):
            # Train period
            train_end = start + self.train_size
            test_end = min(train_end + self.test_size, T)

            train_returns = self.returns[start:train_end]
            test_returns = self.returns[train_end:test_end]

            # Train strategy
            # (This would require generating scenarios, mu, etc. from train_returns)
            # Simplified version: just collect results

            # TODO: Implement full walk-forward logic

        return {}


def compute_strategy_comparison(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create comparison table of strategies.

    Parameters
    ----------
    results : Dict[str, Dict]
        Results from multiple strategies

    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    from .metrics import create_summary_table
    return create_summary_table(results)
