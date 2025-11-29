"""Data preprocessing and scenario generation."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """
    Compute returns from prices.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data, shape (T, n)
    method : str
        'log' for log returns or 'simple' for simple returns

    Returns
    -------
    pd.DataFrame
        Returns, shape (T-1, n)
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown method: {method}")

    return returns.dropna()


def generate_scenarios(
    returns: pd.DataFrame,
    window_size: int = 252,
    method: str = "historical",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate return scenarios for CVaR calculation.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns, shape (T, n)
    window_size : int
        Rolling window size for scenarios
    method : str
        'historical' or 'parametric'

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - scenarios: shape (T, S, n) where S = window_size
        - probabilities: shape (S,), uniform by default
    """
    T, n = returns.shape

    if method == "historical":
        # Use rolling historical window
        scenarios_list = []

        for t in range(window_size, T):
            # Take past window_size days as scenarios
            hist_scenarios = returns.iloc[t - window_size:t].values  # (S, n)
            scenarios_list.append(hist_scenarios)

        scenarios = np.array(scenarios_list)  # (T - window_size, S, n)
        S = window_size
        probabilities = np.ones(S) / S  # Uniform

    elif method == "parametric":
        # Gaussian scenarios based on estimated mean and covariance
        scenarios_list = []
        S = 1000  # Number of Monte Carlo samples

        for t in range(window_size, T):
            hist_data = returns.iloc[t - window_size:t]
            mu = hist_data.mean().values
            cov = hist_data.cov().values

            # Sample from multivariate normal
            sampled = np.random.multivariate_normal(mu, cov, size=S)
            scenarios_list.append(sampled)

        scenarios = np.array(scenarios_list)  # (T - window_size, S, n)
        probabilities = np.ones(S) / S

    else:
        raise ValueError(f"Unknown method: {method}")

    return scenarios, probabilities


def estimate_expected_returns(
    returns: pd.DataFrame,
    window_size: int = 252,
    method: str = "mean",
    decay: float = 0.01,
) -> pd.DataFrame:
    """
    Estimate expected returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns
    window_size : int
        Rolling window size
    method : str
        'mean' for simple average or 'ewma' for exponential weighting
    decay : float
        Decay parameter for EWMA

    Returns
    -------
    pd.DataFrame
        Expected returns, shape (T - window_size, n)
    """
    T, n = returns.shape
    expected_returns = []

    for t in range(window_size, T):
        hist_data = returns.iloc[t - window_size:t]

        if method == "mean":
            mu = hist_data.mean()
        elif method == "ewma":
            # Exponentially weighted moving average
            weights = np.exp(-decay * np.arange(window_size)[::-1])
            weights /= weights.sum()
            mu = (hist_data.values * weights[:, None]).sum(axis=0)
            mu = pd.Series(mu, index=returns.columns)
        else:
            raise ValueError(f"Unknown method: {method}")

        expected_returns.append(mu)

    return pd.DataFrame(expected_returns, index=returns.index[window_size:])


def estimate_covariance(
    returns: pd.DataFrame,
    window_size: int = 252,
    method: str = "sample",
    shrinkage: float = 0.1,
) -> np.ndarray:
    """
    Estimate covariance matrix with optional shrinkage.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns
    window_size : int
        Rolling window size
    method : str
        'sample' or 'ledoit_wolf'
    shrinkage : float
        Shrinkage intensity (0 = sample cov, 1 = identity)

    Returns
    -------
    np.ndarray
        Covariance matrix, shape (T - window_size, n, n)
    """
    from sklearn.covariance import LedoitWolf

    T, n = returns.shape
    cov_matrices = []

    for t in range(window_size, T):
        hist_data = returns.iloc[t - window_size:t].values

        if method == "sample":
            cov = np.cov(hist_data, rowvar=False)

            # Apply shrinkage towards identity
            if shrinkage > 0:
                target = np.eye(n) * np.trace(cov) / n
                cov = (1 - shrinkage) * cov + shrinkage * target

        elif method == "ledoit_wolf":
            lw = LedoitWolf()
            lw.fit(hist_data)
            cov = lw.covariance_

        else:
            raise ValueError(f"Unknown method: {method}")

        cov_matrices.append(cov)

    return np.array(cov_matrices)


def split_train_test(
    data: pd.DataFrame,
    train_end: str = "2022-12-31",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset
    train_end : str
        End date of training set

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Training and testing sets
    """
    train = data.loc[:train_end]
    test = data.loc[train_end:]

    # Remove first day of test (same as last day of train)
    if len(test) > 1:
        test = test.iloc[1:]

    print(f"Train: {train.index[0]} to {train.index[-1]} ({len(train)} days)")
    print(f"Test:  {test.index[0]} to {test.index[-1]} ({len(test)} days)")

    return train, test


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalize portfolio weights to sum to 1.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights

    Returns
    -------
    np.ndarray
        Normalized weights
    """
    return weights / weights.sum()


def handle_missing_data(data: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """
    Handle missing data in returns/prices.

    Parameters
    ----------
    data : pd.DataFrame
        Data with potential missing values
    method : str
        'ffill' (forward fill), 'drop', or 'interpolate'

    Returns
    -------
    pd.DataFrame
        Data with missing values handled
    """
    if method == "ffill":
        return data.fillna(method='ffill').fillna(method='bfill')
    elif method == "drop":
        return data.dropna()
    elif method == "interpolate":
        return data.interpolate(method='linear')
    else:
        raise ValueError(f"Unknown method: {method}")
