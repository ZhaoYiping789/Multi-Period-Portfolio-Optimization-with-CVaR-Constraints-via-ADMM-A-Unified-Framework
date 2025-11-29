"""Data loading from Yahoo Finance."""

import pandas as pd
import yfinance as yf
import numpy as np
from typing import List, Optional
from tqdm import tqdm


def load_sp500_data(
    start_date: str = "2017-01-01",
    end_date: str = "2025-01-01",
    n_assets: int = 50,
    cache_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load S&P 500 stock data from Yahoo Finance.

    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    n_assets : int
        Number of assets to select (most liquid)
    cache_file : str, optional
        Path to cache file to avoid re-downloading

    Returns
    -------
    pd.DataFrame
        DataFrame with adjusted close prices, shape (T, n_assets)
    """
    if cache_file and pd.io.common.file_exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    # Get S&P 500 tickers
    tickers = get_sp500_tickers()

    # Select most liquid stocks
    selected_tickers = select_liquid_stocks(tickers, n_assets, start_date, end_date)

    # Download data
    print(f"Downloading data for {len(selected_tickers)} stocks from {start_date} to {end_date}...")
    data = yf.download(selected_tickers, start=start_date, end=end_date, progress=True)

    # Extract adjusted close prices
    if isinstance(data.columns, pd.MultiIndex):
        # Get the column level names
        level_values_0 = data.columns.get_level_values(0).unique().tolist()
        level_values_1 = data.columns.get_level_values(1).unique().tolist()

        # Check which level has price types (Close, Open, etc.)
        if 'Adj Close' in level_values_0 or 'Close' in level_values_0:
            # Structure is (price_type, ticker)
            if 'Adj Close' in level_values_0:
                prices = data['Adj Close']
            else:
                prices = data['Close']
        elif 'Adj Close' in level_values_1 or 'Close' in level_values_1:
            # Structure is (ticker, price_type)
            if 'Adj Close' in level_values_1:
                prices = data.xs('Adj Close', axis=1, level=1)
            else:
                prices = data.xs('Close', axis=1, level=1)
        else:
            # Fallback: use all data
            raise ValueError(f"Cannot find 'Adj Close' or 'Close' in columns. Available levels: {level_values_0}, {level_values_1}")
    else:
        # Single ticker case
        if 'Adj Close' in data.columns:
            prices = data[['Adj Close']]
        elif 'Close' in data.columns:
            prices = data[['Close']]
        else:
            prices = data

    # Remove stocks with missing data
    prices = prices.dropna(axis=1, how='any')

    # Ensure we have enough stocks
    if prices.shape[1] < n_assets * 0.8:
        raise ValueError(f"Only {prices.shape[1]} valid stocks found, expected at least {int(n_assets * 0.8)}")

    # Take top n_assets
    prices = prices.iloc[:, :n_assets]

    print(f"Data loaded: {prices.shape[0]} days, {prices.shape[1]} stocks")

    # Cache if requested
    if cache_file:
        prices.to_csv(cache_file)
        print(f"Data cached to {cache_file}")

    return prices


def get_sp500_tickers() -> List[str]:
    """
    Get list of S&P 500 tickers.

    Returns
    -------
    List[str]
        List of ticker symbols
    """
    # Common liquid S&P 500 stocks (hardcoded for reliability)
    tickers = [
        # Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ADBE", "CRM",
        "CSCO", "INTC", "QCOM", "TXN", "AMD", "INTU", "AMAT", "MU", "ADI", "LRCX",
        # Finance
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "SPGI",
        # Healthcare
        "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
        # Consumer
        "PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "COST", "SBUX", "TGT",
        # Industrials
        "BA", "CAT", "GE", "MMM", "HON", "UPS", "RTX", "DE", "LMT", "UNP",
        # Energy
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL",
        # Utilities & Others
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PEG", "XEL", "ED",
    ]
    return tickers


def select_liquid_stocks(
    tickers: List[str],
    n_assets: int,
    start_date: str,
    end_date: str,
) -> List[str]:
    """
    Select most liquid stocks based on average trading volume.

    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols
    n_assets : int
        Number of stocks to select
    start_date : str
        Start date
    end_date : str
        End date

    Returns
    -------
    List[str]
        Selected ticker symbols
    """
    print(f"Selecting {n_assets} most liquid stocks...")

    # Download volume data for all tickers
    volumes = {}
    for ticker in tqdm(tickers[:min(len(tickers), n_assets * 2)], desc="Checking liquidity"):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if 'Volume' in data.columns and len(data) > 0:
                avg_volume = data['Volume'].mean()
                # Ensure it's a scalar
                if hasattr(avg_volume, 'item'):
                    volumes[ticker] = avg_volume.item()
                else:
                    volumes[ticker] = float(avg_volume)
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            continue

    # Sort by volume and select top n
    sorted_tickers = sorted(volumes.items(), key=lambda x: x[1], reverse=True)
    selected = [ticker for ticker, _ in sorted_tickers[:n_assets]]

    print(f"Selected {len(selected)} stocks")
    return selected


def get_risk_free_rate(start_date: str, end_date: str) -> pd.Series:
    """
    Get risk-free rate (3-month Treasury bill).

    Parameters
    ----------
    start_date : str
        Start date
    end_date : str
        End date

    Returns
    -------
    pd.Series
        Daily risk-free rate
    """
    # Download 3-month Treasury bill rate (^IRX)
    try:
        tbill = yf.download("^IRX", start=start_date, end=end_date, progress=False)
        rf_rate = tbill['Adj Close'] / 100 / 252  # Convert annual % to daily
        return rf_rate
    except Exception as e:
        print(f"Could not download risk-free rate: {e}")
        print("Using default 2% annual rate")
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        return pd.Series(0.02 / 252, index=dates)
