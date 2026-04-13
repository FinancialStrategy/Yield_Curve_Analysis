"""
Data Layer Module - FRED and Yahoo Finance data fetching
"""

import time
import pandas as pd
import requests
import streamlit as st
from typing import Optional
from config import FRED_SERIES, RECESSION_SERIES, CFG

# =============================================================================
# YFINANCE AVAILABILITY CHECK
# =============================================================================

YFINANCE_AVAILABLE = True
try:
    import yfinance as yf
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False

# =============================================================================
# FRED API FUNCTIONS
# =============================================================================

@st.cache_data(ttl=CFG.cache_ttl_sec, show_spinner=False)
def fetch_fred_series(api_key: str, series_id: str, start_date: str, end_date: str) -> pd.Series:
    """
    Fetch a single series from FRED API
    
    Parameters
    ----------
    api_key : str
        FRED API key
    series_id : str
        FRED series identifier (e.g., "DGS10")
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    
    Returns
    -------
    pd.Series
        Time series data from FRED
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
        "sort_order": "asc",
    }
    
    for attempt in range(CFG.max_retries):
        try:
            response = requests.get(url, params=params, timeout=CFG.timeout)
            response.raise_for_status()
            data = response.json()
            observations = data.get("observations", [])
            
            if not observations:
                return pd.Series(dtype="float64")
            
            dates, values = [], []
            for obs in observations:
                val = obs.get("value")
                if val not in (".", None):
                    dates.append(pd.to_datetime(obs["date"]))
                    values.append(float(val))
            
            if dates:
                return pd.Series(values, index=dates, name=series_id)
            return pd.Series(dtype="float64")
        
        except Exception:
            if attempt == CFG.max_retries - 1:
                return pd.Series(dtype="float64")
            time.sleep(0.6 * (attempt + 1))
    
    return pd.Series(dtype="float64")


@st.cache_data(ttl=CFG.cache_ttl_sec, show_spinner=False)
def fetch_yield_curve(api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch complete yield curve from FRED
    
    Parameters
    ----------
    api_key : str
        FRED API key
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    
    Returns
    -------
    pd.DataFrame
        Yield curve data with maturities as columns
    """
    data = {}
    for name, series_id in FRED_SERIES.items():
        series = fetch_fred_series(api_key, series_id, start_date, end_date)
        if not series.empty:
            data[name] = series
        else:
            st.warning(f"Could not fetch {name} data (FRED series: {series_id})")
    
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    df = df.ffill().bfill()
    return df


@st.cache_data(ttl=CFG.cache_ttl_sec, show_spinner=False)
def fetch_recession_series(api_key: str, start_date: str, end_date: str) -> pd.Series:
    """
    Fetch NBER recession indicator from FRED
    
    Parameters
    ----------
    api_key : str
        FRED API key
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    
    Returns
    -------
    pd.Series
        Recession indicator (1 = recession, 0 = expansion)
    """
    return fetch_fred_series(api_key, RECESSION_SERIES, start_date, end_date)


def validate_fred_api_key(api_key: str) -> bool:
    """
    Validate FRED API key
    
    Parameters
    ----------
    api_key : str
        FRED API key to validate
    
    Returns
    -------
    bool
        True if valid, False otherwise
    """
    if not api_key or len(api_key) < 10:
        return False
    
    test_series = fetch_fred_series(api_key, "DGS10", "2023-01-01", "2023-12-31")
    return not test_series.empty


# =============================================================================
# YAHOO FINANCE FUNCTIONS
# =============================================================================

@st.cache_data(ttl=CFG.cache_ttl_sec, show_spinner=False)
def fetch_market_data(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """
    Fetch market data from Yahoo Finance
    
    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    
    Returns
    -------
    pd.Series
        Adjusted close price series
    """
    if not YFINANCE_AVAILABLE:
        return pd.Series(dtype="float64")
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return pd.Series(dtype="float64")
        
        if "Adj Close" in data.columns:
            series = data["Adj Close"]
        elif "Close" in data.columns:
            series = data["Close"]
        else:
            series = data.iloc[:, 0]
        
        return series.tz_localize(None)
    
    except Exception:
        return pd.Series(dtype="float64")


@st.cache_data(ttl=CFG.cache_ttl_sec, show_spinner=False)
def fetch_ohlc_data(ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """
    Fetch OHLC data from Yahoo Finance for technical analysis
    
    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol
    period : str
        Data period (e.g., "1y", "2y", "5y")
    
    Returns
    -------
    pd.DataFrame or None
        OHLC DataFrame with columns: Open, High, Low, Close, Volume
    """
    if not YFINANCE_AVAILABLE:
        return None
    
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=False)
        if data is None or data.empty:
            return None
        return data
    except Exception:
        return None


@st.cache_data(ttl=CFG.cache_ttl_sec, show_spinner=False)
def fetch_market_bundle(start_date: str, end_date: str):
    """
    Fetch multiple market indicators for volatility and correlation analysis
    
    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    
    Returns
    -------
    tuple
        (volatility_df, correlation_df) as DataFrames
    """
    from config import VOLATILITY_TICKERS, CORRELATION_TICKERS
    
    vol_data = {}
    for ticker, name in VOLATILITY_TICKERS.items():
        series = fetch_market_data(ticker, start_date, end_date)
        if not series.empty:
            vol_data[name] = series
    
    corr_data = {}
    for ticker, name in CORRELATION_TICKERS.items():
        series = fetch_market_data(ticker, start_date, end_date)
        if not series.empty:
            corr_data[name] = series
    
    volatility_df = pd.DataFrame(vol_data) if vol_data else pd.DataFrame()
    correlation_df = pd.DataFrame(corr_data) if corr_data else pd.DataFrame()
    
    return volatility_df, correlation_df