"""
Technical Analysis Module - Indicators for OHLC data
FIXED: Proper handling of pandas Series values
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union


def safe_float_from_series(value: Union[pd.Series, float, None], default: float = 50.0) -> float:
    """
    Safely convert a value to float, handling pandas Series and NaN values
    
    Parameters
    ----------
    value : float, pd.Series, or None
        Input value to convert
    default : float
        Default value if conversion fails
    
    Returns
    -------
    float
        Safely converted float value
    """
    if value is None:
        return default
    if isinstance(value, pd.Series):
        if value.empty:
            return default
        try:
            val = value.iloc[-1]
            if pd.isna(val):
                return default
            return float(val)
        except Exception:
            return default
    if isinstance(value, (int, float)):
        if np.isnan(value):
            return default
        return float(value)
    return default


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def sma(x: pd.Series, n: int) -> pd.Series:
    """Simple Moving Average"""
    return x.rolling(n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    """Exponential Moving Average"""
    return x.ewm(span=n, adjust=False).mean()


def rsi(x: pd.Series, n: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI)
    
    Parameters
    ----------
    x : pd.Series
        Price series
    n : int
        Period for calculation
    
    Returns
    -------
    pd.Series
        RSI values between 0 and 100
    """
    delta = x.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(x: pd.Series):
    """
    Moving Average Convergence Divergence (MACD)
    
    Parameters
    ----------
    x : pd.Series
        Price series
    
    Returns
    -------
    tuple
        (MACD line, Signal line, Histogram)
    """
    m = ema(x, 12) - ema(x, 26)
    s = ema(m, 9)
    return m, s, m - s


def bollinger_bands(x: pd.Series, n: int = 20, k: float = 2.0):
    """
    Bollinger Bands
    
    Parameters
    ----------
    x : pd.Series
        Price series
    n : int
        Period for moving average
    k : float
        Number of standard deviations
    
    Returns
    -------
    tuple
        (Upper band, Middle band, Lower band)
    """
    mid = sma(x, n)
    std = x.rolling(n).std()
    return mid + k * std, mid, mid - k * std


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """
    Average True Range (ATR)
    
    Parameters
    ----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
    n : int
        Period for calculation
    
    Returns
    -------
    pd.Series
        Average True Range
    """
    tr_df = pd.DataFrame(index=high.index)
    tr_df['hl'] = high - low
    tr_df['hc'] = abs(high - close.shift(1))
    tr_df['lc'] = abs(low - close.shift(1))
    tr = tr_df.max(axis=1)
    return tr.rolling(n).mean()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to OHLC DataFrame
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume'
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added technical indicators
    """
    if df is None or df.empty:
        return df
    
    out = df.copy()
    
    # Moving averages
    out["SMA_20"] = sma(out["Close"], 20)
    out["SMA_50"] = sma(out["Close"], 50)
    out["SMA_200"] = sma(out["Close"], 200)
    out["EMA_12"] = ema(out["Close"], 12)
    out["EMA_26"] = ema(out["Close"], 26)
    
    # Momentum indicators
    out["RSI"] = rsi(out["Close"], 14)
    out["MACD"], out["MACD_Signal"], out["MACD_Hist"] = macd(out["Close"])
    
    # Volatility indicators
    out["BB_Upper"], out["BB_Mid"], out["BB_Lower"] = bollinger_bands(out["Close"])
    
    # ATR
    if all(x in out.columns for x in ["High", "Low", "Close"]):
        out["ATR"] = atr(out["High"], out["Low"], out["Close"], 14)
    
    return out


def get_technical_signals(tech_df: pd.DataFrame) -> Dict[str, str]:
    """
    Generate trading signals based on technical indicators
    
    Parameters
    ----------
    tech_df : pd.DataFrame
        DataFrame with technical indicators
    
    Returns
    -------
    dict
        Current technical signals for each indicator
    """
    if tech_df is None or tech_df.empty:
        return {
            "RSI": "N/A",
            "MACD": "N/A",
            "Trend": "N/A",
            "Bollinger": "N/A"
        }
    
    signals = {}
    
    # RSI signal - safely extract float value
    if "RSI" in tech_df.columns:
        rsi_val = safe_float_from_series(tech_df["RSI"], 50.0)
        
        if rsi_val < 30:
            signals["RSI"] = "Oversold (Buy Signal)"
        elif rsi_val > 70:
            signals["RSI"] = "Overbought (Sell Signal)"
        else:
            signals["RSI"] = "Neutral"
    else:
        signals["RSI"] = "N/A"
    
    # MACD signal - safely extract values
    if "MACD" in tech_df.columns and "MACD_Signal" in tech_df.columns:
        macd_val = safe_float_from_series(tech_df["MACD"], 0.0)
        signal_val = safe_float_from_series(tech_df["MACD_Signal"], 0.0)
        
        if macd_val > signal_val:
            signals["MACD"] = "Bullish (MACD above Signal)"
        else:
            signals["MACD"] = "Bearish (MACD below Signal)"
    else:
        signals["MACD"] = "N/A"
    
    # Price vs SMA trend - safely extract values
    if "Close" in tech_df.columns and "SMA_50" in tech_df.columns:
        price = safe_float_from_series(tech_df["Close"], 0.0)
        sma50 = safe_float_from_series(tech_df["SMA_50"], price)
        
        if price > sma50:
            signals["Trend"] = "Above SMA50 (Uptrend)"
        else:
            signals["Trend"] = "Below SMA50 (Downtrend)"
    else:
        signals["Trend"] = "N/A"
    
    # Bollinger Bands signal
    if "Close" in tech_df.columns and "BB_Upper" in tech_df.columns and "BB_Lower" in tech_df.columns:
        price = safe_float_from_series(tech_df["Close"], 0.0)
        bb_upper = safe_float_from_series(tech_df["BB_Upper"], price + 1)
        bb_lower = safe_float_from_series(tech_df["BB_Lower"], price - 1)
        
        if price > bb_upper:
            signals["Bollinger"] = "Above Upper Band (Overextended)"
        elif price < bb_lower:
            signals["Bollinger"] = "Below Lower Band (Oversold)"
        else:
            signals["Bollinger"] = "Within Bands (Normal)"
    else:
        signals["Bollinger"] = "N/A"
    
    return signals