"""
Technical Analysis Module - Indicators for OHLC data
"""

import pandas as pd
import numpy as np


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


def get_technical_signals(tech_df: pd.DataFrame) -> dict:
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
    if tech_df.empty:
        return {}
    
    latest = tech_df.iloc[-1]
    signals = {}
    
    # RSI signal
    rsi_val = latest.get("RSI", 50)
    if rsi_val < 30:
        signals["RSI"] = "Oversold (Buy Signal)"
    elif rsi_val > 70:
        signals["RSI"] = "Overbought (Sell Signal)"
    else:
        signals["RSI"] = "Neutral"
    
    # MACD signal
    macd_val = latest.get("MACD", 0)
    signal_val = latest.get("MACD_Signal", 0)
    if macd_val > signal_val:
        signals["MACD"] = "Bullish (MACD above Signal)"
    else:
        signals["MACD"] = "Bearish (MACD below Signal)"
    
    # Price vs SMA
    price = latest.get("Close", 0)
    sma50 = latest.get("SMA_50", price)
    if price > sma50:
        signals["Trend"] = "Above SMA50 (Uptrend)"
    else:
        signals["Trend"] = "Below SMA50 (Downtrend)"
    
    # Bollinger Bands
    bb_upper = latest.get("BB_Upper", price + 1)
    bb_lower = latest.get("BB_Lower", price - 1)
    if price > bb_upper:
        signals["Bollinger"] = "Above Upper Band (Overextended)"
    elif price < bb_lower:
        signals["Bollinger"] = "Below Lower Band (Oversold)"
    else:
        signals["Bollinger"] = "Within Bands (Normal)"
    
    return signals