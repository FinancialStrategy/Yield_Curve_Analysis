"""
Technical Analysis Module - Indicators for OHLC data
FULLY REWRITTEN - All figures visible, no pandas Series comparison errors
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def safe_float_from_series(value: Union[pd.Series, float, None], default: float = 50.0) -> float:
    """
    Safely convert a value to float, handling pandas Series and NaN values
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
    """Relative Strength Index (RSI)"""
    delta = x.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(x: pd.Series):
    """Moving Average Convergence Divergence (MACD)"""
    m = ema(x, 12) - ema(x, 26)
    s = ema(m, 9)
    return m, s, m - s


def bollinger_bands(x: pd.Series, n: int = 20, k: float = 2.0):
    """Bollinger Bands"""
    mid = sma(x, n)
    std = x.rolling(n).std()
    return mid + k * std, mid, mid - k * std


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to OHLC DataFrame"""
    if df is None or df.empty:
        return df
    
    out = df.copy()
    
    # Moving averages
    out["SMA_20"] = sma(out["Close"], 20)
    out["SMA_50"] = sma(out["Close"], 50)
    out["EMA_12"] = ema(out["Close"], 12)
    out["EMA_26"] = ema(out["Close"], 26)
    
    # Momentum indicators
    out["RSI"] = rsi(out["Close"], 14)
    out["MACD"], out["MACD_Signal"], out["MACD_Hist"] = macd(out["Close"])
    
    # Volatility indicators
    out["BB_Upper"], out["BB_Mid"], out["BB_Lower"] = bollinger_bands(out["Close"])
    
    return out


def get_technical_signals(tech_df: pd.DataFrame) -> Dict[str, str]:
    """Generate trading signals based on technical indicators"""
    if tech_df is None or tech_df.empty:
        return {"RSI": "N/A", "MACD": "N/A", "Trend": "N/A"}
    
    signals = {}
    
    # RSI signal
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
    
    # MACD signal
    if "MACD" in tech_df.columns and "MACD_Signal" in tech_df.columns:
        macd_val = safe_float_from_series(tech_df["MACD"], 0.0)
        signal_val = safe_float_from_series(tech_df["MACD_Signal"], 0.0)
        signals["MACD"] = "Bullish" if macd_val > signal_val else "Bearish"
    else:
        signals["MACD"] = "N/A"
    
    # Trend signal
    if "Close" in tech_df.columns and "SMA_50" in tech_df.columns:
        price = safe_float_from_series(tech_df["Close"], 0.0)
        sma50 = safe_float_from_series(tech_df["SMA_50"], price)
        signals["Trend"] = "Above SMA50" if price > sma50 else "Below SMA50"
    else:
        signals["Trend"] = "N/A"
    
    return signals


def plot_technical_chart(df: pd.DataFrame, ticker: str) -> Optional[go.Figure]:
    """
    Create professional technical analysis chart with visible outputs
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC and technical indicators
    ticker : str
        Ticker symbol for title
    
    Returns
    -------
    plotly.graph_objects.Figure or None
        Interactive technical chart
    """
    if df is None or df.empty:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=(
            f"{ticker} - Price with Moving Averages & Bollinger Bands",
            "RSI (14) - Relative Strength Index",
            "MACD - Moving Average Convergence Divergence",
            "Trading Volume"
        )
    )
    
    # ========================================================================
    # ROW 1: Price with MA and Bollinger Bands
    # ========================================================================
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            showlegend=True
        ),
        row=1, col=1
    )
    
    # SMA 20
    if "SMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_20"],
                mode="lines",
                name="SMA 20",
                line=dict(color="#4a7c59", width=1.5)
            ),
            row=1, col=1
        )
    
    # SMA 50
    if "SMA_50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_50"],
                mode="lines",
                name="SMA 50",
                line=dict(color="#c17f3a", width=1.5)
            ),
            row=1, col=1
        )
    
    # Bollinger Bands
    if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        # Upper band
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Upper"],
                mode="lines",
                name="BB Upper",
                line=dict(color="#667085", width=1, dash="dash")
            ),
            row=1, col=1
        )
        
        # Lower band
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Lower"],
                mode="lines",
                name="BB Lower",
                line=dict(color="#667085", width=1, dash="dash"),
                fill="tonexty",
                fillcolor="rgba(108, 142, 173, 0.1)"
            ),
            row=1, col=1
        )
    
    # ========================================================================
    # ROW 2: RSI
    # ========================================================================
    
    if "RSI" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI"],
                mode="lines",
                name="RSI",
                line=dict(color="#2c5f8a", width=2)
            ),
            row=2, col=1
        )
        
        # Overbought line
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=2, col=1)
        fig.add_annotation(
            x=df.index[-1], y=72,
            text="Overbought (70)",
            showarrow=False,
            font=dict(size=10, color="#ef4444"),
            row=2, col=1
        )
        
        # Oversold line
        fig.add_hline(y=30, line_dash="dash", line_color="#10b981", row=2, col=1)
        fig.add_annotation(
            x=df.index[-1], y=28,
            text="Oversold (30)",
            showarrow=False,
            font=dict(size=10, color="#10b981"),
            row=2, col=1
        )
        
        # Middle line
        fig.add_hline(y=50, line_dash="dot", line_color="#667085", row=2, col=1)
    
    # ========================================================================
    # ROW 3: MACD
    # ========================================================================
    
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        # MACD line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD"],
                mode="lines",
                name="MACD",
                line=dict(color="#2c5f8a", width=2)
            ),
            row=3, col=1
        )
        
        # Signal line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD_Signal"],
                mode="lines",
                name="Signal",
                line=dict(color="#c17f3a", width=2)
            ),
            row=3, col=1
        )
        
        # MACD Histogram
        if "MACD_Hist" in df.columns:
            colors = ["#ef4444" if x < 0 else "#10b981" for x in df["MACD_Hist"].values]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df["MACD_Hist"],
                    name="Histogram",
                    marker_color=colors,
                    opacity=0.7
                ),
                row=3, col=1
            )
        
        # Zero line
        fig.add_hline(y=0, line_dash="solid", line_color="#667085", row=3, col=1)
    
    # ========================================================================
    # ROW 4: Volume
    # ========================================================================
    
    if "Volume" in df.columns:
        # Color volume bars based on price movement
        if "Close" in df.columns:
            colors = ["#10b981" if df["Close"].iloc[i] >= df["Close"].iloc[i-1] else "#ef4444" 
                     for i in range(len(df))]
            colors[0] = "#667085"
        else:
            colors = "#667085"
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.5
            ),
            row=4, col=1
        )
    
    # ========================================================================
    # Layout Updates
    # ========================================================================
    
    fig.update_layout(
        title=dict(
            text=f"<b>Technical Analysis Dashboard - {ticker}</b>",
            x=0.5,
            font=dict(size=16)
        ),
        template="plotly_white",
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    # Update x-axis
    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_xaxes(rangeslider=dict(visible=False), row=1, col=1)
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    
    return fig