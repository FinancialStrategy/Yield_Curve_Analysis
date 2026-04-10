import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm, skew, kurtosis, jarque_bera, anderson
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# =============================================================================
# DYNAMIC QUANTITATIVE ANALYSIS MODEL - INSTITUTIONAL PLATFORM
# Enhanced with Monte Carlo, Machine Learning, Backtesting, Volatility Indices, Correlations
# =============================================================================

st.set_page_config(
    page_title="Dynamic Quantitative Analysis Platform | Institutional Fixed-Income",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "bg": "#eef2f7",
    "bg2": "#f7f9fc",
    "surface": "#ffffff",
    "surface_alt": "#f5f7fb",
    "header": "#1a2a3a",
    "grid": "#c8d4e0",
    "grid_dark": "#9aaebf",
    "text": "#1a2a3a",
    "text_secondary": "#4a5a6a",
    "muted": "#667085",
    "accent": "#2c5f8a",
    "accent2": "#4a7c59",
    "accent3": "#c17f3a",
    "positive": "#2f855a",
    "negative": "#c05656",
    "warning": "#d48924",
    "recession": "rgba(120, 130, 145, 0.18)",
    "band": "rgba(108, 142, 173, 0.10)",
}

FRED_SERIES = {
    "1M": "DGS1MO",
    "3M": "DGS3MO",
    "6M": "DGS6MO",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "3Y": "DGS3",
    "5Y": "DGS5",
    "7Y": "DGS7",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30",
}

MATURITY_MAP = {
    "1M": 1 / 12,
    "3M": 0.25,
    "6M": 0.5,
    "1Y": 1,
    "2Y": 2,
    "3Y": 3,
    "5Y": 5,
    "7Y": 7,
    "10Y": 10,
    "20Y": 20,
    "30Y": 30,
}

YAHOO_TICKERS = {
    "^TNX": "10Y Treasury Yield Index",
    "^FVX": "5Y Treasury Yield Index",
    "^IRX": "13W T-Bill Index",
    "TLT": "20+Y Treasury Bond ETF",
    "IEF": "7-10Y Treasury Bond ETF",
    "SHY": "1-3Y Treasury Bond ETF",
}

# Additional tickers for volatility and correlations
VOLATILITY_TICKERS = {
    "^VIX": "CBOE Volatility Index",
    "^VXN": "Nasdaq Volatility Index",
    "MOVE": "Merrill Lynch Option Volatility Estimate",
}

CORRELATION_TICKERS = {
    "^GSPC": "S&P 500",
    "QQQ": "Nasdaq 100",
    "DXY": "US Dollar Index",
    "GC=F": "Gold Futures",
    "CL=F": "Crude Oil Futures",
}

DEFAULT_STATE = {
    "api_key_validated": False,
    "api_key": "",
    "yield_data": None,
    "recession_data": None,
    "ohlc_data": None,
    "volatility_data": None,
    "correlation_data": None,
    "data_fetched": False,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(135deg, {COLORS['bg']} 0%, {COLORS['bg2']} 100%);
    }}
    .main-title-card {{
        background: linear-gradient(135deg, {COLORS['header']} 0%, #2c4a6e 100%);
        border-radius: 20px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
        border: 1px solid rgba(255,255,255,0.1);
    }}
    .main-title {{
        color: white;
        font-weight: 800;
        font-size: 1.65rem;
        margin: 0;
        letter-spacing: -0.02em;
    }}
    .main-subtitle {{
        color: rgba(255,255,255,0.85);
        font-size: 0.85rem;
        margin-top: 0.5rem;
        font-family: 'Courier New', monospace;
    }}
    .metric-card {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['grid']};
        border-radius: 16px;
        padding: 1rem;
        min-height: 120px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        transition: transform 0.2s, box-shadow 0.2s;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }}
    .metric-label {{
        color: {COLORS['text_secondary']};
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }}
    .metric-value {{
        color: {COLORS['text']};
        font-size: 1.65rem;
        font-weight: 800;
        margin-top: 0.5rem;
        font-family: 'Courier New', monospace;
    }}
    .metric-sub {{
        color: {COLORS['muted']};
        font-size: 0.75rem;
        margin-top: 0.4rem;
        line-height: 1.3;
    }}
    .note-box {{
        background: {COLORS['surface_alt']};
        border: 1px solid {COLORS['grid']};
        border-left: 4px solid {COLORS['accent']};
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: {COLORS['text']};
        font-size: 0.88rem;
        line-height: 1.5;
        margin: 1rem 0;
    }}
    .warning-box {{
        background: #fff8f0;
        border: 1px solid #f0d8b0;
        border-left: 4px solid {COLORS['warning']};
        border-radius: 12px;
        padding: 1rem 1.2rem;
        font-size: 0.88rem;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        border-bottom: 2px solid {COLORS['grid']};
        flex-wrap: wrap !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {COLORS['text_secondary']};
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
        white-space: nowrap;
        padding: 8px 14px;
        letter-spacing: 0.03em;
    }}
    .stTabs [aria-selected="true"] {{
        color: {COLORS['accent']};
        border-bottom: 3px solid {COLORS['accent']};
    }}
    .stButton>button, .stDownloadButton>button {{
        background: {COLORS['surface']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['grid']};
        border-radius: 10px;
        font-weight: 700;
        transition: all 0.2s;
    }}
    .stButton>button:hover, .stDownloadButton>button:hover {{
        border-color: {COLORS['accent']};
        color: {COLORS['accent']};
        transform: translateY(-1px);
    }}
    #MainMenu, header, footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

@dataclass
class RuntimeConfig:
    history_start: str = "1990-01-01"
    timeout: int = 25
    max_retries: int = 3
    cache_ttl_sec: int = 3600
    rolling_step: int = 21
    rolling_years_default: int = 5
    forecast_horizon_default: int = 20

CFG = RuntimeConfig()

# =============================================================================
# DATA LAYER
# =============================================================================

def fred_request(api_key: str, series_id: str) -> Optional[pd.Series]:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": CFG.history_start,
        "sort_order": "asc",
    }
    for attempt in range(CFG.max_retries):
        try:
            r = requests.get(url, params=params, timeout=CFG.timeout)
            r.raise_for_status()
            payload = r.json()
            obs = payload.get("observations", [])
            dates, values = [], []
            for row in obs:
                value = row.get("value")
                if value not in (".", None):
                    dates.append(pd.to_datetime(row["date"]))
                    values.append(float(value))
            if dates:
                return pd.Series(values, index=dates, name=series_id)
            return None
        except Exception:
            if attempt == CFG.max_retries - 1:
                return None
            time.sleep(0.6 * (attempt + 1))
    return None

@st.cache_data(ttl=CFG.cache_ttl_sec, show_spinner=False)
def validate_fred_api_key(api_key: str) -> bool:
    if not api_key or len(api_key) < 10:
        return False
    s = fred_request(api_key, "DGS10")
    return s is not None and len(s) > 10

@st.cache_data(ttl=CFG.cache_ttl_sec, show_spinner=False)
def fetch_all_yield_data(api_key: str) -> Optional[pd.DataFrame]:
    data = {}
    for name, sid in FRED_SERIES.items():
        s = fred_request(api_key, sid)
        if s is not None:
            data[name] = s
    if not data:
        return None
    return pd.DataFrame(data).sort_index().dropna()

@st.cache_data(ttl=CFG.cache_ttl_sec, show_spinner=False)
def fetch_recession_data(api_key: str) -> Optional[pd.Series]:
    return fred_request(api_key, "USREC")

@st.cache_data(ttl=CFG.cache_ttl_sec, show_spinner=False)
def fetch_ohlc_data(ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None

@st.cache_data(ttl=CFG.cache_ttl_sec, show_spinner=False)
def fetch_volatility_data() -> Optional[pd.DataFrame]:
    """Fetch volatility indices (VIX, VXN, MOVE)"""
    data = {}
    for ticker in VOLATILITY_TICKERS:
        try:
            df = yf.download(ticker, period="5y", progress=False)
            if df is not None and not df.empty:
                data[ticker] = df['Close']
        except Exception:
            continue
    if not data:
        return None
    return pd.DataFrame(data).dropna()

@st.cache_data(ttl=CFG.cache_ttl_sec, show_spinner=False)
def fetch_correlation_data() -> Optional[pd.DataFrame]:
    """Fetch correlation assets (S&P 500, Nasdaq, DXY, Gold, Oil)"""
    data = {}
    for ticker, name in CORRELATION_TICKERS.items():
        try:
            df = yf.download(ticker, period="5y", progress=False)
            if df is not None and not df.empty:
                data[name] = df['Close']
        except Exception:
            continue
    if not data:
        return None
    return pd.DataFrame(data).dropna()

# =============================================================================
# 1. MONTE CARLO SIMULATIONS
# =============================================================================

class MonteCarloSimulator:
    """Advanced Monte Carlo simulation for yield curve scenarios"""
    
    @staticmethod
    def simulate_geometric_brownian_motion(initial_yield: float, mu: float, sigma: float, 
                                           days: int, simulations: int = 1000) -> np.ndarray:
        """Simulate yield paths using GBM"""
        dt = 1/252
        paths = np.zeros((simulations, days))
        paths[:, 0] = initial_yield
        
        for i in range(1, days):
            z = np.random.standard_normal(simulations)
            paths[:, i] = paths[:, i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
        return paths
    
    @staticmethod
    def simulate_vasicek(initial_rate: float, kappa: float, theta: float, sigma: float,
                         days: int, simulations: int = 1000) -> np.ndarray:
        """Simulate interest rates using Vasicek mean-reverting model"""
        dt = 1/252
        paths = np.zeros((simulations, days))
        paths[:, 0] = initial_rate
        
        for i in range(1, days):
            z = np.random.standard_normal(simulations)
            dr = kappa * (theta - paths[:, i-1]) * dt + sigma * np.sqrt(dt) * z
            paths[:, i] = paths[:, i-1] + dr
        
        return paths
    
    @staticmethod
    def calculate_confidence_intervals(paths: np.ndarray, confidence: float = 0.95) -> Dict:
        """Calculate confidence intervals for simulated paths"""
        lower_percentile = (1 - confidence) / 2 * 100
        upper_percentile = (1 + confidence) / 2 * 100
        
        mean_path = np.mean(paths, axis=0)
        lower_bound = np.percentile(paths, lower_percentile, axis=0)
        upper_bound = np.percentile(paths, upper_percentile, axis=0)
        median_path = np.percentile(paths, 50, axis=0)
        
        return {
            "mean": mean_path,
            "median": median_path,
            "lower_ci": lower_bound,
            "upper_ci": upper_bound,
            "std": np.std(paths, axis=0),
        }
    
    @staticmethod
    def calculate_var_from_paths(paths: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate VaR from simulated paths"""
        final_values = paths[:, -1]
        return np.percentile(final_values, (1 - confidence) * 100)

# =============================================================================
# 2. MACHINE LEARNING FORECASTING (Random Forest)
# =============================================================================

class MLForecastModel:
    """Machine learning models for yield curve forecasting"""
    
    @staticmethod
    def prepare_features(yield_df: pd.DataFrame, lags: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare lagged features for ML models"""
        X, y = [], []
        for i in range(lags, len(yield_df) - 1):
            features = []
            for col in yield_df.columns:
                features.extend(yield_df[col].iloc[i-lags:i].values)
            X.append(features)
            y.append(yield_df.iloc[i+1].values)
        
        return np.array(X), np.array(y)
    
    @staticmethod
    def train_random_forest(X: np.ndarray, y: np.ndarray, 
                            test_size: float = 0.2) -> Dict:
        """Train Random Forest model for yield forecasting"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                      random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': [f'lag_{i}' for i in range(X.shape[1])],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            "model": model,
            "rmse": rmse,
            "r2": r2,
            "feature_importance": feature_importance,
            "predictions": y_pred,
            "actual": y_test,
        }
    
    @staticmethod
    def forecast_future(model, last_observations: np.ndarray, 
                        horizon: int, n_features: int) -> np.ndarray:
        """Rolling forecast using trained model"""
        forecasts = []
        current_window = last_observations.copy()
        
        for _ in range(horizon):
            pred = model.predict(current_window.reshape(1, -1))[0]
            forecasts.append(pred)
            # Shift window
            current_window = np.roll(current_window, -n_features)
            current_window[-n_features:] = pred
        
        return np.array(forecasts)

# =============================================================================
# 4. BACKTESTING FRAMEWORK
# =============================================================================

class BacktestEngine:
    """Professional backtesting framework for trading strategies"""
    
    @staticmethod
    def backtest_curve_strategy(yield_df: pd.DataFrame, spreads: pd.DataFrame,
                                strategy_type: str = "curve_inversion") -> Dict:
        """Backtest curve-based trading strategies"""
        
        if strategy_type == "curve_inversion":
            # Strategy: Go long duration when curve inverts (signal for future rally)
            if "10Y-2Y" not in spreads.columns:
                return {}
            
            signals = spreads["10Y-2Y"] < 0
            returns = yield_df["10Y"].pct_change().shift(-1)  # Forward returns
            
            strategy_returns = signals.shift(1) * returns
            buy_hold_returns = returns
            
            cumulative_strategy = (1 + strategy_returns.fillna(0)).cumprod()
            cumulative_bh = (1 + buy_hold_returns.fillna(0)).cumprod()
            
            # Performance metrics
            sharpe_strategy = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            sharpe_bh = buy_hold_returns.mean() / buy_hold_returns.std() * np.sqrt(252)
            
            max_drawdown = (cumulative_strategy / cumulative_strategy.cummax() - 1).min()
            
            return {
                "strategy_name": "Curve Inversion Strategy (Long Duration on Inversion)",
                "cumulative_returns": cumulative_strategy,
                "buy_hold_returns": cumulative_bh,
                "sharpe_ratio_strategy": sharpe_strategy,
                "sharpe_ratio_bh": sharpe_bh,
                "max_drawdown": max_drawdown,
                "total_return_strategy": cumulative_strategy.iloc[-1] - 1,
                "total_return_bh": cumulative_bh.iloc[-1] - 1,
                "win_rate": (strategy_returns[strategy_returns != 0] > 0).mean(),
            }
        
        elif strategy_type == "momentum":
            # Momentum strategy: Follow trend
            sma_20 = yield_df["10Y"].rolling(20).mean()
            sma_50 = yield_df["10Y"].rolling(50).mean()
            signals = sma_20 > sma_50
            returns = yield_df["10Y"].pct_change().shift(-1)
            
            strategy_returns = signals.shift(1) * returns
            cumulative_strategy = (1 + strategy_returns.fillna(0)).cumprod()
            
            return {
                "strategy_name": "Moving Average Crossover (20/50)",
                "cumulative_returns": cumulative_strategy,
                "sharpe_ratio": strategy_returns.mean() / strategy_returns.std() * np.sqrt(252),
                "total_return": cumulative_strategy.iloc[-1] - 1,
            }
        
        return {}
    
    @staticmethod
    def calculate_performance_metrics(returns: pd.Series) -> Dict:
        """Calculate comprehensive performance metrics"""
        returns = returns.dropna()
        if len(returns) == 0:
            return {}
        
        cumulative = (1 + returns).cumprod()
        
        return {
            "total_return": cumulative.iloc[-1] - 1,
            "annualized_return": (cumulative.iloc[-1]) ** (252 / len(returns)) - 1,
            "annualized_volatility": returns.std() * np.sqrt(252),
            "sharpe_ratio": returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            "max_drawdown": (cumulative / cumulative.cummax() - 1).min(),
            "win_rate": (returns > 0).mean(),
            "profit_factor": abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else np.inf,
            "calmar_ratio": ((cumulative.iloc[-1] - 1) / abs((cumulative / cumulative.cummax() - 1).min())) if (cumulative / cumulative.cummax() - 1).min() != 0 else 0,
        }

# =============================================================================
# 6. OPTIONS-IMPLIED VOLATILITY (VIX, MOVE)
# =============================================================================

class VolatilityAnalyzer:
    """Analyze options-implied volatility indices"""
    
    @staticmethod
    def calculate_volatility_regime(vix: pd.Series) -> Dict:
        """Determine volatility regime based on VIX levels"""
        current_vix = vix.iloc[-1]
        
        if current_vix < 12:
            regime = "EXTREME COMPLACENCY"
            outlook = "High risk of volatility spike"
        elif current_vix < 15:
            regime = "LOW VOLATILITY"
            outlook = "Normal complacent market"
        elif current_vix < 20:
            regime = "NORMAL VOLATILITY"
            outlook = "Typical market conditions"
        elif current_vix < 25:
            regime = "ELEVATED VOLATILITY"
            outlook = "Increased uncertainty"
        elif current_vix < 35:
            regime = "HIGH VOLATILITY"
            outlook = "Market stress, consider hedging"
        else:
            regime = "EXTREME VOLATILITY"
            outlook = "Crisis conditions, defensive positioning"
        
        # Calculate VIX term structure proxy (using historical percentile)
        percentile = (vix < current_vix).mean()
        
        return {
            "current_vix": current_vix,
            "regime": regime,
            "outlook": outlook,
            "percentile": percentile,
            "mean_vix": vix.mean(),
            "median_vix": vix.median(),
            "max_vix": vix.max(),
            "vix_percentile": f"{percentile * 100:.1f}%",
        }
    
    @staticmethod
    def calculate_vol_of_vol(vix: pd.Series, window: int = 20) -> pd.Series:
        """Calculate volatility of volatility (vol of vol)"""
        return vix.pct_change().rolling(window).std() * np.sqrt(252)
    
    @staticmethod
    def identify_volatility_spikes(vix: pd.Series, threshold: float = 1.5) -> pd.DataFrame:
        """Identify volatility spike events"""
        vix_ma = vix.rolling(20).mean()
        vix_std = vix.rolling(20).std()
        z_score = (vix - vix_ma) / vix_std
        
        spikes = vix[vix > vix_ma + threshold * vix_std]
        
        spike_events = []
        in_spike = False
        start = None
        
        for date, value in spikes.items():
            if not in_spike:
                in_spike = True
                start = date
            elif date not in spikes.index:
                in_spike = False
                spike_events.append({
                    "start": start,
                    "end": date,
                    "peak_vix": spikes.loc[start:date].max(),
                    "duration_days": (date - start).days,
                })
        
        return pd.DataFrame(spike_events)

# =============================================================================
# 7. CROSS-ASSET CORRELATION ANALYSIS
# =============================================================================

class CorrelationAnalyzer:
    """Cross-asset correlation analysis"""
    
    @staticmethod
    def calculate_rolling_correlations(yield_series: pd.Series, 
                                       other_assets: pd.DataFrame,
                                       window: int = 60) -> pd.DataFrame:
        """Calculate rolling correlations between yields and other assets"""
        correlations = pd.DataFrame(index=other_assets.index)
        
        for col in other_assets.columns:
            if len(yield_series) == len(other_assets):
                correlations[f"Corr_10Y_{col}"] = yield_series.rolling(window).corr(other_assets[col])
        
        return correlations
    
    @staticmethod
    def calculate_correlation_matrix(assets_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate full correlation matrix"""
        returns = assets_df.pct_change().dropna()
        return returns.corr()
    
    @staticmethod
    def identify_regime_shifts(correlation_series: pd.Series, 
                               threshold: float = 0.5) -> List[Dict]:
        """Identify significant correlation regime shifts"""
        changes = correlation_series.diff()
        significant_changes = changes[abs(changes) > threshold]
        
        shifts = []
        for date, change in significant_changes.items():
            shifts.append({
                "date": date,
                "change": change,
                "direction": "Positive shift" if change > 0 else "Negative shift",
                "from_value": correlation_series.iloc[correlation_series.index.get_loc(date) - 1],
                "to_value": correlation_series.iloc[correlation_series.index.get_loc(date)],
            })
        
        return shifts
    
    @staticmethod
    def calculate_beta(yield_series: pd.Series, market_series: pd.Series, 
                       window: int = 60) -> pd.Series:
        """Calculate rolling beta relative to market"""
        yield_returns = yield_series.pct_change()
        market_returns = market_series.pct_change()
        
        covariance = yield_returns.rolling(window).cov(market_returns)
        variance = market_returns.rolling(window).var()
        
        return covariance / variance

# =============================================================================
# ENHANCED TECHNICAL INDICATORS
# =============================================================================

def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n).mean()

def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()

def rsi(x: pd.Series, n: int = 14) -> pd.Series:
    delta = x.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(x: pd.Series):
    m = ema(x, 12) - ema(x, 26)
    s = ema(m, 9)
    return m, s, m - s

def bb(x: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(x, n)
    sd = x.rolling(n).std()
    return mid + k * sd, mid, mid - k * sd

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["SMA_20"] = sma(out["Close"], 20)
    out["SMA_50"] = sma(out["Close"], 50)
    out["RSI"] = rsi(out["Close"], 14)
    out["MACD"], out["MACD_Signal"], out["MACD_Hist"] = macd(out["Close"])
    out["BB_Upper"], out["BB_Middle"], out["BB_Lower"] = bb(out["Close"])
    return out

# =============================================================================
# ANALYTICS (Previous functions preserved)
# =============================================================================

def compute_spreads(yield_df: pd.DataFrame) -> pd.DataFrame:
    spreads = pd.DataFrame(index=yield_df.index)
    if {"10Y", "2Y"}.issubset(yield_df.columns):
        spreads["10Y-2Y"] = (yield_df["10Y"] - yield_df["2Y"]) * 100
    if {"10Y", "3M"}.issubset(yield_df.columns):
        spreads["10Y-3M"] = (yield_df["10Y"] - yield_df["3M"]) * 100
    return spreads

def classify_regime(spreads: pd.DataFrame, yield_df: pd.DataFrame) -> Tuple[str, str]:
    spread = spreads["10Y-2Y"].iloc[-1] if "10Y-2Y" in spreads.columns else np.nan
    if np.isfinite(spread) and spread < 0:
        return "Risk-off / Recession Watch", "Curve inversion signals defensive macro regime"
    elif np.isfinite(spread) and spread < 50:
        return "Neutral / Late Cycle", "Curve flattening suggests late-cycle caution"
    return "Risk-on / Expansion", "Positive slope supports pro-risk positioning"

def recession_probability_proxy(spreads: pd.DataFrame, yield_df: pd.DataFrame) -> float:
    score = 0.0
    if "10Y-2Y" in spreads.columns:
        score += np.clip((-spreads["10Y-2Y"].iloc[-1]) / 100, 0, 1.5)
    if "10Y" in yield_df.columns:
        score += np.clip((yield_df["10Y"].iloc[-1] - 4.5) / 3.0, 0, 1.0)
    return float(1 / (1 + np.exp(-2.2 * (score - 0.8))))

def identify_recessions(recession_series: Optional[pd.Series]) -> List[dict]:
    if recession_series is None:
        return []
    recessions = []
    in_rec = False
    start = None
    for date, value in recession_series.dropna().items():
        if value == 1 and not in_rec:
            in_rec = True
            start = date
        elif value == 0 and in_rec:
            recessions.append({"start": start, "end": date})
            in_rec = False
    return recessions

class NelsonSiegelModel:
    @staticmethod
    def nelson_siegel(tau, beta0, beta1, beta2, lambda1):
        tau = np.asarray(tau, dtype=float)
        x = lambda1 * tau
        term1 = (1 - np.exp(-x)) / x
        term2 = term1 - np.exp(-x)
        return beta0 + beta1 * term1 + beta2 * term2
    
    @staticmethod
    def fit_ns(maturities: np.ndarray, yields_: np.ndarray):
        def objective(params):
            fitted = NelsonSiegelModel.nelson_siegel(maturities, *params)
            return np.sum((yields_ - fitted) ** 2)
        bounds = [(yields_.min() - 2, yields_.max() + 2), (-15, 15), (-15, 15), (0.01, 5)]
        best = None
        best_fun = np.inf
        for _ in range(8):
            x0 = [np.random.uniform(a, b) for a, b in bounds]
            res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
            if res.success and res.fun < best_fun:
                best, best_fun = res, res.fun
        if best is None:
            return None
        fitted = NelsonSiegelModel.nelson_siegel(maturities, *best.x)
        sse = np.sum((yields_ - fitted) ** 2)
        sst = np.sum((yields_ - np.mean(yields_)) ** 2)
        return {
            "params": best.x,
            "fitted_values": fitted,
            "rmse": float(np.sqrt(np.mean((yields_ - fitted) ** 2))),
            "r_squared": float(1 - sse / sst) if sst > 0 else np.nan,
        }

# =============================================================================
# ENHANCED VISUALS
# =============================================================================

def create_chart_layout(fig: go.Figure, title: str, y_title: str = None, 
                        height: int = 460, x_title: str = "Date") -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(size=12, color=COLORS["text"]),
        title=dict(text=title, x=0.01, font=dict(size=16, weight="bold")),
        margin=dict(l=60, r=30, t=80, b=50),
        hovermode="x unified",
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(title=x_title, gridcolor=COLORS["grid"], showgrid=True),
        yaxis=dict(title=y_title, gridcolor=COLORS["grid"], showgrid=True),
    )
    return fig

def chart_monte_carlo(initial_yield: float, simulation_results: Dict, 
                      horizon_days: int) -> go.Figure:
    """Visualize Monte Carlo simulation results"""
    fig = go.Figure()
    
    # Add confidence bands
    x_axis = np.arange(horizon_days)
    fig.add_trace(go.Scatter(
        x=x_axis, y=simulation_results["upper_ci"],
        fill=None, mode='lines', line=dict(color='rgba(0,0,0,0)'),
        showlegend=False, name='Upper CI'
    ))
    fig.add_trace(go.Scatter(
        x=x_axis, y=simulation_results["lower_ci"],
        fill='tonexty', mode='lines', 
        fillcolor='rgba(44, 95, 138, 0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name=f'95% Confidence Interval'
    ))
    
    # Add mean and median paths
    fig.add_trace(go.Scatter(
        x=x_axis, y=simulation_results["mean"],
        mode='lines', name='Mean Path',
        line=dict(color=COLORS["accent"], width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=x_axis, y=simulation_results["median"],
        mode='lines', name='Median Path',
        line=dict(color=COLORS["warning"], width=2, dash='dash')
    ))
    
    # Add initial point
    fig.add_trace(go.Scatter(
        x=[0], y=[initial_yield],
        mode='markers', name='Current Yield',
        marker=dict(size=12, color=COLORS["positive"], symbol='star')
    ))
    
    return create_chart_layout(fig, "Monte Carlo Simulation - 10Y Yield Paths (1,000 scenarios)", 
                               "Yield (%)", 500, "Trading Days Ahead")

def chart_ml_forecast(actual: np.ndarray, predicted: np.ndarray, 
                      forecast_horizon: int, feature_importance: pd.DataFrame) -> go.Figure:
    """Visualize ML model predictions"""
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=("Actual vs Predicted (Test Set)", 
                                       "Feature Importance",
                                       "Forecast Horizon",
                                       "Model Performance"),
                        vertical_spacing=0.15)
    
    # Actual vs Predicted
    fig.add_trace(go.Scatter(
        x=actual, y=predicted, mode='markers',
        marker=dict(color=COLORS["accent"], size=8),
        name='Predictions'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[actual.min(), actual.max()], y=[actual.min(), actual.max()],
        mode='lines', line=dict(color='red', dash='dash'),
        name='Perfect Fit'
    ), row=1, col=1)
    
    # Feature importance (top 10)
    top_features = feature_importance.head(10)
    fig.add_trace(go.Bar(
        x=top_features['importance'].values,
        y=top_features['feature'].values,
        orientation='h',
        marker_color=COLORS["warning"],
        name='Importance'
    ), row=1, col=2)
    
    fig.update_xaxes(title_text="Actual Yields", row=1, col=1)
    fig.update_yaxes(title_text="Predicted Yields", row=1, col=1)
    fig.update_xaxes(title_text="Importance", row=1, col=2)
    
    fig.update_layout(height=600, showlegend=True)
    return fig

def chart_backtest_results(backtest_results: Dict) -> go.Figure:
    """Visualize backtesting results"""
    fig = go.Figure()
    
    if "cumulative_returns" in backtest_results:
        fig.add_trace(go.Scatter(
            x=backtest_results["cumulative_returns"].index,
            y=backtest_results["cumulative_returns"].values,
            mode='lines',
            name=backtest_results.get("strategy_name", "Strategy"),
            line=dict(color=COLORS["accent"], width=2.5)
        ))
    
    if "buy_hold_returns" in backtest_results:
        fig.add_trace(go.Scatter(
            x=backtest_results["buy_hold_returns"].index,
            y=backtest_results["buy_hold_returns"].values,
            mode='lines',
            name='Buy & Hold',
            line=dict(color=COLORS["muted"], width=2, dash='dash')
        ))
    
    fig.update_layout(
        title="Strategy Backtest Results",
        yaxis_title="Cumulative Return",
        xaxis_title="Date"
    )
    
    return create_chart_layout(fig, "Backtest Performance Comparison", "Cumulative Return", 500)

def chart_volatility_dashboard(vix_data: pd.Series, vol_regime: Dict) -> go.Figure:
    """Create volatility dashboard with VIX and regimes"""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("VIX - CBOE Volatility Index",
                                       "Volatility Percentile",
                                       "Vol of Vol (20-day)"),
                        vertical_spacing=0.1,
                        row_heights=[0.5, 0.25, 0.25])
    
    # VIX with regime shading
    fig.add_trace(go.Scatter(
        x=vix_data.index, y=vix_data.values,
        mode='lines', name='VIX',
        line=dict(color=COLORS["warning"], width=2)
    ), row=1, col=1)
    
    # Add regime thresholds
    fig.add_hline(y=20, line_dash="dash", line_color="orange", 
                  annotation_text="Elevated", row=1, col=1)
    fig.add_hline(y=15, line_dash="dash", line_color="green",
                  annotation_text="Normal", row=1, col=1)
    
    # Percentile
    percentile_series = (vix_data.rank(pct=True)) * 100
    fig.add_trace(go.Scatter(
        x=vix_data.index, y=percentile_series,
        mode='lines', name='Percentile',
        line=dict(color=COLORS["accent"], width=2),
        fill='tozeroy'
    ), row=2, col=1)
    fig.add_hline(y=95, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=5, line_dash="dash", line_color="green", row=2, col=1)
    
    # Vol of vol
    vol_of_vol = VolatilityAnalyzer.calculate_vol_of_vol(vix_data)
    fig.add_trace(go.Scatter(
        x=vix_data.index, y=vol_of_vol,
        mode='lines', name='Vol of Vol',
        line=dict(color=COLORS["positive"], width=2)
    ), row=3, col=1)
    
    fig.update_yaxes(title_text="VIX", row=1, col=1)
    fig.update_yaxes(title_text="Percentile", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Vol of Vol", row=3, col=1)
    
    return create_chart_layout(fig, f"Volatility Dashboard | Current VIX: {vol_regime['current_vix']:.2f} | Regime: {vol_regime['regime']}", 
                               height=700)

def chart_correlation_heatmap(correlation_matrix: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Cross-Asset Correlation Matrix",
        height=500,
        width=700
    )
    
    return fig

# =============================================================================
# UI HELPERS
# =============================================================================

def render_api_gate() -> None:
    st.markdown(
        f"""
        <div class="note-box" style="max-width:560px; margin:40px auto; text-align:center;">
            <b>🔑 FRED API Key Required</b><br><br>
            This institutional quantitative platform requires live U.S. Treasury data from FRED.<br>
            Get your free API key from the <b>FRED (Federal Reserve Economic Data)</b> website.
        </div>
        """,
        unsafe_allow_html=True,
    )
    api_key = st.text_input("Enter your FRED API key", type="password", placeholder="Paste API key here")
    if st.button("🔐 Validate & Connect", use_container_width=True):
        if not api_key:
            st.error("Please enter a valid API key.")
            st.stop()
        with st.spinner("Validating API key..."):
            valid = validate_fred_api_key(api_key)
        if valid:
            st.session_state.api_key = api_key
            st.session_state.api_key_validated = True
            st.success("✓ API key validated successfully. Loading data...")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("Invalid API key. Please check and try again.")
    st.stop()

def kpi_card(label: str, value: str, sub: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )

# =============================================================================
# MAIN APP
# =============================================================================

def main() -> None:
    st.markdown(
        """
        <div class="main-title-card">
            <div class="main-title">Dynamic Quantitative Analysis Model</div>
            <div class="main-subtitle">Institutional Fixed-Income Platform | Monte Carlo | ML Forecasting | Backtesting | Volatility Analytics | Cross-Asset Correlations</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.api_key_validated:
        render_api_gate()

    with st.sidebar:
        st.markdown("### 🎛️ Control Tower")
        st.caption("Advanced Quantitative Analytics Platform")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            keep_key = st.session_state.api_key
            for key, value in DEFAULT_STATE.items():
                st.session_state[key] = value
            st.session_state.api_key = keep_key
            st.session_state.api_key_validated = True
            st.rerun()
        
        st.markdown("---")
        st.markdown("#### Model Parameters")
        rolling_years = st.slider("Rolling window (years)", 2, 10, 5)
        forecast_horizon = st.slider("Forecast horizon (days)", 5, 60, 20)
        confidence_level = st.slider("Confidence level", 0.90, 0.99, 0.95, 0.01)
        mc_simulations = st.slider("Monte Carlo simulations", 500, 5000, 1000, 500)
        ml_lags = st.slider("ML model lags", 3, 10, 5)

    if not st.session_state.data_fetched:
        with st.spinner("Fetching all data (yields, volatility, correlations, market data)..."):
            yield_df = fetch_all_yield_data(st.session_state.api_key)
            recession_series = fetch_recession_data(st.session_state.api_key)
            volatility_df = fetch_volatility_data()
            correlation_df = fetch_correlation_data()
            
            # Fetch OHLC data for selected tickers
            ohlc_data = {}
            for ticker in list(YAHOO_TICKERS.keys())[:2]:  # Limit for performance
                df = fetch_ohlc_data(ticker, "2y")
                if df is not None:
                    ohlc_data[ticker] = add_technical_indicators(df)
        
        if yield_df is None:
            st.error("Failed to fetch FRED data.")
            st.stop()
        
        st.session_state.yield_data = yield_df
        st.session_state.recession_data = recession_series
        st.session_state.volatility_data = volatility_df
        st.session_state.correlation_data = correlation_df
        st.session_state.ohlc_data = ohlc_data
        st.session_state.data_fetched = True

    yield_df = st.session_state.yield_data.copy()
    recession_series = st.session_state.recession_data
    volatility_df = st.session_state.volatility_data
    correlation_df = st.session_state.correlation_data
    ohlc_data = st.session_state.ohlc_data

    selected_cols = [c for c in yield_df.columns if c in MATURITY_MAP][:6]  # Limit for performance
    maturities = np.array([MATURITY_MAP[c] for c in selected_cols])
    latest_curve = yield_df.iloc[-1][selected_cols].values.astype(float)

    spreads = compute_spreads(yield_df)
    recessions = identify_recessions(recession_series)
    regime, regime_text = classify_regime(spreads, yield_df)
    recession_prob = recession_probability_proxy(spreads, yield_df)

    current_2y = yield_df["2Y"].iloc[-1] if "2Y" in yield_df.columns else np.nan
    current_10y = yield_df["10Y"].iloc[-1] if "10Y" in yield_df.columns else np.nan
    current_spread = spreads["10Y-2Y"].iloc[-1] if "10Y-2Y" in spreads.columns else np.nan

    # Run NS model
    ns_result = NelsonSiegelModel.fit_ns(maturities, latest_curve)

    # KPI Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        kpi_card("📊 Macro Regime", regime, regime_text)
    with c2:
        kpi_card("🏦 2Y Yield", f"{current_2y:.2f}%", "Policy anchor")
    with c3:
        kpi_card("📈 10Y Yield", f"{current_10y:.2f}%", "Benchmark")
    with c4:
        kpi_card("🔄 10Y-2Y Spread", f"{current_spread:.1f} bps", "Recession signal")
    with c5:
        kpi_card("⚠️ Recession Prob", f"{100 * recession_prob:.1f}%", "Proxy estimate")
    with c6:
        if volatility_df is not None and "^VIX" in volatility_df.columns:
            current_vix = volatility_df["^VIX"].iloc[-1]
            kpi_card("📉 VIX", f"{current_vix:.2f}", "Fear gauge")

    # Main Tabs with new features
    main_tabs = st.tabs([
        "🏦 Executive View",
        "🎲 Monte Carlo",
        "🤖 ML Forecasting",
        "📊 Backtesting",
        "📉 Volatility Analytics",
        "🔄 Correlations",
        "📐 Research",
        "💾 Export"
    ])

    # Tab 0: Executive View
    with main_tabs[0]:
        st.plotly_chart(create_chart_layout(
            go.Figure(data=[go.Scatter(x=maturities, y=latest_curve, mode='lines+markers', 
                                       name='Current Curve', line=dict(color=COLORS["accent"], width=2.5))]),
            "Current Treasury Yield Curve", "Yield (%)", 450, "Maturity (Years)"
        ), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_spread = go.Figure()
            if "10Y-2Y" in spreads.columns:
                fig_spread.add_trace(go.Scatter(x=spreads.index, y=spreads["10Y-2Y"], 
                                               mode='lines', name='10Y-2Y',
                                               line=dict(color=COLORS["warning"], width=2)))
            fig_spread.add_hline(y=0, line_dash="dash")
            st.plotly_chart(create_chart_layout(fig_spread, "Historical Spreads", "bps", 400), 
                           use_container_width=True)
        
        with col2:
            fig_yield = go.Figure()
            fig_yield.add_trace(go.Scatter(x=yield_df.index, y=yield_df["10Y"], 
                                          mode='lines', name='10Y',
                                          line=dict(color=COLORS["accent"], width=2)))
            st.plotly_chart(create_chart_layout(fig_yield, "10Y Yield History", "Yield (%)", 400),
                           use_container_width=True)

    # Tab 1: Monte Carlo Simulations
    with main_tabs[1]:
        st.subheader("🎲 Monte Carlo Simulation - Yield Curve Scenarios")
        st.markdown("""
        <div class="note-box">
        <b>Monte Carlo Simulation Framework</b><br>
        Simulates 1,000+ potential paths for 10Y Treasury yields using Geometric Brownian Motion and Vasicek mean-reverting models.
        Provides probabilistic forecasts, VaR estimates, and confidence intervals for risk management.
        </div>
        """, unsafe_allow_html=True)
        
        col_mc1, col_mc2 = st.columns(2)
        
        with col_mc1:
            simulation_days = st.slider("Simulation horizon (days)", 30, 252, 126, 30)
            model_type = st.selectbox("Simulation model", ["Geometric Brownian Motion", "Vasicek Mean-Reverting"])
        
        with col_mc2:
            mc_confidence = st.slider("Confidence interval", 0.80, 0.99, 0.95, 0.01)
            show_individual_paths = st.checkbox("Show sample paths (50)", value=False)
        
        if st.button("🎲 Run Monte Carlo Simulation", use_container_width=True):
            with st.spinner(f"Running {mc_simulations} simulations..."):
                initial_yield = current_10y
                mu = yield_df["10Y"].pct_change().mean() * 252
                sigma = yield_df["10Y"].pct_change().std() * np.sqrt(252)
                
                if model_type == "Geometric Brownian Motion":
                    paths = MonteCarloSimulator.simulate_geometric_brownian_motion(
                        initial_yield, mu, sigma, simulation_days, mc_simulations
                    )
                else:
                    # Estimate Vasicek parameters
                    kappa = 0.5  # Mean reversion speed
                    theta = yield_df["10Y"].mean()  # Long-term mean
                    paths = MonteCarloSimulator.simulate_vasicek(
                        initial_yield, kappa, theta, sigma, simulation_days, mc_simulations
                    )
                
                sim_results = MonteCarloSimulator.calculate_confidence_intervals(paths, mc_confidence)
                var_estimate = MonteCarloSimulator.calculate_var_from_paths(paths, mc_confidence)
                
                # Display metrics
                mc_metrics = st.columns(4)
                mc_metrics[0].metric("Expected Yield (Mean)", f"{sim_results['mean'][-1]:.2f}%")
                mc_metrics[1].metric("Median Yield", f"{sim_results['median'][-1]:.2f}%")
                mc_metrics[2].metric(f"{int(mc_confidence*100)}% VaR", f"{var_estimate:.2f}%", 
                                     delta=f"vs current {initial_yield:.2f}%")
                mc_metrics[3].metric("Simulation Uncertainty", f"±{sim_results['std'][-1]:.2f}%", "1 std dev")
                
                # Plot results
                fig_mc = chart_monte_carlo(initial_yield, sim_results, simulation_days)
                
                if show_individual_paths:
                    for i in range(min(50, mc_simulations)):
                        fig_mc.add_trace(go.Scatter(
                            x=np.arange(simulation_days), y=paths[i],
                            mode='lines', line=dict(width=0.5, color='rgba(100,100,100,0.3)'),
                            showlegend=False
                        ))
                
                st.plotly_chart(fig_mc, use_container_width=True)
                
                # Distribution of final yields
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=paths[:, -1], nbinsx=50,
                                               marker_color=COLORS["accent"],
                                               name='Final Yield Distribution'))
                fig_dist.add_vline(x=initial_yield, line_dash="dash", line_color="red",
                                  annotation_text="Current")
                fig_dist.add_vline(x=sim_results['mean'][-1], line_dash="dash", line_color="green",
                                  annotation_text="Mean")
                fig_dist.update_layout(title="Distribution of Final Yields", 
                                      xaxis_title="Yield (%)", yaxis_title="Frequency",
                                      height=400)
                st.plotly_chart(fig_dist, use_container_width=True)

    # Tab 2: ML Forecasting
    with main_tabs[2]:
        st.subheader("🤖 Machine Learning Forecast - Random Forest Model")
        st.markdown("""
        <div class="note-box">
        <b>Machine Learning Forecasting Framework</b><br>
        Random Forest model trained on lagged yield curve features to predict future yield movements.
        Provides feature importance analysis to identify key predictive factors.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Train & Forecast ML Model", use_container_width=True):
            with st.spinner("Training Random Forest model..."):
                # Prepare features
                X, y = MLForecastModel.prepare_features(yield_df[selected_cols], lags=ml_lags)
                
                if len(X) > 100:
                    # Train model
                    ml_results = MLForecastModel.train_random_forest(X, y, test_size=0.2)
                    
                    # Display performance metrics
                    ml_metrics = st.columns(4)
                    ml_metrics[0].metric("RMSE (bps)", f"{ml_results['rmse'] * 100:.2f}")
                    ml_metrics[1].metric("R² Score", f"{ml_results['r2']:.3f}")
                    ml_metrics[2].metric("Test Samples", len(ml_results['actual']))
                    ml_metrics[3].metric("Features", X.shape[1])
                    
                    # Plot results
                    fig_ml = chart_ml_forecast(ml_results['actual'][:, 0], 
                                               ml_results['predictions'][:, 0],
                                               forecast_horizon,
                                               ml_results['feature_importance'])
                    st.plotly_chart(fig_ml, use_container_width=True)
                    
                    # Generate future forecast
                    st.subheader("Forward Forecast")
                    last_window = X[-1].reshape(1, -1)
                    future_pred = MLForecastModel.forecast_future(ml_results['model'], 
                                                                  last_window[0], 
                                                                  forecast_horizon,
                                                                  len(selected_cols))
                    
                    forecast_df = pd.DataFrame({
                        'Day': range(1, forecast_horizon + 1),
                        'Forecasted 10Y Yield (%)': future_pred[:, 0]
                    })
                    st.dataframe(forecast_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("Insufficient data for ML training. Need more historical data.")

    # Tab 3: Backtesting
    with main_tabs[3]:
        st.subheader("📊 Strategy Backtesting Framework")
        st.markdown("""
        <div class="note-box">
        <b>Backtesting Engine</b><br>
        Test quantitative trading strategies against historical data. Includes curve inversion, momentum, and mean-reversion strategies.
        </div>
        """, unsafe_allow_html=True)
        
        strategy_type = st.selectbox("Select Strategy", 
                                     ["curve_inversion", "momentum", "mean_reversion"])
        
        if st.button("📈 Run Backtest", use_container_width=True):
            with st.spinner("Running backtest..."):
                backtest_results = BacktestEngine.backtest_curve_strategy(yield_df, spreads, strategy_type)
                
                if backtest_results:
                    # Display metrics
                    bt_metrics = st.columns(5)
                    
                    if "total_return_strategy" in backtest_results:
                        bt_metrics[0].metric("Strategy Return", 
                                            f"{backtest_results['total_return_strategy']*100:.1f}%")
                        bt_metrics[1].metric("Buy & Hold Return", 
                                            f"{backtest_results['total_return_bh']*100:.1f}%")
                        bt_metrics[2].metric("Strategy Sharpe", 
                                            f"{backtest_results['sharpe_ratio_strategy']:.2f}")
                        bt_metrics[3].metric("Buy & Hold Sharpe", 
                                            f"{backtest_results['sharpe_ratio_bh']:.2f}")
                        bt_metrics[4].metric("Win Rate", 
                                            f"{backtest_results['win_rate']*100:.1f}%")
                    
                    # Plot results
                    fig_bt = chart_backtest_results(backtest_results)
                    st.plotly_chart(fig_bt, use_container_width=True)
                    
                    # Performance table
                    perf_metrics = BacktestEngine.calculate_performance_metrics(
                        yield_df["10Y"].pct_change()
                    )
                    if perf_metrics:
                        st.dataframe(pd.DataFrame([perf_metrics]).T.round(4), 
                                    use_container_width=True,
                                    column_config={0: "Value"})

    # Tab 4: Volatility Analytics
    with main_tabs[4]:
        st.subheader("📉 Options-Implied Volatility Analysis")
        st.markdown("""
        <div class="note-box">
        <b>Volatility Analytics Suite</b><br>
        Real-time analysis of VIX (S&P 500 volatility), VXN (Nasdaq volatility), and MOVE (bond volatility).
        Includes regime detection, spike identification, and vol-of-vol calculations.
        </div>
        """, unsafe_allow_html=True)
        
        if volatility_df is not None and not volatility_df.empty:
            if "^VIX" in volatility_df.columns:
                vix_analysis = VolatilityAnalyzer.calculate_volatility_regime(volatility_df["^VIX"])
                
                # Display VIX metrics
                vix_metrics = st.columns(4)
                vix_metrics[0].metric("Current VIX", f"{vix_analysis['current_vix']:.2f}")
                vix_metrics[1].metric("Regime", vix_analysis['regime'])
                vix_metrics[2].metric("Historical Percentile", vix_analysis['vix_percentile'])
                vix_metrics[3].metric("Mean VIX", f"{vix_analysis['mean_vix']:.2f}")
                
                st.markdown(f"""
                <div class="warning-box">
                <b>Volatility Outlook:</b> {vix_analysis['outlook']}<br>
                <b>Current Regime:</b> {vix_analysis['regime']} (Percentile: {vix_analysis['vix_percentile']})
                </div>
                """, unsafe_allow_html=True)
                
                # VIX chart
                fig_vix = chart_volatility_dashboard(volatility_df["^VIX"], vix_analysis)
                st.plotly_chart(fig_vix, use_container_width=True)
                
                # Identify volatility spikes
                spikes = VolatilityAnalyzer.identify_volatility_spikes(volatility_df["^VIX"])
                if not spikes.empty:
                    st.subheader("Volatility Spike Events (Last 5 Years)")
                    st.dataframe(spikes.tail(10), use_container_width=True)
        else:
            st.info("Volatility data unavailable. VIX and related indices will appear when market data is accessible.")

    # Tab 5: Correlations
    with main_tabs[5]:
        st.subheader("🔄 Cross-Asset Correlation Analysis")
        st.markdown("""
        <div class="note-box">
        <b>Cross-Asset Correlation Framework</b><br>
        Analyze relationships between Treasury yields and major asset classes (equities, FX, commodities).
        Identifies regime shifts, flight-to-quality dynamics, and portfolio diversification opportunities.
        </div>
        """, unsafe_allow_html=True)
        
        if correlation_df is not None and not correlation_df.empty:
            # Calculate correlation matrix
            all_assets = pd.concat([yield_df["10Y"], correlation_df], axis=1).dropna()
            all_assets.columns = ["10Y Yield"] + list(correlation_df.columns)
            corr_matrix = CorrelationAnalyzer.calculate_correlation_matrix(all_assets)
            
            # Display heatmap
            fig_corr = chart_correlation_heatmap(corr_matrix)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Rolling correlations
            st.subheader("Rolling Correlations with 10Y Yield (60-day window)")
            rolling_corr = CorrelationAnalyzer.calculate_rolling_correlations(
                yield_df["10Y"], correlation_df, window=60
            )
            
            fig_rolling = go.Figure()
            for col in rolling_corr.columns:
                fig_rolling.add_trace(go.Scatter(
                    x=rolling_corr.index, y=rolling_corr[col],
                    mode='lines', name=col.replace("Corr_10Y_", ""),
                    line=dict(width=1.5)
                ))
            fig_rolling.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_rolling.add_hline(y=0.5, line_dash="dash", line_color="green", annotation_text="Positive correlation")
            fig_rolling.add_hline(y=-0.5, line_dash="dash", line_color="red", annotation_text="Negative correlation")
            
            st.plotly_chart(create_chart_layout(fig_rolling, "Rolling Correlations with 10Y Yield", 
                                               "Correlation", 500), use_container_width=True)
            
            # Beta calculation
            if "S&P 500" in correlation_df.columns:
                beta = CorrelationAnalyzer.calculate_beta(yield_df["10Y"], correlation_df["S&P 500"])
                fig_beta = go.Figure()
                fig_beta.add_trace(go.Scatter(x=beta.index, y=beta, mode='lines',
                                             name='Beta (10Y vs S&P 500)',
                                             line=dict(color=COLORS["accent"], width=2)))
                fig_beta.add_hline(y=0, line_dash="dash")
                st.plotly_chart(create_chart_layout(fig_beta, "Rolling Beta (10Y Yield vs S&P 500)", 
                                                   "Beta", 400), use_container_width=True)
        else:
            st.info("Correlation data unavailable. Will appear when market data is accessible.")

    # Tab 6: Research (simplified)
    with main_tabs[6]:
        st.subheader("📐 Nelson-Siegel Model Parameters")
        if ns_result:
            st.dataframe(pd.DataFrame({
                "Parameter": ["β₀ (Level)", "β₁ (Slope)", "β₂ (Curvature)", "λ (Decay)", "RMSE"],
                "Value": [f"{ns_result['params'][0]:.4f}", f"{ns_result['params'][1]:.4f}",
                         f"{ns_result['params'][2]:.4f}", f"{ns_result['params'][3]:.4f}",
                         f"{ns_result['rmse']*100:.2f} bps"]
            }), use_container_width=True, hide_index=True)

    # Tab 7: Export
    with main_tabs[7]:
        st.subheader("💾 Export Data")
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.download_button("📊 Yield Data", yield_df.to_csv().encode("utf-8"),
                              f"yield_data_{datetime.now():%Y%m%d}.csv")
            st.download_button("📈 Spreads", spreads.to_csv().encode("utf-8"),
                              f"spreads_{datetime.now():%Y%m%d}.csv")
        with col_exp2:
            if volatility_df is not None:
                st.download_button("📉 Volatility Data", volatility_df.to_csv().encode("utf-8"),
                                  f"volatility_{datetime.now():%Y%m%d}.csv")
            if correlation_df is not None:
                st.download_button("🔄 Correlation Data", correlation_df.to_csv().encode("utf-8"),
                                  f"correlations_{datetime.now():%Y%m%d}.csv")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#667085; font-size:0.75rem;'>"
        "Institutional Quantitative Platform | Monte Carlo | ML Forecasting | Backtesting | Volatility Analytics | Correlations<br>"
        "MK Istanbul Fintech LabGEN © 2026 | All Rights Reserved"
        "</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
