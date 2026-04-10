import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# FULLY FIXED VERSION - All errors resolved
# =============================================================================

st.set_page_config(
    page_title="Dynamic Quantitative Analysis Platform | Institutional Fixed-Income",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# COLOR SCHEME
# =============================================================================

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

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

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
    "TLT": "20+ Year Treasury Bond ETF",
    "IEF": "7-10 Year Treasury Bond ETF",
    "SHY": "1-3 Year Treasury Bond ETF",
}

VOLATILITY_TICKERS = {
    "^VIX": "CBOE Volatility Index",
}

CORRELATION_TICKERS = {
    "^GSPC": "S&P 500",
    "QQQ": "Nasdaq 100",
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

# =============================================================================
# CUSTOM CSS
# =============================================================================

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

# =============================================================================
# CONFIGURATION
# =============================================================================

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
# DATA LAYER - FIXED
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
    df = pd.DataFrame(data).sort_index().dropna()
    if df.empty:
        return None
    return df

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
    data = {}
    for ticker in VOLATILITY_TICKERS:
        try:
            df = yf.download(ticker, period="2y", progress=False)
            if df is not None and not df.empty and 'Close' in df.columns:
                series = df['Close'].dropna()
                if len(series) > 0:
                    data[ticker] = series
        except Exception:
            continue
    
    if not data:
        return None
    
    try:
        df_result = pd.DataFrame(data)
        if df_result is not None and not df_result.empty:
            df_result = df_result.dropna()
            if not df_result.empty:
                return df_result
    except Exception:
        return None
    
    return None

@st.cache_data(ttl=CFG.cache_ttl_sec, show_spinner=False)
def fetch_correlation_data() -> Optional[pd.DataFrame]:
    data = {}
    for ticker, name in CORRELATION_TICKERS.items():
        try:
            df = yf.download(ticker, period="2y", progress=False)
            if df is not None and not df.empty and 'Close' in df.columns:
                series = df['Close'].dropna()
                if len(series) > 0:
                    data[name] = series
        except Exception:
            continue
    
    if not data:
        return None
    
    try:
        df_result = pd.DataFrame(data)
        if df_result is not None and not df_result.empty:
            df_result = df_result.dropna()
            if not df_result.empty:
                return df_result
    except Exception:
        return None
    
    return None

# =============================================================================
# ENHANCED TECHNICAL INDICATORS - FIXED
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

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range"""
    try:
        high = df['High']
        low = df['Low']
        close = df['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(n).mean()
    except Exception:
        return pd.Series(index=df.index, dtype=float)

def stochastic_oscillator(df: pd.DataFrame, n: int = 14) -> Tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator"""
    try:
        low_n = df['Low'].rolling(n).min()
        high_n = df['High'].rolling(n).max()
        k = 100 * (df['Close'] - low_n) / (high_n - low_n)
        d = k.rolling(3).mean()
        return k, d
    except Exception:
        return pd.Series(index=df.index, dtype=float), pd.Series(index=df.index, dtype=float)

def volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Volume analysis indicators - FIXED"""
    out = df.copy()
    if 'Volume' in df.columns and df['Volume'] is not None:
        try:
            out['Volume_SMA'] = df['Volume'].rolling(20).mean()
            # Avoid division by zero
            out['Volume_Ratio'] = df['Volume'] / out['Volume_SMA'].replace(0, np.nan)
            out['Volume_Ratio'] = out['Volume_Ratio'].fillna(1.0)
        except Exception:
            out['Volume_SMA'] = np.nan
            out['Volume_Ratio'] = 1.0
    else:
        out['Volume_SMA'] = np.nan
        out['Volume_Ratio'] = 1.0
    return out

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators - FIXED"""
    if df is None or df.empty:
        return df
    
    out = df.copy()
    
    try:
        out["SMA_20"] = sma(out["Close"], 20)
        out["SMA_50"] = sma(out["Close"], 50)
        out["SMA_200"] = sma(out["Close"], 200)
        out["EMA_12"] = ema(out["Close"], 12)
        out["EMA_26"] = ema(out["Close"], 26)
        out["RSI"] = rsi(out["Close"], 14)
        out["RSI_21"] = rsi(out["Close"], 21)
        out["MACD"], out["MACD_Signal"], out["MACD_Hist"] = macd(out["Close"])
        out["BB_Upper"], out["BB_Middle"], out["BB_Lower"] = bb(out["Close"])
        out["ATR"] = atr(out, 14)
        k, d = stochastic_oscillator(out, 14)
        out["Stoch_K"] = k
        out["Stoch_D"] = d
        out = volume_indicators(out)
    except Exception as e:
        st.warning(f"Technical indicator calculation warning: {str(e)}")
        # Fill with default values
        for col in ["SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26", "RSI", "RSI_21", 
                    "MACD", "MACD_Signal", "MACD_Hist", "BB_Upper", "BB_Middle", "BB_Lower", 
                    "ATR", "Stoch_K", "Stoch_D", "Volume_SMA", "Volume_Ratio"]:
            if col not in out.columns:
                out[col] = np.nan
    
    return out

def generate_technical_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive technical trading signals"""
    signals = pd.DataFrame(index=df.index)
    
    try:
        if 'RSI' in df.columns:
            signals['RSI_Oversold'] = df['RSI'] < 30
            signals['RSI_Overbought'] = df['RSI'] > 70
        
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            signals['MACD_Bullish'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
            signals['MACD_Bearish'] = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
        
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            signals['Golden_Cross'] = (df['SMA_20'] > df['SMA_50']) & (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1))
            signals['Death_Cross'] = (df['SMA_20'] < df['SMA_50']) & (df['SMA_20'].shift(1) >= df['SMA_50'].shift(1))
        
        if 'Close' in df.columns and 'BB_Lower' in df.columns and 'BB_Upper' in df.columns:
            signals['BB_Oversold'] = df['Close'] < df['BB_Lower']
            signals['BB_Overbought'] = df['Close'] > df['BB_Upper']
        
        if 'Stoch_K' in df.columns and 'Stoch_D' in df.columns:
            signals['Stoch_Oversold'] = df['Stoch_K'] < 20
            signals['Stoch_Overbought'] = df['Stoch_K'] > 80
            signals['Stoch_Bullish'] = (df['Stoch_K'] > df['Stoch_D']) & (df['Stoch_K'].shift(1) <= df['Stoch_D'].shift(1))
    except Exception:
        pass
    
    return signals

# =============================================================================
# MONTE CARLO SIMULATIONS
# =============================================================================

class MonteCarloSimulator:
    @staticmethod
    def simulate_geometric_brownian_motion(initial_yield: float, mu: float, sigma: float, 
                                           days: int, simulations: int = 1000) -> np.ndarray:
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
        lower_percentile = (1 - confidence) / 2 * 100
        upper_percentile = (1 + confidence) / 2 * 100
        
        return {
            "mean": np.mean(paths, axis=0),
            "median": np.percentile(paths, 50, axis=0),
            "lower_ci": np.percentile(paths, lower_percentile, axis=0),
            "upper_ci": np.percentile(paths, upper_percentile, axis=0),
            "std": np.std(paths, axis=0),
        }
    
    @staticmethod
    def calculate_var_from_paths(paths: np.ndarray, confidence: float = 0.95) -> float:
        final_values = paths[:, -1]
        return np.percentile(final_values, (1 - confidence) * 100)

# =============================================================================
# ML FORECASTING
# =============================================================================

class MLForecastModel:
    @staticmethod
    def prepare_features(yield_df: pd.DataFrame, lags: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(lags, len(yield_df) - 1):
            features = []
            for col in yield_df.columns:
                features.extend(yield_df[col].iloc[i-lags:i].values)
            X.append(features)
            y.append(yield_df.iloc[i+1].values)
        
        if not X:
            return np.array([]), np.array([])
        return np.array(X), np.array(y)
    
    @staticmethod
    def train_random_forest(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict:
        if len(X) == 0:
            return {}
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        feature_importance = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(X.shape[1])],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            "model": model,
            "rmse": rmse,
            "r2": r2,
            "feature_importance": feature_importance,
        }

# =============================================================================
# BACKTESTING
# =============================================================================

class BacktestEngine:
    @staticmethod
    def backtest_curve_strategy(yield_df: pd.DataFrame, spreads: pd.DataFrame,
                                strategy_type: str = "curve_inversion") -> Dict:
        if strategy_type == "curve_inversion":
            if "10Y-2Y" not in spreads.columns or "10Y" not in yield_df.columns:
                return {}
            
            signals = spreads["10Y-2Y"] < 0
            returns = yield_df["10Y"].pct_change().shift(-1)
            strategy_returns = signals.shift(1) * returns
            cumulative_strategy = (1 + strategy_returns.fillna(0)).cumprod()
            
            if strategy_returns.std() == 0:
                sharpe = 0
            else:
                sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            
            return {
                "strategy_name": "Curve Inversion Strategy",
                "cumulative_returns": cumulative_strategy,
                "sharpe_ratio": sharpe,
                "total_return": cumulative_strategy.iloc[-1] - 1,
                "win_rate": (strategy_returns[strategy_returns != 0] > 0).mean() if len(strategy_returns[strategy_returns != 0]) > 0 else 0,
            }
        return {}

# =============================================================================
# VOLATILITY ANALYZER
# =============================================================================

class VolatilityAnalyzer:
    @staticmethod
    def calculate_volatility_regime(vix: pd.Series) -> Dict:
        if vix is None or len(vix) == 0:
            return {"current_vix": 0, "regime": "N/A", "outlook": "Data unavailable"}
        
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
            outlook = "Crisis conditions"
        
        percentile = (vix < current_vix).mean() if len(vix) > 0 else 0.5
        
        return {
            "current_vix": current_vix,
            "regime": regime,
            "outlook": outlook,
            "vix_percentile": f"{percentile * 100:.1f}%",
        }
    
    @staticmethod
    def calculate_vol_of_vol(vix: pd.Series, window: int = 20) -> pd.Series:
        if vix is None or len(vix) < window:
            return pd.Series()
        return vix.pct_change().rolling(window).std() * np.sqrt(252)

# =============================================================================
# CORRELATION ANALYZER
# =============================================================================

class CorrelationAnalyzer:
    @staticmethod
    def calculate_correlation_matrix(assets_df: pd.DataFrame) -> pd.DataFrame:
        if assets_df is None or assets_df.empty:
            return pd.DataFrame()
        returns = assets_df.pct_change().dropna()
        return returns.corr()

# =============================================================================
# ANALYTICS FUNCTIONS
# =============================================================================

def compute_spreads(yield_df: pd.DataFrame) -> pd.DataFrame:
    spreads = pd.DataFrame(index=yield_df.index)
    if {"10Y", "2Y"}.issubset(yield_df.columns):
        spreads["10Y-2Y"] = (yield_df["10Y"] - yield_df["2Y"]) * 100
    if {"10Y", "3M"}.issubset(yield_df.columns):
        spreads["10Y-3M"] = (yield_df["10Y"] - yield_df["3M"]) * 100
    return spreads

def classify_regime(spreads: pd.DataFrame, yield_df: pd.DataFrame) -> Tuple[str, str]:
    if "10Y-2Y" not in spreads.columns or spreads.empty:
        return "Data Loading", "Please wait for data to load"
    
    spread = spreads["10Y-2Y"].iloc[-1]
    if np.isfinite(spread) and spread < 0:
        return "Risk-off / Recession Watch", "Curve inversion signals defensive macro regime"
    elif np.isfinite(spread) and spread < 50:
        return "Neutral / Late Cycle", "Curve flattening suggests late-cycle caution"
    return "Risk-on / Expansion", "Positive slope supports pro-risk positioning"

def recession_probability_proxy(spreads: pd.DataFrame, yield_df: pd.DataFrame) -> float:
    if "10Y-2Y" not in spreads.columns or "10Y" not in yield_df.columns:
        return 0.5
    
    score = np.clip((-spreads["10Y-2Y"].iloc[-1]) / 100, 0, 1.5)
    score += np.clip((yield_df["10Y"].iloc[-1] - 4.5) / 3.0, 0, 1.0)
    return float(1 / (1 + np.exp(-2.2 * (score - 0.8))))

def identify_recessions(recession_series: Optional[pd.Series]) -> List[dict]:
    if recession_series is None or len(recession_series) == 0:
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
        if len(maturities) == 0 or len(yields_) == 0:
            return None
            
        def objective(params):
            fitted = NelsonSiegelModel.nelson_siegel(maturities, *params)
            return np.sum((yields_ - fitted) ** 2)
        
        bounds = [(yields_.min() - 2, yields_.max() + 2), (-15, 15), (-15, 15), (0.01, 5)]
        best = None
        best_fun = np.inf
        
        for _ in range(5):
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
            "rmse": float(np.sqrt(np.mean((yields_ - fitted) ** 2))),
            "r_squared": float(1 - sse / sst) if sst > 0 else np.nan,
        }

# =============================================================================
# VISUALIZATION FUNCTIONS
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

def add_recession_bands(fig: go.Figure, recessions: List[dict]) -> go.Figure:
    for rec in recessions:
        fig.add_vrect(x0=rec["start"], x1=rec["end"], 
                      fillcolor=COLORS["recession"], opacity=0.35, 
                      layer="below", line_width=0)
    return fig

def chart_current_curve(maturities: np.ndarray, yields_: np.ndarray, 
                        ns_result: Optional[dict], nss_result: Optional[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=maturities, y=yields_, mode="markers+lines", 
        name="Actual Market Curve",
        marker=dict(size=12, color=COLORS["accent"], symbol="circle"),
        line=dict(color=COLORS["accent"], width=2.5),
    ))
    if ns_result:
        fig.add_trace(go.Scatter(
            x=maturities, y=ns_result["fitted_values"], 
            mode="lines", name="Nelson-Siegel (3-factor)",
            line=dict(color=COLORS["accent2"], width=2.5)
        ))
    if nss_result:
        fig.add_trace(go.Scatter(
            x=maturities, y=nss_result["fitted_values"], 
            mode="lines", name="Nelson-Siegel-Svensson (4-factor)",
            line=dict(color=COLORS["warning"], width=2.5, dash="dot")
        ))
    return create_chart_layout(fig, "Current Treasury Yield Curve with Model Fits", 
                               "Yield (%)", 480, "Maturity (Years)")

def chart_monte_carlo(initial_yield: float, simulation_results: Dict, 
                      horizon_days: int) -> go.Figure:
    fig = go.Figure()
    x_axis = np.arange(horizon_days)
    
    fig.add_trace(go.Scatter(
        x=x_axis, y=simulation_results["upper_ci"],
        fill=None, mode='lines', line=dict(color='rgba(0,0,0,0)'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x_axis, y=simulation_results["lower_ci"],
        fill='tonexty', mode='lines', 
        fillcolor='rgba(44, 95, 138, 0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name='95% Confidence Interval'
    ))
    
    fig.add_trace(go.Scatter(
        x=x_axis, y=simulation_results["mean"],
        mode='lines', name='Mean Path',
        line=dict(color=COLORS["accent"], width=2.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0], y=[initial_yield],
        mode='markers', name='Current Yield',
        marker=dict(size=12, color=COLORS["positive"], symbol='star')
    ))
    
    return create_chart_layout(fig, "Monte Carlo Simulation - 10Y Yield Paths", 
                               "Yield (%)", 500, "Trading Days Ahead")

def chart_backtest_results(backtest_results: Dict) -> go.Figure:
    fig = go.Figure()
    
    if "cumulative_returns" in backtest_results and backtest_results["cumulative_returns"] is not None:
        fig.add_trace(go.Scatter(
            x=backtest_results["cumulative_returns"].index,
            y=backtest_results["cumulative_returns"].values,
            mode='lines',
            name=backtest_results.get("strategy_name", "Strategy"),
            line=dict(color=COLORS["accent"], width=2.5)
        ))
    
    return create_chart_layout(fig, "Backtest Performance", "Cumulative Return", 500)

def chart_volatility_dashboard(vix_data: pd.Series, vol_regime: Dict) -> go.Figure:
    if vix_data is None or len(vix_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Volatility data unavailable", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("VIX - CBOE Volatility Index", "Vol of Vol (20-day)"),
                        vertical_spacing=0.15,
                        row_heights=[0.6, 0.4])
    
    fig.add_trace(go.Scatter(
        x=vix_data.index, y=vix_data.values,
        mode='lines', name='VIX',
        line=dict(color=COLORS["warning"], width=2)
    ), row=1, col=1)
    
    fig.add_hline(y=20, line_dash="dash", line_color="orange", row=1, col=1)
    fig.add_hline(y=15, line_dash="dash", line_color="green", row=1, col=1)
    
    vol_of_vol = VolatilityAnalyzer.calculate_vol_of_vol(vix_data)
    if len(vol_of_vol) > 0:
        fig.add_trace(go.Scatter(
            x=vix_data.index, y=vol_of_vol,
            mode='lines', name='Vol of Vol',
            line=dict(color=COLORS["accent"], width=2)
        ), row=2, col=1)
    
    fig.update_yaxes(title_text="VIX", row=1, col=1)
    fig.update_yaxes(title_text="Vol of Vol", row=2, col=1)
    
    return create_chart_layout(fig, f"Volatility Dashboard | VIX: {vol_regime.get('current_vix', 0):.2f}", height=600)

def chart_correlation_heatmap(correlation_matrix: pd.DataFrame) -> go.Figure:
    if correlation_matrix.empty:
        fig = go.Figure()
        fig.add_annotation(text="Correlation data unavailable", x=0.5, y=0.5, showarrow=False)
        return fig
    
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
    
    fig.update_layout(title="Cross-Asset Correlation Matrix", height=500, width=700)
    return fig

def chart_ohlc(ohlc_df: pd.DataFrame, ticker: str) -> Optional[go.Figure]:
    if ohlc_df is None or ohlc_df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=ohlc_df.index, open=ohlc_df["Open"], high=ohlc_df["High"], 
        low=ohlc_df["Low"], close=ohlc_df["Close"],
        increasing=dict(line=dict(color=COLORS["positive"]), fillcolor=COLORS["positive"]),
        decreasing=dict(line=dict(color=COLORS["negative"]), fillcolor=COLORS["negative"]),
        name=ticker,
    ))
    if "SMA_20" in ohlc_df.columns:
        fig.add_trace(go.Scatter(
            x=ohlc_df.index, y=ohlc_df["SMA_20"], mode="lines", 
            name="SMA 20", line=dict(color=COLORS["accent"], width=1.5)
        ))
    if "SMA_50" in ohlc_df.columns:
        fig.add_trace(go.Scatter(
            x=ohlc_df.index, y=ohlc_df["SMA_50"], mode="lines", 
            name="SMA 50", line=dict(color=COLORS["warning"], width=1.5)
        ))
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))
    return create_chart_layout(fig, f"OHLC Price Analysis | {ticker}", "Price (USD)", 550)

def chart_technical_panels(ohlc_df: pd.DataFrame, ticker: str, signals_df: pd.DataFrame) -> Optional[go.Figure]:
    if ohlc_df is None or ohlc_df.empty:
        return None
    
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(
            f"{ticker} Price with Technical Overlays",
            "RSI (14) - Momentum Oscillator",
            "MACD - Trend & Momentum",
        ),
    )
    
    # Price panel
    fig.add_trace(go.Scatter(
        x=ohlc_df.index, y=ohlc_df["Close"], mode="lines", 
        name="Close Price", line=dict(color=COLORS["accent"], width=2.5)
    ), row=1, col=1)
    
    if "BB_Upper" in ohlc_df.columns:
        fig.add_trace(go.Scatter(
            x=ohlc_df.index, y=ohlc_df["BB_Upper"], mode="lines", 
            name="BB Upper (2σ)", line=dict(color=COLORS["muted"], width=1.2, dash="dash")
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=ohlc_df.index, y=ohlc_df["BB_Lower"], mode="lines", 
            name="BB Lower (2σ)", line=dict(color=COLORS["muted"], width=1.2, dash="dash"),
            fill='tonexty', fillcolor=COLORS["band"]
        ), row=1, col=1)
    
    # RSI panel
    if "RSI" in ohlc_df.columns:
        fig.add_trace(go.Scatter(
            x=ohlc_df.index, y=ohlc_df["RSI"], mode="lines", 
            name="RSI", line=dict(color=COLORS["accent"], width=2)
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=COLORS["negative"], 
                      line_width=1.5, row=2, col=1, annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color=COLORS["positive"], 
                      line_width=1.5, row=2, col=1, annotation_text="Oversold (30)")
    
    # MACD panel
    if "MACD" in ohlc_df.columns:
        fig.add_trace(go.Scatter(
            x=ohlc_df.index, y=ohlc_df["MACD"], mode="lines", 
            name="MACD Line", line=dict(color=COLORS["positive"], width=2)
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=ohlc_df.index, y=ohlc_df["MACD_Signal"], mode="lines", 
            name="Signal Line", line=dict(color=COLORS["negative"], width=2)
        ), row=3, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color=COLORS["text_secondary"], row=3, col=1)
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return create_chart_layout(fig, f"Technical Analysis Dashboard | {ticker}", height=800)

# =============================================================================
# UI HELPERS
# =============================================================================

def render_api_gate() -> None:
    st.markdown(
        f"""
        <div class="note-box" style="max-width:560px; margin:40px auto; text-align:center;">
            <b>🔑 FRED API Key Required</b><br><br>
            This platform requires live U.S. Treasury data from FRED.<br>
            Get your free API key from the FRED website.
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
# MAIN APPLICATION
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
        mc_simulations = st.slider("Monte Carlo simulations", 500, 2000, 1000, 500)
        ml_lags = st.slider("ML model lags", 3, 8, 5)
        ohlc_period = st.selectbox("OHLC data period", ["6mo", "1y", "2y", "5y"], index=2)

    if not st.session_state.data_fetched:
        with st.spinner("Fetching all data..."):
            yield_df = fetch_all_yield_data(st.session_state.api_key)
            recession_series = fetch_recession_data(st.session_state.api_key)
            volatility_df = fetch_volatility_data()
            correlation_df = fetch_correlation_data()
            
            ohlc_data = {}
            for ticker in YAHOO_TICKERS.keys():
                try:
                    df = fetch_ohlc_data(ticker, ohlc_period)
                    if df is not None and not df.empty:
                        df_with_indicators = add_technical_indicators(df)
                        if df_with_indicators is not None and not df_with_indicators.empty:
                            ohlc_data[ticker] = df_with_indicators
                except Exception as e:
                    st.warning(f"Could not load {ticker}: {str(e)}")
                    continue
        
        if yield_df is None:
            st.error("Failed to fetch FRED data. Please check your API key and try again.")
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

    selected_cols = [c for c in yield_df.columns if c in MATURITY_MAP][:6]
    maturities = np.array([MATURITY_MAP[c] for c in selected_cols])
    latest_curve = yield_df.iloc[-1][selected_cols].values.astype(float)

    spreads = compute_spreads(yield_df)
    recessions = identify_recessions(recession_series)
    regime, regime_text = classify_regime(spreads, yield_df)
    recession_prob = recession_probability_proxy(spreads, yield_df)

    current_2y = yield_df["2Y"].iloc[-1] if "2Y" in yield_df.columns else np.nan
    current_10y = yield_df["10Y"].iloc[-1] if "10Y" in yield_df.columns else np.nan
    current_spread = spreads["10Y-2Y"].iloc[-1] if "10Y-2Y" in spreads.columns else np.nan

    ns_result = NelsonSiegelModel.fit_ns(maturities, latest_curve)

    # KPI Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        kpi_card("📊 Macro Regime", regime, regime_text[:40] + "...")
    with c2:
        kpi_card("🏦 2Y Yield", f"{current_2y:.2f}%" if not np.isnan(current_2y) else "N/A", "Policy anchor")
    with c3:
        kpi_card("📈 10Y Yield", f"{current_10y:.2f}%" if not np.isnan(current_10y) else "N/A", "Benchmark")
    with c4:
        kpi_card("🔄 10Y-2Y Spread", f"{current_spread:.1f} bps" if not np.isnan(current_spread) else "N/A", "Recession signal")
    with c5:
        kpi_card("⚠️ Recession Prob", f"{100 * recession_prob:.1f}%", "Proxy estimate")
    with c6:
        if volatility_df is not None and not volatility_df.empty and "^VIX" in volatility_df.columns:
            current_vix = volatility_df["^VIX"].iloc[-1]
            kpi_card("📉 VIX", f"{current_vix:.2f}", "Fear gauge")
        else:
            kpi_card("📉 VIX", "N/A", "Data loading")

    # Main Tabs
    main_tabs = st.tabs([
        "🏦 Executive View",
        "🎲 Monte Carlo",
        "🤖 ML Forecasting",
        "📊 Backtesting",
        "📉 Volatility",
        "🔄 Correlations",
        "⚡ Technical Analysis",
        "📐 Research",
        "💾 Export"
    ])

    # Tab 0: Executive View
    with main_tabs[0]:
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=maturities, y=latest_curve, mode='lines+markers', 
                                       name='Current Curve', line=dict(color=COLORS["accent"], width=2.5),
                                       marker=dict(size=10)))
        st.plotly_chart(create_chart_layout(fig_curve, "Current Treasury Yield Curve", 
                                           "Yield (%)", 450, "Maturity (Years)"), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_spread = go.Figure()
            if "10Y-2Y" in spreads.columns:
                fig_spread.add_trace(go.Scatter(x=spreads.index, y=spreads["10Y-2Y"], 
                                               mode='lines', name='10Y-2Y',
                                               line=dict(color=COLORS["warning"], width=2)))
                fig_spread.add_hline(y=0, line_dash="dash")
                add_recession_bands(fig_spread, recessions)
                st.plotly_chart(create_chart_layout(fig_spread, "10Y-2Y Spread History", 
                                                   "Basis Points", 400), use_container_width=True)
        
        with col2:
            fig_yield = go.Figure()
            fig_yield.add_trace(go.Scatter(x=yield_df.index, y=yield_df["10Y"], 
                                          mode='lines', name='10Y Yield',
                                          line=dict(color=COLORS["accent"], width=2)))
            st.plotly_chart(create_chart_layout(fig_yield, "10Y Yield History", 
                                               "Yield (%)", 400), use_container_width=True)

    # Tab 1: Monte Carlo
    with main_tabs[1]:
        st.subheader("🎲 Monte Carlo Simulation")
        st.markdown('<div class="note-box">Simulates 1,000+ potential paths for 10Y yields.</div>', 
                   unsafe_allow_html=True)
        
        if st.button("Run Monte Carlo Simulation", use_container_width=True):
            with st.spinner(f"Running {mc_simulations} simulations..."):
                initial_yield = current_10y if not np.isnan(current_10y) else 4.0
                mu = yield_df["10Y"].pct_change().mean() * 252 if len(yield_df) > 1 else 0
                sigma = yield_df["10Y"].pct_change().std() * np.sqrt(252) if len(yield_df) > 1 else 0.1
                
                paths = MonteCarloSimulator.simulate_geometric_brownian_motion(
                    initial_yield, mu, sigma, forecast_horizon, mc_simulations
                )
                sim_results = MonteCarloSimulator.calculate_confidence_intervals(paths, confidence_level)
                var_estimate = MonteCarloSimulator.calculate_var_from_paths(paths, confidence_level)
                
                col_mc1, col_mc2, col_mc3 = st.columns(3)
                col_mc1.metric("Expected Yield", f"{sim_results['mean'][-1]:.2f}%")
                col_mc2.metric(f"{int(confidence_level*100)}% VaR", f"{var_estimate:.2f}%")
                col_mc3.metric("Uncertainty Range", f"±{sim_results['std'][-1]:.2f}%")
                
                fig_mc = chart_monte_carlo(initial_yield, sim_results, forecast_horizon)
                st.plotly_chart(fig_mc, use_container_width=True)

    # Tab 2: ML Forecasting
    with main_tabs[2]:
        st.subheader("🤖 Machine Learning Forecast")
        
        if st.button("Train ML Model", use_container_width=True):
            with st.spinner("Training Random Forest..."):
                X, y = MLForecastModel.prepare_features(yield_df[selected_cols], lags=ml_lags)
                
                if len(X) > 50:
                    ml_results = MLForecastModel.train_random_forest(X, y)
                    
                    col_ml1, col_ml2 = st.columns(2)
                    col_ml1.metric("RMSE", f"{ml_results.get('rmse', 0)*100:.2f} bps")
                    col_ml2.metric("R² Score", f"{ml_results.get('r2', 0):.3f}")
                    
                    st.success(f"Model trained on {len(X)} samples")
                else:
                    st.warning(f"Insufficient data. Need 50+ samples, have {len(X)}.")

    # Tab 3: Backtesting
    with main_tabs[3]:
        st.subheader("📊 Strategy Backtesting")
        
        if st.button("Run Backtest", use_container_width=True):
            with st.spinner("Running backtest..."):
                backtest_results = BacktestEngine.backtest_curve_strategy(yield_df, spreads, "curve_inversion")
                
                if backtest_results:
                    col_bt1, col_bt2, col_bt3, col_bt4 = st.columns(4)
                    col_bt1.metric("Strategy Return", f"{backtest_results.get('total_return', 0)*100:.1f}%")
                    col_bt2.metric("Sharpe Ratio", f"{backtest_results.get('sharpe_ratio', 0):.2f}")
                    col_bt3.metric("Win Rate", f"{backtest_results.get('win_rate', 0)*100:.1f}%")
                    
                    fig_bt = chart_backtest_results(backtest_results)
                    st.plotly_chart(fig_bt, use_container_width=True)
                else:
                    st.warning("Insufficient data for backtesting")

    # Tab 4: Volatility
    with main_tabs[4]:
        st.subheader("📉 Volatility Analytics")
        
        if volatility_df is not None and not volatility_df.empty and "^VIX" in volatility_df.columns:
            vix_analysis = VolatilityAnalyzer.calculate_volatility_regime(volatility_df["^VIX"])
            
            col_vix1, col_vix2, col_vix3 = st.columns(3)
            col_vix1.metric("Current VIX", f"{vix_analysis['current_vix']:.2f}")
            col_vix2.metric("Regime", vix_analysis['regime'])
            col_vix3.metric("Percentile", vix_analysis['vix_percentile'])
            
            st.markdown(f'<div class="warning-box">📊 {vix_analysis["outlook"]}</div>', unsafe_allow_html=True)
            
            fig_vix = chart_volatility_dashboard(volatility_df["^VIX"], vix_analysis)
            st.plotly_chart(fig_vix, use_container_width=True)
        else:
            st.info("Volatility data unavailable. Will appear when market data is accessible.")

    # Tab 5: Correlations
    with main_tabs[5]:
        st.subheader("🔄 Cross-Asset Correlations")
        
        if correlation_df is not None and not correlation_df.empty:
            all_assets = pd.concat([yield_df["10Y"], correlation_df], axis=1).dropna()
            all_assets.columns = ["10Y Yield"] + list(correlation_df.columns)
            corr_matrix = CorrelationAnalyzer.calculate_correlation_matrix(all_assets)
            
            if not corr_matrix.empty:
                fig_corr = chart_correlation_heatmap(corr_matrix)
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Correlation data unavailable. Will appear when market data is accessible.")

    # Tab 6: Technical Analysis
    with main_tabs[6]:
        st.subheader("⚡ Technical Analysis")
        
        ticker = st.selectbox("Select Instrument", list(YAHOO_TICKERS.keys()), 
                             format_func=lambda x: f"{x} | {YAHOO_TICKERS[x]}")
        ohlc_df = ohlc_data.get(ticker)
        
        if ohlc_df is not None and not ohlc_df.empty:
            signals_df = generate_technical_signals(ohlc_df)
            
            # Current signals summary
            st.subheader("Current Technical Signals")
            sig_cols = st.columns(4)
            with sig_cols[0]:
                rsi_val = ohlc_df["RSI"].iloc[-1] if "RSI" in ohlc_df.columns else 50
                rsi_signal = "🟢 Oversold" if rsi_val < 30 else ("🔴 Overbought" if rsi_val > 70 else "⚪ Neutral")
                st.metric("RSI", f"{rsi_val:.1f}", rsi_signal)
            with sig_cols[1]:
                macd_val = ohlc_df["MACD"].iloc[-1] if "MACD" in ohlc_df.columns else 0
                macd_signal = ohlc_df["MACD_Signal"].iloc[-1] if "MACD_Signal" in ohlc_df.columns else 0
                macd_cross = "🟢 Bullish" if macd_val > macd_signal else ("🔴 Bearish" if macd_val < macd_signal else "⚪ Neutral")
                st.metric("MACD", f"{macd_val:.4f}", macd_cross)
            with sig_cols[2]:
                bb_position = "🟢 Below Lower" if ohlc_df["Close"].iloc[-1] < ohlc_df["BB_Lower"].iloc[-1] else ("🔴 Above Upper" if ohlc_df["Close"].iloc[-1] > ohlc_df["BB_Upper"].iloc[-1] else "⚪ Within Bands")
                st.metric("Bollinger Bands", bb_position, "Mean reversion signal")
            with sig_cols[3]:
                vol_ratio = ohlc_df["Volume_Ratio"].iloc[-1] if "Volume_Ratio" in ohlc_df.columns else 1.0
                vol_signal = "📈 High Volume" if vol_ratio > 1.5 else ("📉 Low Volume" if vol_ratio < 0.7 else "📊 Normal")
                st.metric("Volume Ratio", f"{vol_ratio:.2f}x", vol_signal)
            
            left, right = st.columns([1.15, 1])
            with left:
                fig_ohlc = chart_ohlc(ohlc_df, ticker)
                if fig_ohlc:
                    st.plotly_chart(fig_ohlc, use_container_width=True)
            with right:
                fig_ta = chart_technical_panels(ohlc_df, ticker, signals_df)
                if fig_ta:
                    st.plotly_chart(fig_ta, use_container_width=True)
        else:
            st.warning(f"No technical data available for {ticker}")

    # Tab 7: Research
    with main_tabs[7]:
        st.subheader("📐 Nelson-Siegel Model")
        if ns_result:
            st.dataframe(pd.DataFrame({
                "Parameter": ["β₀ (Level)", "β₁ (Slope)", "β₂ (Curvature)", "λ (Decay)", "RMSE"],
                "Value": [f"{ns_result['params'][0]:.4f}", f"{ns_result['params'][1]:.4f}",
                         f"{ns_result['params'][2]:.4f}", f"{ns_result['params'][3]:.4f}",
                         f"{ns_result['rmse']*100:.2f} bps"]
            }), use_container_width=True, hide_index=True)

    # Tab 8: Export
    with main_tabs[8]:
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

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#667085; font-size:0.75rem;'>"
        "Institutional Quantitative Platform | MK Istanbul Fintech LabGEN © 2026 | All Rights Reserved"
        "</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
