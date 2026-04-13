import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm, skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# =============================================================================
# V36 ENHANCED INSTITUTIONAL PLATFORM
# - Smarter KPI Layout with trend indicators
# - Enhanced visualizations with recession bands
# - Integrated Monte Carlo, ML, and Backtesting
# - Professional risk metrics with Cornish-Fisher VaR
# - Real-time market correlation analysis
# =============================================================================

st.set_page_config(
    page_title="Yield Curve Institutional Platform v36",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# ENHANCED DESIGN SYSTEM
# =============================================================================

COLORS = {
    "bg": "#eef2f7",
    "bg2": "#f7f9fc",
    "surface": "#ffffff",
    "surface_alt": "#f5f7fb",
    "header": "#1a2a3a",
    "grid": "#c8d4e0",
    "grid_dark": "#97a8b8",
    "text": "#1a2a3a",
    "text_secondary": "#4a5a6a",
    "muted": "#667085",
    "accent": "#2c5f8a",
    "accent2": "#4a7c59",
    "accent3": "#c17f3a",
    "accent4": "#8b5cf6",
    "positive": "#10b981",
    "negative": "#ef4444",
    "warning": "#f59e0b",
    "info": "#3b82f6",
    "recession": "rgba(120, 130, 145, 0.18)",
    "band": "rgba(108, 142, 173, 0.10)",
    "shadow": "rgba(16, 24, 40, 0.08)",
    "gradient_start": "#1a2a3a",
    "gradient_end": "#2c5f8a",
}

# Enhanced CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(135deg, {COLORS['bg']} 0%, {COLORS['bg2']} 100%);
    }}
    .main-title-card {{
        background: linear-gradient(135deg, {COLORS['gradient_start']} 0%, {COLORS['gradient_end']} 100%);
        border-radius: 20px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
        border: 1px solid rgba(255,255,255,0.08);
    }}
    .main-title {{
        color: white;
        font-weight: 800;
        font-size: 1.65rem;
        margin: 0;
        letter-spacing: -0.02em;
    }}
    .main-subtitle {{
        color: rgba(255,255,255,0.86);
        font-size: 0.86rem;
        margin-top: 0.55rem;
        font-family: "Courier New", monospace;
    }}
    .metric-card {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['grid']};
        border-radius: 16px;
        padding: 1rem;
        min-height: 120px;
        box-shadow: 0 4px 16px {COLORS['shadow']};
        transition: all 0.2s ease;
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
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 0.45rem;
        font-family: "Courier New", monospace;
    }}
    .metric-sub {{
        color: {COLORS['muted']};
        font-size: 0.75rem;
        margin-top: 0.35rem;
        line-height: 1.3;
    }}
    .metric-trend-up {{
        color: {COLORS['positive']};
        font-weight: 600;
    }}
    .metric-trend-down {{
        color: {COLORS['negative']};
        font-weight: 600;
    }}
    .note-box {{
        background: {COLORS['surface_alt']};
        border: 1px solid {COLORS['grid']};
        border-left: 4px solid {COLORS['accent']};
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: {COLORS['text']};
        font-size: 0.88rem;
        line-height: 1.55;
        margin: 1rem 0;
    }}
    .warning-box {{
        background: #fff8f0;
        border: 1px solid #f0d8b0;
        border-left: 4px solid {COLORS['warning']};
        border-radius: 12px;
        padding: 1rem 1.2rem;
        font-size: 0.88rem;
        line-height: 1.55;
        margin: 1rem 0;
    }}
    .success-box {{
        background: #f3fbf6;
        border: 1px solid #bfe3ca;
        border-left: 4px solid {COLORS['positive']};
        border-radius: 12px;
        padding: 1rem 1.2rem;
        font-size: 0.88rem;
        line-height: 1.55;
        margin: 1rem 0;
    }}
    .info-box {{
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-left: 4px solid {COLORS['info']};
        border-radius: 12px;
        padding: 1rem 1.2rem;
        font-size: 0.88rem;
        line-height: 1.55;
        margin: 1rem 0;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        border-bottom: 2px solid {COLORS['grid']};
        flex-wrap: wrap !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {COLORS['text_secondary']};
        font-weight: 700;
        font-size: 0.74rem;
        text-transform: uppercase;
        white-space: nowrap;
        padding: 8px 14px;
        letter-spacing: 0.03em;
    }}
    .stTabs [aria-selected="true"] {{
        color: {COLORS['accent']};
        border-bottom: 3px solid {COLORS['accent']};
    }}
    .stButton > button, .stDownloadButton > button {{
        background: {COLORS['surface']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['grid']};
        border-radius: 10px;
        font-weight: 700;
        transition: all 0.2s ease;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        border-color: {COLORS['accent']};
        color: {COLORS['accent']};
        transform: translateY(-1px);
    }}
    .stMetric {{
        background: {COLORS['surface']};
        border-radius: 12px;
        padding: 0.8rem;
    }}
    #MainMenu, header, footer {{
        visibility: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# ENHANCED CONFIGURATION
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
    "1Y": 1.0,
    "2Y": 2.0,
    "3Y": 3.0,
    "5Y": 5.0,
    "7Y": 7.0,
    "10Y": 10.0,
    "20Y": 20.0,
    "30Y": 30.0,
}

YAHOO_TICKERS = {
    "^TNX": "10Y Treasury Yield Index",
    "^FVX": "5Y Treasury Yield Index",
    "^IRX": "13W T-Bill Index",
    "TLT": "20+Y Treasury Bond ETF",
    "IEF": "7-10Y Treasury Bond ETF",
    "SHY": "1-3Y Treasury Bond ETF",
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq 100 ETF",
}

VOLATILITY_TICKERS = {
    "^VIX": "CBOE Volatility Index",
    "^VXN": "Nasdaq Volatility Index",
    "^MOVE": "MOVE Index",
}

CORRELATION_TICKERS = {
    "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq Composite",
    "GLD": "Gold",
    "UUP": "US Dollar Index",
    "TLT": "Long-term Treasuries",
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

for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v

@dataclass
class RuntimeConfig:
    history_start: str = "1990-01-01"
    timeout: int = 25
    max_retries: int = 3
    cache_ttl_sec: int = 3600
    rolling_step: int = 21
    rolling_years_default: int = 5
    forecast_horizon_default: int = 20
    mc_simulations_default: int = 5000
    var_confidence_default: float = 0.95

CFG = RuntimeConfig()

# =============================================================================
# ENHANCED DATA LAYER
# =============================================================================

def fred_request(api_key: str, series_id: str) -> Optional[pd.Series]:
    """Fetch data from FRED API with improved error handling"""
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
            obs = r.json().get("observations", [])
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
    return pd.DataFrame(data).sort_index().dropna(how="all")

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

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_bundle(start_date=None, end_date=None):
    """Fetch comprehensive market data for volatility and correlation analysis"""
    vol_data = {}
    corr_data = {}
    
    for ticker, name in VOLATILITY_TICKERS.items():
        try:
            raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if raw is not None and not raw.empty:
                if 'Adj Close' in raw.columns:
                    s = raw['Adj Close']
                elif 'Close' in raw.columns:
                    s = raw['Close']
                else:
                    s = raw.iloc[:, 0]
                s = pd.to_numeric(s, errors='coerce').dropna()
                if not s.empty:
                    s.index = s.index.tz_localize(None)
                    vol_data[name] = s
        except Exception:
            continue
    
    for ticker, name in CORRELATION_TICKERS.items():
        try:
            raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if raw is not None and not raw.empty:
                if 'Adj Close' in raw.columns:
                    s = raw['Adj Close']
                elif 'Close' in raw.columns:
                    s = raw['Close']
                else:
                    s = raw.iloc[:, 0]
                s = pd.to_numeric(s, errors='coerce').dropna()
                if not s.empty:
                    s.index = s.index.tz_localize(None)
                    corr_data[name] = s
        except Exception:
            continue
    
    volatility_df = pd.DataFrame(vol_data) if vol_data else pd.DataFrame()
    correlation_df = pd.DataFrame(corr_data) if corr_data else pd.DataFrame()
    
    if not volatility_df.empty:
        volatility_df = volatility_df.sort_index()
    if not correlation_df.empty:
        correlation_df = correlation_df.sort_index()
    
    return volatility_df, correlation_df

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

def bollinger_bands(x: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(x, n)
    sd = x.rolling(n).std()
    return mid + k * sd, mid, mid - k * sd

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift(1)),
        'lc': abs(low - close.shift(1))
    }).max(axis=1)
    return tr.rolling(n).mean()

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["SMA_20"] = sma(out["Close"], 20)
    out["SMA_50"] = sma(out["Close"], 50)
    out["SMA_200"] = sma(out["Close"], 200)
    out["EMA_12"] = ema(out["Close"], 12)
    out["EMA_26"] = ema(out["Close"], 26)
    out["RSI"] = rsi(out["Close"], 14)
    out["MACD"], out["MACD_Signal"], out["MACD_Hist"] = macd(out["Close"])
    out["BB_Upper"], out["BB_Mid"], out["BB_Lower"] = bollinger_bands(out["Close"])
    if all(x in out.columns for x in ["High", "Low", "Close"]):
        out["ATR"] = atr(out["High"], out["Low"], out["Close"], 14)
    return out

# =============================================================================
# ENHANCED ANALYTICS CORE
# =============================================================================

def compute_spreads(yield_df: pd.DataFrame) -> pd.DataFrame:
    spreads = pd.DataFrame(index=yield_df.index)
    if {"10Y", "2Y"}.issubset(yield_df.columns):
        spreads["10Y-2Y"] = (yield_df["10Y"] - yield_df["2Y"]) * 100
    if {"10Y", "3M"}.issubset(yield_df.columns):
        spreads["10Y-3M"] = (yield_df["10Y"] - yield_df["3M"]) * 100
    if {"5Y", "2Y"}.issubset(yield_df.columns):
        spreads["5Y-2Y"] = (yield_df["5Y"] - yield_df["2Y"]) * 100
    if {"30Y", "10Y"}.issubset(yield_df.columns):
        spreads["30Y-10Y"] = (yield_df["30Y"] - yield_df["10Y"]) * 100
    return spreads

def classify_regime(spreads: pd.DataFrame, yield_df: pd.DataFrame) -> Tuple[str, str, str]:
    if "10Y-2Y" not in spreads.columns or spreads.empty:
        return "Data Loading", "Please wait for data to load.", "neutral"
    spread = spreads["10Y-2Y"].iloc[-1]
    y10 = yield_df["10Y"].iloc[-1] if "10Y" in yield_df.columns else np.nan
    
    if np.isfinite(spread) and spread < 0:
        return "Risk-off / Recession Watch", "Curve inversion signals a defensive macro regime.", "bearish"
    if np.isfinite(spread) and spread < 50:
        return "Neutral / Late Cycle", "Curve flattening suggests late-cycle caution.", "neutral"
    if np.isfinite(y10) and y10 > 5.5:
        return "Restrictive Long-End", "Positive slope but restrictive long-end rates.", "cautious"
    return "Risk-on / Expansion", "Positive slope supports pro-risk positioning.", "bullish"

def recession_probability_proxy(spreads: pd.DataFrame, yield_df: pd.DataFrame) -> float:
    score = 0.0
    if "10Y-2Y" in spreads.columns:
        score += np.clip((-spreads["10Y-2Y"].iloc[-1]) / 100, 0, 1.5)
    if "10Y-3M" in spreads.columns:
        score += np.clip((-spreads["10Y-3M"].iloc[-1]) / 100, 0, 1.5)
    if "10Y" in yield_df.columns:
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

def calculate_inversion_periods(spreads: pd.DataFrame) -> List[dict]:
    if "10Y-2Y" not in spreads.columns:
        return []
    s = spreads["10Y-2Y"].dropna()
    periods = []
    in_inv = False
    start = None
    for date, value in s.items():
        if value < 0 and not in_inv:
            in_inv = True
            start = date
        elif value >= 0 and in_inv:
            periods.append({
                "start": start,
                "end": date,
                "depth": float(s.loc[start:date].min()),
                "duration_days": (date - start).days,
            })
            in_inv = False
    return periods

# =============================================================================
# ENHANCED NELSON-SIEGEL / NSS
# =============================================================================

class NelsonSiegelModel:
    @staticmethod
    def nelson_siegel(tau, beta0, beta1, beta2, lambda1):
        tau = np.asarray(tau, dtype=float)
        tau_safe = np.where(tau == 0, 1e-8, tau)
        x = lambda1 * tau_safe
        term1 = (1 - np.exp(-x)) / x
        term2 = term1 - np.exp(-x)
        out = beta0 + beta1 * term1 + beta2 * term2
        return np.where(tau == 0, beta0 + beta1 + beta2, out)

    @staticmethod
    def nss(tau, beta0, beta1, beta2, beta3, lambda1, lambda2):
        tau = np.asarray(tau, dtype=float)
        tau_safe = np.where(tau == 0, 1e-8, tau)
        x1 = lambda1 * tau_safe
        x2 = lambda2 * tau_safe
        term1 = (1 - np.exp(-x1)) / x1
        term2 = term1 - np.exp(-x1)
        term3 = ((1 - np.exp(-x2)) / x2) - np.exp(-x2)
        out = beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3
        return np.where(tau == 0, beta0 + beta1 + beta2 + beta3, out)

    @staticmethod
    def fit_ns(maturities: np.ndarray, yields_: np.ndarray):
        if len(maturities) == 0 or len(yields_) == 0:
            return None

        def objective(params):
            fitted = NelsonSiegelModel.nelson_siegel(maturities, *params)
            return np.sum((yields_ - fitted) ** 2)

        bounds = [
            (yields_.min() - 2, yields_.max() + 2),
            (-15, 15), (-15, 15), (0.01, 5),
        ]
        best, best_fun = None, np.inf
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
            "mae": float(np.mean(np.abs(yields_ - fitted))),
            "r_squared": float(1 - sse / sst) if sst > 0 else np.nan,
            "residuals": yields_ - fitted,
        }

    @staticmethod
    def fit_nss(maturities: np.ndarray, yields_: np.ndarray):
        if len(maturities) == 0 or len(yields_) == 0:
            return None

        def objective(params):
            fitted = NelsonSiegelModel.nss(maturities, *params)
            weights = 1 / (maturities + 0.25)
            return np.sum(weights * (yields_ - fitted) ** 2)

        bounds = [
            (yields_.min() - 2, yields_.max() + 2),
            (-20, 20), (-20, 20), (-20, 20),
            (0.01, 10), (0.01, 10),
        ]
        try:
            res = differential_evolution(objective, bounds=bounds, maxiter=220, popsize=10, polish=True, seed=42)
            if not res.success:
                return None
            fitted = NelsonSiegelModel.nss(maturities, *res.x)
        except Exception:
            return None

        sse = np.sum((yields_ - fitted) ** 2)
        sst = np.sum((yields_ - np.mean(yields_)) ** 2)
        return {
            "params": res.x,
            "fitted_values": fitted,
            "rmse": float(np.sqrt(np.mean((yields_ - fitted) ** 2))),
            "mae": float(np.mean(np.abs(yields_ - fitted))),
            "r_squared": float(1 - sse / sst) if sst > 0 else np.nan,
            "residuals": yields_ - fitted,
        }

def model_governance(ns_result: Optional[dict], nss_result: Optional[dict]) -> pd.DataFrame:
    rows = []
    for model_name, result in [("NS", ns_result), ("NSS", nss_result)]:
        if result is None:
            continue
        max_abs_residual = float(np.max(np.abs(result["residuals"])))
        rmse = result["rmse"]
        r2 = result["r_squared"]
        if rmse < 0.05 and r2 > 0.98:
            confidence = "High"
        elif rmse < 0.10 and r2 > 0.95:
            confidence = "Moderate"
        else:
            confidence = "Low"
        flags = []
        if max_abs_residual > 0.15:
            flags.append("Residual outlier")
        if r2 < 0.95:
            flags.append("Low fit quality")
        rows.append({
            "Model": model_name,
            "RMSE": rmse,
            "MAE": result["mae"],
            "R2": r2,
            "MaxAbsResidual": max_abs_residual,
            "FitConfidence": confidence,
            "WarningFlags": ", ".join(flags) if flags else "None",
        })
    return pd.DataFrame(rows)

def rolling_ns_parameters(yield_df: pd.DataFrame, maturities: np.ndarray, selected_cols: List[str], years: int) -> pd.DataFrame:
    window_size = years * 252
    if len(yield_df) <= window_size + 5:
        return pd.DataFrame()
    rows = []
    for i in range(window_size, len(yield_df), CFG.rolling_step):
        curve = yield_df.iloc[i][selected_cols].values
        res = NelsonSiegelModel.fit_ns(maturities, curve)
        if res:
            rows.append({
                "date": yield_df.index[i],
                "beta0": res["params"][0],
                "beta1": res["params"][1],
                "beta2": res["params"][2],
                "lambda": res["params"][3],
                "rmse": res["rmse"],
            })
    return pd.DataFrame(rows)

def factor_contributions(yield_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=yield_df.index)
    if "10Y" in yield_df.columns:
        out["Level"] = yield_df["10Y"]
    if {"10Y", "3M"}.issubset(yield_df.columns):
        out["Slope"] = (yield_df["10Y"] - yield_df["3M"]) * 100
    if {"3M", "5Y", "10Y"}.issubset(yield_df.columns):
        out["Curvature"] = (2 * yield_df["5Y"] - (yield_df["3M"] + yield_df["10Y"])) * 100
    if {"2Y", "10Y", "30Y"}.issubset(yield_df.columns):
        out["Butterfly"] = (2 * yield_df["10Y"] - (yield_df["2Y"] + yield_df["30Y"])) * 100
    return out

def pca_risk_decomp(yield_df: pd.DataFrame, n_components: int = 3) -> Optional[dict]:
    returns = yield_df.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 20 or returns.shape[1] < 2:
        return None
    scaler = StandardScaler()
    x = scaler.fit_transform(returns)
    k = min(n_components, x.shape[1], x.shape[0] - 1)
    pca = PCA(n_components=k)
    pcs = pca.fit_transform(x)
    loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(k)], index=returns.columns)
    factors = pd.DataFrame(pcs, index=returns.index, columns=[f"PC{i+1}" for i in range(k)])
    return {
        "explained_variance": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        "loadings": loadings,
        "factors": factors,
    }

def calculate_var_metrics(returns: pd.Series, confidence: float = 0.95, horizon: int = 10) -> Optional[dict]:
    returns = returns.dropna()
    if len(returns) < 20:
        return None
    var_hist = np.percentile(returns, (1 - confidence) * 100)
    cvar = returns[returns <= var_hist].mean()
    var_param = norm.ppf(1 - confidence) * returns.std()
    skewness = returns.skew()
    kurt = returns.kurtosis()
    z = norm.ppf(1 - confidence)
    z_cf = z + (z**2 - 1) * skewness / 6 + (z**3 - 3 * z) * kurt / 24 - (2 * z**3 - 5 * z) * skewness**2 / 36
    return {
        "VaR_Historical": float(var_hist * np.sqrt(horizon)),
        "VaR_Parametric": float(var_param * np.sqrt(horizon)),
        "VaR_CornishFisher": float(z_cf * returns.std() * np.sqrt(horizon)),
        "CVaR": float(cvar * np.sqrt(horizon)),
        "Skewness": float(skewness),
        "Kurtosis": float(kurt),
    }

# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

class MonteCarloSimulator:
    @staticmethod
    def simulate_gbm(initial_price: float, mu: float, sigma: float, days: int, simulations: int = 1000) -> np.ndarray:
        dt = 1 / 252
        paths = np.zeros((simulations, days))
        paths[:, 0] = initial_price
        for i in range(1, days):
            z = np.random.standard_normal(simulations)
            paths[:, i] = paths[:, i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        return paths

    @staticmethod
    def simulate_vasicek(initial_rate: float, kappa: float, theta: float, sigma: float, days: int, simulations: int = 1000) -> np.ndarray:
        dt = 1 / 252
        paths = np.zeros((simulations, days))
        paths[:, 0] = initial_rate
        for i in range(1, days):
            z = np.random.standard_normal(simulations)
            dr = kappa * (theta - paths[:, i-1]) * dt + sigma * np.sqrt(dt) * z
            paths[:, i] = paths[:, i-1] + dr
        return paths

    @staticmethod
    def calculate_confidence_intervals(paths: np.ndarray, confidence: float = 0.95) -> Dict[str, np.ndarray]:
        lower_p = (1 - confidence) / 2 * 100
        upper_p = (1 + confidence) / 2 * 100
        return {
            "mean": np.mean(paths, axis=0),
            "median": np.percentile(paths, 50, axis=0),
            "lower_ci": np.percentile(paths, lower_p, axis=0),
            "upper_ci": np.percentile(paths, upper_p, axis=0),
            "std": np.std(paths, axis=0),
        }

    @staticmethod
    def calculate_var_from_paths(paths: np.ndarray, confidence: float = 0.95) -> float:
        return np.percentile(paths[:, -1], (1 - confidence) * 100)

# =============================================================================
# MACHINE LEARNING FORECAST
# =============================================================================

class MLForecastModel:
    @staticmethod
    def prepare_features(yield_df: pd.DataFrame, lags: int = 5) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
        X, y = [], []
        for i in range(lags, len(yield_df) - 1):
            feats = []
            for col in yield_df.columns:
                feats.extend(yield_df[col].iloc[i-lags:i].values)
            X.append(feats)
            y.append(yield_df.iloc[i + 1].values)
        if not X:
            return np.array([]), np.array([]), None
        X_arr, y_arr = np.array(X), np.array(y)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)
        return X_scaled, y_arr, scaler

    @staticmethod
    def train_model(X: np.ndarray, y: np.ndarray, model_type: str = "Random Forest", test_size: float = 0.2) -> Dict:
        if len(X) == 0:
            return {}
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if model_type == "Gradient Boosting":
            target_train = y_train[:, -1] if y_train.ndim > 1 else y_train
            target_test = y_test[:, -1] if y_test.ndim > 1 else y_test
            model = GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=3, random_state=42)
            model.fit(X_train, target_train)
            y_pred = model.predict(X_test)
            y_eval = target_test
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred_full = model.predict(X_test)
            if y_pred_full.ndim > 1:
                y_pred = y_pred_full[:, -1]
                y_eval = y_test[:, -1]
            else:
                y_pred = y_pred_full
                y_eval = y_test

        rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
        mae = mean_absolute_error(y_eval, y_pred)
        r2 = r2_score(y_eval, y_pred)

        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "feature": [f"Lagged_Feature_{i}" for i in range(X.shape[1])],
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False).head(12)
        else:
            importance_df = pd.DataFrame()

        return {
            "model_used": model_type,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "feature_importance": importance_df,
        }

# =============================================================================
# ENHANCED BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    @staticmethod
    def backtest_strategy(yield_df: pd.DataFrame, spreads: pd.DataFrame, strategy_type: str = "Curve Inversion") -> Dict:
        if "10Y" not in yield_df.columns:
            return {}
        returns = yield_df["10Y"].pct_change().shift(-1)

        if strategy_type == "Curve Inversion":
            if "10Y-2Y" not in spreads.columns:
                return {}
            signals = spreads["10Y-2Y"] < 0
        elif strategy_type == "Macro Trend (50-Day SMA)":
            sma_50 = yield_df["10Y"].rolling(50).mean()
            signals = yield_df["10Y"] > sma_50
        elif strategy_type == "Momentum Strategy":
            momentum = yield_df["10Y"].diff(20)
            signals = momentum > 0
        else:
            return {}

        strategy_returns = signals.shift(1) * returns
        buy_hold_returns = returns
        cumulative_strategy = (1 + strategy_returns.fillna(0)).cumprod()
        cumulative_bh = (1 + buy_hold_returns.fillna(0)).cumprod()

        sharpe_strategy = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() > 0 else 0
        sharpe_bh = (buy_hold_returns.mean() / buy_hold_returns.std() * np.sqrt(252)) if buy_hold_returns.std() > 0 else 0
        drawdown = cumulative_strategy / cumulative_strategy.cummax() - 1
        max_drawdown = drawdown.min()
        calmar = ((cumulative_strategy.iloc[-1] ** (252 / max(len(cumulative_strategy), 1)) - 1) / abs(max_drawdown)) if max_drawdown < 0 else 0

        active_returns = strategy_returns[strategy_returns != 0]
        win_rate = (active_returns > 0).mean() if len(active_returns) > 0 else 0

        return {
            "strategy_name": strategy_type,
            "cumulative_returns": cumulative_strategy,
            "buy_hold_returns": cumulative_bh,
            "sharpe_ratio_strategy": sharpe_strategy,
            "sharpe_ratio_bh": sharpe_bh,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "total_return_strategy": cumulative_strategy.iloc[-1] - 1,
            "total_return_bh": cumulative_bh.iloc[-1] - 1,
            "win_rate": win_rate,
        }

# =============================================================================
# VOLATILITY AND CORRELATION ANALYZER
# =============================================================================

class VolatilityAnalyzer:
    @staticmethod
    def calculate_volatility_regime(vix: pd.Series) -> Dict:
        if vix is None or len(vix) == 0:
            return {"current_vix": 0, "regime": "N/A", "outlook": "Data unavailable", "vix_percentile": "N/A"}
        current_vix = vix.iloc[-1]
        if current_vix < 12:
            regime, outlook = "EXTREME COMPLACENCY", "High risk of volatility spike."
        elif current_vix < 15:
            regime, outlook = "LOW VOLATILITY", "Normal complacent market."
        elif current_vix < 20:
            regime, outlook = "NORMAL VOLATILITY", "Typical market conditions."
        elif current_vix < 25:
            regime, outlook = "ELEVATED VOLATILITY", "Increased uncertainty."
        elif current_vix < 35:
            regime, outlook = "HIGH VOLATILITY", "Market stress, consider hedging."
        else:
            regime, outlook = "EXTREME VOLATILITY", "Crisis conditions, defensive positioning."
        percentile = (vix < current_vix).mean()
        return {
            "current_vix": current_vix,
            "regime": regime,
            "outlook": outlook,
            "vix_percentile": f"{percentile * 100:.1f}%",
        }

    @staticmethod
    def calculate_vol_of_vol(vix: pd.Series, window: int = 20) -> pd.Series:
        if vix is None or len(vix) < window:
            return pd.Series(dtype=float)
        return vix.pct_change().rolling(window).std() * np.sqrt(252)

class CorrelationAnalyzer:
    @staticmethod
    def calculate_correlation_matrix(assets_df: pd.DataFrame) -> pd.DataFrame:
        if assets_df is None or assets_df.empty:
            return pd.DataFrame()
        returns = assets_df.pct_change().dropna()
        return returns.corr()

    @staticmethod
    def calculate_rolling_correlation(asset1: pd.Series, asset2: pd.Series, window: int = 60) -> pd.Series:
        if asset1 is None or asset2 is None or len(asset1) < window:
            return pd.Series(dtype=float)
        returns1 = asset1.pct_change()
        returns2 = asset2.pct_change()
        return returns1.rolling(window).corr(returns2)

# =============================================================================
# SCENARIO ENGINE
# =============================================================================

def scenario_engine(yield_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    latest = yield_df.iloc[-1].copy()
    scenarios = {}
    tenor_order = list(yield_df.columns)

    bull = latest.copy()
    for col in tenor_order:
        m = MATURITY_MAP.get(col, 1)
        bull[col] = bull[col] - (0.08 + 0.06 * min(m / 10, 1.5))
    scenarios["Bull Steepener"] = pd.DataFrame({"Current": latest, "Scenario": bull})

    bear = latest.copy()
    for col in tenor_order:
        m = MATURITY_MAP.get(col, 1)
        bear[col] = bear[col] + (0.14 if m <= 2 else 0.07)
    scenarios["Bear Flattener"] = pd.DataFrame({"Current": latest, "Scenario": bear})

    recession = latest.copy()
    for col in tenor_order:
        m = MATURITY_MAP.get(col, 1)
        recession[col] = recession[col] - (0.22 if m <= 2 else 0.14 if m <= 10 else 0.10)
    scenarios["Recession Case"] = pd.DataFrame({"Current": latest, "Scenario": recession})

    easing = latest.copy()
    for col in tenor_order:
        m = MATURITY_MAP.get(col, 1)
        easing[col] = easing[col] - (0.25 if m <= 2 else 0.12 if m <= 10 else 0.06)
    scenarios["Policy Easing"] = pd.DataFrame({"Current": latest, "Scenario": easing})

    return scenarios

# =============================================================================
# ENHANCED VISUALIZATION FUNCTIONS
# =============================================================================

def add_recession_bands(fig: go.Figure, recessions: List[dict]) -> go.Figure:
    for rec in recessions:
        fig.add_vrect(
            x0=rec["start"], x1=rec["end"],
            fillcolor=COLORS["recession"], opacity=0.35,
            layer="below", line_width=0,
            annotation_text="Recession" if rec == recessions[-1] else None,
            annotation_position="top left"
        )
    return fig

def create_chart_layout(fig: go.Figure, title: str, y_title: Optional[str] = None, height: int = 460, x_title: str = "Date") -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(size=12, color=COLORS["text"]),
        title=dict(text=title, x=0.01, font=dict(size=16, color=COLORS["text"])),
        margin=dict(l=60, r=30, t=80, b=50),
        hovermode="x unified",
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title=x_title, gridcolor=COLORS["grid"], showgrid=True, tickfont=dict(color=COLORS["text"], size=11), title_font=dict(color=COLORS["text"], size=12))
    fig.update_yaxes(title=y_title, gridcolor=COLORS["grid"], showgrid=True, tickfont=dict(color=COLORS["text"], size=11), title_font=dict(color=COLORS["text"], size=12))
    return fig

def chart_current_curve(maturities: np.ndarray, latest_curve: np.ndarray, ns_result: Optional[dict], nss_result: Optional[dict], recessions: List[dict] = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=maturities, y=latest_curve, mode='lines+markers', name='Current Curve',
        line=dict(color=COLORS["accent"], width=2.5), marker=dict(size=10, color=COLORS["accent"])
    ))
    if ns_result:
        fig.add_trace(go.Scatter(
            x=maturities, y=ns_result["fitted_values"], mode='lines', name='NS Fit',
            line=dict(color=COLORS["positive"], width=2.0)
        ))
    if nss_result:
        fig.add_trace(go.Scatter(
            x=maturities, y=nss_result["fitted_values"], mode='lines', name='NSS Fit',
            line=dict(color=COLORS["accent3"], width=2.0, dash="dash")
        ))
    return create_chart_layout(fig, "Current Treasury Yield Curve and Model Fits", "Yield (%)", 460, "Maturity (Years)")

def chart_spreads(spreads: pd.DataFrame, recessions: List[dict]) -> go.Figure:
    fig = go.Figure()
    palette = [COLORS["negative"], COLORS["accent"], COLORS["warning"], COLORS["positive"]]
    for i, col in enumerate(spreads.columns):
        fig.add_trace(go.Scatter(
            x=spreads.index, y=spreads[col], mode='lines', name=col,
            line=dict(color=palette[i % len(palette)], width=2)
        ))
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"])
    fig.add_hline(y=-50, line_dash="dot", line_color=COLORS["warning"], line_width=1)
    add_recession_bands(fig, recessions)
    return create_chart_layout(fig, "Treasury Spread Dashboard", "Basis Points", 520)

def chart_yield_with_recessions(yield_df: pd.DataFrame, tenor: str, color: str, title: str, recessions: List[dict]) -> Optional[go.Figure]:
    if tenor not in yield_df.columns:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yield_df.index, y=yield_df[tenor], mode='lines', name=tenor, line=dict(color=color, width=2.2)))
    add_recession_bands(fig, recessions)
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))
    return create_chart_layout(fig, title, "Yield (%)", 440)

def chart_monte_carlo(initial_value: float, simulation_results: Dict[str, np.ndarray], horizon_days: int, title_prefix: str) -> go.Figure:
    fig = go.Figure()
    x_axis = np.arange(horizon_days)
    fig.add_trace(go.Scatter(x=x_axis, y=simulation_results["upper_ci"], fill=None, mode='lines', line=dict(color='rgba(0,0,0,0)'), showlegend=False))
    fig.add_trace(go.Scatter(x=x_axis, y=simulation_results["lower_ci"], fill='tonexty', mode='lines', fillcolor='rgba(44, 95, 138, 0.20)', line=dict(color='rgba(0,0,0,0)'), name='Confidence Interval'))
    fig.add_trace(go.Scatter(x=x_axis, y=simulation_results["mean"], mode='lines', name='Mean Path', line=dict(color=COLORS["accent"], width=2.5)))
    fig.add_trace(go.Scatter(x=[0], y=[initial_value], mode='markers', name='Current', marker=dict(size=12, color=COLORS["positive"], symbol='star')))
    return create_chart_layout(fig, f"{title_prefix} | 10Y Yield Paths", "Yield (%)", 500, "Trading Days Ahead")

def chart_backtest_results(backtest_results: Dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=backtest_results["cumulative_returns"].index, y=backtest_results["cumulative_returns"].values, mode='lines', name=backtest_results["strategy_name"], line=dict(color=COLORS["accent"], width=2.5)))
    fig.add_trace(go.Scatter(x=backtest_results["buy_hold_returns"].index, y=backtest_results["buy_hold_returns"].values, mode='lines', name='Buy & Hold', line=dict(color=COLORS["muted"], width=2, dash='dash')))
    return create_chart_layout(fig, f"Backtest Performance: {backtest_results['strategy_name']}", "Cumulative Return", 500)

def chart_volatility_dashboard(vix_data: pd.Series, vol_regime: Dict) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("VIX", "Vol of Vol"), vertical_spacing=0.15, row_heights=[0.6, 0.4])
    fig.add_trace(go.Scatter(x=vix_data.index, y=vix_data.values, mode='lines', name='VIX', line=dict(color=COLORS["warning"], width=2)), row=1, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="orange", row=1, col=1)
    fig.add_hline(y=15, line_dash="dash", line_color="green", row=1, col=1)
    vol_of_vol = VolatilityAnalyzer.calculate_vol_of_vol(vix_data)
    if len(vol_of_vol) > 0:
        fig.add_trace(go.Scatter(x=vol_of_vol.index, y=vol_of_vol.values, mode='lines', name='Vol of Vol', line=dict(color=COLORS["accent"], width=2)), row=2, col=1)
    return create_chart_layout(fig, f"Volatility Dashboard | Current VIX: {vol_regime['current_vix']:.2f}", height=600)

def chart_correlation_heatmap(correlation_matrix: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation"),
    ))
    fig.update_layout(title="Cross-Asset Correlation Matrix", height=550)
    return fig

def chart_rate_dynamics(rate_directions: pd.DataFrame) -> Optional[go.Figure]:
    if rate_directions is None or rate_directions.empty:
        return None
    fig = make_subplots(rows=2, cols=2, subplot_titles=tuple(rate_directions.columns[:4]))
    cols = list(rate_directions.columns[:4])
    positions = [(1,1), (1,2), (2,1), (2,2)]
    palette = [COLORS["accent"], COLORS["warning"], COLORS["positive"], COLORS["accent3"]]
    for idx, col in enumerate(cols):
        r, c = positions[idx]
        fig.add_trace(go.Scatter(x=rate_directions.index, y=rate_directions[col], mode='lines', name=col, line=dict(color=palette[idx], width=2)), row=r, col=c)
    return create_chart_layout(fig, "Directional Rate Dynamics", height=620)

def chart_technical(ohlc_df: pd.DataFrame, ticker: str) -> Optional[go.Figure]:
    if ohlc_df is None or ohlc_df.empty:
        return None
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.52, 0.24, 0.24],
        subplot_titles=(f"{ticker} Price and Bands", "RSI (14)", "MACD"),
    )
    fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["Close"], mode="lines", name="Close", line=dict(color=COLORS["accent"], width=2)), row=1, col=1)
    if "SMA_20" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["SMA_20"], mode="lines", name="SMA 20", line=dict(color=COLORS["positive"], width=1.4)), row=1, col=1)
    if "SMA_50" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["SMA_50"], mode="lines", name="SMA 50", line=dict(color=COLORS["warning"], width=1.4)), row=1, col=1)
    if "BB_Upper" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["BB_Upper"], mode="lines", name="BB Upper", line=dict(color=COLORS["muted"], width=1.0)), row=1, col=1)
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["BB_Lower"], mode="lines", name="BB Lower", line=dict(color=COLORS["muted"], width=1.0), fill='tonexty', fillcolor=COLORS["band"]), row=1, col=1)
    if "RSI" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["RSI"], mode="lines", name="RSI", line=dict(color=COLORS["accent2"], width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=COLORS["negative"], row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=COLORS["positive"], row=2, col=1)
    if "MACD" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["MACD"], mode="lines", name="MACD", line=dict(color=COLORS["accent"], width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["MACD_Signal"], mode="lines", name="Signal", line=dict(color=COLORS["accent3"], width=1.5)), row=3, col=1)
    return create_chart_layout(fig, f"Technical Analysis | {ticker}", height=760)

def chart_model_residuals(selected_cols: List[str], ns_result: Optional[dict], nss_result: Optional[dict]) -> Optional[go.Figure]:
    if ns_result is None and nss_result is None:
        return None
    fig = go.Figure()
    if ns_result:
        fig.add_trace(go.Bar(x=selected_cols, y=ns_result["residuals"], name="NS Residuals", marker_color=COLORS["positive"], opacity=0.65))
    if nss_result:
        fig.add_trace(go.Bar(x=selected_cols, y=nss_result["residuals"], name="NSS Residuals", marker_color=COLORS["warning"], opacity=0.65))
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"])
    fig.add_hline(y=0.15, line_dash="dot", line_color=COLORS["negative"], line_width=1)
    fig.add_hline(y=-0.15, line_dash="dot", line_color=COLORS["negative"], line_width=1)
    return create_chart_layout(fig, "Model Residual Quality and Warning Flags", "Residual (Yield %)", 420, "Tenor")

def chart_factor_contributions(factor_df: pd.DataFrame) -> Optional[go.Figure]:
    if factor_df is None or factor_df.empty:
        return None
    fig = go.Figure()
    palette = [COLORS["accent"], COLORS["warning"], COLORS["positive"], COLORS["accent2"]]
    for i, col in enumerate(factor_df.columns):
        fig.add_trace(go.Scatter(x=factor_df.index, y=factor_df[col], mode="lines", name=col, line=dict(color=palette[i % len(palette)], width=1.8)))
    return create_chart_layout(fig, "Historical Factor Contributions", "Value", 430)

def chart_pca_variance(pca_risk: Optional[dict]) -> Optional[go.Figure]:
    if not pca_risk:
        return None
    fig = go.Figure()
    ev = pca_risk["explained_variance"] * 100
    fig.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(ev))], y=ev, marker_color=COLORS["accent"]))
    fig.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(ev))], y=np.cumsum(ev), mode='lines+markers', name='Cumulative', line=dict(color=COLORS["warning"], width=2), yaxis='y2'))
    fig.update_layout(yaxis2=dict(title="Cumulative %", overlaying='y', side='right'))
    return create_chart_layout(fig, "PCA Variance Explained", "Percent", 430)

def chart_dynamic_parameters(dynamic_params: pd.DataFrame) -> Optional[go.Figure]:
    if dynamic_params is None or dynamic_params.empty:
        return None
    fig = make_subplots(rows=2, cols=2, subplot_titles=("β0 Level", "β1 Slope", "β2 Curvature", "RMSE"))
    fig.add_trace(go.Scatter(x=dynamic_params["date"], y=dynamic_params["beta0"], mode="lines", name="β0", line=dict(color=COLORS["positive"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=dynamic_params["date"], y=dynamic_params["beta1"], mode="lines", name="β1", line=dict(color=COLORS["accent"])), row=1, col=2)
    fig.add_trace(go.Scatter(x=dynamic_params["date"], y=dynamic_params["beta2"], mode="lines", name="β2", line=dict(color=COLORS["warning"])), row=2, col=1)
    fig.add_trace(go.Scatter(x=dynamic_params["date"], y=dynamic_params["rmse"], mode="lines", name="RMSE", line=dict(color=COLORS["muted"])), row=2, col=2)
    return create_chart_layout(fig, "Rolling Nelson-Siegel Parameters", height=620)

def chart_scenario_analysis(scenario_df: pd.DataFrame, scenario_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=scenario_df.index, y=scenario_df["Current"], name="Current", marker_color=COLORS["accent2"]))
    fig.add_trace(go.Bar(x=scenario_df.index, y=scenario_df["Scenario"], name=scenario_name, marker_color=COLORS["accent3"]))
    fig.update_layout(barmode="group")
    return create_chart_layout(fig, f"Scenario Analysis | {scenario_name}", "Yield (%)", 460, "Tenor")

def chart_ohlc_with_indicators(ohlc_df: pd.DataFrame, ticker: str) -> Optional[go.Figure]:
    if ohlc_df is None or ohlc_df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=ohlc_df.index, open=ohlc_df["Open"], high=ohlc_df["High"], low=ohlc_df["Low"], close=ohlc_df["Close"],
        increasing=dict(line=dict(color=COLORS["positive"]), fillcolor=COLORS["positive"]),
        decreasing=dict(line=dict(color=COLORS["negative"]), fillcolor=COLORS["negative"]),
        name=ticker,
    ))
    if "SMA_20" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["SMA_20"], mode="lines", name="SMA 20", line=dict(color=COLORS["accent"], width=1.2)))
    if "SMA_50" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["SMA_50"], mode="lines", name="SMA 50", line=dict(color=COLORS["warning"], width=1.2)))
    if "BB_Upper" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["BB_Upper"], mode="lines", name="BB Upper", line=dict(color=COLORS["muted"], width=1)))
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["BB_Lower"], mode="lines", name="BB Lower", line=dict(color=COLORS["muted"], width=1), fill='tonexty', fillcolor=COLORS["band"]))
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))
    return create_chart_layout(fig, f"OHLC | {ticker}", "Price", 520)

# =============================================================================
# ENHANCED UI HELPERS
# =============================================================================

def render_api_gate() -> None:
    st.markdown(
        f"""
        <div class="note-box" style="max-width:560px; margin:40px auto; text-align:center;">
            <b>🔑 FRED API Key Required</b><br><br>
            This institutional platform requires live U.S. Treasury data.<br>
            Get your free API key from the <b>FRED</b> website.
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
            st.session_state["api_key"] = api_key
            st.session_state["api_key_validated"] = True
            st.success("✓ API key validated successfully. Loading data...")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("Invalid API key. Please check and try again.")
    st.stop()

def kpi_card(label: str, value: str, sub: str, trend: Optional[float] = None, trend_label: str = "") -> None:
    trend_html = ""
    if trend is not None:
        trend_class = "metric-trend-up" if trend >= 0 else "metric-trend-down"
        trend_symbol = "▲" if trend >= 0 else "▼"
        trend_html = f'<div class="{trend_class}" style="font-size:0.7rem; margin-top:0.25rem;">{trend_symbol} {abs(trend):.2f}% {trend_label}</div>'
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{sub}</div>{trend_html}</div>',
        unsafe_allow_html=True,
    )

def create_smart_kpi_row(yield_df: pd.DataFrame, spreads: pd.DataFrame, regime: str, regime_text: str, recession_prob: float, volatility_df: pd.DataFrame) -> None:
    current_2y = yield_df["2Y"].iloc[-1] if "2Y" in yield_df.columns else np.nan
    current_10y = yield_df["10Y"].iloc[-1] if "10Y" in yield_df.columns else np.nan
    current_30y = yield_df["30Y"].iloc[-1] if "30Y" in yield_df.columns else np.nan
    current_spread = spreads["10Y-2Y"].iloc[-1] if "10Y-2Y" in spreads.columns else np.nan
    
    # Calculate trends (20-day change)
    if "2Y" in yield_df.columns and len(yield_df) > 20:
        trend_2y = (yield_df["2Y"].iloc[-1] - yield_df["2Y"].iloc[-20]) / yield_df["2Y"].iloc[-20] * 100
        trend_10y = (yield_df["10Y"].iloc[-1] - yield_df["10Y"].iloc[-20]) / yield_df["10Y"].iloc[-20] * 100
        trend_spread = spreads["10Y-2Y"].iloc[-1] - spreads["10Y-2Y"].iloc[-20] if "10Y-2Y" in spreads.columns else None
    else:
        trend_2y = trend_10y = None
        trend_spread = None
    
    current_vix = volatility_df["CBOE Volatility Index"].iloc[-1] if volatility_df is not None and "CBOE Volatility Index" in volatility_df.columns and not volatility_df.empty else np.nan
    trend_vix = (volatility_df["CBOE Volatility Index"].iloc[-1] - volatility_df["CBOE Volatility Index"].iloc[-20]) / volatility_df["CBOE Volatility Index"].iloc[-20] * 100 if volatility_df is not None and "CBOE Volatility Index" in volatility_df.columns and len(volatility_df) > 20 else None
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        kpi_card("📊 Macro Regime", regime, regime_text)
    with c2:
        kpi_card("🏦 2Y Yield", f"{current_2y:.2f}%" if np.isfinite(current_2y) else "N/A", "Policy anchor", trend_2y, "20d")
    with c3:
        kpi_card("📈 10Y Yield", f"{current_10y:.2f}%" if np.isfinite(current_10y) else "N/A", "Benchmark", trend_10y, "20d")
    with c4:
        spread_display = f"{current_spread:.1f} bps" if np.isfinite(current_spread) else "N/A"
        spread_delta = f"{trend_spread:.1f} bps 20d" if trend_spread is not None and np.isfinite(trend_spread) else ""
        kpi_card("🔄 10Y-2Y Spread", spread_display, "Recession signal", trend_spread if trend_spread is not None else None, "bps 20d")
    with c5:
        kpi_card("⚠️ Recession Prob", f"{100 * recession_prob:.1f}%", "Proxy estimate")
    with c6:
        kpi_card("📉 VIX", f"{current_vix:.2f}" if np.isfinite(current_vix) else "N/A", "Fear gauge", trend_vix, "20d")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main() -> None:
    st.markdown(
        """
        <div class="main-title-card">
            <div class="main-title">Dynamic Quantitative Analysis Model</div>
            <div class="main-subtitle">Institutional Fixed-Income Platform | V36 Enhanced | Real-time Market Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state["api_key_validated"]:
        render_api_gate()

    with st.sidebar:
        st.markdown("### 🎛️ Control Tower")
        st.caption("Institutional fixed-income macro monitoring")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            keep_key = st.session_state["api_key"]
            for key, value in DEFAULT_STATE.items():
                st.session_state[key] = value
            st.session_state["api_key"] = keep_key
            st.session_state["api_key_validated"] = True
            st.rerun()

        with st.expander("📊 Analysis Parameters", expanded=True):
            rolling_years = st.slider("Rolling Window (years)", 2, 10, CFG.rolling_years_default)
            forecast_horizon = st.slider("Forecast Horizon (days)", 5, 60, CFG.forecast_horizon_default)
            confidence_level = st.slider("VaR Confidence Level", 0.90, 0.99, 0.95, 0.01)

        with st.expander("🎲 Monte Carlo Parameters", expanded=False):
            mc_model = st.selectbox("Simulation Model", ["Geometric Brownian Motion", "Vasicek Mean-Reverting"])
            mc_simulations = st.slider("Number of Simulations", 500, 10000, 5000, 500)
            mc_horizon = st.slider("Forecast Horizon (days)", 5, 252, 20)

        with st.expander("🤖 Machine Learning Engine", expanded=False):
            ml_model_type = st.selectbox("Algorithm", ["Random Forest", "Gradient Boosting"])
            ml_lags = st.slider("Autoregressive Lags", 3, 21, 5)

        with st.expander("📊 Backtesting Rules", expanded=False):
            bt_strategy = st.selectbox("Strategy Type", ["Curve Inversion", "Macro Trend (50-Day SMA)", "Momentum Strategy"])

        with st.expander("📡 Technical Analysis", expanded=False):
            ohlc_ticker = st.selectbox("OHLC Instrument", list(YAHOO_TICKERS.keys()), format_func=lambda x: f"{x} | {YAHOO_TICKERS[x]}", index=0)
            ohlc_period = st.selectbox("OHLC Period", ["6mo", "1y", "2y", "5y"], index=2)

    if not st.session_state["data_fetched"]:
        with st.spinner("Fetching macro and market data..."):
            st.session_state["yield_data"] = fetch_all_yield_data(st.session_state["api_key"])
            st.session_state["recession_data"] = fetch_recession_data(st.session_state["api_key"])
            volatility_df, correlation_df = fetch_market_bundle()
            st.session_state["volatility_data"] = volatility_df
            st.session_state["correlation_data"] = correlation_df
            
            ohlc_data = {}
            for ticker in YAHOO_TICKERS:
                df = fetch_ohlc_data(ticker, ohlc_period)
                if df is not None and not df.empty:
                    ohlc_data[ticker] = add_technical_indicators(df)
            st.session_state["ohlc_data"] = ohlc_data

        if st.session_state["yield_data"] is None:
            st.error("Failed to fetch FRED data.")
            st.stop()
        st.session_state["data_fetched"] = True

    yield_df = st.session_state["yield_data"].copy()
    recession_series = st.session_state["recession_data"]
    volatility_df = st.session_state["volatility_data"]
    correlation_df = st.session_state["correlation_data"]
    ohlc_map = st.session_state["ohlc_data"] if st.session_state["ohlc_data"] else {}

    selected_cols = [c for c in yield_df.columns if c in MATURITY_MAP]
    maturities = np.array([MATURITY_MAP[c] for c in selected_cols], dtype=float)
    latest_curve = yield_df.iloc[-1][selected_cols].values.astype(float)
    spreads = compute_spreads(yield_df)
    recessions = identify_recessions(recession_series)
    regime, regime_text, regime_type = classify_regime(spreads, yield_df)
    recession_prob = recession_probability_proxy(spreads, yield_df)

    # Run model calculations
    with st.spinner("Running model layer..."):
        ns_result = NelsonSiegelModel.fit_ns(maturities, latest_curve)
        nss_result = NelsonSiegelModel.fit_nss(maturities, latest_curve)
        governance_df = model_governance(ns_result, nss_result)
        factor_df = factor_contributions(yield_df)
        pca_risk = pca_risk_decomp(yield_df[selected_cols])
        dynamic_params = rolling_ns_parameters(yield_df[selected_cols], maturities, selected_cols, rolling_years)
        
        # Rate direction metrics
        rate_directions = pd.DataFrame(index=yield_df.index)
        if "2Y" in yield_df.columns:
            rate_directions["2Y_20D_Change"] = yield_df["2Y"].diff(20)
        if "10Y" in yield_df.columns:
            rate_directions["10Y_20D_Change"] = yield_df["10Y"].diff(20)
            rate_directions["10Y_RealizedVol_20D"] = yield_df["10Y"].pct_change().rolling(20).std() * np.sqrt(252)
        if "10Y-2Y" in spreads.columns:
            rate_directions["Slope_Momentum_20D"] = spreads["10Y-2Y"].diff(20)
        rate_directions = rate_directions.dropna(how="all")
        
        # Scenarios
        scenarios = scenario_engine(yield_df[selected_cols])

    # Smart KPI Row
    create_smart_kpi_row(yield_df, spreads, regime, regime_text, recession_prob, volatility_df)

    # Main Tabs
    main_tabs = st.tabs([
        "🏦 Executive Summary",
        "📐 Model Parameters",
        "🎲 Monte Carlo Simulation",
        "🤖 Machine Learning",
        "📊 Algorithmic Backtesting",
        "📉 Risk & Volatility",
        "📡 Directional Dynamics",
        "🛠 Technical Analysis",
        "💾 System Export",
    ])

    # =====================================================================
    # TAB 1: Executive Summary
    # =====================================================================
    with main_tabs[0]:
        st.markdown(
            f"""
            <div class="note-box">
            <b>📋 Macro Analysis Summary</b><br><br>
            The current macro regime is identified as <b>{regime}</b>. The 10Y Treasury yield sits at
            <b>{yield_df['10Y'].iloc[-1]:.2f}%</b>. The 10Y-2Y spread is <b>{spreads['10Y-2Y'].iloc[-1]:.1f} bps</b>,
            with a recession probability of <b>{100 * recession_prob:.1f}%</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(chart_current_curve(maturities, latest_curve, ns_result, nss_result, recessions), use_container_width=True)
        with right:
            fig_spread = go.Figure()
            if "10Y-2Y" in spreads.columns:
                fig_spread.add_trace(go.Scatter(x=spreads.index, y=spreads["10Y-2Y"], mode="lines", name="10Y-2Y", line=dict(color=COLORS["warning"], width=2)))
                fig_spread.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"])
                add_recession_bands(fig_spread, recessions)
            st.plotly_chart(create_chart_layout(fig_spread, "10Y-2Y Spread History with Recessions", "Basis Points", 440), use_container_width=True)

        # Yield curves comparison
        st.subheader("Historical Yield Curves")
        col1, col2 = st.columns(2)
        with col1:
            fig2 = chart_yield_with_recessions(yield_df, "2Y", COLORS["warning"], "2-Year Treasury Yield", recessions)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
        with col2:
            fig10 = chart_yield_with_recessions(yield_df, "10Y", COLORS["accent"], "10-Year Treasury Yield", recessions)
            if fig10:
                st.plotly_chart(fig10, use_container_width=True)

        st.plotly_chart(chart_spreads(spreads, recessions), use_container_width=True)

    # =====================================================================
    # TAB 2: Model Parameters
    # =====================================================================
    with main_tabs[1]:
        sub_tabs = st.tabs(["NS Parameters", "NSS Parameters", "Governance", "Dynamic Analysis", "Factor / PCA"])

        with sub_tabs[0]:
            st.subheader("Nelson-Siegel Parameters")
            if ns_result:
                st.dataframe(pd.DataFrame({
                    "Parameter Variable": ["β₀ (Long-Term Level)", "β₁ (Short-Term Slope)", "β₂ (Medium-Term Curvature)", "λ (Decay Horizon)", "Optimization RMSE"],
                    "Calibrated Value": [
                        f"{ns_result['params'][0]:.4f}",
                        f"{ns_result['params'][1]:.4f}",
                        f"{ns_result['params'][2]:.4f}",
                        f"{ns_result['params'][3]:.4f}",
                        f"{ns_result['rmse'] * 100:.2f} bps",
                    ]
                }), use_container_width=True, hide_index=True)
                st.markdown(
                    '<div class="note-box"><b>📖 Interpretation</b><br><br>β₀ approximates the long-run level of the curve, β₁ governs short-end slope, β₂ governs hump/curvature behavior, and λ controls how quickly the loading structure decays across maturities.</div>',
                    unsafe_allow_html=True,
                )

        with sub_tabs[1]:
            st.subheader("Nelson-Siegel-Svensson Parameters")
            if nss_result:
                st.dataframe(pd.DataFrame({
                    "Parameter Variable": ["β₀", "β₁", "β₂", "β₃", "λ₁", "λ₂", "Optimization RMSE"],
                    "Calibrated Value": [
                        f"{nss_result['params'][0]:.4f}",
                        f"{nss_result['params'][1]:.4f}",
                        f"{nss_result['params'][2]:.4f}",
                        f"{nss_result['params'][3]:.4f}",
                        f"{nss_result['params'][4]:.4f}",
                        f"{nss_result['params'][5]:.4f}",
                        f"{nss_result['rmse'] * 100:.2f} bps",
                    ]
                }), use_container_width=True, hide_index=True)
                st.markdown(
                    '<div class="note-box"><b>📖 Interpretation</b><br><br>The NSS model extends NS by adding β₃ and λ₂ to better capture an additional hump or twist in the curve. It is often more flexible when the term structure exhibits multiple local shape features.</div>',
                    unsafe_allow_html=True,
                )

        with sub_tabs[2]:
            st.subheader("Model Governance & Residual Quality")
            if not governance_df.empty:
                st.dataframe(governance_df.round(4), use_container_width=True, hide_index=True)
            fig_resid = chart_model_residuals(selected_cols, ns_result, nss_result)
            if fig_resid:
                st.plotly_chart(fig_resid, use_container_width=True)
            st.markdown(
                '<div class="info-box"><b>✅ Governance Logic</b><br><br>High fit confidence requires both low RMSE and high R². Warning flags identify outlier residuals or weak fit conditions. Use these diagnostics before drawing trading conclusions from calibrated parameters.</div>',
                unsafe_allow_html=True,
            )

        with sub_tabs[3]:
            fig_dyn = chart_dynamic_parameters(dynamic_params)
            if fig_dyn:
                st.plotly_chart(fig_dyn, use_container_width=True)
            else:
                st.info("Insufficient data for rolling parameter analysis. Try a longer date range or smaller rolling window.")

        with sub_tabs[4]:
            left_pca, right_pca = st.columns([1, 1])
            with left_pca:
                fig_factor = chart_factor_contributions(factor_df)
                if fig_factor:
                    st.plotly_chart(fig_factor, use_container_width=True)
            with right_pca:
                fig_pca = chart_pca_variance(pca_risk)
                if fig_pca:
                    st.plotly_chart(fig_pca, use_container_width=True)
                if pca_risk:
                    st.dataframe(pca_risk["loadings"].round(4), use_container_width=True)
                    st.markdown(
                        '<div class="note-box"><b>📖 PCA Interpretation</b><br><br>PC1 usually captures broad level moves, PC2 often captures slope changes, and PC3 frequently reflects curvature or twist. Their exact meaning depends on the loading signs and magnitudes shown above.</div>',
                        unsafe_allow_html=True,
                    )

    # =====================================================================
    # TAB 3: Monte Carlo Simulation
    # =====================================================================
    with main_tabs[2]:
        st.subheader(f"🎲 {mc_model} Engine")
        st.markdown(
            '<div class="note-box">This module simulates stochastic yield paths. Use it to understand terminal dispersion, downside tail thresholds, and distributional uncertainty rather than only a point forecast.</div>',
            unsafe_allow_html=True,
        )
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info(f"**Current 10Y Yield:** {yield_df['10Y'].iloc[-1]:.2f}%")
            st.info(f"**Historical Volatility (10Y):** {yield_df['10Y'].pct_change().std() * np.sqrt(252):.2%}")
        
        if st.button("🚀 Initialize Simulation Array", use_container_width=True):
            with st.spinner(f"Computing {mc_simulations:,} stochastic paths..."):
                initial_y = yield_df['10Y'].iloc[-1] if np.isfinite(yield_df['10Y'].iloc[-1]) else 4.0
                returns = yield_df["10Y"].pct_change().dropna() if "10Y" in yield_df.columns else pd.Series([0])
                
                if mc_model == "Geometric Brownian Motion":
                    mu = returns.mean() * 252 if len(returns) > 0 else 0
                    sigma = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.10
                    paths = MonteCarloSimulator.simulate_gbm(initial_y, mu, sigma, mc_horizon, mc_simulations)
                    model_name = "GBM"
                else:
                    theta = yield_df["10Y"].mean() if "10Y" in yield_df.columns else initial_y
                    sigma_v = yield_df["10Y"].diff().dropna().std() * np.sqrt(252) if "10Y" in yield_df.columns else 0.10
                    paths = MonteCarloSimulator.simulate_vasicek(initial_y, 0.5, theta, sigma_v, mc_horizon, mc_simulations)
                    model_name = "Vasicek"

                sim_results = MonteCarloSimulator.calculate_confidence_intervals(paths, 0.95)
                var_est = MonteCarloSimulator.calculate_var_from_paths(paths, 0.95)

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Terminal Expected Value", f"{sim_results['mean'][-1]:.2f}%")
                mc2.metric(f"95% VaR", f"{var_est:.2f}%")
                mc3.metric("Volatility Dispersion", f"±{sim_results['std'][-1]:.2f}%")
                mc4.metric("Simulation Paths", f"{mc_simulations:,}")
                
                st.plotly_chart(chart_monte_carlo(initial_y, sim_results, mc_horizon, model_name), use_container_width=True)
                
                # Distribution plot
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=paths[:, -1], nbinsx=50, name='Terminal Distribution', marker_color=COLORS["accent"], opacity=0.7))
                fig_dist.add_vline(x=sim_results['mean'][-1], line_dash="dash", line_color="red", annotation_text=f"Mean: {sim_results['mean'][-1]:.2f}%")
                fig_dist.add_vline(x=var_est, line_dash="dash", line_color="orange", annotation_text=f"VaR: {var_est:.2f}%")
                fig_dist.update_layout(title="Terminal Value Distribution", xaxis_title="Yield (%)", yaxis_title="Frequency", template="plotly_white", height=400)
                st.plotly_chart(fig_dist, use_container_width=True)

    # =====================================================================
    # TAB 4: Machine Learning
    # =====================================================================
    with main_tabs[3]:
        st.subheader("🤖 Algorithmic ML Yield Prediction")
        st.markdown(
            f'<div class="note-box">The selected algorithm is <b>{ml_model_type}</b>. Features are built from lagged term-structure observations, scaled, and evaluated on holdout data. Use RMSE, MAE, and R² jointly.</div>',
            unsafe_allow_html=True,
        )
        
        if st.button("🧠 Compile & Train Model", use_container_width=True):
            with st.spinner(f"Training {ml_model_type} topology..."):
                X, y, scaler = MLForecastModel.prepare_features(yield_df[selected_cols], lags=ml_lags)
                if len(X) > 50:
                    ml_res = MLForecastModel.train_model(X, y, model_type=ml_model_type)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("RMSE", f"{ml_res.get('rmse', 0) * 100:.2f} bps")
                    m2.metric("MAE", f"{ml_res.get('mae', 0) * 100:.2f} bps")
                    m3.metric("R²", f"{ml_res.get('r2', 0):.3f}")
                    st.success(f"✅ Optimized on {len(X)} historical tensors.")
                    if not ml_res["feature_importance"].empty:
                        st.subheader("Feature Importance")
                        st.dataframe(ml_res["feature_importance"], use_container_width=True)
                else:
                    st.warning(f"Insufficient data vectors for algorithmic convergence. Need >50 samples, have {len(X)}.")

    # =====================================================================
    # TAB 5: Algorithmic Backtesting
    # =====================================================================
    with main_tabs[4]:
        st.subheader("📊 Quantitative Strategy Backtesting")
        st.markdown(
            f'<div class="note-box">Historical loop execution for <b>{bt_strategy}</b> versus a buy-and-hold benchmark. Focus on total return, Sharpe, drawdown, and Calmar rather than one metric in isolation.</div>',
            unsafe_allow_html=True,
        )
        
        if st.button("⚡ Execute Backtest Sequence", use_container_width=True):
            with st.spinner("Processing historical order blocks..."):
                bt_res = BacktestEngine.backtest_strategy(yield_df, spreads, bt_strategy)
                if bt_res:
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Total Return", f"{bt_res['total_return_strategy'] * 100:.1f}%")
                    b2.metric("Max Drawdown", f"{bt_res['max_drawdown'] * 100:.2f}%")
                    b3.metric("Sharpe Ratio", f"{bt_res['sharpe_ratio_strategy']:.2f}")
                    b4.metric("Calmar Ratio", f"{bt_res['calmar_ratio']:.2f}")
                    st.plotly_chart(chart_backtest_results(bt_res), use_container_width=True)
                    
                    # Additional metrics
                    with st.expander("Detailed Performance Metrics"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Win Rate", f"{bt_res['win_rate']:.2%}")
                            st.metric("Total Return (Buy & Hold)", f"{bt_res['total_return_bh'] * 100:.1f}%")
                        with col2:
                            st.metric("Excess Return", f"{(bt_res['total_return_strategy'] - bt_res['total_return_bh']) * 100:.1f}%")
                            st.metric("Sharpe Ratio (Benchmark)", f"{bt_res['sharpe_ratio_bh']:.2f}")
                else:
                    st.warning("Insufficient historical topology for strategy computation.")

    # =====================================================================
    # TAB 6: Risk & Volatility
    # =====================================================================
    with main_tabs[5]:
        left, right = st.columns([1, 1])

        with left:
            st.subheader("📊 Volatility Analytics")
            if volatility_df is not None and not volatility_df.empty and "CBOE Volatility Index" in volatility_df.columns:
                vix_analysis = VolatilityAnalyzer.calculate_volatility_regime(volatility_df["CBOE Volatility Index"])
                st.markdown(f'<div class="warning-box"><b>{vix_analysis["regime"]}</b><br><br>{vix_analysis["outlook"]}</div>', unsafe_allow_html=True)
                st.plotly_chart(chart_volatility_dashboard(volatility_df["CBOE Volatility Index"], vix_analysis), use_container_width=True)
            else:
                st.info("Volatility data unavailable.")
            
            # VaR Metrics
            st.subheader("📉 Value at Risk (VaR)")
            if "10Y" in yield_df.columns:
                risk_metrics = calculate_var_metrics(yield_df["10Y"].pct_change(), confidence_level, 10)
                if risk_metrics:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Historical VaR (10d)", f"{risk_metrics['VaR_Historical']:.4f}")
                        st.metric("Parametric VaR (10d)", f"{risk_metrics['VaR_Parametric']:.4f}")
                    with col2:
                        st.metric("Cornish-Fisher VaR", f"{risk_metrics['VaR_CornishFisher']:.4f}")
                        st.metric("CVaR (Expected Shortfall)", f"{risk_metrics['CVaR']:.4f}")
                    st.caption(f"Skewness: {risk_metrics['Skewness']:.3f} | Kurtosis: {risk_metrics['Kurtosis']:.3f}")

        with right:
            st.subheader("🔄 Cross-Asset Correlation")
            if correlation_df is not None and not correlation_df.empty and "10Y" in yield_df.columns:
                all_assets = pd.concat([yield_df["10Y"], correlation_df], axis=1).dropna()
                all_assets.columns = ["10Y Yield"] + list(correlation_df.columns)
                corr_matrix = CorrelationAnalyzer.calculate_correlation_matrix(all_assets)
                if not corr_matrix.empty:
                    st.plotly_chart(chart_correlation_heatmap(corr_matrix), use_container_width=True)
                    
                    # Rolling correlation with 10Y
                    st.subheader("Rolling Correlation with 10Y")
                    if "S&P 500" in correlation_df.columns:
                        rolling_corr = CorrelationAnalyzer.calculate_rolling_correlation(yield_df["10Y"], correlation_df["S&P 500"], 60)
                        if not rolling_corr.empty:
                            fig_roll = go.Figure()
                            fig_roll.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr.values, mode='lines', name='S&P 500 vs 10Y', line=dict(color=COLORS["accent"], width=2)))
                            fig_roll.add_hline(y=0, line_dash='dash', line_color='gray')
                            fig_roll.update_layout(title="60-Day Rolling Correlation: S&P 500 vs 10Y Yield", xaxis_title="Date", yaxis_title="Correlation", template="plotly_white", height=300)
                            st.plotly_chart(fig_roll, use_container_width=True)
            else:
                st.info("Correlation data unavailable.")

    # =====================================================================
    # TAB 7: Directional Dynamics
    # =====================================================================
    with main_tabs[6]:
        st.subheader("📡 Directional Rate Dynamics")
        fig_dyn = chart_rate_dynamics(rate_directions)
        if fig_dyn:
            st.plotly_chart(fig_dyn, use_container_width=True)
        
        st.subheader("🎭 Scenario Analysis")
        scenario_name = st.selectbox("Select Scenario", list(scenarios.keys()))
        scenario_df = scenarios[scenario_name]
        st.plotly_chart(chart_scenario_analysis(scenario_df, scenario_name), use_container_width=True)
        
        st.dataframe(scenario_df.assign(Change=scenario_df["Scenario"] - scenario_df["Current"]).round(4), use_container_width=True)
        
        scenario_notes = {
            "Bull Steepener": "Short and intermediate yields decline modestly, while the long end declines more. This is usually associated with falling inflation expectations, softer growth, or a rally led by duration.",
            "Bear Flattener": "Front-end yields rise more than the long end. This often reflects hawkish central-bank policy or repricing of the policy path.",
            "Recession Case": "Broad rally across the curve, with the strongest fall at the short end. This scenario emphasizes defensive allocation and recession hedging.",
            "Policy Easing": "Short-end yields fall sharply after expected policy cuts, while the long end reacts less. This helps users isolate the transmission of a central-bank easing cycle."
        }
        st.markdown(f'<div class="info-box"><b>{scenario_name}</b><br><br>{scenario_notes.get(scenario_name, "Analyze the impact of different market scenarios on the yield curve.")}</div>', unsafe_allow_html=True)

    # =====================================================================
    # TAB 8: Technical Analysis
    # =====================================================================
    with main_tabs[7]:
        st.subheader("🛠 Technical Analysis Layer")
        if not YFINANCE_AVAILABLE:
            st.error("yfinance is not installed in this environment. Add it to requirements.txt and redeploy.")
        else:
            ohlc_df = ohlc_map.get(ohlc_ticker)
            if ohlc_df is not None and not ohlc_df.empty:
                col1, col2 = st.columns([1, 1])
                with col1:
                    fig_ohlc = chart_ohlc_with_indicators(ohlc_df, ohlc_ticker)
                    if fig_ohlc:
                        st.plotly_chart(fig_ohlc, use_container_width=True)
                with col2:
                    fig_ta = chart_technical(ohlc_df, ohlc_ticker)
                    if fig_ta:
                        st.plotly_chart(fig_ta, use_container_width=True)
                
                st.markdown(
                    '<div class="note-box"><b>📖 Technical Interpretation</b><br><br>Use RSI for momentum stress, MACD for trend shifts, and Bollinger Bands for volatility envelope analysis. This technical layer is tactical and should be interpreted alongside the macro curve regime.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info(f"Technical data unavailable for {ohlc_ticker}.")

    # =====================================================================
    # TAB 9: System Export
    # =====================================================================
    with main_tabs[8]:
        st.subheader("💾 System Export")
        e1, e2 = st.columns(2)
        with e1:
            st.download_button("📥 Export Yield Matrix (CSV)", yield_df.to_csv().encode("utf-8"), f"yield_matrix_{datetime.now():%Y%m%d}.csv", "text/csv")
            st.download_button("📥 Export Spreads (CSV)", spreads.to_csv().encode("utf-8"), f"spreads_{datetime.now():%Y%m%d}.csv", "text/csv")
            if not governance_df.empty:
                st.download_button("📥 Export Governance Table (CSV)", governance_df.to_csv(index=False).encode("utf-8"), f"governance_{datetime.now():%Y%m%d}.csv", "text/csv")
            if dynamic_params is not None and not dynamic_params.empty:
                st.download_button("📥 Export Dynamic Parameters (CSV)", dynamic_params.to_csv(index=False).encode("utf-8"), f"dynamic_params_{datetime.now():%Y%m%d}.csv", "text/csv")
        with e2:
            if volatility_df is not None and not volatility_df.empty:
                st.download_button("📥 Export Volatility Data (CSV)", volatility_df.to_csv().encode("utf-8"), f"volatility_{datetime.now():%Y%m%d}.csv", "text/csv")
            if correlation_df is not None and not correlation_df.empty:
                st.download_button("📥 Export Correlation Data (CSV)", correlation_df.to_csv().encode("utf-8"), f"correlation_{datetime.now():%Y%m%d}.csv", "text/csv")
            if pca_risk:
                st.download_button("📥 Export PCA Loadings (CSV)", pca_risk["loadings"].to_csv().encode("utf-8"), f"pca_loadings_{datetime.now():%Y%m%d}.csv", "text/csv")
            if not rate_directions.empty:
                st.download_button("📥 Export Rate Dynamics (CSV)", rate_directions.to_csv().encode("utf-8"), f"rate_dynamics_{datetime.now():%Y%m%d}.csv", "text/csv")

        st.markdown(
            '<div class="success-box"><b>🚀 Deployment Notes</b><br><br>For Streamlit Cloud, keep <code>app.py</code> and <code>requirements.txt</code> in the repository root. Make sure <code>yfinance</code> and <code>scikit-learn</code> are present in requirements. The app expects the user to enter the FRED API key inside the UI.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#667085; font-size:0.75rem;'>"
        "Institutional Quantitative Platform | Algorithmic Forecasting | Advanced Monte Carlo Engine<br>"
        "MK Istanbul Fintech LabGEN © 2026 | All Rights Reserved"
        "</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
