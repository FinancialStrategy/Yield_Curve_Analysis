import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import jarque_bera, anderson

warnings.filterwarnings("ignore")

# =============================================================================
# DYNAMIC QUANTITATIVE ANALYSIS MODEL - INSTITUTIONAL PLATFORM
# Professional fixed-income analytics with enhanced visual clarity
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
    "band": "rgba(108, 142, 173, 0.10"),
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

DEFAULT_STATE = {
    "api_key_validated": False,
    "api_key": "",
    "yield_data": None,
    "recession_data": None,
    "ohlc_data": None,
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

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def stochastic_oscillator(df: pd.DataFrame, n: int = 14) -> Tuple[pd.Series, pd.Series]:
    low_n = df['Low'].rolling(n).min()
    high_n = df['High'].rolling(n).max()
    k = 100 * (df['Close'] - low_n) / (high_n - low_n)
    d = k.rolling(3).mean()
    return k, d

def volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'Volume' in df.columns:
        out['Volume_SMA'] = df['Volume'].rolling(20).mean()
        out['Volume_Ratio'] = df['Volume'] / out['Volume_SMA']
    return out

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
    out["RSI_21"] = rsi(out["Close"], 21)
    out["MACD"], out["MACD_Signal"], out["MACD_Hist"] = macd(out["Close"])
    out["BB_Upper"], out["BB_Middle"], out["BB_Lower"] = bb(out["Close"])
    out["ATR"] = atr(out, 14)
    k, d = stochastic_oscillator(out, 14)
    out["Stoch_K"] = k
    out["Stoch_D"] = d
    out = volume_indicators(out)
    return out

def detect_support_resistance(df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
    """Detect support and resistance levels"""
    support = df['Low'].rolling(window, center=True).min()
    resistance = df['High'].rolling(window, center=True).max()
    return support, resistance

def generate_technical_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive technical trading signals"""
    signals = pd.DataFrame(index=df.index)
    
    # RSI signals
    signals['RSI_Oversold'] = df['RSI'] < 30
    signals['RSI_Overbought'] = df['RSI'] > 70
    
    # MACD signals
    signals['MACD_Bullish'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
    signals['MACD_Bearish'] = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
    
    # Moving average signals
    signals['Golden_Cross'] = (df['SMA_20'] > df['SMA_50']) & (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1))
    signals['Death_Cross'] = (df['SMA_20'] < df['SMA_50']) & (df['SMA_20'].shift(1) >= df['SMA_50'].shift(1))
    
    # Bollinger Band signals
    signals['BB_Oversold'] = df['Close'] < df['BB_Lower']
    signals['BB_Overbought'] = df['Close'] > df['BB_Upper']
    
    # Stochastic signals
    signals['Stoch_Oversold'] = df['Stoch_K'] < 20
    signals['Stoch_Overbought'] = df['Stoch_K'] > 80
    signals['Stoch_Bullish'] = (df['Stoch_K'] > df['Stoch_D']) & (df['Stoch_K'].shift(1) <= df['Stoch_D'].shift(1))
    
    return signals

# =============================================================================
# ANALYTICS
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

def classify_regime(spreads: pd.DataFrame, yield_df: pd.DataFrame) -> Tuple[str, str]:
    spread = spreads["10Y-2Y"].iloc[-1] if "10Y-2Y" in spreads.columns else np.nan
    y10 = yield_df["10Y"].iloc[-1] if "10Y" in yield_df.columns else np.nan
    if np.isfinite(spread) and spread < 0:
        return "Risk-off / Recession Watch", "Curve inversion signals defensive macro regime and elevated recession risk."
    if np.isfinite(spread) and spread < 50:
        return "Neutral / Late Cycle", "Curve flattening suggests late-cycle caution and potential policy inflection."
    if np.isfinite(y10) and y10 > 5.5:
        return "Neutral / Restrictive", "Elevated long-end rates indicate restrictive financial conditions."
    return "Risk-on / Expansion", "Positive slope supports pro-risk positioning and cyclical growth."

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
            recessions.append({
                "start": start,
                "end": date,
                "duration_days": (date - start).days,
                "duration_months": (date - start).days / 30.44,
            })
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
            seg = s.loc[start:date]
            periods.append({
                "start": start,
                "end": date,
                "depth": float(seg.min()),
                "duration_days": (date - start).days,
                "duration_months": (date - start).days / 30.44,
            })
            in_inv = False
    return periods

def calculate_lead_times(inversions: List[dict], recessions: List[dict]) -> List[dict]:
    lead_times = []
    for inv in inversions:
        for rec in recessions:
            if inv["start"] < rec["start"]:
                days = (rec["start"] - inv["start"]).days
                lead_times.append({
                    "inversion_start": inv["start"],
                    "recession_start": rec["start"],
                    "lead_days": days,
                    "lead_months": days / 30.44,
                    "inversion_depth": inv["depth"],
                })
                break
    return lead_times

class NelsonSiegelModel:
    @staticmethod
    def nelson_siegel(tau, beta0, beta1, beta2, lambda1):
        tau = np.asarray(tau, dtype=float)
        ts = np.where(tau == 0, 1e-8, tau)
        x = lambda1 * ts
        term1 = (1 - np.exp(-x)) / x
        term2 = term1 - np.exp(-x)
        out = beta0 + beta1 * term1 + beta2 * term2
        return np.where(tau == 0, beta0 + beta1 + beta2, out)

    @staticmethod
    def nss(tau, beta0, beta1, beta2, beta3, lambda1, lambda2):
        tau = np.asarray(tau, dtype=float)
        ts = np.where(tau == 0, 1e-8, tau)
        x1 = lambda1 * ts
        x2 = lambda2 * ts
        t1 = (1 - np.exp(-x1)) / x1
        t2 = t1 - np.exp(-x1)
        t3 = ((1 - np.exp(-x2)) / x2) - np.exp(-x2)
        out = beta0 + beta1 * t1 + beta2 * t2 + beta3 * t3
        return np.where(tau == 0, beta0 + beta1 + beta2 + beta3, out)

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
            "mae": float(np.mean(np.abs(yields_ - fitted))),
            "r_squared": float(1 - sse / sst) if sst > 0 else np.nan,
            "residuals": yields_ - fitted,
        }

    @staticmethod
    def fit_nss(maturities: np.ndarray, yields_: np.ndarray):
        def objective(params):
            fitted = NelsonSiegelModel.nss(maturities, *params)
            weights = 1 / (maturities + 0.25)
            return np.sum(weights * (yields_ - fitted) ** 2)
        bounds = [
            (yields_.min() - 2, yields_.max() + 2),
            (-20, 20), (-20, 20), (-20, 20),
            (0.01, 10), (0.01, 10),
        ]
        res = differential_evolution(objective, bounds=bounds, maxiter=220, popsize=10, polish=True, seed=42)
        if not res.success:
            return None
        fitted = NelsonSiegelModel.nss(maturities, *res.x)
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
    for name, res in [("Nelson-Siegel (NS)", ns_result), ("Nelson-Siegel-Svensson (NSS)", nss_result)]:
        if res is None:
            continue
        rmse = res["rmse"]
        r2 = res["r_squared"]
        max_abs = float(np.max(np.abs(res["residuals"])))
        
        # Enhanced confidence assessment
        if rmse < 0.04 and r2 > 0.99:
            confidence = "HIGH - Excellent fit for institutional use"
            color = "green"
        elif rmse < 0.07 and r2 > 0.98:
            confidence = "GOOD - Suitable for most applications"
            color = "blue"
        elif rmse < 0.12 and r2 > 0.95:
            confidence = "MODERATE - Use with caution for precise pricing"
            color = "orange"
        else:
            confidence = "LOW - Significant model limitations, not recommended for pricing"
            color = "red"
        
        flags = []
        if max_abs > 0.15:
            flags.append("⚠️ Large residual outlier detected")
        if r2 < 0.95:
            flags.append("⚠️ Low R-squared indicates poor fit")
        if rmse > 0.10:
            flags.append("⚠️ High RMSE suggests model inadequacy")
        if any(abs(r) > 0.10 for r in res["residuals"]):
            flags.append("⚠️ Individual maturity residuals exceed 10bps")
        
        rows.append({
            "Model": name,
            "RMSE (bps)": rmse * 100,
            "MAE (bps)": res["mae"] * 100,
            "R²": r2,
            "Max Abs Residual (bps)": max_abs * 100,
            "Fit Confidence": confidence,
            "Warning Flags": ", ".join(flags) if flags else "✓ No significant issues detected",
        })
    return pd.DataFrame(rows)

def rolling_ns_parameters(yield_df: pd.DataFrame, maturities: np.ndarray, selected_cols: List[str], years: int) -> pd.DataFrame:
    window_size = years * 252
    if len(yield_df) <= window_size + 5:
        return pd.DataFrame()
    out = []
    for i in range(window_size, len(yield_df), CFG.rolling_step):
        curve = yield_df.iloc[i][selected_cols].values
        res = NelsonSiegelModel.fit_ns(maturities, curve)
        if res:
            out.append({
                "date": yield_df.index[i],
                "beta0": res["params"][0],
                "beta1": res["params"][1],
                "beta2": res["params"][2],
                "lambda": res["params"][3],
                "rmse": res["rmse"],
            })
    return pd.DataFrame(out)

def factor_contributions(yield_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=yield_df.index)
    if "10Y" in yield_df.columns:
        out["Level (Long-term trend)"] = yield_df["10Y"]
    if {"10Y", "3M"}.issubset(yield_df.columns):
        out["Slope (Monetary policy stance)"] = (yield_df["10Y"] - yield_df["3M"]) * 100
    if {"3M", "5Y", "10Y"}.issubset(yield_df.columns):
        out["Curvature (Medium-term premium)"] = (2 * yield_df["5Y"] - (yield_df["3M"] + yield_df["10Y"])) * 100
    if {"2Y", "10Y", "30Y"}.issubset(yield_df.columns):
        out["Butterfly (Long-term expectations)"] = (2 * yield_df["10Y"] - (yield_df["2Y"] + yield_df["30Y"])) * 100
    return out

def pca_risk_decomp(yield_df: pd.DataFrame, n_components: int = 3) -> Optional[dict]:
    returns = yield_df.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if returns.shape[0] < 20 or returns.shape[1] < 2:
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
        "n_components": k,
    }

def calculate_var_metrics(returns: pd.Series, confidence: float = 0.95, horizon: int = 10) -> Optional[dict]:
    returns = returns.dropna()
    if len(returns) < 20:
        return None
    var_hist = np.percentile(returns, (1 - confidence) * 100)
    cvar = returns[returns <= var_hist].mean()
    var_param = norm.ppf(1 - confidence) * returns.std()
    skew_val = returns.skew()
    kurt_val = returns.kurtosis()
    z = norm.ppf(1 - confidence)
    z_cf = z + (z**2 - 1) * skew_val / 6 + (z**3 - 3 * z) * kurt_val / 24 - (2 * z**3 - 5 * z) * skew_val**2 / 36
    
    # Additional risk metrics
    jb_stat, jb_pvalue = jarque_bera(returns.dropna())
    anderson_result = anderson(returns.dropna())
    
    return {
        "VaR_Historical": float(var_hist * np.sqrt(horizon)),
        "VaR_Parametric": float(var_param * np.sqrt(horizon)),
        "VaR_CornishFisher": float(z_cf * returns.std() * np.sqrt(horizon)),
        "CVaR": float(cvar * np.sqrt(horizon)),
        "Skewness": float(skew_val),
        "Kurtosis": float(kurt_val),
        "Jarque_Bera_Stat": float(jb_stat),
        "Jarque_Bera_Pvalue": float(jb_pvalue),
        "Anderson_Darling": float(anderson_result.statistic),
        "Volatility_Annualized": float(returns.std() * np.sqrt(252)),
    }

def forecast_curve(yield_df: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    if len(yield_df) < 120:
        return pd.DataFrame()
    x = np.arange(len(yield_df)).reshape(-1, 1)
    future_x = np.arange(len(yield_df), len(yield_df) + horizon).reshape(-1, 1)
    out = {}
    for col in yield_df.columns:
        model = LinearRegression()
        model.fit(x, yield_df[col].values)
        out[col] = model.predict(future_x)
    dates = pd.bdate_range(yield_df.index[-1] + pd.Timedelta(days=1), periods=horizon)
    return pd.DataFrame(out, index=dates)

def scenario_engine(yield_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    latest = yield_df.iloc[-1].copy()
    scenarios = {}
    tenor_order = list(yield_df.columns)

    # Bull steepener: long-end down more than short-end
    bull = latest.copy()
    for col in tenor_order:
        m = MATURITY_MAP.get(col, 1)
        bull[col] = bull[col] - (0.08 + 0.06 * min(m / 10, 1.5))
    scenarios["Bull Steepener (Growth expectations decline)"] = pd.DataFrame({"Current": latest, "Scenario": bull})

    # Bear flattener: front-end up more than long-end
    bear = latest.copy()
    for col in tenor_order:
        m = MATURITY_MAP.get(col, 1)
        bear[col] = bear[col] + (0.14 if m <= 2 else 0.07)
    scenarios["Bear Flattener (Hawkish policy pivot)"] = pd.DataFrame({"Current": latest, "Scenario": bear})

    # Recession case: broad rally, short end falls sharply
    recession = latest.copy()
    for col in tenor_order:
        m = MATURITY_MAP.get(col, 1)
        recession[col] = recession[col] - (0.22 if m <= 2 else 0.14 if m <= 10 else 0.10)
    scenarios["Recession Scenario (Defensive positioning)"] = pd.DataFrame({"Current": latest, "Scenario": recession})

    # Policy easing: front-end down, moderate long-end response
    easing = latest.copy()
    for col in tenor_order:
        m = MATURITY_MAP.get(col, 1)
        easing[col] = easing[col] - (0.25 if m <= 2 else 0.12 if m <= 10 else 0.06)
    scenarios["Policy Easing (Central bank cuts)"] = pd.DataFrame({"Current": latest, "Scenario": easing})

    return scenarios

def recession_hit_stats(inversions: List[dict], recessions: List[dict]) -> dict:
    if not inversions:
        return {"HitRatio": np.nan, "FalsePositiveRate": np.nan, "Matches": 0, "Signals": 0}
    leads = calculate_lead_times(inversions, recessions)
    matches = len(leads)
    signals = len(inversions)
    false_positives = signals - matches
    return {
        "HitRatio": matches / signals if signals else np.nan,
        "FalsePositiveRate": false_positives / signals if signals else np.nan,
        "Matches": matches,
        "Signals": signals,
    }

def current_inversion_vs_history(spreads: pd.DataFrame, inversions: List[dict]) -> pd.DataFrame:
    if "10Y-2Y" not in spreads.columns:
        return pd.DataFrame()
    current = spreads["10Y-2Y"].iloc[-1]
    rows = [{"Metric": "Current 10Y-2Y Spread (bps)", "Value": current}]
    if inversions:
        rows.append({"Metric": "Historical Minimum Inversion Depth (bps)", "Value": min(x["depth"] for x in inversions)})
        rows.append({"Metric": "Historical Average Inversion Depth (bps)", "Value": np.mean([x["depth"] for x in inversions])})
        rows.append({"Metric": "Historical Average Inversion Duration (days)", "Value": np.mean([x["duration_days"] for x in inversions])})
    return pd.DataFrame(rows)

def arbitrage_diagnostics(yield_df: pd.DataFrame, maturities: np.ndarray) -> Optional[dict]:
    latest = yield_df.iloc[-1].values[: len(maturities)]
    nss = NelsonSiegelModel.fit_nss(maturities, latest)
    if nss is None:
        return None
    theoretical = nss["fitted_values"]
    residuals = latest - theoretical
    rows = []
    for i, (m, r) in enumerate(zip(maturities, residuals)):
        if abs(r) > 0.10:
            rows.append({
                "Tenor (Years)": m,
                "Actual Yield (%)": latest[i],
                "Theoretical Yield (%)": theoretical[i],
                "Difference (bps)": r * 100,
                "Relative Value Signal": "UNDERVALUED (Buy)" if r > 0 else "OVERVALUED (Sell)",
            })
    return {
        "mean_abs_error_bps": float(np.mean(np.abs(residuals)) * 100),
        "max_abs_error_bps": float(np.max(np.abs(residuals)) * 100),
        "std_error_bps": float(np.std(residuals) * 100),
        "mispriced_count": len(rows),
        "mispriced_table": pd.DataFrame(rows) if rows else pd.DataFrame(),
    }

# =============================================================================
# ENHANCED VISUALS WITH BETTER AXIS LABELS
# =============================================================================

def add_recession_bands(fig: go.Figure, recessions: List[dict]) -> go.Figure:
    for rec in recessions:
        fig.add_vrect(x0=rec["start"], x1=rec["end"], fillcolor=COLORS["recession"], opacity=0.35, layer="below", line_width=0)
    return fig

def create_chart_layout(fig: go.Figure, title: str, y_title: Optional[str] = None, height: int = 460, x_title: str = "Date") -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(size=12, color=COLORS["text"], family="Arial, sans-serif"),
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=16, color=COLORS["text"], weight="bold")),
        margin=dict(l=60, r=30, t=80, b=50),
        hovermode="x unified",
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=13, weight="bold")),
            tickfont=dict(size=11),
            gridcolor=COLORS["grid"],
            gridwidth=1,
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=13, weight="bold")),
            tickfont=dict(size=11),
            gridcolor=COLORS["grid"],
            gridwidth=1,
            showgrid=True,
            zeroline=True,
            zerolinecolor=COLORS["grid_dark"],
            zerolinewidth=1,
        ),
    )
    return fig

def chart_current_curve(maturities: np.ndarray, yields_: np.ndarray, ns_result: Optional[dict], nss_result: Optional[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=maturities, y=yields_, mode="markers+lines", name="Actual Market Curve",
        marker=dict(size=12, color=COLORS["accent"], symbol="circle"),
        line=dict(color=COLORS["accent"], width=2.5),
    ))
    if ns_result:
        fig.add_trace(go.Scatter(x=maturities, y=ns_result["fitted_values"], mode="lines", name="Nelson-Siegel (3-factor)", line=dict(color=COLORS["accent2"], width=2.5, dash="solid")))
    if nss_result:
        fig.add_trace(go.Scatter(x=maturities, y=nss_result["fitted_values"], mode="lines", name="Nelson-Siegel-Svensson (4-factor)", line=dict(color=COLORS["warning"], width=2.5, dash="dot")))
    return create_chart_layout(fig, "Current Treasury Yield Curve with Model Fits", "Yield (%)", 480, "Maturity (Years)")

def chart_yield(yield_df: pd.DataFrame, tenor: str, color: str, title: str) -> Optional[go.Figure]:
    if tenor not in yield_df.columns:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yield_df.index, y=yield_df[tenor], mode="lines", name=tenor, line=dict(color=color, width=2.5)))
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))
    return create_chart_layout(fig, title, "Yield (%)", 440)

def chart_spreads(spreads: pd.DataFrame, recessions: List[dict]) -> go.Figure:
    fig = go.Figure()
    palette = [COLORS["negative"], COLORS["accent"], COLORS["warning"], COLORS["accent2"]]
    for i, col in enumerate(spreads.columns):
        fig.add_trace(go.Scatter(x=spreads.index, y=spreads[col], mode="lines", name=col, line=dict(color=palette[i % len(palette)], width=2)))
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["text_secondary"], line_width=2)
    add_recession_bands(fig, recessions)
    return create_chart_layout(fig, "Treasury Spread Dashboard - Key Yield Differentials", "Basis Points (bps)", 520)

def chart_model_residuals(selected_cols: List[str], ns_result: Optional[dict], nss_result: Optional[dict]) -> Optional[go.Figure]:
    if ns_result is None and nss_result is None:
        return None
    fig = go.Figure()
    if ns_result:
        fig.add_trace(go.Bar(x=selected_cols, y=ns_result["residuals"] * 100, name="NS Residuals", marker_color=COLORS["accent2"], opacity=0.7))
    if nss_result:
        fig.add_trace(go.Bar(x=selected_cols, y=nss_result["residuals"] * 100, name="NSS Residuals", marker_color=COLORS["warning"], opacity=0.7))
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["text_secondary"], line_width=2)
    fig.add_hrect(y0=-5, y1=5, line_width=0, fillcolor="green", opacity=0.1, annotation_text="Acceptable Range (±5bps)", annotation_position="top left")
    fig.add_hrect(y0=-10, y1=10, line_width=0, fillcolor="yellow", opacity=0.1)
    return create_chart_layout(fig, "Model Residual Analysis - Pricing Errors by Maturity", "Residual (bps)", 480, "Tenor")

def chart_dynamic(dynamic_params: pd.DataFrame) -> Optional[go.Figure]:
    if dynamic_params is None or dynamic_params.empty:
        return None
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=("β₀ - Level Factor (Long-term Trend)", "β₁ - Slope Factor (Monetary Policy Stance)", 
                        "β₂ - Curvature Factor (Medium-term Premium)", "RMSE - Model Fit Quality"),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )
    fig.add_trace(go.Scatter(x=dynamic_params["date"], y=dynamic_params["beta0"], mode="lines", name="β₀", line=dict(color=COLORS["accent2"], width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dynamic_params["date"], y=dynamic_params["beta1"], mode="lines", name="β₁", line=dict(color=COLORS["accent"], width=2.5)), row=1, col=2)
    fig.add_trace(go.Scatter(x=dynamic_params["date"], y=dynamic_params["beta2"], mode="lines", name="β₂", line=dict(color=COLORS["warning"], width=2.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=dynamic_params["date"], y=dynamic_params["rmse"] * 100, mode="lines", name="RMSE", line=dict(color=COLORS["muted"], width=2.5)), row=2, col=2)
    
    fig.update_yaxes(title_text="Yield (%)", row=1, col=1)
    fig.update_yaxes(title_text="Spread (bps)", row=1, col=2)
    fig.update_yaxes(title_text="Spread (bps)", row=2, col=1)
    fig.update_yaxes(title_text="Basis Points", row=2, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    
    return create_chart_layout(fig, "Dynamic Nelson-Siegel Parameter Evolution", height=680)

def chart_factors(factor_df: pd.DataFrame) -> Optional[go.Figure]:
    if factor_df is None or factor_df.empty:
        return None
    fig = go.Figure()
    palette = [COLORS["accent"], COLORS["warning"], COLORS["accent2"], COLORS["positive"]]
    for i, col in enumerate(factor_df.columns):
        fig.add_trace(go.Scatter(x=factor_df.index, y=factor_df[col], mode="lines", name=col, line=dict(color=palette[i % len(palette)], width=2)))
    return create_chart_layout(fig, "Historical Factor Contributions - Curve Dynamics Decomposition", "Value", 480)

def chart_pca_variance(pca_risk: Optional[dict]) -> Optional[go.Figure]:
    if not pca_risk:
        return None
    fig = go.Figure()
    ev = pca_risk["explained_variance"] * 100
    cumulative = pca_risk["cumulative_variance"] * 100
    fig.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(ev))], y=ev, name="Individual", marker_color=COLORS["accent"], text=[f"{x:.1f}%" for x in ev], textposition="outside"))
    fig.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(cumulative))], y=cumulative, name="Cumulative", mode="lines+markers", line=dict(color=COLORS["warning"], width=3), marker=dict(size=10)))
    fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="95% Threshold")
    return create_chart_layout(fig, "Principal Component Analysis - Variance Decomposition", "Variance Explained (%)", 500, "Principal Component")

def chart_scenario(scenario_df: pd.DataFrame, scenario_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=scenario_df.index, y=scenario_df["Current"], name="Current Curve", marker_color=COLORS["accent2"], text=[f"{x:.2f}%" for x in scenario_df["Current"]], textposition="outside"))
    fig.add_trace(go.Bar(x=scenario_df.index, y=scenario_df["Scenario"], name=scenario_name, marker_color=COLORS["warning"], text=[f"{x:.2f}%" for x in scenario_df["Scenario"]], textposition="outside"))
    fig.update_layout(barmode="group", bargap=0.15)
    return create_chart_layout(fig, f"Scenario Analysis | {scenario_name}", "Yield (%)", 500, "Tenor")

def chart_ohlc(ohlc_df: pd.DataFrame, ticker: str) -> Optional[go.Figure]:
    if ohlc_df is None or ohlc_df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=ohlc_df.index, open=ohlc_df["Open"], high=ohlc_df["High"], low=ohlc_df["Low"], close=ohlc_df["Close"],
        increasing=dict(line=dict(color=COLORS["positive"], width=1.5), fillcolor=COLORS["positive"]),
        decreasing=dict(line=dict(color=COLORS["negative"], width=1.5), fillcolor=COLORS["negative"]),
        name=ticker,
    ))
    if "SMA_20" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["SMA_20"], mode="lines", name="SMA 20 (Short-term)", line=dict(color=COLORS["accent"], width=1.8)))
    if "SMA_50" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["SMA_50"], mode="lines", name="SMA 50 (Medium-term)", line=dict(color=COLORS["warning"], width=1.8)))
    if "SMA_200" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["SMA_200"], mode="lines", name="SMA 200 (Long-term)", line=dict(color=COLORS["positive"], width=1.8, dash="dash")))
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))
    return create_chart_layout(fig, f"OHLC Price Analysis | {ticker}", "Price (USD)", 550)

def chart_technical_panels(ohlc_df: pd.DataFrame, ticker: str, signals_df: pd.DataFrame) -> Optional[go.Figure]:
    if ohlc_df is None or ohlc_df.empty:
        return None
    
    fig = make_subplots(
        rows=5, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.04,
        row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
        subplot_titles=(
            f"{ticker} Price with Technical Overlays",
            "RSI (14) - Momentum Oscillator",
            "MACD - Trend & Momentum",
            "Stochastic Oscillator (14,3)",
            "Volume Analysis"
        ),
    )
    
    # Price panel with Bollinger Bands
    fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["Close"], mode="lines", name="Close Price", line=dict(color=COLORS["accent"], width=2.5)), row=1, col=1)
    if "BB_Upper" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["BB_Upper"], mode="lines", name="BB Upper (2σ)", line=dict(color=COLORS["muted"], width=1.2, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["BB_Lower"], mode="lines", name="BB Lower (2σ)", line=dict(color=COLORS["muted"], width=1.2, dash="dash"), fill='tonexty', fillcolor=COLORS["band"]), row=1, col=1)
    
    # RSI panel
    if "RSI" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["RSI"], mode="lines", name="RSI", line=dict(color=COLORS["accent"], width=2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=COLORS["negative"], line_width=1.5, row=2, col=1, annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color=COLORS["positive"], line_width=1.5, row=2, col=1, annotation_text="Oversold (30)")
        fig.add_hrect(y0=30, y1=70, line_width=0, fillcolor="green", opacity=0.05, row=2, col=1)
    
    # MACD panel
    if "MACD" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["MACD"], mode="lines", name="MACD Line", line=dict(color=COLORS["positive"], width=2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["MACD_Signal"], mode="lines", name="Signal Line", line=dict(color=COLORS["negative"], width=2)), row=3, col=1)
        # Histogram
        colors = ["red" if x < 0 else "green" for x in ohlc_df["MACD_Hist"]]
        fig.add_trace(go.Bar(x=ohlc_df.index, y=ohlc_df["MACD_Hist"], name="Histogram", marker_color=colors, opacity=0.6), row=3, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color=COLORS["text_secondary"], line_width=1, row=3, col=1)
    
    # Stochastic panel
    if "Stoch_K" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["Stoch_K"], mode="lines", name="%K (Fast)", line=dict(color=COLORS["accent"], width=2)), row=4, col=1)
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["Stoch_D"], mode="lines", name="%D (Slow)", line=dict(color=COLORS["warning"], width=2)), row=4, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color=COLORS["negative"], line_width=1.5, row=4, col=1, annotation_text="Overbought")
        fig.add_hline(y=20, line_dash="dash", line_color=COLORS["positive"], line_width=1.5, row=4, col=1, annotation_text="Oversold")
    
    # Volume panel
    if "Volume" in ohlc_df.columns:
        colors_vol = ["red" if ohlc_df["Close"].iloc[i] < ohlc_df["Close"].iloc[i-1] else "green" for i in range(len(ohlc_df))]
        fig.add_trace(go.Bar(x=ohlc_df.index, y=ohlc_df["Volume"], name="Volume", marker_color=colors_vol, opacity=0.6), row=5, col=1)
        if "Volume_SMA" in ohlc_df.columns:
            fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["Volume_SMA"], mode="lines", name="Volume SMA 20", line=dict(color=COLORS["accent"], width=1.5, dash="dash")), row=5, col=1)
    
    # Add signal markers
    if signals_df is not None and len(signals_df) > 0:
        # Highlight buy/sell signals on price chart
        buy_signals = signals_df[signals_df['RSI_Oversold'] | signals_df['MACD_Bullish'] | signals_df['Stoch_Bullish']]
        sell_signals = signals_df[signals_df['RSI_Overbought'] | signals_df['MACD_Bearish']]
        
        if len(buy_signals) > 0:
            fig.add_trace(go.Scatter(
                x=buy_signals.index, y=ohlc_df.loc[buy_signals.index, 'Close'],
                mode="markers", name="BUY Signal", marker=dict(symbol="triangle-up", size=12, color=COLORS["positive"], line=dict(width=1, color="white"))
            ), row=1, col=1)
        
        if len(sell_signals) > 0:
            fig.add_trace(go.Scatter(
                x=sell_signals.index, y=ohlc_df.loc[sell_signals.index, 'Close'],
                mode="markers", name="SELL Signal", marker=dict(symbol="triangle-down", size=12, color=COLORS["negative"], line=dict(width=1, color="white"))
            ), row=1, col=1)
    
    fig.update_xaxes(title_text="Date", row=5, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Stochastic", row=4, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Volume", row=5, col=1)
    
    return create_chart_layout(fig, f"Professional Technical Analysis Dashboard | {ticker}", height=1000)

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
            <div class="main-subtitle">Institutional Fixed-Income Platform | Real-Time Yield Curve Analytics | Risk Management Suite</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.api_key_validated:
        render_api_gate()

    with st.sidebar:
        st.markdown("### 🎛️ Control Tower")
        st.caption("Institutional fixed-income macro monitoring and risk analytics")
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
        rolling_years = st.slider("Rolling window (years)", 2, 10, CFG.rolling_years_default, help="Number of years for rolling parameter estimation")
        forecast_horizon = st.slider("Forecast horizon (business days)", 5, 60, CFG.forecast_horizon_default, help="Forward yield curve projection period")
        confidence_level = st.slider("VaR confidence level", 0.90, 0.99, 0.95, 0.01, help="Statistical confidence for risk metrics")
        ohlc_period = st.selectbox("OHLC data period", ["6mo", "1y", "2y", "5y"], index=2, help="Historical period for technical analysis")

    if not st.session_state.data_fetched:
        with st.spinner("Fetching Treasury, recession, and OHLC data..."):
            yield_df = fetch_all_yield_data(st.session_state.api_key)
            recession_series = fetch_recession_data(st.session_state.api_key)
            ohlc_data = {}
            for ticker in YAHOO_TICKERS:
                df = fetch_ohlc_data(ticker, ohlc_period)
                if df is not None and not df.empty:
                    ohlc_data[ticker] = add_technical_indicators(df)
        if yield_df is None:
            st.error("Failed to fetch FRED data. Please check your API key and try again.")
            st.stop()
        st.session_state.yield_data = yield_df
        st.session_state.recession_data = recession_series
        st.session_state.ohlc_data = ohlc_data
        st.session_state.data_fetched = True

    yield_df = st.session_state.yield_data.copy()
    recession_series = st.session_state.recession_data.copy() if st.session_state.recession_data is not None else None
    ohlc_data = st.session_state.ohlc_data if st.session_state.ohlc_data is not None else {}

    selected_cols = [c for c in yield_df.columns if c in MATURITY_MAP]
    maturities = np.array([MATURITY_MAP[c] for c in selected_cols], dtype=float)
    latest_curve = yield_df.iloc[-1][selected_cols].values.astype(float)

    spreads = compute_spreads(yield_df)
    recessions = identify_recessions(recession_series)
    inversions = calculate_inversion_periods(spreads)
    lead_times = calculate_lead_times(inversions, recessions)
    hit_stats = recession_hit_stats(inversions, recessions)
    regime, regime_text = classify_regime(spreads, yield_df)
    recession_prob = recession_probability_proxy(spreads, yield_df)

    current_2y = yield_df["2Y"].iloc[-1] if "2Y" in yield_df.columns else np.nan
    current_10y = yield_df["10Y"].iloc[-1] if "10Y" in yield_df.columns else np.nan
    current_30y = yield_df["30Y"].iloc[-1] if "30Y" in yield_df.columns else np.nan
    current_spread = spreads["10Y-2Y"].iloc[-1] if "10Y-2Y" in spreads.columns else np.nan

    with st.spinner("Running quantitative model layer..."):
        ns_result = NelsonSiegelModel.fit_ns(maturities, latest_curve)
        nss_result = NelsonSiegelModel.fit_nss(maturities, latest_curve)
        governance_df = model_governance(ns_result, nss_result)
        dynamic_params = rolling_ns_parameters(yield_df[selected_cols], maturities, selected_cols, rolling_years)
        factor_df = factor_contributions(yield_df)
        pca_risk = pca_risk_decomp(yield_df[selected_cols])
        forecast_df = forecast_curve(yield_df[selected_cols], forecast_horizon)
        arb = arbitrage_diagnostics(yield_df[selected_cols], maturities)
        scenarios = scenario_engine(yield_df[selected_cols])

    # KPI Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        kpi_card("📊 Macro Regime", regime, regime_text[:50] + "...")
    with c2:
        kpi_card("🏦 2Y Yield", f"{current_2y:.2f}%", "Short-end policy anchor")
    with c3:
        kpi_card("📈 10Y Yield", f"{current_10y:.2f}%", "Long-end benchmark")
    with c4:
        kpi_card("🏛️ 30Y Yield", f"{current_30y:.2f}%", "Long duration reference")
    with c5:
        kpi_card("🔄 10Y-2Y Spread", f"{current_spread:.1f} bps", "Primary recession signal")
    with c6:
        kpi_card("⚠️ Recession Probability", f"{100 * recession_prob:.1f}%", "Institutional proxy estimate")

    top_tabs = st.tabs([
        "📊 Executive View",
        "🔬 Research View",
        "📐 Risk Metrics",
        "🎭 Scenario Analysis",
        "📉 Recession Analysis",
        "⚡ Technical Analysis",
        "💾 Export & Deployment",
    ])

    with top_tabs[0]:
        st.plotly_chart(chart_current_curve(maturities, latest_curve, ns_result, nss_result), use_container_width=True)
        left, right = st.columns([1, 1])
        with left:
            fig2 = chart_yield(yield_df, "2Y", COLORS["warning"], "2-Year Treasury Yield - Policy Expectations")
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
        with right:
            fig10 = chart_yield(yield_df, "10Y", COLORS["accent"], "10-Year Treasury Yield - Growth & Inflation Expectations")
            if fig10:
                st.plotly_chart(fig10, use_container_width=True)
        st.plotly_chart(chart_spreads(spreads, recessions), use_container_width=True)
        
        # Executive summary
        st.markdown(
            f"""
            <div class="note-box">
            <b>📋 Executive Summary</b><br><br>
            Current regime: <b>{regime}</b><br>
            Recession probability estimate: <b>{100 * recession_prob:.1f}%</b><br>
            Curve slope (10Y-2Y): <b>{current_spread:.1f} bps</b><br><br>
            <b>Key observations:</b> {regime_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_tabs[1]:
        research_tabs = st.tabs(["📋 Data Table", "📐 NS Model", "📏 NSS Model", "⚙️ Model Governance", "📊 Dynamic Analysis", "🔬 Factor & PCA"])
        
        with research_tabs[0]:
            display_df = yield_df.iloc[::-1].reset_index()
            display_df.columns = ["Date"] + list(yield_df.columns)
            for col in display_df.columns[1:]:
                display_df[col] = display_df[col].map(lambda x: f"{x:.3f}%")
            display_df["Date"] = pd.to_datetime(display_df["Date"]).dt.strftime("%Y-%m-%d")
            st.dataframe(display_df, use_container_width=True, height=450)
        
        with research_tabs[1]:
            if ns_result:
                st.dataframe(pd.DataFrame({
                    "Parameter": ["β₀ (Level)", "β₁ (Slope)", "β₂ (Curvature)", "λ (Decay Rate)", "RMSE (bps)", "MAE (bps)", "R²"],
                    "Value": [
                        f"{ns_result['params'][0]:.4f}", f"{ns_result['params'][1]:.4f}", f"{ns_result['params'][2]:.4f}",
                        f"{ns_result['params'][3]:.4f}", f"{ns_result['rmse'] * 100:.2f}", f"{ns_result['mae'] * 100:.2f}",
                        f"{ns_result['r_squared']:.4f}",
                    ]
                }), use_container_width=True, hide_index=True)
                
                st.markdown(
                    """
                    <div class="note-box">
                    <b>📐 Nelson-Siegel (NS) Model Parameter Interpretation</b><br><br>
                    <b>β₀ (Level Factor):</b> Represents the long-term equilibrium interest rate. Higher values indicate expectations of higher future rates. This factor typically explains 80-90% of yield curve variation.<br><br>
                    <b>β₁ (Slope Factor):</b> Captures the difference between short and long-term rates. Negative values indicate an inverted curve (recession signal). Positive values suggest normal upward-sloping curve.<br><br>
                    <b>β₂ (Curvature Factor):</b> Measures the hump or dip in the medium-term section of the curve. Positive values indicate a hump (higher medium-term yields), negative values suggest a valley.<br><br>
                    <b>λ (Decay Rate):</b> Determines where the slope and curvature factors have maximum impact. Typical values range from 0.5-3.0. Lower λ means slower decay (more influence at longer maturities).<br><br>
                    <b>RMSE & MAE:</b> Model fit quality metrics. Lower values (<5bps) indicate excellent fit suitable for pricing applications.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        
        with research_tabs[2]:
            if nss_result:
                st.dataframe(pd.DataFrame({
                    "Parameter": ["β₀", "β₁", "β₂", "β₃", "λ₁", "λ₂", "RMSE (bps)", "MAE (bps)", "R²"],
                    "Value": [
                        f"{nss_result['params'][0]:.4f}", f"{nss_result['params'][1]:.4f}", f"{nss_result['params'][2]:.4f}",
                        f"{nss_result['params'][3]:.4f}", f"{nss_result['params'][4]:.4f}", f"{nss_result['params'][5]:.4f}",
                        f"{nss_result['rmse'] * 100:.2f}", f"{nss_result['mae'] * 100:.2f}", f"{nss_result['r_squared']:.4f}",
                    ]
                }), use_container_width=True, hide_index=True)
                
                st.markdown(
                    """
                    <div class="note-box">
                    <b>📏 Nelson-Siegel-Svensson (NSS) Model Parameter Interpretation</b><br><br>
                    The NSS model extends NS by adding a second curvature factor (β₃, λ₂), providing additional flexibility for complex curve shapes.<br><br>
                    <b>β₀ (Level Factor):</b> Same as NS - long-term equilibrium rate.<br>
                    <b>β₁ (Slope Factor):</b> Same as NS - short-term vs long-term differential.<br>
                    <b>β₂ (First Curvature):</b> Captures medium-term hump with decay rate λ₁.<br>
                    <b>β₃ (Second Curvature):</b> Captures additional shape features (e.g., butterfly patterns) with decay rate λ₂.<br>
                    <b>λ₁, λ₂ (Decay Rates):</b> Two decay parameters allow the model to fit more complex curve shapes, including those with multiple inflection points.<br><br>
                    <b>When to use NSS vs NS:</b> NSS is preferred for curves with pronounced humps or when high precision is required. NS is more parsimonious and stable for most market conditions.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        
        with research_tabs[3]:
            st.dataframe(governance_df.round(4), use_container_width=True, hide_index=True)
            fig_resid = chart_model_residuals(selected_cols, ns_result, nss_result)
            if fig_resid:
                st.plotly_chart(fig_resid, use_container_width=True)
            st.markdown(
                """
                <div class="note-box">
                <b>⚙️ Model Governance Framework</b><br><br>
                <b>Fit Confidence Levels:</b>
                <ul>
                <li><b>HIGH:</b> RMSE &lt; 4bps AND R² &gt; 0.99 → Excellent fit suitable for pricing and risk management</li>
                <li><b>GOOD:</b> RMSE &lt; 7bps AND R² &gt; 0.98 → Suitable for most analytical applications</li>
                <li><b>MODERATE:</b> RMSE &lt; 12bps AND R² &gt; 0.95 → Use caution, not recommended for precise pricing</li>
                <li><b>LOW:</b> Otherwise → Model inadequacy, consider alternative approaches</li>
                </ul>
                <br>
                <b>Warning Flags Interpretation:</b>
                <ul>
                <li><b>Large residual outlier:</b> Individual maturity with &gt;15bps error → Review market data for that tenor</li>
                <li><b>Low R-squared:</b> Model explains &lt;95% of variation → Curve has unusual shape requiring more flexible model</li>
                <li><b>High RMSE:</b> &gt;10bps average error → Model not recommended for analytical use</li>
                </ul>
                <br>
                The residual chart above shows pricing errors by maturity. Green band (±5bps) represents acceptable range for institutional applications.
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with research_tabs[4]:
            fig_dyn = chart_dynamic(dynamic_params)
            if fig_dyn:
                st.plotly_chart(fig_dyn, use_container_width=True)
            st.markdown(
                """
                <div class="note-box">
                <b>📊 Dynamic Parameter Analysis</b><br><br>
                <b>β₀ (Level):</b> Trends over time reflect changing long-term interest rate expectations. Rising β₀ indicates increasing neutral rate expectations.<br>
                <b>β₁ (Slope):</b> Becomes negative during inversion periods, typically preceding recessions by 6-18 months.<br>
                <b>β₂ (Curvature):</b> Increases when medium-term rates are elevated relative to short and long ends (humped curve).<br>
                <b>RMSE Evolution:</b> Tracking model fit over time helps identify structural breaks or regime changes.
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with research_tabs[5]:
            l, r = st.columns(2)
            with l:
                fig_factor = chart_factors(factor_df)
                if fig_factor:
                    st.plotly_chart(fig_factor, use_container_width=True)
                
                st.markdown(
                    """
                    <div class="note-box">
                    <b>🔬 Factor Analysis Interpretation</b><br><br>
                    <b>Level Factor:</b> Represents parallel shifts in the yield curve (all maturities move together). Driven by inflation expectations and monetary policy stance.<br>
                    <b>Slope Factor:</b> Represents curve steepness (short vs long rates). Negative slope signals recession risk.<br>
                    <b>Curvature Factor:</b> Represents medium-term premiums. High curvature suggests uncertainty about medium-term policy path.<br>
                    <b>Butterfly Factor:</b> Captures relative movement of intermediate vs short/long maturities. Important for relative value trading.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            with r:
                fig_pca = chart_pca_variance(pca_risk)
                if fig_pca:
                    st.plotly_chart(fig_pca, use_container_width=True)
                if pca_risk:
                    st.dataframe(pca_risk["loadings"].round(4), use_container_width=True)
                    
                    st.markdown(
                        f"""
                        <div class="note-box">
                        <b>📈 Principal Component Analysis (PCA) Explanation</b><br><br>
                        <b>Purpose:</b> PCA decomposes yield curve movements into uncorrelated risk factors, identifying the main drivers of curve dynamics.<br><br>
                        <b>PC1 (Level Component):</b> Explains {pca_risk['explained_variance'][0]*100:.1f}% of variation. Represents parallel shifts. All tenors load positively with similar magnitude.<br>
                        <b>PC2 (Slope Component):</b> Explains {pca_risk['explained_variance'][1]*100:.1f}% of variation. Short vs long-term divergence. Loadings positive at long end, negative at short end.<br>
                        <b>PC3 (Curvature Component):</b> Explains {pca_risk['explained_variance'][2]*100:.1f}% of variation. Medium-term hump. Loadings positive for medium maturities, negative for extremes.<br><br>
                        <b>Institutional Application:</b> The first {pca_risk['n_components']} components explain {pca_risk['cumulative_variance'][-1]*100:.1f}% of curve variance, suggesting that a {pca_risk['n_components']}-factor model is sufficient for risk management.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    with top_tabs[2]:
        if "10Y" in yield_df.columns:
            risk = calculate_var_metrics(yield_df["10Y"].pct_change(), confidence_level, 10)
            if risk:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("📊 Historical VaR (10-day)", f"{risk['VaR_Historical']:.2%}", help="Historical simulation using empirical return distribution")
                col2.metric("📈 Parametric VaR (10-day)", f"{risk['VaR_Parametric']:.2%}", help="Normal distribution assumption")
                col3.metric("🎯 Cornish-Fisher VaR (10-day)", f"{risk['VaR_CornishFisher']:.2%}", help="Adjusts for skewness and kurtosis")
                col4.metric("⚠️ CVaR (10-day)", f"{risk['CVaR']:.2%}", help="Expected shortfall beyond VaR")
                
                col5, col6, col7, col8 = st.columns(4)
                col5.metric("📐 Skewness", f"{risk['Skewness']:.3f}", help="Negative = left-tail risk")
                col6.metric("📊 Kurtosis", f"{risk['Kurtosis']:.3f}", help=">3 indicates fat tails")
                col7.metric("📉 Annualized Volatility", f"{risk['Volatility_Annualized']:.2%}", help="252-day annualization")
                col8.metric("🔬 Jarque-Bera p-value", f"{risk['Jarque_Bera_Pvalue']:.4f}", help="Normality test (p<0.05 rejects normality)")
                
                st.markdown(
                    """
                    <div class="note-box">
                    <b>📐 Quantitative Risk Metrics - Institutional Framework</b><br><br>
                    <b>VaR (Value at Risk):</b> Maximum expected loss over 10 days at given confidence level. Three methodologies provide robustness checks:<br>
                    • <b>Historical VaR:</b> Non-parametric, captures actual tail behavior but limited by sample period<br>
                    • <b>Parametric VaR:</b> Assumes normality, efficient but may underestimate tail risk<br>
                    • <b>Cornish-Fisher VaR:</b> Adjusts for skewness/kurtosis, more accurate for non-normal returns<br><br>
                    <b>CVaR (Conditional VaR):</b> Average loss beyond VaR threshold. More coherent risk measure for tail events.<br><br>
                    <b>Normality Assessment:</b> Jarque-Bera test p-value &lt; 0.05 indicates significant non-normality, favoring Cornish-Fisher VaR over parametric.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                # Additional risk analytics
                st.subheader("Risk Factor Correlation Matrix")
                returns_df = yield_df[selected_cols].pct_change().dropna()
                corr_matrix = returns_df.corr()
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.values.round(3),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                ))
                fig_corr.update_layout(title="Yield Changes Correlation Matrix", height=500, width=700)
                st.plotly_chart(fig_corr, use_container_width=True)

    with top_tabs[3]:
        scenario_name = st.selectbox("Select Scenario", list(scenarios.keys()))
        st.plotly_chart(chart_scenario(scenarios[scenario_name], scenario_name), use_container_width=True)
        
        scenario_notes = {
            "Bull Steepener (Growth expectations decline)": "Short and intermediate yields decline modestly, while the long end declines more. This scenario reflects falling inflation expectations, softer growth, or a rally led by duration. Recommended: Extend duration, prefer long-term bonds.",
            "Bear Flattener (Hawkish policy pivot)": "Front-end yields rise more than the long end. This reflects hawkish central-bank policy or repricing of the policy path. Recommended: Reduce duration, prefer floating rate or short-term instruments.",
            "Recession Scenario (Defensive positioning)": "Broad rally across the curve, with the strongest fall at the short end. This scenario emphasizes defensive allocation and recession hedging. Recommended: Increase government bond exposure, reduce credit risk.",
            "Policy Easing (Central bank cuts)": "Short-end yields fall sharply after expected policy cuts, while the long end reacts less. Recommended: Focus on curve steepeners, benefit from policy accommodation."
        }
        st.markdown(f'<div class="note-box"><b>🎭 Scenario Analysis Framework</b><br><br>{scenario_notes[scenario_name]}</div>', unsafe_allow_html=True)

        # Historical scenario comparison
        hist_table = scenarios[scenario_name].copy()
        hist_table["Change (bps)"] = (hist_table["Scenario"] - hist_table["Current"]) * 100
        hist_table.index.name = "Tenor"
        st.dataframe(hist_table.round(4), use_container_width=True)
        
        # Relative value impact
        st.markdown(
            """
            <div class="note-box">
            <b>💰 Relative Value Implications</b><br><br>
            The table above shows absolute yield levels and changes under the selected scenario. Positive changes indicate yields rising (prices falling), negative changes indicate yields falling (prices rising).<br><br>
            Use this analysis to assess portfolio convexity, duration positioning, and relative value trades across the curve.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_tabs[4]:
        rr_tabs = st.tabs(["📉 NBER Chart", "🎯 Hit Ratio", "⚠️ False Positives", "📊 Current vs History", "⏱️ Lead-Time Summary"])
        
        with rr_tabs[0]:
            fig = go.Figure()
            if "10Y-2Y" in spreads.columns:
                fig.add_trace(go.Scatter(x=spreads.index, y=spreads["10Y-2Y"], mode="lines", name="10Y-2Y Spread", line=dict(color=COLORS["negative"], width=2.5)))
            fig.add_hline(y=0, line_dash="dash", line_color=COLORS["text_secondary"], line_width=2)
            add_recession_bands(fig, recessions)
            st.plotly_chart(create_chart_layout(fig, "NBER Recession Overlay | 10Y-2Y Spread Historical Analysis", "Basis Points (bps)", 550), use_container_width=True)
            
            st.markdown(
                """
                <div class="note-box">
                <b>📉 Recession Signal Interpretation</b><br><br>
                Gray shaded areas represent NBER-dated recessions. The 10Y-2Y spread (red line) typically inverts (crosses below 0) 6-18 months before recession onset.<br><br>
                <b>Key observations:</b>
                <ul>
                <li>Not every inversion leads to recession (false positives)</li>
                <li>Lead time varies significantly across cycles</li>
                <li>Depth and duration of inversion matter for signal strength</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with rr_tabs[1]:
            hit_df = pd.DataFrame([hit_stats])
            st.dataframe(hit_df.round(4), use_container_width=True, hide_index=True)
            st.markdown('<div class="note-box"><b>🎯 Hit Ratio Analysis</b><br><br>Hit Ratio shows the proportion of inversion signals that were followed by NBER recessions. Higher ratios indicate more reliable signal. Historical average is approximately 0.65-0.75 (65-75% accuracy).</div>', unsafe_allow_html=True)
        
        with rr_tabs[2]:
            false_df = pd.DataFrame([{
                "Total Signals": hit_stats["Signals"],
                "Correct Signals (Matches)": hit_stats["Matches"],
                "False Positives": hit_stats["Signals"] - hit_stats["Matches"],
                "False Positive Rate": hit_stats["FalsePositiveRate"],
            }])
            st.dataframe(false_df.round(4), use_container_width=True, hide_index=True)
            st.markdown('<div class="note-box"><b>⚠️ False Positive Analysis</b><br><br>False positives occur when curve inverts but recession does not follow. Common causes: technical factors, Fed policy credibility, structural changes. Monitor both false positive rate AND lead time consistency.</div>', unsafe_allow_html=True)
        
        with rr_tabs[3]:
            st.dataframe(current_inversion_vs_history(spreads, inversions).round(4), use_container_width=True, hide_index=True)
            st.markdown('<div class="note-box"><b>📊 Historical Comparison</b><br><br>Compare current inversion characteristics (depth, duration) with historical episodes to assess potential severity.</div>', unsafe_allow_html=True)
        
        with rr_tabs[4]:
            if lead_times:
                st.dataframe(pd.DataFrame(lead_times).round(4), use_container_width=True, hide_index=True)
                avg_lead = np.mean([lt["lead_months"] for lt in lead_times])
                st.markdown(f'<div class="note-box"><b>⏱️ Lead-Time Analysis</b><br><br>Average lead time from inversion start to recession onset: <b>{avg_lead:.1f} months</b>.<br><br>Lead times vary significantly (range: 5-24 months). Use this distribution to assess timing uncertainty.</div>', unsafe_allow_html=True)
            else:
                st.info("No lead-time episodes identified in current sample alignment.")

    with top_tabs[5]:
        ticker = st.selectbox("Select OHLC Instrument", list(YAHOO_TICKERS.keys()), format_func=lambda x: f"{x} | {YAHOO_TICKERS[x]}")
        ohlc_df = ohlc_data.get(ticker)
        
        if ohlc_df is not None and not ohlc_df.empty:
            # Generate technical signals
            signals_df = generate_technical_signals(ohlc_df)
            
            # Current signals summary
            st.subheader("Current Technical Signals")
            sig_cols = st.columns(5)
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
                stoch_val = ohlc_df["Stoch_K"].iloc[-1] if "Stoch_K" in ohlc_df.columns else 50
                stoch_signal = "🟢 Oversold" if stoch_val < 20 else ("🔴 Overbought" if stoch_val > 80 else "⚪ Neutral")
                st.metric("Stochastic", f"{stoch_val:.1f}", stoch_signal)
            with sig_cols[3]:
                bb_position = "🟢 Below Lower" if ohlc_df["Close"].iloc[-1] < ohlc_df["BB_Lower"].iloc[-1] else ("🔴 Above Upper" if ohlc_df["Close"].iloc[-1] > ohlc_df["BB_Upper"].iloc[-1] else "⚪ Within Bands")
                st.metric("Bollinger Bands", bb_position, "Mean reversion signal")
            with sig_cols[4]:
                vol_ratio = ohlc_df["Volume_Ratio"].iloc[-1] if "Volume_Ratio" in ohlc_df.columns else 1.0
                vol_signal = "📈 High Volume" if vol_ratio > 1.5 else ("📉 Low Volume" if vol_ratio < 0.7 else "📊 Normal")
                st.metric("Volume Ratio", f"{vol_ratio:.2f}x", vol_signal)
            
            # Main technical charts
            left, right = st.columns([1.15, 1])
            with left:
                fig_ohlc = chart_ohlc(ohlc_df, ticker)
                if fig_ohlc:
                    st.plotly_chart(fig_ohlc, use_container_width=True)
            with right:
                fig_ta = chart_technical_panels(ohlc_df, ticker, signals_df)
                if fig_ta:
                    st.plotly_chart(fig_ta, use_container_width=True)
            
            # Recent signal history
            st.subheader("Recent Technical Events (Last 30 days)")
            recent_signals = signals_df.iloc[-30:].copy()
            signal_columns = [col for col in recent_signals.columns if recent_signals[col].any()]
            if signal_columns:
                signal_summary = recent_signals[signal_columns].replace({True: "✓", False: ""})
                signal_summary.index = recent_signals.index.strftime("%Y-%m-%d")
                st.dataframe(signal_summary, use_container_width=True, height=300)
            
            st.markdown(
                """
                <div class="note-box">
                <b>⚡ Advanced Technical Analysis Framework</b><br><br>
                <b>Signal Generation Logic:</b>
                <ul>
                <li><b>RSI:</b> &lt;30 = Oversold (potential bounce), &gt;70 = Overbought (potential pullback)</li>
                <li><b>MACD:</b> Bullish cross = MACD line crosses above signal line, Bearish cross = below signal line</li>
                <li><b>Golden Cross:</b> SMA 20 crosses above SMA 50 → Long-term bullish signal</li>
                <li><b>Death Cross:</b> SMA 20 crosses below SMA 50 → Long-term bearish signal</li>
                <li><b>Bollinger Bands:</b> Price below lower band suggests oversold conditions, above upper band suggests overbought</li>
                <li><b>Stochastic:</b> %K crosses above %D in oversold region (&lt;20) = bullish, cross in overbought (&gt;80) = bearish</li>
                </ul>
                <br>
                <b>Institutional Application:</b> Technical indicators should complement fundamental and macro analysis. Use multiple confirmations before acting on signals.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning(f"No data available for {ticker}")

    with top_tabs[6]:
        export_tabs = st.tabs(["📥 Download Data", "📦 Requirements", "🚀 Run Instructions", "☁️ Streamlit Cloud"])
        
        with export_tabs[0]:
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button("📊 Yield Data (CSV)", yield_df.to_csv().encode("utf-8"), f"yield_data_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
                st.download_button("📈 Spread Data (CSV)", spreads.to_csv().encode("utf-8"), f"spreads_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
                if dynamic_params is not None and not dynamic_params.empty:
                    st.download_button("📉 Dynamic Parameters", dynamic_params.to_csv(index=False).encode("utf-8"), f"dynamic_params_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
            with col_dl2:
                if pca_risk:
                    st.download_button("📐 PCA Loadings", pca_risk["loadings"].to_csv().encode("utf-8"), f"pca_loadings_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
                if lead_times:
                    st.download_button("⏱️ Lead Times", pd.DataFrame(lead_times).to_csv(index=False).encode("utf-8"), f"lead_times_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
                if arb and not arb["mispriced_table"].empty:
                    st.download_button("💰 Relative Value Opportunities", arb["mispriced_table"].to_csv(index=False).encode("utf-8"), f"relative_value_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
        
        with export_tabs[1]:
            st.code(
                "streamlit>=1.28.0\npandas>=2.0.0\nnumpy>=1.24.0\nplotly>=5.17.0\nrequests>=2.31.0\nyfinance>=0.2.28\nscipy>=1.10.0\nscikit-learn>=1.3.0",
                language="text",
            )
        
        with export_tabs[2]:
            st.code(
                "# Installation\npip install -r requirements.txt\n\n# Run application\nstreamlit run yield_curve_v35_5_full_fold_in_final.py\n\n# Alternative with specific port\nstreamlit run yield_curve_v35_5_full_fold_in_final.py --server.port 8501",
                language="bash",
            )
        
        with export_tabs[3]:
            st.markdown(
                """
                <div class="note-box">
                <b>☁️ Streamlit Cloud Deployment Guide</b><br><br>
                <b>Step 1:</b> Create a GitHub repository with your app file and requirements.txt<br>
                <b>Step 2:</b> Sign in to <b>share.streamlit.io</b> with GitHub<br>
                <b>Step 3:</b> Deploy from repository, set main file path<br>
                <b>Step 4:</b> Launch app and enter FRED API key in UI (never hardcode)<br><br>
                <b>Performance Optimization:</b>
                <ul>
                <li>Reduce rolling window length (2-3 years) for faster computation</li>
                <li>Decrease forecast horizon (10-20 days)</li>
                <li>Use Streamlit's caching (already implemented)</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center; color:{COLORS['muted']}; font-size:0.75rem; font-family: monospace;'>Institutional Quantitative Platform | MK Istanbul Fintech LabGEN © 2026 | All Rights Reserved</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
