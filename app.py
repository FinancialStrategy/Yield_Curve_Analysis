
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
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# V35.5 FULL FOLD-IN FINAL VERSION
# Institutional architecture refactor
# - Light professional design
# - Fixed tab overlap
# - Executive / Research / Risk / Scenario / Recession / Technical / Export
# - NS / NSS / governance / PCA / VaR / NBER / OHLC / deployment notes
# =============================================================================

st.set_page_config(
    page_title="Yield Curve Institutional Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "bg": "#eef2f7",
    "bg2": "#f7f9fc",
    "surface": "#ffffff",
    "surface_alt": "#f5f7fb",
    "header": "#253246",
    "grid": "#d6dee8",
    "text": "#1f2937",
    "muted": "#667085",
    "accent": "#355c7d",
    "accent2": "#6c8ead",
    "positive": "#2f855a",
    "negative": "#c05656",
    "warning": "#b7791f",
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
        background: linear-gradient(180deg, {COLORS['bg']} 0%, {COLORS['bg2']} 100%);
    }}
    .main-title-card {{
        background: linear-gradient(90deg, {COLORS['header']} 0%, {COLORS['accent']} 100%);
        border-radius: 16px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 22px rgba(0,0,0,0.08);
    }}
    .main-title {{
        color: white;
        font-weight: 700;
        font-size: 1.35rem;
        margin: 0;
    }}
    .metric-card {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['grid']};
        border-radius: 12px;
        padding: 0.95rem;
        min-height: 108px;
        box-shadow: 0 4px 14px rgba(16,24,40,0.04);
    }}
    .metric-label {{
        color: {COLORS['muted']};
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    .metric-value {{
        color: {COLORS['text']};
        font-size: 1.42rem;
        font-weight: 800;
        margin-top: 0.4rem;
        font-family: 'Courier New', monospace;
    }}
    .metric-sub {{
        color: {COLORS['muted']};
        font-size: 0.73rem;
        margin-top: 0.3rem;
        line-height: 1.25;
    }}
    .note-box {{
        background: {COLORS['surface_alt']};
        border: 1px solid {COLORS['grid']};
        border-left: 4px solid {COLORS['accent']};
        border-radius: 10px;
        padding: 0.8rem 0.9rem;
        color: {COLORS['text']};
        font-size: 0.86rem;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        border-bottom: 1px solid {COLORS['grid']};
        flex-wrap: wrap !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {COLORS['muted']};
        font-weight: 700;
        font-size: 0.70rem;
        text-transform: uppercase;
        white-space: nowrap;
        padding: 6px 10px;
    }}
    .stTabs [aria-selected="true"] {{
        color: {COLORS['accent']};
        border-bottom: 2px solid {COLORS['accent']};
    }}
    .stButton>button, .stDownloadButton>button {{
        background: {COLORS['surface']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['grid']};
        border-radius: 8px;
        font-weight: 700;
    }}
    .stButton>button:hover, .stDownloadButton>button:hover {{
        border-color: {COLORS['accent']};
        color: {COLORS['accent']};
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

def _safe_series_from_yf(obj, name=None):
    try:
        if obj is None:
            return None

        if isinstance(obj, pd.Series):
            s = obj.copy()
            s = pd.to_numeric(s, errors="coerce").dropna()
            if s.empty:
                return None
            if name is not None:
                s.name = name
            return s

        if isinstance(obj, pd.DataFrame):
            if obj.empty:
                return None

            candidate_cols = [c for c in ["Adj Close", "Close"] if c in obj.columns]
            if candidate_cols:
                s = obj[candidate_cols[0]].copy()
            else:
                numeric_cols = obj.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    return None
                s = obj[numeric_cols[0]].copy()

            s = pd.to_numeric(s, errors="coerce").dropna()
            if s.empty:
                return None
            if name is not None:
                s.name = name
            return s

        if np.isscalar(obj):
            return None

        return None
    except Exception:
        return None


def _safe_frame_from_dict_of_series(series_dict):
    clean = {}
    for key, value in series_dict.items():
        s = _safe_series_from_yf(value, name=key)
        if s is not None and len(s) > 1:
            clean[key] = s

    if not clean:
        return pd.DataFrame()

    df = pd.concat(clean.values(), axis=1, join="outer")
    df.columns = list(clean.keys())
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_bundle(start_date=None, end_date=None):
    vol_data = {}
    corr_data = {}

    volatility_map = {
        "^VIX": "VIX",
        "^MOVE": "MOVE"
    }

    correlation_map = {
        "^GSPC": "SP500",
        "QQQ": "NASDAQ100",
        "GLD": "Gold",
        "UUP": "USD"
    }

    for ticker, label in volatility_map.items():
        try:
            raw = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            s = _safe_series_from_yf(raw, name=label)
            if s is not None:
                vol_data[label] = s
        except Exception:
            continue

    for ticker, label in correlation_map.items():
        try:
            raw = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            s = _safe_series_from_yf(raw, name=label)
            if s is not None:
                corr_data[label] = s
        except Exception:
            continue

    volatility_df = _safe_frame_from_dict_of_series(vol_data)
    correlation_df = _safe_frame_from_dict_of_series(corr_data)

    if not volatility_df.empty:
        volatility_df = volatility_df.sort_index()
    if not correlation_df.empty:
        correlation_df = correlation_df.sort_index()

    return volatility_df, correlation_df


# =============================================================================
# TECHNICALS
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
        return "Risk-off", "Curve inversion implies a defensive macro regime."
    if np.isfinite(spread) and spread < 50:
        return "Neutral / Late Cycle", "Curve flattening suggests late-cycle caution."
    if np.isfinite(y10) and y10 > 5.5:
        return "Neutral / Restrictive", "Long-end rates remain restrictive."
    return "Risk-on", "Positive slope supports pro-risk macro positioning."

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
    for name, res in [("NS", ns_result), ("NSS", nss_result)]:
        if res is None:
            continue
        rmse = res["rmse"]
        r2 = res["r_squared"]
        max_abs = float(np.max(np.abs(res["residuals"])))
        if rmse < 0.05 and r2 > 0.98:
            confidence = "High"
        elif rmse < 0.10 and r2 > 0.95:
            confidence = "Moderate"
        else:
            confidence = "Low"
        flags = []
        if max_abs > 0.15:
            flags.append("Residual outlier")
        if r2 < 0.95:
            flags.append("Low fit")
        rows.append({
            "Model": name,
            "RMSE": rmse,
            "MAE": res["mae"],
            "R2": r2,
            "MaxAbsResidual": max_abs,
            "FitConfidence": confidence,
            "WarningFlags": ", ".join(flags) if flags else "None",
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
    }

def calculate_var_metrics(returns: pd.Series, confidence: float = 0.95, horizon: int = 10) -> Optional[dict]:
    returns = returns.dropna()
    if len(returns) < 20:
        return None
    var_hist = np.percentile(returns, (1 - confidence) * 100)
    cvar = returns[returns <= var_hist].mean()
    var_param = norm.ppf(1 - confidence) * returns.std()
    skew = returns.skew()
    kurt = returns.kurtosis()
    z = norm.ppf(1 - confidence)
    z_cf = z + (z**2 - 1) * skew / 6 + (z**3 - 3 * z) * kurt / 24 - (2 * z**3 - 5 * z) * skew**2 / 36
    return {
        "VaR_Historical": float(var_hist * np.sqrt(horizon)),
        "VaR_Parametric": float(var_param * np.sqrt(horizon)),
        "VaR_CornishFisher": float(z_cf * returns.std() * np.sqrt(horizon)),
        "CVaR": float(cvar * np.sqrt(horizon)),
        "Skewness": float(skew),
        "Kurtosis": float(kurt),
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
    scenarios["Bull Steepener"] = pd.DataFrame({"Current": latest, "Scenario": bull})

    # Bear flattener: front-end up more than long-end
    bear = latest.copy()
    for col in tenor_order:
        m = MATURITY_MAP.get(col, 1)
        bear[col] = bear[col] + (0.14 if m <= 2 else 0.07)
    scenarios["Bear Flattener"] = pd.DataFrame({"Current": latest, "Scenario": bear})

    # Recession case: broad rally, short end falls sharply
    recession = latest.copy()
    for col in tenor_order:
        m = MATURITY_MAP.get(col, 1)
        recession[col] = recession[col] - (0.22 if m <= 2 else 0.14 if m <= 10 else 0.10)
    scenarios["Recession Case"] = pd.DataFrame({"Current": latest, "Scenario": recession})

    # Policy easing: front-end down, moderate long-end response
    easing = latest.copy()
    for col in tenor_order:
        m = MATURITY_MAP.get(col, 1)
        easing[col] = easing[col] - (0.25 if m <= 2 else 0.12 if m <= 10 else 0.06)
    scenarios["Policy Easing"] = pd.DataFrame({"Current": latest, "Scenario": easing})

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
    rows = [{"Metric": "Current 10Y-2Y", "Value": current}]
    if inversions:
        rows.append({"Metric": "Historical Min Inversion", "Value": min(x["depth"] for x in inversions)})
        rows.append({"Metric": "Historical Avg Inversion Depth", "Value": np.mean([x["depth"] for x in inversions])})
        rows.append({"Metric": "Historical Avg Duration Days", "Value": np.mean([x["duration_days"] for x in inversions])})
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
                "Tenor": m,
                "Actual": latest[i],
                "Theoretical": theoretical[i],
                "Difference": r,
                "Signal": "Undervalued" if r > 0 else "Overvalued",
            })
    return {
        "mean_abs_error": float(np.mean(np.abs(residuals))),
        "max_abs_error": float(np.max(np.abs(residuals))),
        "std_error": float(np.std(residuals)),
        "mispriced_count": len(rows),
        "mispriced_table": pd.DataFrame(rows) if rows else pd.DataFrame(),
    }

# =============================================================================
# VISUALS
# =============================================================================

def add_recession_bands(fig: go.Figure, recessions: List[dict]) -> go.Figure:
    for rec in recessions:
        fig.add_vrect(x0=rec["start"], x1=rec["end"], fillcolor=COLORS["recession"], opacity=0.35, layer="below", line_width=0)
    return fig

def create_chart_layout(fig: go.Figure, title: str, y_title: Optional[str] = None, height: int = 460) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(size=11, color="#1f2937"),
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=15, color="#1f2937")),
        margin=dict(l=60, r=28, t=72, b=52),
        hovermode="x unified",
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#334155")),
    )
    fig.update_xaxes(
        gridcolor=COLORS["grid"],
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="#9aa7b8",
        tickfont=dict(color="#334155", size=11),
        title_font=dict(color="#1f2937", size=12),
    )
    fig.update_yaxes(
        gridcolor=COLORS["grid"],
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="#9aa7b8",
        tickfont=dict(color="#334155", size=11),
        title_font=dict(color="#1f2937", size=12),
    )
    if y_title:
        fig.update_yaxes(title_text=y_title)
    return fig

def chart_current_curve(maturities: np.ndarray, yields_: np.ndarray, ns_result: Optional[dict], nss_result: Optional[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=maturities, y=yields_, mode="markers+lines", name="Actual",
        marker=dict(size=10, color=COLORS["accent"]),
        line=dict(color=COLORS["accent2"], width=2),
    ))
    if ns_result:
        fig.add_trace(go.Scatter(x=maturities, y=ns_result["fitted_values"], mode="lines", name="NS", line=dict(color=COLORS["positive"], width=2.2)))
    if nss_result:
        fig.add_trace(go.Scatter(x=maturities, y=nss_result["fitted_values"], mode="lines", name="NSS", line=dict(color=COLORS["warning"], width=2.2, dash="dash")))
    return create_chart_layout(fig, "Current Treasury Curve and Model Fits", "Yield (%)", 440)

def chart_yield(yield_df: pd.DataFrame, tenor: str, color: str, title: str) -> Optional[go.Figure]:
    if tenor not in yield_df.columns:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yield_df.index, y=yield_df[tenor], mode="lines", name=tenor, line=dict(color=color, width=2.2)))
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))
    return create_chart_layout(fig, title, "Yield (%)", 420)

def chart_spreads(spreads: pd.DataFrame, recessions: List[dict]) -> go.Figure:
    fig = go.Figure()
    palette = [COLORS["negative"], COLORS["accent"], COLORS["warning"], COLORS["positive"]]
    for i, col in enumerate(spreads.columns):
        fig.add_trace(go.Scatter(x=spreads.index, y=spreads[col], mode="lines", name=col, line=dict(color=palette[i % len(palette)], width=2)))
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"])
    add_recession_bands(fig, recessions)
    return create_chart_layout(fig, "Treasury Spread Dashboard", "bps", 520)

def chart_model_residuals(selected_cols: List[str], ns_result: Optional[dict], nss_result: Optional[dict]) -> Optional[go.Figure]:
    if ns_result is None and nss_result is None:
        return None
    fig = go.Figure()
    if ns_result:
        fig.add_trace(go.Bar(x=selected_cols, y=ns_result["residuals"], name="NS Residuals", marker_color=COLORS["positive"], opacity=0.65))
    if nss_result:
        fig.add_trace(go.Bar(x=selected_cols, y=nss_result["residuals"], name="NSS Residuals", marker_color=COLORS["warning"], opacity=0.65))
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"])
    warning_threshold = 0.15
    base = nss_result if nss_result is not None else ns_result
    if base is not None:
        for x_label, y_val in zip(selected_cols, base["residuals"]):
            if abs(y_val) >= warning_threshold:
                fig.add_annotation(
                    x=x_label,
                    y=y_val,
                    text="⚠",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-22 if y_val >= 0 else 22,
                    font=dict(color="#c05656", size=16),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#c05656",
                    borderwidth=1,
                )
    return create_chart_layout(fig, "Residual Quality and Warning Flags", "Residual", 420)

def chart_dynamic(dynamic_params: pd.DataFrame) -> Optional[go.Figure]:
    if dynamic_params is None or dynamic_params.empty:
        return None
    fig = make_subplots(rows=2, cols=2, subplot_titles=("β0 Level", "β1 Slope", "β2 Curvature", "RMSE"))
    fig.add_trace(go.Scatter(x=dynamic_params["date"], y=dynamic_params["beta0"], mode="lines", name="β0", line=dict(color=COLORS["positive"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=dynamic_params["date"], y=dynamic_params["beta1"], mode="lines", name="β1", line=dict(color=COLORS["accent"])), row=1, col=2)
    fig.add_trace(go.Scatter(x=dynamic_params["date"], y=dynamic_params["beta2"], mode="lines", name="β2", line=dict(color=COLORS["warning"])), row=2, col=1)
    fig.add_trace(go.Scatter(x=dynamic_params["date"], y=dynamic_params["rmse"], mode="lines", name="RMSE", line=dict(color=COLORS["muted"])), row=2, col=2)
    return create_chart_layout(fig, "Rolling Nelson-Siegel Parameters", height=620)

def chart_factors(factor_df: pd.DataFrame) -> Optional[go.Figure]:
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
    return create_chart_layout(fig, "PCA Variance Explained", "Percent", 430)

def chart_scenario(scenario_df: pd.DataFrame, scenario_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=scenario_df.index, y=scenario_df["Current"], name="Current", marker_color=COLORS["accent2"]))
    fig.add_trace(go.Bar(x=scenario_df.index, y=scenario_df["Scenario"], name=scenario_name, marker_color=COLORS["warning"]))
    fig.update_layout(barmode="group")
    return create_chart_layout(fig, f"Scenario Analysis | {scenario_name}", "Yield (%)", 460)

def chart_ohlc(ohlc_df: pd.DataFrame, ticker: str) -> Optional[go.Figure]:
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
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))
    return create_chart_layout(fig, f"OHLC | {ticker}", "Price", 520)

def chart_technical_panels(ohlc_df: pd.DataFrame, ticker: str) -> Optional[go.Figure]:
    if ohlc_df is None or ohlc_df.empty:
        return None
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f"{ticker} Close", "RSI (14)", "MACD"),
    )
    fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["Close"], mode="lines", name="Close", line=dict(color=COLORS["accent2"], width=2)), row=1, col=1)
    if "BB_Upper" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["BB_Upper"], mode="lines", name="BB Upper", line=dict(color=COLORS["muted"], width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["BB_Lower"], mode="lines", name="BB Lower", line=dict(color=COLORS["muted"], width=1), fill='tonexty', fillcolor=COLORS["band"]), row=1, col=1)
    if "RSI" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["RSI"], mode="lines", name="RSI", line=dict(color=COLORS["accent"], width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=COLORS["negative"], row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=COLORS["positive"], row=2, col=1)
    if "MACD" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["MACD"], mode="lines", name="MACD", line=dict(color=COLORS["positive"], width=1.4)), row=3, col=1)
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["MACD_Signal"], mode="lines", name="Signal", line=dict(color=COLORS["negative"], width=1.4)), row=3, col=1)
    return create_chart_layout(fig, f"Technical Analysis | {ticker}", height=760)

def chart_rate_dynamics(yield_df: pd.DataFrame, spreads: pd.DataFrame) -> Optional[go.Figure]:
    if not {"2Y", "10Y"}.issubset(yield_df.columns):
        return None
    df = pd.DataFrame(index=yield_df.index)
    df["2Y_20D_Change"] = yield_df["2Y"].diff(20)
    df["10Y_20D_Change"] = yield_df["10Y"].diff(20)
    if "10Y-2Y" in spreads.columns:
        df["SlopeMomentum_20D"] = spreads["10Y-2Y"].diff(20)
    df["10Y_RealizedVol_60D"] = yield_df["10Y"].diff().rolling(60).std()
    df = df.dropna()
    if df.empty:
        return None
    fig = make_subplots(rows=2, cols=2, subplot_titles=("2Y 20D Change", "10Y 20D Change", "Slope Momentum (10Y-2Y)", "10Y Realized Vol 60D"))
    fig.add_trace(go.Scatter(x=df.index, y=df["2Y_20D_Change"], mode="lines", name="2Y 20D Change", line=dict(color=COLORS["warning"], width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["10Y_20D_Change"], mode="lines", name="10Y 20D Change", line=dict(color=COLORS["accent"], width=1.8)), row=1, col=2)
    if "SlopeMomentum_20D" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SlopeMomentum_20D"], mode="lines", name="Slope Momentum", line=dict(color=COLORS["negative"], width=1.8)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["10Y_RealizedVol_60D"], mode="lines", name="10Y Realized Vol 60D", line=dict(color=COLORS["positive"], width=1.8)), row=2, col=2)
    return create_chart_layout(fig, "Quantitative Rate-Direction Diagnostics", height=640)

# =============================================================================
# UI HELPERS
# =============================================================================

def render_api_gate() -> None:
    st.markdown(
        f"""
        <div class="note-box" style="max-width:560px; margin:40px auto; text-align:center;">
            <b>FRED API Key Required</b><br><br>
            This institutional dashboard uses live U.S. Treasury data from FRED.<br>
            Get a free API key from the official FRED documentation page.
        </div>
        """,
        unsafe_allow_html=True,
    )
    api_key = st.text_input("Enter your FRED API key", type="password", placeholder="Paste API key here")
    if st.button("Validate & Connect", use_container_width=True):
        if not api_key:
            st.error("Please enter a valid API key.")
            st.stop()
        with st.spinner("Validating API key..."):
            valid = validate_fred_api_key(api_key)
        if valid:
            st.session_state.api_key = api_key
            st.session_state.api_key_validated = True
            st.success("API key validated.")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("Invalid API key.")
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
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.api_key_validated:
        render_api_gate()

    with st.sidebar:
        st.markdown("### Control Tower")
        st.caption("Institutional fixed-income macro monitoring")
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            keep_key = st.session_state.api_key
            for key, value in DEFAULT_STATE.items():
                st.session_state[key] = value
            st.session_state.api_key = keep_key
            st.session_state.api_key_validated = True
            st.rerun()
        rolling_years = st.slider("Rolling window (years)", 2, 10, CFG.rolling_years_default)
        forecast_horizon = st.slider("Forecast horizon (business days)", 5, 60, CFG.forecast_horizon_default)
        confidence_level = st.slider("VaR confidence", 0.90, 0.99, 0.95, 0.01)
        ohlc_period = st.selectbox("OHLC period", ["6mo", "1y", "2y", "5y"], index=2)

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
            st.error("Failed to fetch FRED data.")
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

    with st.spinner("Running model layer..."):
        ns_result = NelsonSiegelModel.fit_ns(maturities, latest_curve)
        nss_result = NelsonSiegelModel.fit_nss(maturities, latest_curve)
        governance_df = model_governance(ns_result, nss_result)
        dynamic_params = rolling_ns_parameters(yield_df[selected_cols], maturities, selected_cols, rolling_years)
        factor_df = factor_contributions(yield_df)
        pca_risk = pca_risk_decomp(yield_df[selected_cols])
        forecast_df = forecast_curve(yield_df[selected_cols], forecast_horizon)
        arb = arbitrage_diagnostics(yield_df[selected_cols], maturities)
        scenarios = scenario_engine(yield_df[selected_cols])

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        kpi_card("Regime", regime, regime_text)
    with c2:
        kpi_card("2Y Yield", f"{current_2y:.2f}%", "Short-end policy anchor")
    with c3:
        kpi_card("10Y Yield", f"{current_10y:.2f}%", "Long-end benchmark")
    with c4:
        kpi_card("30Y Yield", f"{current_30y:.2f}%", "Long duration reference")
    with c5:
        kpi_card("10Y-2Y Spread", f"{current_spread:.1f} bps", "Primary recession signal")
    with c6:
        kpi_card("Recession Probability", f"{100 * recession_prob:.1f}%", "Institutional proxy")

    top_tabs = st.tabs([
        "Executive View",
        "Research View",
        "Risk Metrics",
        "Historical Scenario Analysis and Simulations",
        "Recession Analysis",
        "Technical Analysis",
        "Export and Deployment",
    ])

    with top_tabs[0]:
        st.plotly_chart(chart_current_curve(maturities, latest_curve, ns_result, nss_result), use_container_width=True)
        left, right = st.columns([1, 1])
        with left:
            fig2 = chart_yield(yield_df, "2Y", COLORS["warning"], "2-Year Treasury Yield")
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
        with right:
            fig10 = chart_yield(yield_df, "10Y", COLORS["accent"], "10-Year Treasury Yield")
            if fig10:
                st.plotly_chart(fig10, use_container_width=True)
        st.plotly_chart(chart_spreads(spreads, recessions), use_container_width=True)

    with top_tabs[1]:
        research_tabs = st.tabs(["Data Table", "NS Model", "NSS Model", "Model Governance", "Dynamic Analysis", "Factor and PCA"])
        with research_tabs[0]:
            display_df = yield_df.iloc[::-1].reset_index()
            display_df.columns = ["Date"] + list(yield_df.columns)
            for col in display_df.columns[1:]:
                display_df[col] = display_df[col].map(lambda x: f"{x:.2f}%")
            display_df["Date"] = pd.to_datetime(display_df["Date"]).dt.strftime("%Y-%m-%d")
            st.dataframe(display_df, use_container_width=True, height=450)
        with research_tabs[1]:
            if ns_result:
                st.dataframe(pd.DataFrame({
                    "Parameter": ["β0 Level", "β1 Slope", "β2 Curvature", "λ Decay", "RMSE", "MAE", "R²"],
                    "Value": [
                        f"{ns_result['params'][0]:.4f}", f"{ns_result['params'][1]:.4f}", f"{ns_result['params'][2]:.4f}",
                        f"{ns_result['params'][3]:.4f}", f"{ns_result['rmse']:.4f}", f"{ns_result['mae']:.4f}",
                        f"{ns_result['r_squared']:.4f}",
                    ]
                }), use_container_width=True, hide_index=True)
                st.markdown(
                    """
                    <div class="note-box">
                    <b>Nelson-Siegel Parameter Guide</b><br><br>
                    <b>β0 (Level)</b> captures the long-run average yield level across maturities.<br>
                    <b>β1 (Slope)</b> measures the short-vs-long maturity steepness of the curve.<br>
                    <b>β2 (Curvature)</b> captures the hump or belly shape, typically around intermediate maturities.<br>
                    <b>λ (Decay)</b> controls where slope and curvature effects are strongest along the maturity spectrum.<br><br>
                    Users should read these parameters together: level tells the broad rate regime, slope shows cycle direction, and curvature shows how concentrated the move is in the belly of the curve.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with research_tabs[2]:
            if nss_result:
                st.dataframe(pd.DataFrame({
                    "Parameter": ["β0", "β1", "β2", "β3", "λ1", "λ2", "RMSE", "MAE", "R²"],
                    "Value": [
                        f"{nss_result['params'][0]:.4f}", f"{nss_result['params'][1]:.4f}", f"{nss_result['params'][2]:.4f}",
                        f"{nss_result['params'][3]:.4f}", f"{nss_result['params'][4]:.4f}", f"{nss_result['params'][5]:.4f}",
                        f"{nss_result['rmse']:.4f}", f"{nss_result['mae']:.4f}", f"{nss_result['r_squared']:.4f}",
                    ]
                }), use_container_width=True, hide_index=True)
                st.markdown(
                    """
                    <div class="note-box">
                    <b>NSS Parameter Guide</b><br><br>
                    <b>β0</b> is the long-run level factor.<br>
                    <b>β1</b> captures the slope of the curve.<br>
                    <b>β2</b> captures medium-term curvature.<br>
                    <b>β3</b> adds a second curvature component, allowing more flexible shapes than the basic Nelson-Siegel model.<br>
                    <b>λ1</b> and <b>λ2</b> determine where these curvature effects peak across maturities.<br><br>
                    The NSS model is useful when the curve has multiple bends or when the belly and long end move differently. It is often more flexible, but that flexibility also requires stronger governance and residual checks.
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
                <b>Model Governance Guidance</b><br><br>
                <b>Fit Confidence</b> summarizes whether the model is close enough to the observed curve for institutional use.<br>
                <b>Residual Quality</b> highlights which maturities are poorly explained by the model.<br>
                <b>Warning Flags</b> identify outliers or weak fit conditions where investment conclusions should be treated more cautiously.
                </div>
                """,
                unsafe_allow_html=True,
            )
        with research_tabs[4]:
            fig_dyn = chart_dynamic(dynamic_params)
            if fig_dyn:
                st.plotly_chart(fig_dyn, use_container_width=True)
        with research_tabs[5]:
            l, r = st.columns(2)
            with l:
                fig_factor = chart_factors(factor_df)
                if fig_factor:
                    st.plotly_chart(fig_factor, use_container_width=True)
            with r:
                fig_pca = chart_pca_variance(pca_risk)
                if fig_pca:
                    st.plotly_chart(fig_pca, use_container_width=True)
                if pca_risk:
                    st.dataframe(pca_risk["loadings"].round(4), use_container_width=True)
                    st.markdown(
                        """
                        <div class="note-box">
                        <b>PCA Interpretation Guide</b><br><br>
                        <b>PC1</b> usually represents the <b>Level</b> factor: the whole curve shifts up or down together.<br>
                        <b>PC2</b> often represents the <b>Slope</b> factor: short and long maturities move differently.<br>
                        <b>PC3</b> often represents <b>Curvature</b>: the belly of the curve behaves differently from the wings.<br><br>
                        The purpose of PCA is dimensionality reduction. Instead of tracking every maturity separately, users can summarize the dominant rate dynamics with a few orthogonal factors and see whether the current market is being driven by level, slope, or curvature shocks.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    fig_rate_dyn = chart_rate_dynamics(yield_df[selected_cols], spreads)
                    if fig_rate_dyn:
                        st.plotly_chart(fig_rate_dyn, use_container_width=True)

    with top_tabs[2]:
        if "10Y" in yield_df.columns:
            risk = calculate_var_metrics(yield_df["10Y"].pct_change(), confidence_level, 10)
            if risk:
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Historical VaR", f"{risk['VaR_Historical']:.4f}")
                r2.metric("Parametric VaR", f"{risk['VaR_Parametric']:.4f}")
                r3.metric("Cornish-Fisher VaR", f"{risk['VaR_CornishFisher']:.4f}")
                r4.metric("CVaR", f"{risk['CVaR']:.4f}")
                st.markdown(
                    """
                    <div class="note-box">
                    <b>Risk Metrics Explanation</b><br><br>
                    <b>Historical VaR</b> estimates the loss threshold using empirical return history.<br>
                    <b>Parametric VaR</b> assumes returns are approximately normal and scales risk using mean and volatility.<br>
                    <b>Cornish-Fisher VaR</b> adjusts for skewness and kurtosis, helping users see whether tail shape matters.<br>
                    <b>CVaR</b> measures the average loss beyond VaR and is more informative for tail-risk management.<br><br>
                    Users should compare these metrics side by side: wide divergence suggests non-normal risk, asymmetric shocks, or unstable volatility.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with top_tabs[3]:
        scenario_name = st.selectbox("Select Scenario", list(scenarios.keys()))
        st.plotly_chart(chart_scenario(scenarios[scenario_name], scenario_name), use_container_width=True)
        scenario_notes = {
            "Bull Steepener": "Short and intermediate yields decline modestly, while the long end declines more. This is usually associated with falling inflation expectations, softer growth, or a rally led by duration.",
            "Bear Flattener": "Front-end yields rise more than the long end. This often reflects hawkish central-bank policy or repricing of the policy path.",
            "Recession Case": "Broad rally across the curve, with the strongest fall at the short end. This scenario emphasizes defensive allocation and recession hedging.",
            "Policy Easing": "Short-end yields fall sharply after expected policy cuts, while the long end reacts less. This helps users isolate the transmission of a central-bank easing cycle."
        }
        st.markdown(f'<div class="note-box"><b>{scenario_name}</b><br><br>{scenario_notes[scenario_name]}</div>', unsafe_allow_html=True)

        # historical scenario comparison table
        hist_table = scenarios[scenario_name].copy()
        hist_table["Change"] = hist_table["Scenario"] - hist_table["Current"]
        hist_table.index.name = "Tenor"
        st.dataframe(hist_table.round(4), use_container_width=True)

    with top_tabs[4]:
        rr_tabs = st.tabs(["NBER Chart", "Hit Ratio", "False Positives", "Current Inversion vs History", "Lead-Time Summary"])
        with rr_tabs[0]:
            fig = go.Figure()
            if "10Y-2Y" in spreads.columns:
                fig.add_trace(go.Scatter(x=spreads.index, y=spreads["10Y-2Y"], mode="lines", name="10Y-2Y", line=dict(color=COLORS["negative"], width=2)))
            fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"])
            add_recession_bands(fig, recessions)
            st.plotly_chart(create_chart_layout(fig, "NBER Recession Overlay | 10Y-2Y Spread", "bps", 500), use_container_width=True)
        with rr_tabs[1]:
            hit_df = pd.DataFrame([hit_stats])
            st.dataframe(hit_df.round(4), use_container_width=True, hide_index=True)
            st.markdown('<div class="note-box"><b>Hit Ratio</b> shows how many inversion signals were eventually followed by recessions in this sample.</div>', unsafe_allow_html=True)
        with rr_tabs[2]:
            false_df = pd.DataFrame([{
                "Signals": hit_stats["Signals"],
                "Matches": hit_stats["Matches"],
                "FalsePositives": hit_stats["Signals"] - hit_stats["Matches"],
                "FalsePositiveRate": hit_stats["FalsePositiveRate"],
            }])
            st.dataframe(false_df.round(4), use_container_width=True, hide_index=True)
            st.markdown('<div class="note-box"><b>False Positives</b> matter because not every inversion becomes a recession immediately. Users should evaluate both hit quality and timing dispersion.</div>', unsafe_allow_html=True)
        with rr_tabs[3]:
            st.dataframe(current_inversion_vs_history(spreads, inversions).round(4), use_container_width=True, hide_index=True)
        with rr_tabs[4]:
            if lead_times:
                st.dataframe(pd.DataFrame(lead_times).round(4), use_container_width=True, hide_index=True)
            else:
                st.info("No lead-time episodes were identified in the current sample alignment.")
            st.markdown('<div class="note-box"><b>Lead-Time Summary</b> helps users compare the current inversion with past episodes and estimate how much time historically elapsed before recession onset.</div>', unsafe_allow_html=True)

    with top_tabs[5]:
        tech_tabs = st.tabs(["OHLC and Indicators", "Rate Direction Techniques", "Interpretation Notes"])
        with tech_tabs[0]:
            ticker = st.selectbox("Select OHLC Instrument", list(YAHOO_TICKERS.keys()), format_func=lambda x: f"{x} | {YAHOO_TICKERS[x]}")
            ohlc_df = ohlc_data.get(ticker)
            left, right = st.columns([1.15, 1])
            with left:
                fig_ohlc = chart_ohlc(ohlc_df, ticker)
                if fig_ohlc:
                    st.plotly_chart(fig_ohlc, use_container_width=True)
            with right:
                fig_ta = chart_technical_panels(ohlc_df, ticker)
                if fig_ta:
                    st.plotly_chart(fig_ta, use_container_width=True)
        with tech_tabs[1]:
            fig_rate_dyn = chart_rate_dynamics(yield_df[selected_cols], spreads)
            if fig_rate_dyn:
                st.plotly_chart(fig_rate_dyn, use_container_width=True)
            st.markdown(
                """
                <div class="note-box">
                <b>Quantitative Techniques for Changing Rate Direction</b><br><br>
                This panel tracks how interest-rate direction changes over time using:
                <b>20-day changes</b> in 2Y and 10Y yields,
                <b>slope momentum</b> for the 10Y-2Y spread,
                and <b>60-day realized volatility</b> for the 10Y yield.
                Together, these help the user distinguish between simple rate moves, curve-shape changes, and volatility regime shifts.
                </div>
                """,
                unsafe_allow_html=True,
            )
        with tech_tabs[2]:
            st.markdown(
                """
                <div class="note-box">
                <b>Technical Layer Explanation</b><br><br>
                OHLC and indicator panels are provided as a supplemental tactical layer rather than the main macro signal engine.
                <b>RSI</b> highlights overbought / oversold momentum,
                <b>MACD</b> helps users identify trend shifts,
                and <b>Bollinger Bands</b> show volatility envelopes around price behavior.
                For institutional use, this layer should be read together with regime classification, curve shape, recession diagnostics, and scenario analysis.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with top_tabs[6]:
        export_tabs = st.tabs(["Download Data", "Requirements", "Run Instructions", "Streamlit Cloud"])
        with export_tabs[0]:
            st.download_button("Download Yield Data (CSV)", yield_df.to_csv().encode("utf-8"), f"yield_data_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
            st.download_button("Download Spread Data (CSV)", spreads.to_csv().encode("utf-8"), f"spreads_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
            if dynamic_params is not None and not dynamic_params.empty:
                st.download_button("Download Dynamic Parameters", dynamic_params.to_csv(index=False).encode("utf-8"), f"dynamic_params_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
            if pca_risk:
                st.download_button("Download PCA Loadings", pca_risk["loadings"].to_csv().encode("utf-8"), f"pca_loadings_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
            if lead_times:
                st.download_button("Download Lead Times", pd.DataFrame(lead_times).to_csv(index=False).encode("utf-8"), f"lead_times_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
        with export_tabs[1]:
            st.code(
                "streamlit\npandas\nnumpy\nplotly\nrequests\nyfinance\nscipy\nscikit-learn",
                language="text",
            )
        with export_tabs[2]:
            st.code(
                "pip install -r requirements.txt\nstreamlit run yield_curve_v35_5_full_fold_in_final.py",
                language="bash",
            )
        with export_tabs[3]:
            st.markdown(
                """
                <div class="note-box">
                <b>Streamlit Cloud Compatibility</b><br><br>
                1. Upload the app file and a <code>requirements.txt</code> file.<br>
                2. Set the main file to <code>yield_curve_v35_5_full_fold_in_final.py</code>.<br>
                3. Launch the app, then enter your FRED API key inside the UI.<br>
                4. If compute feels slow, reduce rolling window length and forecast horizon.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center; color:{COLORS['muted']}; font-size:0.72rem;'>Institutional Quantitative Platform | MK Istanbul Fintech LabGEN @2026</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
