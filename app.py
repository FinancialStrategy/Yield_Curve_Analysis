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
from plotly.subplots import make_subplots
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    yf = None
    YFINANCE_AVAILABLE = False

st.set_page_config(
    page_title="Dynamic Quantitative Analysis Model | Institutional Fixed-Income",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "bg": "#eef2f7", "bg2": "#f7f9fc", "surface": "#ffffff", "surface_alt": "#f5f7fb",
    "header": "#1a2a3a", "grid": "#c8d4e0", "grid_dark": "#97a8b8",
    "text": "#1a2a3a", "text_secondary": "#4a5a6a", "muted": "#667085",
    "accent": "#2c5f8a", "accent2": "#4a7c59", "accent3": "#c17f3a",
    "positive": "#2f855a", "negative": "#c05656", "warning": "#d48924",
    "recession": "rgba(120, 130, 145, 0.18)", "band": "rgba(108, 142, 173, 0.10)",
}

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #eef2f7 0%, #f7f9fc 100%); }
.main-title-card { background: linear-gradient(135deg, #1a2a3a 0%, #2c4a6e 100%); border-radius: 20px; padding: 1.5rem 1.8rem; margin-bottom: 1.5rem; }
.main-title { color: white; font-weight: 800; font-size: 1.65rem; margin: 0; }
.main-subtitle { color: rgba(255,255,255,0.86); font-size: 0.86rem; margin-top: 0.55rem; }
.metric-card { background: white; border: 1px solid #c8d4e0; border-radius: 16px; padding: 1rem; min-height: 118px; }
.metric-label { color: #4a5a6a; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; }
.metric-value { color: #1a2a3a; font-size: 1.55rem; font-weight: 800; font-family: monospace; }
.metric-sub { color: #667085; font-size: 0.75rem; margin-top: 0.35rem; }
.note-box { background: #f5f7fb; border: 1px solid #c8d4e0; border-left: 4px solid #2c5f8a; border-radius: 12px; padding: 1rem 1.2rem; margin: 1rem 0; }
.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 2px solid #c8d4e0; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { color: #4a5a6a; font-weight: 700; font-size: 0.74rem; padding: 8px 14px; }
.stTabs [aria-selected="true"] { color: #2c5f8a; border-bottom: 3px solid #2c5f8a; }
.stButton>button { background: white; border: 1px solid #c8d4e0; border-radius: 10px; font-weight: 700; }
.stButton>button:hover { border-color: #2c5f8a; color: #2c5f8a; }
#MainMenu, header, footer { visibility: hidden; }
</style>""", unsafe_allow_html=True)

FRED_SERIES = {
    "1M": "DGS1MO", "3M": "DGS3MO", "6M": "DGS6MO",
    "1Y": "DGS1", "2Y": "DGS2", "3Y": "DGS3",
    "5Y": "DGS5", "7Y": "DGS7", "10Y": "DGS10",
    "20Y": "DGS20", "30Y": "DGS30",
}

MATURITY_MAP = {
    "1M": 1/12, "3M": 0.25, "6M": 0.5,
    "1Y": 1.0, "2Y": 2.0, "3Y": 3.0,
    "5Y": 5.0, "7Y": 7.0, "10Y": 10.0,
    "20Y": 20.0, "30Y": 30.0,
}

YAHOO_TICKERS = {"^TNX": "10Y Treasury", "^FVX": "5Y Treasury", "^IRX": "13W T-Bill"}
VOLATILITY_TICKERS = {"^VIX": "CBOE VIX"}
CORRELATION_TICKERS = {"^GSPC": "S&P 500", "QQQ": "Nasdaq 100", "GLD": "Gold", "UUP": "US Dollar"}

DEFAULT_STATE = {
    "api_key_validated": False, "api_key": "",
    "yield_data": None, "recession_data": None,
    "ohlc_data": None, "volatility_data": None,
    "correlation_data": None, "data_fetched": False,
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

CFG = RuntimeConfig()

# =============================================================================
# DATA FUNCTIONS
# =============================================================================

def fred_request(api_key: str, series_id: str) -> Optional[pd.Series]:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json",
              "observation_start": CFG.history_start, "sort_order": "asc"}
    for attempt in range(CFG.max_retries):
        try:
            r = requests.get(url, params=params, timeout=CFG.timeout)
            r.raise_for_status()
            obs = r.json().get("observations", [])
            dates, values = [], []
            for row in obs:
                val = row.get("value")
                if val not in (".", None):
                    dates.append(pd.to_datetime(row["date"]))
                    values.append(float(val))
            return pd.Series(values, index=dates, name=series_id) if dates else None
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
def fetch_yahoo_close(ticker: str, period: str = "2y") -> Optional[pd.Series]:
    if not YFINANCE_AVAILABLE:
        return None
    try:
        df = yf.download(ticker, period=period, progress=False)
        return df["Close"].rename(ticker) if df is not None and not df.empty else None
    except Exception:
        return None

@st.cache_data(ttl=CFG.cache_ttl_sec, show_spinner=False)
def fetch_ohlc_data(ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
    if not YFINANCE_AVAILABLE:
        return None
    try:
        return yf.download(ticker, period=period, progress=False)
    except Exception:
        return None

@st.cache_data(ttl=CFG.cache_ttl_sec, show_spinner=False)
def fetch_market_bundle():
    vol_dict = {}
    for t in VOLATILITY_TICKERS:
        s = fetch_yahoo_close(t)
        if s is not None:
            vol_dict[t] = s
    vol_df = pd.DataFrame(vol_dict).dropna() if vol_dict else None
    corr_dict = {}
    for t, name in CORRELATION_TICKERS.items():
        s = fetch_yahoo_close(t)
        if s is not None:
            corr_dict[name] = s
    corr_df = pd.DataFrame(corr_dict).dropna() if corr_dict else None
    ohlc_map = {t: fetch_ohlc_data(t, "2y") for t in YAHOO_TICKERS}
    return vol_df, corr_df, ohlc_map

# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def sma(x, n): return x.rolling(n).mean()
def ema(x, n): return x.ewm(span=n, adjust=False).mean()

def rsi(x, n=14):
    d = x.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(x):
    m = ema(x, 12) - ema(x, 26)
    s = ema(m, 9)
    return m, s, m - s

def bollinger_bands(x, n=20, k=2.0):
    mid = sma(x, n)
    sd = x.rolling(n).std()
    return mid + k * sd, mid, mid - k * sd

def add_technical_indicators(df):
    if df is None or df.empty:
        return df
    out = df.copy()
    out["SMA_20"] = sma(out["Close"], 20)
    out["SMA_50"] = sma(out["Close"], 50)
    out["RSI"] = rsi(out["Close"], 14)
    out["MACD"], out["MACD_Signal"], out["MACD_Hist"] = macd(out["Close"])
    out["BB_Upper"], out["BB_Mid"], out["BB_Lower"] = bollinger_bands(out["Close"])
    return out

# =============================================================================
# ANALYTICS
# =============================================================================

def compute_spreads(df):
    spreads = pd.DataFrame(index=df.index)
    if "10Y" in df and "2Y" in df:
        spreads["10Y-2Y"] = (df["10Y"] - df["2Y"]) * 100
    if "10Y" in df and "3M" in df:
        spreads["10Y-3M"] = (df["10Y"] - df["3M"]) * 100
    return spreads

def classify_regime(spreads, df):
    if "10Y-2Y" not in spreads or spreads.empty:
        return "Data Loading", "Please wait"
    s = spreads["10Y-2Y"].iloc[-1]
    if s < 0:
        return "Risk-off / Recession Watch", "Curve inversion signals defensive regime"
    if s < 50:
        return "Neutral / Late Cycle", "Curve flattening suggests caution"
    return "Risk-on / Expansion", "Positive slope supports risk-on"

def recession_probability(spreads, df):
    if "10Y-2Y" not in spreads:
        return 0.5
    score = np.clip((-spreads["10Y-2Y"].iloc[-1]) / 100, 0, 1.5)
    return float(1 / (1 + np.exp(-2.2 * (score - 0.8))))

def identify_recessions(rec_series):
    if rec_series is None or len(rec_series) == 0:
        return []
    recessions = []
    in_rec = False
    start = None
    for date, val in rec_series.dropna().items():
        if val == 1 and not in_rec:
            in_rec = True
            start = date
        elif val == 0 and in_rec:
            recessions.append({"start": start, "end": date})
            in_rec = False
    return recessions

class NelsonSiegelModel:
    @staticmethod
    def ns(tau, b0, b1, b2, l1):
        tau = np.asarray(tau, dtype=float)
        tau = np.where(tau == 0, 1e-8, tau)
        x = l1 * tau
        t1 = (1 - np.exp(-x)) / x
        t2 = t1 - np.exp(-x)
        return b0 + b1 * t1 + b2 * t2

    @staticmethod
    def fit_ns(maturities, yields_):
        if len(maturities) == 0:
            return None
        def obj(params):
            fitted = NelsonSiegelModel.ns(maturities, *params)
            return np.sum((yields_ - fitted) ** 2)
        bounds = [(yields_.min()-2, yields_.max()+2), (-15,15), (-15,15), (0.01,5)]
        best, best_fun = None, np.inf
        for _ in range(6):
            x0 = [np.random.uniform(a,b) for a,b in bounds]
            res = minimize(obj, x0=x0, bounds=bounds, method="L-BFGS-B")
            if res.success and res.fun < best_fun:
                best, best_fun = res, res.fun
        if best is None:
            return None
        fitted = NelsonSiegelModel.ns(maturities, *best.x)
        sse = np.sum((yields_ - fitted)**2)
        sst = np.sum((yields_ - np.mean(yields_))**2)
        return {
            "params": best.x,
            "fitted_values": fitted,
            "rmse": np.sqrt(np.mean((yields_ - fitted)**2)),
            "r_squared": 1 - sse/sst if sst > 0 else np.nan,
        }

# =============================================================================
# MONTE CARLO
# =============================================================================

class MonteCarloSimulator:
    @staticmethod
    def gbm(initial, mu, sigma, days, sims=1000):
        dt = 1/252
        paths = np.zeros((sims, days))
        paths[:,0] = initial
        for i in range(1, days):
            z = np.random.standard_normal(sims)
            paths[:,i] = paths[:,i-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
        return paths

    @staticmethod
    def ci(paths, conf=0.95):
        low = (1-conf)/2*100
        high = (1+conf)/2*100
        return {
            "mean": np.mean(paths, axis=0),
            "median": np.percentile(paths, 50, axis=0),
            "lower": np.percentile(paths, low, axis=0),
            "upper": np.percentile(paths, high, axis=0),
            "std": np.std(paths, axis=0),
        }

# =============================================================================
# ML FORECAST
# =============================================================================

class MLForecastModel:
    @staticmethod
    def prepare_features(df, lags=5):
        X, y = [], []
        for i in range(lags, len(df)-1):
            feats = []
            for col in df.columns:
                feats.extend(df[col].iloc[i-lags:i].values)
            X.append(feats)
            y.append(df.iloc[i+1].values)
        if not X:
            return np.array([]), np.array([]), None
        X_arr, y_arr = np.array(X), np.array(y)
        scaler = StandardScaler()
        return scaler.fit_transform(X_arr), y_arr, scaler

    @staticmethod
    def train(X, y, model_type="Random Forest"):
        if len(X) == 0:
            return {}
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if model_type == "Gradient Boosting":
            model = GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=3, random_state=42)
            model.fit(X_train, y_train[:, -1])
            y_pred = model.predict(X_test)
            y_eval = y_test[:, -1]
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)[:, -1]
            y_eval = y_test[:, -1]
        return {
            "rmse": np.sqrt(mean_squared_error(y_eval, y_pred)),
            "r2": r2_score(y_eval, y_pred),
        }

# =============================================================================
# BACKTEST
# =============================================================================

class BacktestEngine:
    @staticmethod
    def backtest(df, spreads, strategy="Curve Inversion"):
        if "10Y" not in df:
            return {}
        returns = df["10Y"].pct_change().shift(-1)
        if strategy == "Curve Inversion":
            if "10Y-2Y" not in spreads:
                return {}
            signals = spreads["10Y-2Y"] < 0
        else:
            signals = df["10Y"] > df["10Y"].rolling(50).mean()
        strategy_returns = signals.shift(1) * returns
        cum = (1 + strategy_returns.fillna(0)).cumprod()
        return {
            "cumulative": cum,
            "total_return": cum.iloc[-1] - 1,
            "sharpe": (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() > 0 else 0,
        }

# =============================================================================
# VOLATILITY
# =============================================================================

class VolatilityAnalyzer:
    @staticmethod
    def regime(vix):
        if vix is None or len(vix) == 0:
            return {"current": 0, "regime": "N/A"}
        current = vix.iloc[-1]
        if current < 12:
            regime = "EXTREME COMPLACENCY"
        elif current < 15:
            regime = "LOW VOLATILITY"
        elif current < 20:
            regime = "NORMAL VOLATILITY"
        elif current < 25:
            regime = "ELEVATED VOLATILITY"
        elif current < 35:
            regime = "HIGH VOLATILITY"
        else:
            regime = "EXTREME VOLATILITY"
        return {"current": current, "regime": regime}

# =============================================================================
# PCA & FACTORS - DÜZELTİLDİ
# =============================================================================

def factor_contributions(df):
    out = pd.DataFrame(index=df.index)
    if "10Y" in df:
        out["Level"] = df["10Y"]
    if "10Y" in df and "3M" in df:
        out["Slope"] = (df["10Y"] - df["3M"]) * 100
    if "3M" in df and "5Y" in df and "10Y" in df:
        out["Curvature"] = (2 * df["5Y"] - (df["3M"] + df["10Y"])) * 100
    return out

def pca_decomp(df, n=3):
    """PCA decomposition - HATALARI GİDERİLDİ"""
    if df is None or df.empty:
        return None
    
    # Get returns and clean NaN/inf values
    returns = df.pct_change()
    
    # Replace inf with NaN, then drop NaN
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna()
    
    if len(returns) < 20:
        return None
    
    if returns.shape[1] < 2:
        return None
    
    try:
        scaler = StandardScaler()
        x = scaler.fit_transform(returns)
        
        k = min(n, x.shape[1], x.shape[0]-1)
        if k < 1:
            return None
            
        pca = PCA(n_components=k)
        pca.fit(x)
        
        loadings = pd.DataFrame(
            pca.components_.T, 
            columns=[f"PC{i+1}" for i in range(k)], 
            index=returns.columns
        )
        
        return {
            "explained": pca.explained_variance_ratio_,
            "cumulative": np.cumsum(pca.explained_variance_ratio_),
            "loadings": loadings,
        }
    except Exception as e:
        return None

# =============================================================================
# SCENARIO ENGINE
# =============================================================================

def scenario_engine(df):
    latest = df.iloc[-1].copy()
    scenarios = {}
    bull = latest.copy()
    for col in df.columns:
        m = MATURITY_MAP.get(col, 1)
        bull[col] = bull[col] - (0.08 + 0.06 * min(m/10, 1.5))
    scenarios["Bull Steepener"] = pd.DataFrame({"Current": latest, "Scenario": bull})
    bear = latest.copy()
    for col in df.columns:
        m = MATURITY_MAP.get(col, 1)
        bear[col] = bear[col] + (0.14 if m <= 2 else 0.07)
    scenarios["Bear Flattener"] = pd.DataFrame({"Current": latest, "Scenario": bear})
    rec = latest.copy()
    for col in df.columns:
        m = MATURITY_MAP.get(col, 1)
        rec[col] = rec[col] - (0.22 if m <= 2 else 0.14 if m <= 10 else 0.10)
    scenarios["Recession Case"] = pd.DataFrame({"Current": latest, "Scenario": rec})
    return scenarios

# =============================================================================
# GRAFİK FONKSİYONLARI
# =============================================================================

def plot_curve(maturities, yields, ns_result):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=maturities, y=yields, mode='lines+markers', name='Actual Curve', line=dict(color='#2c5f8a', width=3), marker=dict(size=10, color='#2c5f8a')))
    if ns_result:
        fig.add_trace(go.Scatter(x=maturities, y=ns_result['fitted_values'], mode='lines', name='NS Fit', line=dict(color='#4a7c59', width=2.5, dash='dash')))
    fig.update_layout(title='Current Treasury Yield Curve', xaxis_title='Maturity (Years)', yaxis_title='Yield (%)', height=500, template='plotly_white')
    return fig

def plot_spread(spreads, recessions):
    fig = go.Figure()
    if "10Y-2Y" in spreads:
        fig.add_trace(go.Scatter(x=spreads.index, y=spreads['10Y-2Y'], mode='lines', name='10Y-2Y Spread', line=dict(color='#d48924', width=2.5)))
    fig.add_hline(y=0, line_dash='dash', line_color='red')
    for r in recessions:
        fig.add_vrect(x0=r['start'], x1=r['end'], fillcolor='rgba(120,130,145,0.2)', layer='below', line_width=0)
    fig.update_layout(title='10Y-2Y Spread History', xaxis_title='Date', yaxis_title='Basis Points (bps)', height=450, template='plotly_white')
    return fig

def plot_yield_history(df, tenor, title):
    if tenor not in df:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[tenor], mode='lines', name=tenor, line=dict(color='#2c5f8a', width=2.5)))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Yield (%)', height=400, template='plotly_white')
    return fig

def plot_monte_carlo(initial, results, days):
    fig = go.Figure()
    x = np.arange(days)
    fig.add_trace(go.Scatter(x=x, y=results['upper'], fill=None, mode='lines', line=dict(color='rgba(0,0,0,0)'), showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=results['lower'], fill='tonexty', mode='lines', fillcolor='rgba(44,95,138,0.2)', line=dict(color='rgba(0,0,0,0)'), name='95% CI'))
    fig.add_trace(go.Scatter(x=x, y=results['mean'], mode='lines', name='Mean Path', line=dict(color='#2c5f8a', width=3)))
    fig.add_trace(go.Scatter(x=[0], y=[initial], mode='markers', name='Current', marker=dict(size=12, color='#2f855a', symbol='star')))
    fig.update_layout(title=f'Monte Carlo Simulation - {days} Days', xaxis_title='Trading Days Ahead', yaxis_title='Yield (%)', height=500, template='plotly_white')
    return fig

def plot_backtest(bt_results):
    if not bt_results:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt_results['cumulative'].index, y=bt_results['cumulative'].values, mode='lines', name='Strategy', line=dict(color='#2c5f8a', width=2.5)))
    fig.update_layout(title='Backtest Performance', xaxis_title='Date', yaxis_title='Cumulative Return', height=450, template='plotly_white')
    return fig

def plot_volatility(vix, regime):
    if vix is None or len(vix) == 0:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vix.index, y=vix.values, mode='lines', name='VIX', line=dict(color='#d48924', width=2.5), fill='tozeroy', fillcolor='rgba(212,137,36,0.1)'))
    fig.add_hline(y=20, line_dash='dash', line_color='orange')
    fig.add_hline(y=15, line_dash='dash', line_color='green')
    fig.update_layout(title=f'VIX Dashboard - Current: {regime["current"]:.1f} ({regime["regime"]})', xaxis_title='Date', yaxis_title='VIX', height=450, template='plotly_white')
    return fig

def plot_correlation(corr_matrix):
    if corr_matrix.empty:
        return None
    fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='RdBu', zmid=0, text=corr_matrix.values.round(2), texttemplate='%{text}'))
    fig.update_layout(title='Cross-Asset Correlation Matrix', height=500, width=650, template='plotly_white')
    return fig

def plot_technical(df, ticker):
    if df is None or df.empty:
        return None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.6, 0.4])
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close', line=dict(color='#2c5f8a', width=2.5)), row=1, col=1)
    if 'RSI' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='#4a7c59', width=2)), row=2, col=1)
        fig.add_hline(y=70, line_dash='dash', line_color='red', row=2, col=1)
        fig.add_hline(y=30, line_dash='dash', line_color='green', row=2, col=1)
    fig.update_layout(title=f'Technical Analysis - {ticker}', height=600, template='plotly_white')
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='RSI', row=2, col=1, range=[0,100])
    return fig

def plot_factor_contributions(factor_df):
    if factor_df is None or factor_df.empty:
        return None
    fig = go.Figure()
    colors = ['#2c5f8a', '#4a7c59', '#d48924', '#c17f3a']
    for i, col in enumerate(factor_df.columns[:4]):
        fig.add_trace(go.Scatter(x=factor_df.index, y=factor_df[col], mode='lines', name=col, line=dict(color=colors[i % len(colors)], width=2)))
    fig.update_layout(title='Factor Contributions', xaxis_title='Date', yaxis_title='Value', height=450, template='plotly_white')
    return fig

def plot_pca(pca_result):
    if pca_result is None:
        return None
    fig = go.Figure()
    ev = pca_result['explained'] * 100
    fig.add_trace(go.Bar(x=[f'PC{i+1}' for i in range(len(ev))], y=ev, marker_color='#2c5f8a', text=[f'{x:.1f}%' for x in ev], textposition='outside'))
    fig.update_layout(title='PCA Variance Explained', xaxis_title='Principal Component', yaxis_title='Variance Explained (%)', height=450, template='plotly_white')
    return fig

def plot_scenario(scenario_df, name):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=scenario_df.index, y=scenario_df['Current'], name='Current', marker_color='#2c5f8a'))
    fig.add_trace(go.Bar(x=scenario_df.index, y=scenario_df['Scenario'], name=name, marker_color='#d48924'))
    fig.update_layout(title=f'Scenario Analysis - {name}', xaxis_title='Tenor', yaxis_title='Yield (%)', height=450, template='plotly_white', barmode='group')
    return fig

# =============================================================================
# UI HELPERS
# =============================================================================

def render_api_gate():
    st.markdown('<div class="note-box" style="max-width:500px; margin:40px auto; text-align:center;"><b>🔑 FRED API Key Required</b><br><br>Get your free API key from the FRED website.</div>', unsafe_allow_html=True)
    api_key = st.text_input("Enter your FRED API key", type="password")
    if st.button("Validate & Connect", use_container_width=True):
        if not api_key:
            st.error("Please enter a valid API key.")
            st.stop()
        with st.spinner("Validating..."):
            valid = validate_fred_api_key(api_key)
        if valid:
            st.session_state["api_key"] = api_key
            st.session_state["api_key_validated"] = True
            st.success("✓ API key validated. Loading data...")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("Invalid API key.")
    st.stop()

def kpi_card(label, value, sub):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

# =============================================================================
# MAIN
# =============================================================================

def main():
    st.markdown('<div class="main-title-card"><div class="main-title">Dynamic Quantitative Analysis Model</div><div class="main-subtitle">Institutional Fixed-Income Platform | Advanced Analytics</div></div>', unsafe_allow_html=True)

    if not st.session_state["api_key_validated"]:
        render_api_gate()

    with st.sidebar:
        st.markdown("### Control Tower")
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            k = st.session_state["api_key"]
            for key in DEFAULT_STATE:
                st.session_state[key] = DEFAULT_STATE[key]
            st.session_state["api_key"] = k
            st.session_state["api_key_validated"] = True
            st.rerun()
        mc_sims = st.slider("Monte Carlo Simulations", 500, 3000, 1000)
        fc_horizon = st.slider("Forecast Horizon (days)", 5, 252, 20)
        ml_lags = st.slider("ML Lags", 3, 10, 5)

    if not st.session_state["data_fetched"]:
        with st.spinner("Fetching data..."):
            st.session_state["yield_data"] = fetch_all_yield_data(st.session_state["api_key"])
            st.session_state["recession_data"] = fetch_recession_data(st.session_state["api_key"])
            vol_df, corr_df, ohlc_map = fetch_market_bundle()
            for k, v in ohlc_map.items():
                if v is not None:
                    ohlc_map[k] = add_technical_indicators(v)
            st.session_state["volatility_data"] = vol_df
            st.session_state["correlation_data"] = corr_df
            st.session_state["ohlc_data"] = ohlc_map
        if st.session_state["yield_data"] is None:
            st.error("Failed to fetch FRED data.")
            st.stop()
        st.session_state["data_fetched"] = True

    df = st.session_state["yield_data"].copy()
    rec_series = st.session_state["recession_data"]
    vol_df = st.session_state["volatility_data"]
    corr_df = st.session_state["correlation_data"]
    ohlc_map = st.session_state["ohlc_data"]

    selected = [c for c in df.columns if c in MATURITY_MAP]
    maturities = np.array([MATURITY_MAP[c] for c in selected])
    latest = df.iloc[-1][selected].values

    spreads = compute_spreads(df)
    recessions = identify_recessions(rec_series)
    regime, regime_text = classify_regime(spreads, df)
    rec_prob = recession_probability(spreads, df)

    current_2y = df["2Y"].iloc[-1] if "2Y" in df else np.nan
    current_10y = df["10Y"].iloc[-1] if "10Y" in df else np.nan
    current_spread = spreads["10Y-2Y"].iloc[-1] if "10Y-2Y" in spreads else np.nan
    current_vix = vol_df["^VIX"].iloc[-1] if vol_df is not None and "^VIX" in vol_df else np.nan

    ns_result = NelsonSiegelModel.fit_ns(maturities, latest)
    factor_df = factor_contributions(df)
    pca_result = pca_decomp(df[selected])  # DÜZELTİLDİ
    scenarios = scenario_engine(df[selected])

    # KPI Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: kpi_card("Macro Regime", regime, regime_text[:30])
    with c2: kpi_card("2Y Yield", f"{current_2y:.2f}%" if not np.isnan(current_2y) else "N/A", "Policy anchor")
    with c3: kpi_card("10Y Yield", f"{current_10y:.2f}%" if not np.isnan(current_10y) else "N/A", "Benchmark")
    with c4: kpi_card("10Y-2Y Spread", f"{current_spread:.1f} bps" if not np.isnan(current_spread) else "N/A", "Recession signal")
    with c5: kpi_card("Recession Prob", f"{rec_prob*100:.1f}%", "Proxy estimate")
    with c6: kpi_card("VIX", f"{current_vix:.2f}" if not np.isnan(current_vix) else "N/A", "Fear gauge")

    tabs = st.tabs(["Executive", "Monte Carlo", "ML Forecast", "Backtest", "Volatility", "Correlations", "Technical", "Research", "Export"])

    with tabs[0]:
        st.plotly_chart(plot_curve(maturities, latest, ns_result), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_spread(spreads, recessions), use_container_width=True)
        with col2:
            st.plotly_chart(plot_yield_history(df, "10Y", "10-Year Treasury Yield History"), use_container_width=True)
        st.plotly_chart(plot_factor_contributions(factor_df), use_container_width=True)

    with tabs[1]:
        st.subheader("Monte Carlo Simulation")
        if st.button("Run Simulation"):
            with st.spinner(f"Running {mc_sims} simulations..."):
                initial = current_10y if not np.isnan(current_10y) else 4.0
                mu = df["10Y"].pct_change().mean() * 252 if "10Y" in df else 0
                sigma = df["10Y"].pct_change().std() * np.sqrt(252) if "10Y" in df else 0.1
                paths = MonteCarloSimulator.gbm(initial, mu, sigma, fc_horizon, mc_sims)
                results = MonteCarloSimulator.ci(paths)
                st.plotly_chart(plot_monte_carlo(initial, results, fc_horizon), use_container_width=True)

    with tabs[2]:
        st.subheader("Machine Learning Forecast")
        if st.button("Train Model"):
            with st.spinner("Training..."):
                X, y, _ = MLForecastModel.prepare_features(df[selected], lags=ml_lags)
                if len(X) > 50:
                    res = MLForecastModel.train(X, y)
                    st.metric("RMSE", f"{res.get('rmse',0)*100:.2f} bps")
                    st.metric("R²", f"{res.get('r2',0):.3f}")
                else:
                    st.warning(f"Need 50+ samples, have {len(X)}")

    with tabs[3]:
        st.subheader("Backtesting")
        if st.button("Run Backtest"):
            res = BacktestEngine.backtest(df, spreads)
            if res:
                st.metric("Total Return", f"{res['total_return']*100:.1f}%")
                st.metric("Sharpe Ratio", f"{res['sharpe']:.2f}")
                st.plotly_chart(plot_backtest(res), use_container_width=True)

    with tabs[4]:
        st.subheader("Volatility Analytics")
        if vol_df is not None and "^VIX" in vol_df:
            vix_regime = VolatilityAnalyzer.regime(vol_df["^VIX"])
            st.plotly_chart(plot_volatility(vol_df["^VIX"], vix_regime), use_container_width=True)
        else:
            st.info("VIX data unavailable")

    with tabs[5]:
        st.subheader("Cross-Asset Correlations")
        if corr_df is not None and not corr_df.empty and "10Y" in df:
            all_data = pd.concat([df["10Y"], corr_df], axis=1).dropna()
            all_data.columns = ["10Y"] + list(corr_df.columns)
            corr_matrix = all_data.pct_change().dropna().corr()
            st.plotly_chart(plot_correlation(corr_matrix), use_container_width=True)
        else:
            st.info("Correlation data unavailable")

    with tabs[6]:
        st.subheader("Technical Analysis")
        ticker = st.selectbox("Instrument", list(YAHOO_TICKERS.keys()))
        ohlc = ohlc_map.get(ticker)
        if ohlc is not None:
            st.plotly_chart(plot_technical(ohlc, ticker), use_container_width=True)
        else:
            st.warning(f"No data for {ticker}")

    with tabs[7]:
        st.subheader("Research")
        st.plotly_chart(plot_pca(pca_result), use_container_width=True)
        if pca_result and "loadings" in pca_result:
            st.dataframe(pca_result["loadings"], use_container_width=True)
        scenario_name = st.selectbox("Scenario", list(scenarios.keys()))
        st.plotly_chart(plot_scenario(scenarios[scenario_name], scenario_name), use_container_width=True)

    with tabs[8]:
        st.subheader("Export Data")
        st.download_button("Yield Data", df.to_csv().encode(), f"yield_{datetime.now():%Y%m%d}.csv")
        st.download_button("Spreads", spreads.to_csv().encode(), f"spreads_{datetime.now():%Y%m%d}.csv")

    st.markdown("---")
    st.markdown("<div style='text-align:center; color:#667085;'>Institutional Quantitative Platform | MK Istanbul Fintech LabGEN © 2026</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
