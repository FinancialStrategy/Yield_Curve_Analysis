import time
import warnings
from dataclasses import dataclass
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
    page_title="Dynamic Quantitative Analysis Model",
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
    "text": "#1a2a3a",
    "text_secondary": "#4a5a6a",
    "muted": "#667085",
    "accent": "#2c5f8a",
    "accent2": "#4a7c59",
    "accent3": "#c17f3a",
    "positive": "#2f855a",
    "negative": "#c05656",
    "warning": "#d48924",
    "recession": "rgba(120,130,145,0.18)",
    "band": "rgba(108,142,173,0.10)",
}

FRED_SERIES = {
    "1M": "DGS1MO", "3M": "DGS3MO", "6M": "DGS6MO", "1Y": "DGS1",
    "2Y": "DGS2", "3Y": "DGS3", "5Y": "DGS5", "7Y": "DGS7",
    "10Y": "DGS10", "20Y": "DGS20", "30Y": "DGS30",
}

MATURITY_MAP = {
    "1M": 1 / 12, "3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0,
    "3Y": 3.0, "5Y": 5.0, "7Y": 7.0, "10Y": 10.0, "20Y": 20.0, "30Y": 30.0,
}

YAHOO_TICKERS = {
    "^TNX": "10Y Treasury Yield Index",
    "^FVX": "5Y Treasury Yield Index",
    "^IRX": "13W T-Bill Index",
}

DEFAULT_STATE = {
    "api_key_validated": False,
    "api_key": "",
    "yield_data": None,
    "recession_data": None,
    "technical_map": None,
    "data_fetched": False,
}
for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown(
    f'''
    <style>
    .stApp {{
        background: linear-gradient(135deg, {COLORS["bg"]} 0%, {COLORS["bg2"]} 100%);
    }}
    .main-title-card {{
        background: linear-gradient(135deg, {COLORS["header"]} 0%, #2c4a6e 100%);
        border-radius: 20px; padding: 1.4rem 1.8rem; margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    }}
    .main-title {{ color: white; font-weight: 800; font-size: 1.6rem; margin: 0; }}
    .main-subtitle {{ color: rgba(255,255,255,0.86); font-size: 0.85rem; margin-top: 0.45rem; font-family: "Courier New", monospace; }}
    .metric-card {{
        background: {COLORS["surface"]}; border: 1px solid {COLORS["grid"]}; border-radius: 16px;
        padding: 1rem; min-height: 118px; box-shadow: 0 4px 16px rgba(16,24,40,0.06);
    }}
    .metric-label {{ color: {COLORS["text_secondary"]}; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; }}
    .metric-value {{ color: {COLORS["text"]}; font-size: 1.5rem; font-weight: 800; margin-top: 0.45rem; font-family: "Courier New", monospace; }}
    .metric-sub {{ color: {COLORS["muted"]}; font-size: 0.76rem; margin-top: 0.35rem; line-height: 1.35; }}
    .note-box {{
        background: {COLORS["surface_alt"]}; border: 1px solid {COLORS["grid"]}; border-left: 4px solid {COLORS["accent"]};
        border-radius: 12px; padding: 1rem 1.2rem; color: {COLORS["text"]}; font-size: 0.88rem; line-height: 1.55; margin: 1rem 0;
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 2px solid {COLORS["grid"]}; flex-wrap: wrap !important; }}
    .stTabs [data-baseweb="tab"] {{
        color: {COLORS["text_secondary"]}; font-weight: 700; font-size: 0.74rem; text-transform: uppercase;
        white-space: nowrap; padding: 8px 14px;
    }}
    .stTabs [aria-selected="true"] {{ color: {COLORS["accent"]}; border-bottom: 3px solid {COLORS["accent"]}; }}
    .stButton > button, .stDownloadButton > button {{
        background: {COLORS["surface"]}; color: {COLORS["text"]}; border: 1px solid {COLORS["grid"]};
        border-radius: 10px; font-weight: 700;
    }}
    #MainMenu, header, footer {{ visibility: hidden; }}
    </style>
    ''',
    unsafe_allow_html=True,
)

@dataclass
class RuntimeConfig:
    history_start: str = "1990-01-01"
    timeout: int = 25
    max_retries: int = 3
    cache_ttl_sec: int = 3600
    rolling_step: int = 21

CFG = RuntimeConfig()

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
            observations = r.json().get("observations", [])
            dates, values = [], []
            for row in observations:
                value = row.get("value")
                if value not in (".", None):
                    dates.append(pd.to_datetime(row["date"]))
                    values.append(float(value))
            return pd.Series(values, index=dates, name=series_id) if dates else None
        except Exception:
            if attempt == CFG.max_retries - 1:
                return None
            time.sleep(0.6 * (attempt + 1))
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def validate_fred_api_key(api_key: str) -> bool:
    if not api_key or len(api_key) < 10:
        return False
    s = fred_request(api_key, "DGS10")
    return s is not None and len(s) > 10

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_yield_data(api_key: str) -> Optional[pd.DataFrame]:
    series_map = {}
    for name, sid in FRED_SERIES.items():
        s = fred_request(api_key, sid)
        if s is not None:
            series_map[name] = s
    if not series_map:
        return None
    df = pd.DataFrame(series_map).sort_index().dropna(how="all")
    return df.dropna()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recession_data(api_key: str) -> Optional[pd.Series]:
    return fred_request(api_key, "USREC")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlc_data(ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
    if not YFINANCE_AVAILABLE:
        return None
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None

def sma(x, n): return x.rolling(n).mean()
def ema(x, n): return x.ewm(span=n, adjust=False).mean()
def rsi(x, n=14):
    delta = x.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
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

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["SMA_20"] = sma(out["Close"], 20)
    out["SMA_50"] = sma(out["Close"], 50)
    out["RSI"] = rsi(out["Close"], 14)
    out["MACD"], out["MACD_Signal"], out["MACD_Hist"] = macd(out["Close"])
    out["BB_Upper"], out["BB_Mid"], out["BB_Lower"] = bollinger_bands(out["Close"])
    return out

def compute_spreads(yield_df: pd.DataFrame) -> pd.DataFrame:
    spreads = pd.DataFrame(index=yield_df.index)
    if {"10Y", "2Y"}.issubset(yield_df.columns):
        spreads["10Y-2Y"] = (yield_df["10Y"] - yield_df["2Y"]) * 100
    if {"10Y", "3M"}.issubset(yield_df.columns):
        spreads["10Y-3M"] = (yield_df["10Y"] - yield_df["3M"]) * 100
    return spreads

def classify_regime(spreads: pd.DataFrame, yield_df: pd.DataFrame) -> Tuple[str, str]:
    if "10Y-2Y" not in spreads.columns or spreads.empty:
        return "Data Loading", "Please wait for the data to load."
    spread = spreads["10Y-2Y"].iloc[-1]
    y10 = yield_df["10Y"].iloc[-1] if "10Y" in yield_df.columns else np.nan
    if np.isfinite(spread) and spread < 0:
        return "Risk-off / Recession Watch", "Curve inversion signals a defensive macro regime."
    if np.isfinite(spread) and spread < 50:
        return "Neutral / Late Cycle", "Curve flattening suggests late-cycle caution."
    if np.isfinite(y10) and y10 > 5.5:
        return "Restrictive Long-End", "Positive slope but restrictive long-end rates."
    return "Risk-on / Expansion", "Positive slope supports pro-risk positioning."

def recession_probability_proxy(spreads: pd.DataFrame, yield_df: pd.DataFrame) -> float:
    if "10Y-2Y" not in spreads.columns or "10Y" not in yield_df.columns:
        return 0.50
    score = 0.0
    score += np.clip((-spreads["10Y-2Y"].iloc[-1]) / 100, 0, 1.5)
    if "10Y-3M" in spreads.columns:
        score += np.clip((-spreads["10Y-3M"].iloc[-1]) / 100, 0, 1.5)
    score += np.clip((yield_df["10Y"].iloc[-1] - 4.5) / 3.0, 0, 1.0)
    return float(1 / (1 + np.exp(-2.2 * (score - 0.8))))

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
        bounds = [(yields_.min() - 2, yields_.max() + 2), (-15, 15), (-15, 15), (0.01, 5)]
        best, best_fun = None, np.inf
        for _ in range(6):
            x0 = [np.random.uniform(a, b) for a, b in bounds]
            res = minimize(objective, x0=x0, method="L-BFGS-B", bounds=bounds)
            if res.success and res.fun < best_fun:
                best, best_fun = res, res.fun
        if best is None:
            return None
        fitted = NelsonSiegelModel.nelson_siegel(maturities, *best.x)
        sse = np.sum((yields_ - fitted) ** 2)
        sst = np.sum((yields_ - np.mean(yields_)) ** 2)
        return {"params": best.x, "fitted_values": fitted, "rmse": float(np.sqrt(np.mean((yields_ - fitted) ** 2))), "mae": float(np.mean(np.abs(yields_ - fitted))), "r_squared": float(1 - sse / sst) if sst > 0 else np.nan}

    @staticmethod
    def fit_nss(maturities: np.ndarray, yields_: np.ndarray):
        if len(maturities) == 0 or len(yields_) == 0:
            return None
        def objective(params):
            fitted = NelsonSiegelModel.nss(maturities, *params)
            weights = 1 / (maturities + 0.25)
            return np.sum(weights * (yields_ - fitted) ** 2)
        bounds = [(yields_.min() - 2, yields_.max() + 2), (-20, 20), (-20, 20), (-20, 20), (0.01, 10), (0.01, 10)]
        try:
            res = differential_evolution(objective, bounds=bounds, maxiter=120, popsize=10, polish=True, seed=42)
            if not res.success:
                return None
        except Exception:
            return None
        fitted = NelsonSiegelModel.nss(maturities, *res.x)
        sse = np.sum((yields_ - fitted) ** 2)
        sst = np.sum((yields_ - np.mean(yields_)) ** 2)
        return {"params": res.x, "fitted_values": fitted, "rmse": float(np.sqrt(np.mean((yields_ - fitted) ** 2))), "mae": float(np.mean(np.abs(yields_ - fitted))), "r_squared": float(1 - sse / sst) if sst > 0 else np.nan}

def model_governance(ns_result, nss_result):
    rows = []
    for name, result in [("NS", ns_result), ("NSS", nss_result)]:
        if result is None:
            continue
        rmse = result["rmse"]
        r2 = result["r_squared"]
        confidence = "High" if rmse < 0.05 and r2 > 0.98 else "Moderate" if rmse < 0.10 and r2 > 0.95 else "Low"
        rows.append({"Model": name, "RMSE": rmse, "MAE": result["mae"], "R2": r2, "FitConfidence": confidence})
    return pd.DataFrame(rows)

def factor_contributions(yield_df):
    out = pd.DataFrame(index=yield_df.index)
    if "10Y" in yield_df.columns:
        out["Level"] = yield_df["10Y"]
    if {"10Y", "3M"}.issubset(yield_df.columns):
        out["Slope"] = (yield_df["10Y"] - yield_df["3M"]) * 100
    if {"3M", "5Y", "10Y"}.issubset(yield_df.columns):
        out["Curvature"] = (2 * yield_df["5Y"] - (yield_df["3M"] + yield_df["10Y"])) * 100
    return out

def pca_risk_decomp(yield_df, n_components=3):
    returns = yield_df.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 20 or returns.shape[1] < 2:
        return None
    scaler = StandardScaler()
    X = scaler.fit_transform(returns)
    k = min(n_components, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=k)
    pca.fit(X)
    loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(k)], index=returns.columns)
    return {"explained_variance": pca.explained_variance_ratio_, "loadings": loadings}

def rolling_ns_parameters(yield_df, maturities, selected_cols, years):
    window_size = years * 252
    if len(yield_df) <= window_size + 5:
        return pd.DataFrame()
    rows = []
    for i in range(window_size, len(yield_df), CFG.rolling_step):
        curve = yield_df.iloc[i][selected_cols].values
        res = NelsonSiegelModel.fit_ns(maturities, curve)
        if res:
            rows.append({"date": yield_df.index[i], "beta0": res["params"][0], "beta1": res["params"][1], "beta2": res["params"][2], "lambda": res["params"][3], "rmse": res["rmse"]})
    return pd.DataFrame(rows)

class MonteCarloSimulator:
    @staticmethod
    def simulate_geometric_brownian_motion(initial_yield, mu, sigma, days, simulations=1000):
        dt = 1 / 252
        paths = np.zeros((simulations, days))
        paths[:, 0] = initial_yield
        for i in range(1, days):
            z = np.random.standard_normal(simulations)
            paths[:, i] = paths[:, i - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        return paths

    @staticmethod
    def simulate_vasicek(initial_rate, kappa, theta, sigma, days, simulations=1000):
        dt = 1 / 252
        paths = np.zeros((simulations, days))
        paths[:, 0] = initial_rate
        for i in range(1, days):
            z = np.random.standard_normal(simulations)
            dr = kappa * (theta - paths[:, i - 1]) * dt + sigma * np.sqrt(dt) * z
            paths[:, i] = paths[:, i - 1] + dr
        return paths

    @staticmethod
    def calculate_confidence_intervals(paths, confidence=0.95):
        low = (1 - confidence) / 2 * 100
        high = (1 + confidence) / 2 * 100
        return {"mean": np.mean(paths, axis=0), "lower_ci": np.percentile(paths, low, axis=0), "upper_ci": np.percentile(paths, high, axis=0)}

class MLForecastModel:
    @staticmethod
    def prepare_features(yield_df, lags=5):
        X, y = [], []
        for i in range(lags, len(yield_df) - 1):
            feats = []
            for col in yield_df.columns:
                feats.extend(yield_df[col].iloc[i-lags:i].values)
            X.append(feats)
            y.append(yield_df.iloc[i + 1].values)
        if not X:
            return np.array([]), np.array([])
        scaler = StandardScaler()
        return scaler.fit_transform(np.array(X)), np.array(y)

    @staticmethod
    def train_model(X, y, model_type="Random Forest"):
        if len(X) == 0:
            return {}
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if model_type == "Gradient Boosting":
            model = GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=3, random_state=42)
            model.fit(X_train, y_train[:, -1])
            pred = model.predict(X_test)
            eval_y = y_test[:, -1]
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            pred_full = model.predict(X_test)
            pred = pred_full[:, -1] if pred_full.ndim > 1 else pred_full
            eval_y = y_test[:, -1] if y_test.ndim > 1 else y_test
        return {"rmse": np.sqrt(mean_squared_error(eval_y, pred)), "mae": mean_absolute_error(eval_y, pred), "r2": r2_score(eval_y, pred)}

class BacktestEngine:
    @staticmethod
    def backtest_strategy(yield_df, spreads, strategy_type):
        if "10Y" not in yield_df.columns:
            return {}
        returns = yield_df["10Y"].pct_change().shift(-1)
        if strategy_type == "Curve Inversion":
            if "10Y-2Y" not in spreads.columns:
                return {}
            signals = spreads["10Y-2Y"] < 0
        else:
            sma50 = yield_df["10Y"].rolling(50).mean()
            signals = yield_df["10Y"] > sma50
        strategy_returns = signals.shift(1) * returns
        bh_returns = returns
        cumulative_strategy = (1 + strategy_returns.fillna(0)).cumprod()
        cumulative_bh = (1 + bh_returns.fillna(0)).cumprod()
        sharpe = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() > 0 else 0
        max_dd = (cumulative_strategy / cumulative_strategy.cummax() - 1).min()
        return {"cumulative_returns": cumulative_strategy, "buy_hold_returns": cumulative_bh, "total_return_strategy": cumulative_strategy.iloc[-1] - 1, "max_drawdown": max_dd, "sharpe_ratio_strategy": sharpe, "strategy_name": strategy_type}

def create_chart_layout(fig, title, y_title=None, height=460, x_title="Date"):
    fig.update_layout(template="plotly_white", paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["surface"], font=dict(size=12, color=COLORS["text"]), title=dict(text=title, x=0.01, font=dict(size=16)), margin=dict(l=60, r=30, t=80, b=50), hovermode="x unified", height=height, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_xaxes(title=x_title, gridcolor=COLORS["grid"], tickfont=dict(color=COLORS["text"], size=11))
    fig.update_yaxes(title=y_title, gridcolor=COLORS["grid"], tickfont=dict(color=COLORS["text"], size=11))
    return fig

def chart_current_curve(maturities, curve, ns_result, nss_result):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=maturities, y=curve, mode="lines+markers", name="Current Curve", line=dict(color=COLORS["accent"], width=2.5), marker=dict(size=10)))
    if ns_result:
        fig.add_trace(go.Scatter(x=maturities, y=ns_result["fitted_values"], mode="lines", name="NS Fit", line=dict(color=COLORS["positive"], width=2)))
    if nss_result:
        fig.add_trace(go.Scatter(x=maturities, y=nss_result["fitted_values"], mode="lines", name="NSS Fit", line=dict(color=COLORS["accent3"], width=2, dash="dash")))
    return create_chart_layout(fig, "Current Treasury Yield Curve", "Yield (%)", 430, "Maturity (Years)")

def chart_spread_history(spreads):
    fig = go.Figure()
    if "10Y-2Y" in spreads.columns:
        fig.add_trace(go.Scatter(x=spreads.index, y=spreads["10Y-2Y"], mode="lines", name="10Y-2Y", line=dict(color=COLORS["warning"], width=2)))
        fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"])
    return create_chart_layout(fig, "10Y-2Y Spread History", "Basis Points", 430)

def chart_dynamic_params(dynamic_df):
    if dynamic_df is None or dynamic_df.empty:
        return None
    fig = make_subplots(rows=2, cols=2, subplot_titles=("β0 Level", "β1 Slope", "β2 Curvature", "RMSE"))
    fig.add_trace(go.Scatter(x=dynamic_df["date"], y=dynamic_df["beta0"], mode="lines", name="β0", line=dict(color=COLORS["accent"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=dynamic_df["date"], y=dynamic_df["beta1"], mode="lines", name="β1", line=dict(color=COLORS["positive"])), row=1, col=2)
    fig.add_trace(go.Scatter(x=dynamic_df["date"], y=dynamic_df["beta2"], mode="lines", name="β2", line=dict(color=COLORS["accent3"])), row=2, col=1)
    fig.add_trace(go.Scatter(x=dynamic_df["date"], y=dynamic_df["rmse"], mode="lines", name="RMSE", line=dict(color=COLORS["muted"])), row=2, col=2)
    return create_chart_layout(fig, "Rolling Nelson-Siegel Parameters", height=620)

def chart_monte_carlo(initial_value, sim, horizon_days, title_prefix):
    fig = go.Figure()
    x = np.arange(horizon_days)
    fig.add_trace(go.Scatter(x=x, y=sim["upper_ci"], fill=None, mode="lines", line=dict(color="rgba(0,0,0,0)"), showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=sim["lower_ci"], fill="tonexty", mode="lines", fillcolor="rgba(44,95,138,0.20)", line=dict(color="rgba(0,0,0,0)"), name="Confidence Interval"))
    fig.add_trace(go.Scatter(x=x, y=sim["mean"], mode="lines", name="Mean Path", line=dict(color=COLORS["accent"], width=2.5)))
    fig.add_trace(go.Scatter(x=[0], y=[initial_value], mode="markers", name="Current", marker=dict(size=12, color=COLORS["positive"], symbol="star")))
    return create_chart_layout(fig, f"{title_prefix} | 10Y Yield Paths", "Yield (%)", 500, "Trading Days Ahead")

def chart_backtest_results(bt_res):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt_res["cumulative_returns"].index, y=bt_res["cumulative_returns"].values, mode="lines", name=bt_res["strategy_name"], line=dict(color=COLORS["accent"], width=2.5)))
    fig.add_trace(go.Scatter(x=bt_res["buy_hold_returns"].index, y=bt_res["buy_hold_returns"].values, mode="lines", name="Buy & Hold", line=dict(color=COLORS["muted"], width=2, dash="dash")))
    return create_chart_layout(fig, f"Backtest Performance: {bt_res['strategy_name']}", "Cumulative Return", 500)

def chart_factor_panel(factor_df):
    if factor_df is None or factor_df.empty:
        return None
    fig = go.Figure()
    palette = [COLORS["accent"], COLORS["positive"], COLORS["accent3"]]
    for i, col in enumerate(factor_df.columns):
        fig.add_trace(go.Scatter(x=factor_df.index, y=factor_df[col], mode="lines", name=col, line=dict(color=palette[i % len(palette)], width=2)))
    return create_chart_layout(fig, "Factor Contributions", "Value", 430)

def chart_pca_variance(pca_risk):
    if not pca_risk:
        return None
    ev = pca_risk["explained_variance"] * 100
    fig = go.Figure(go.Bar(x=[f"PC{i+1}" for i in range(len(ev))], y=ev, marker_color=COLORS["accent"]))
    return create_chart_layout(fig, "PCA Variance Explained", "Percent", 430, "Principal Components")

def chart_technical(ohlc_df, ticker):
    if ohlc_df is None or ohlc_df.empty:
        return None
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.52, 0.24, 0.24], subplot_titles=(f"{ticker} Price and Bands", "RSI (14)", "MACD"))
    fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["Close"], mode="lines", name="Close", line=dict(color=COLORS["accent"], width=2)), row=1, col=1)
    if "SMA_20" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["SMA_20"], mode="lines", name="SMA 20", line=dict(color=COLORS["positive"], width=1.5)), row=1, col=1)
    if "SMA_50" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["SMA_50"], mode="lines", name="SMA 50", line=dict(color=COLORS["warning"], width=1.5)), row=1, col=1)
    if "BB_Upper" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["BB_Upper"], mode="lines", name="BB Upper", line=dict(color=COLORS["muted"], width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["BB_Lower"], mode="lines", name="BB Lower", line=dict(color=COLORS["muted"], width=1), fill="tonexty", fillcolor=COLORS["band"]), row=1, col=1)
    if "RSI" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["RSI"], mode="lines", name="RSI", line=dict(color=COLORS["accent2"], width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=COLORS["negative"], row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=COLORS["positive"], row=2, col=1)
    if "MACD" in ohlc_df.columns:
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["MACD"], mode="lines", name="MACD", line=dict(color=COLORS["accent"], width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=ohlc_df.index, y=ohlc_df["MACD_Signal"], mode="lines", name="Signal", line=dict(color=COLORS["accent3"], width=1.5)), row=3, col=1)
    return create_chart_layout(fig, f"Technical Analysis | {ticker}", height=760)

def render_api_gate():
    st.markdown(f'<div class="note-box" style="max-width:560px; margin:40px auto; text-align:center;"><b>🔑 FRED API Key Required</b><br><br>This app uses live U.S. Treasury data from FRED. Enter your key below to begin.</div>', unsafe_allow_html=True)
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
            st.success("API key validated successfully.")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("Invalid API key. Please check and try again.")
    st.stop()

def kpi_card(label, value, sub):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-title-card"><div class="main-title">Dynamic Quantitative Analysis Model</div><div class="main-subtitle">Institutional Fixed-Income Platform | Stable Deploy Version</div></div>', unsafe_allow_html=True)

    if not st.session_state["api_key_validated"]:
        render_api_gate()

    with st.sidebar:
        st.markdown("### 🎛️ Control Tower")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            keep_key = st.session_state["api_key"]
            for key, value in DEFAULT_STATE.items():
                st.session_state[key] = value
            st.session_state["api_key"] = keep_key
            st.session_state["api_key_validated"] = True
            st.rerun()
        mc_model = st.selectbox("Simulation Model", ["Geometric Brownian Motion", "Vasicek Mean-Reverting"])
        mc_simulations = st.slider("Paths Generation", 500, 3000, 1000, 500)
        forecast_horizon = st.slider("Forecast Horizon (days)", 5, 252, 20)
        ml_model_type = st.selectbox("ML Algorithm", ["Random Forest", "Gradient Boosting"])
        ml_lags = st.slider("Autoregressive Lags", 3, 21, 5)
        bt_strategy = st.selectbox("Backtesting Strategy", ["Curve Inversion", "Macro Trend (50-Day SMA)"])
        rolling_years = st.slider("Rolling Window (years)", 2, 10, 5)
        ohlc_ticker = st.selectbox("Technical Analysis Ticker", list(YAHOO_TICKERS.keys()))

    if not st.session_state["data_fetched"]:
        with st.spinner("Fetching data..."):
            st.session_state["yield_data"] = fetch_all_yield_data(st.session_state["api_key"])
            st.session_state["recession_data"] = fetch_recession_data(st.session_state["api_key"])
            technical_map = {}
            for ticker in YAHOO_TICKERS:
                ohlc = fetch_ohlc_data(ticker, "2y")
                technical_map[ticker] = add_technical_indicators(ohlc) if ohlc is not None else None
            st.session_state["technical_map"] = technical_map
        if st.session_state["yield_data"] is None:
            st.error("Failed to fetch FRED data.")
            st.stop()
        st.session_state["data_fetched"] = True

    yield_df = st.session_state["yield_data"].copy()
    spreads = compute_spreads(yield_df)
    regime, regime_text = classify_regime(spreads, yield_df)
    recession_prob = recession_probability_proxy(spreads, yield_df)
    selected_cols = [c for c in yield_df.columns if c in MATURITY_MAP]
    maturities = np.array([MATURITY_MAP[c] for c in selected_cols], dtype=float)
    latest_curve = yield_df.iloc[-1][selected_cols].values.astype(float)
    current_2y = yield_df["2Y"].iloc[-1] if "2Y" in yield_df.columns else np.nan
    current_10y = yield_df["10Y"].iloc[-1] if "10Y" in yield_df.columns else np.nan
    current_spread = spreads["10Y-2Y"].iloc[-1] if "10Y-2Y" in spreads.columns else np.nan

    ns_result = NelsonSiegelModel.fit_ns(maturities, latest_curve)
    nss_result = NelsonSiegelModel.fit_nss(maturities, latest_curve)
    governance_df = model_governance(ns_result, nss_result)
    factor_df = factor_contributions(yield_df)
    pca_risk = pca_risk_decomp(yield_df[selected_cols])
    dynamic_df = rolling_ns_parameters(yield_df[selected_cols], maturities, selected_cols, rolling_years)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Macro Regime", regime, regime_text)
    with c2: kpi_card("2Y Yield", f"{current_2y:.2f}%" if np.isfinite(current_2y) else "N/A", "Short-end anchor")
    with c3: kpi_card("10Y Yield", f"{current_10y:.2f}%" if np.isfinite(current_10y) else "N/A", "Benchmark")
    with c4: kpi_card("10Y-2Y Spread", f"{current_spread:.1f} bps" if np.isfinite(current_spread) else "N/A", "Slope signal")
    with c5: kpi_card("Recession Prob.", f"{100 * recession_prob:.1f}%", "Proxy estimate")

    tabs = st.tabs(["Executive Summary", "Monte Carlo", "Machine Learning", "Backtesting", "Model Parameters", "Technical Analysis", "Export"])

    with tabs[0]:
        st.markdown(f'<div class="note-box"><b>Summary</b><br><br>The current regime is <b>{regime}</b>. The 10Y yield is <b>{current_10y:.2f}%</b>. This version is simplified to prioritize stability and visible outputs.</div>', unsafe_allow_html=True)
        left, right = st.columns([1,1])
        with left:
            st.plotly_chart(chart_current_curve(maturities, latest_curve, ns_result, nss_result), use_container_width=True)
        with right:
            st.plotly_chart(chart_spread_history(spreads), use_container_width=True)
        fig_dyn = chart_dynamic_params(dynamic_df)
        if fig_dyn:
            st.plotly_chart(fig_dyn, use_container_width=True)

    with tabs[1]:
        if st.button("Run Monte Carlo", use_container_width=True):
            initial_y = current_10y if np.isfinite(current_10y) else 4.0
            if mc_model == "Geometric Brownian Motion":
                mu = yield_df["10Y"].pct_change().dropna().mean() * 252 if "10Y" in yield_df.columns else 0
                sigma = yield_df["10Y"].pct_change().dropna().std() * np.sqrt(252) if "10Y" in yield_df.columns else 0.10
                paths = MonteCarloSimulator.simulate_geometric_brownian_motion(initial_y, mu, sigma, forecast_horizon, mc_simulations)
            else:
                theta = yield_df["10Y"].mean() if "10Y" in yield_df.columns else initial_y
                sigma_v = yield_df["10Y"].diff().dropna().std() * np.sqrt(252) if "10Y" in yield_df.columns else 0.10
                paths = MonteCarloSimulator.simulate_vasicek(initial_y, 0.5, theta, sigma_v, forecast_horizon, mc_simulations)
            sim = MonteCarloSimulator.calculate_confidence_intervals(paths)
            st.plotly_chart(chart_monte_carlo(initial_y, sim, forecast_horizon, mc_model), use_container_width=True)

    with tabs[2]:
        if st.button("Train ML Model", use_container_width=True):
            X, y = MLForecastModel.prepare_features(yield_df[selected_cols], ml_lags)
            if len(X) > 50:
                ml_res = MLForecastModel.train_model(X, y, ml_model_type)
                m1, m2, m3 = st.columns(3)
                m1.metric("RMSE", f"{ml_res.get('rmse', 0) * 100:.2f} bps")
                m2.metric("MAE", f"{ml_res.get('mae', 0) * 100:.2f} bps")
                m3.metric("R²", f"{ml_res.get('r2', 0):.3f}")
            else:
                st.warning("Not enough data vectors to train the model.")

    with tabs[3]:
        if st.button("Run Backtest", use_container_width=True):
            bt_res = BacktestEngine.backtest_strategy(yield_df, spreads, bt_strategy)
            if bt_res:
                b1, b2, b3 = st.columns(3)
                b1.metric("Total Return", f"{bt_res['total_return_strategy'] * 100:.1f}%")
                b2.metric("Max Drawdown", f"{bt_res['max_drawdown'] * 100:.2f}%")
                b3.metric("Sharpe Ratio", f"{bt_res['sharpe_ratio_strategy']:.2f}")
                st.plotly_chart(chart_backtest_results(bt_res), use_container_width=True)
            else:
                st.warning("Backtest could not be computed with current data.")

    with tabs[4]:
        sub = st.tabs(["NS", "NSS", "Governance", "Factor / PCA"])
        with sub[0]:
            if ns_result:
                st.dataframe(pd.DataFrame({"Parameter": ["β0", "β1", "β2", "λ", "RMSE", "R²"], "Value": [f"{ns_result['params'][0]:.4f}", f"{ns_result['params'][1]:.4f}", f"{ns_result['params'][2]:.4f}", f"{ns_result['params'][3]:.4f}", f"{ns_result['rmse']:.4f}", f"{ns_result['r_squared']:.4f}"]}), hide_index=True, use_container_width=True)
        with sub[1]:
            if nss_result:
                st.dataframe(pd.DataFrame({"Parameter": ["β0", "β1", "β2", "β3", "λ1", "λ2", "RMSE", "R²"], "Value": [f"{nss_result['params'][0]:.4f}", f"{nss_result['params'][1]:.4f}", f"{nss_result['params'][2]:.4f}", f"{nss_result['params'][3]:.4f}", f"{nss_result['params'][4]:.4f}", f"{nss_result['params'][5]:.4f}", f"{nss_result['rmse']:.4f}", f"{nss_result['r_squared']:.4f}"]}), hide_index=True, use_container_width=True)
        with sub[2]:
            if not governance_df.empty:
                st.dataframe(governance_df.round(4), hide_index=True, use_container_width=True)
        with sub[3]:
            left_f, right_f = st.columns([1,1])
            with left_f:
                fig_f = chart_factor_panel(factor_df)
                if fig_f:
                    st.plotly_chart(fig_f, use_container_width=True)
            with right_f:
                fig_p = chart_pca_variance(pca_risk)
                if fig_p:
                    st.plotly_chart(fig_p, use_container_width=True)
                if pca_risk:
                    st.dataframe(pca_risk["loadings"].round(4), use_container_width=True)

    with tabs[5]:
        if not YFINANCE_AVAILABLE:
            st.error("yfinance is not installed. Add it to requirements.txt and redeploy.")
        else:
            fig_t = chart_technical(st.session_state["technical_map"].get(ohlc_ticker), ohlc_ticker)
            if fig_t:
                st.plotly_chart(fig_t, use_container_width=True)
            else:
                st.info("Technical analysis data is not available for the selected ticker.")

    with tabs[6]:
        st.download_button("Export Yield Data (CSV)", yield_df.to_csv().encode("utf-8"), "yield_data.csv")
        st.download_button("Export Spread Data (CSV)", spreads.to_csv().encode("utf-8"), "spreads.csv")
        if not governance_df.empty:
            st.download_button("Export Governance Table (CSV)", governance_df.to_csv(index=False).encode("utf-8"), "governance.csv")

    st.markdown("---")
    st.markdown("<div style='text-align:center; color:#667085; font-size:0.75rem;'>Institutional Quantitative Platform | MK Istanbul Fintech LabGEN @2026</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
