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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# V34 INSTITUTIONAL ARCHITECTURE REFACTOR
# - lighter institutional background
# - executive view separated from research view
# - restrained palette and consistent charts
# - lazy compute sections and stronger cache discipline
# - preserves full app scope from prior versions
# =============================================================================

st.set_page_config(
    page_title="Yield Curve Analytics | Institutional v34",
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
    "accent_2": "#6c8ead",
    "positive": "#2f855a",
    "negative": "#c05656",
    "warning": "#b7791f",
    "recession": "rgba(120, 130, 145, 0.18)",
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

DEFAULT_STATE = {
    "api_key_validated": False,
    "api_key": "",
    "yield_data": None,
    "recession_data": None,
    "data_fetched": False,
    "ns_results": None,
    "nss_results": None,
    "dynamic_params": None,
    "factor_data": None,
    "pca_risk": None,
    "forecast_data": None,
    "arbitrage": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.markdown(
    f"""
    <style>
    .stApp {{ background: linear-gradient(180deg, {COLORS['bg']} 0%, {COLORS['bg2']} 100%); }}
    .main-title-card {{
        background: linear-gradient(90deg, {COLORS['header']} 0%, {COLORS['accent']} 100%);
        border-radius: 16px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 22px rgba(0,0,0,0.08);
    }}
    .main-title {{ color: white; font-weight: 700; font-size: 1.35rem; margin: 0; }}
    .main-sub {{ color: #dce6f2; font-size: 0.78rem; margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.06em; }}
    .metric-card {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['grid']};
        border-radius: 12px;
        padding: 0.95rem;
        min-height: 110px;
        box-shadow: 0 4px 14px rgba(16,24,40,0.04);
    }}
    .metric-label {{ color: {COLORS['muted']}; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }}
    .metric-value {{ color: {COLORS['text']}; font-size: 1.45rem; font-weight: 800; margin-top: 0.4rem; font-family: 'Courier New', monospace; }}
    .metric-sub {{ color: {COLORS['muted']}; font-size: 0.73rem; margin-top: 0.3rem; line-height: 1.25; }}
    .section-card {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['grid']};
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 4px 14px rgba(16,24,40,0.04);
    }}
    .section-title {{ color: {COLORS['text']}; font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.65rem; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid {COLORS['grid']}; }}
    .stTabs [data-baseweb="tab"] {{ color: {COLORS['muted']}; font-weight: 700; font-size: 0.72rem; text-transform: uppercase; }}
    .stTabs [aria-selected="true"] {{ color: {COLORS['accent']}; border-bottom: 2px solid {COLORS['accent']}; }}
    .stButton>button, .stDownloadButton>button {{
        background: {COLORS['surface']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['grid']};
        border-radius: 8px;
        font-weight: 700;
    }}
    .stButton>button:hover, .stDownloadButton>button:hover {{ border-color: {COLORS['accent']}; color: {COLORS['accent']}; }}
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
    for name, series_id in FRED_SERIES.items():
        s = fred_request(api_key, series_id)
        if s is not None:
            data[name] = s
    if not data:
        return None
    df = pd.DataFrame(data).sort_index().dropna(how="all")
    return df.dropna()


@st.cache_data(ttl=CFG.cache_ttl_sec, show_spinner=False)
def fetch_recession_data(api_key: str) -> Optional[pd.Series]:
    return fred_request(api_key, "USREC")


# =============================================================================
# ANALYTICS LAYER
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
            resid = yields_ - fitted
            return np.sum(resid ** 2)

        bounds = [
            (yields_.min() - 2, yields_.max() + 2),
            (-15, 15),
            (-15, 15),
            (0.01, 5),
        ]

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
        }

    @staticmethod
    def fit_nss(maturities: np.ndarray, yields_: np.ndarray):
        def objective(params):
            fitted = NelsonSiegelModel.nss(maturities, *params)
            weights = 1 / (maturities + 0.25)
            return np.sum(weights * (yields_ - fitted) ** 2)

        bounds = [
            (yields_.min() - 2, yields_.max() + 2),
            (-20, 20),
            (-20, 20),
            (-20, 20),
            (0.01, 10),
            (0.01, 10),
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
        }


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
# VISUAL LAYER
# =============================================================================

def add_recession_bands(fig: go.Figure, recessions: List[dict]) -> go.Figure:
    for rec in recessions:
        fig.add_vrect(
            x0=rec["start"], x1=rec["end"], fillcolor=COLORS["recession"], opacity=0.35, layer="below", line_width=0
        )
    return fig


def create_chart_layout(fig: go.Figure, title: str, y_title: Optional[str] = None, height: int = 460) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(size=11, color=COLORS["text"]),
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=15, color=COLORS["text"])),
        margin=dict(l=50, r=25, t=70, b=40),
        hovermode="x unified",
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    if y_title:
        fig.update_yaxes(title_text=y_title)
    return fig


def chart_2y(yield_df: pd.DataFrame) -> Optional[go.Figure]:
    if "2Y" not in yield_df.columns:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yield_df.index, y=yield_df["2Y"], mode="lines", name="2Y", line=dict(color=COLORS["warning"], width=2.2)))
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))
    return create_chart_layout(fig, "2-Year Treasury Yield", "Yield (%)")


def chart_10y(yield_df: pd.DataFrame) -> Optional[go.Figure]:
    if "10Y" not in yield_df.columns:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yield_df.index, y=yield_df["10Y"], mode="lines", name="10Y", line=dict(color=COLORS["accent"], width=2.2)))
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))
    return create_chart_layout(fig, "10-Year Treasury Yield", "Yield (%)")


def chart_spreads(spreads: pd.DataFrame, recessions: List[dict]) -> go.Figure:
    fig = go.Figure()
    palette = [COLORS["negative"], COLORS["accent"], COLORS["warning"], COLORS["positive"]]
    for i, col in enumerate(spreads.columns):
        fig.add_trace(go.Scatter(x=spreads.index, y=spreads[col], mode="lines", name=col, line=dict(color=palette[i % len(palette)], width=2)))
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"])
    add_recession_bands(fig, recessions)
    return create_chart_layout(fig, "Treasury Spread Dashboard", "bps", 520)


def chart_current_curve(maturities: np.ndarray, yields_: np.ndarray, ns_result: Optional[dict], nss_result: Optional[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=maturities, y=yields_, mode="markers+lines", name="Actual", marker=dict(size=10, color=COLORS["accent"]), line=dict(color=COLORS["accent_2"], width=2)))
    if ns_result:
        fig.add_trace(go.Scatter(x=maturities, y=ns_result["fitted_values"], mode="lines", name="NS", line=dict(color=COLORS["positive"], width=2.2)))
    if nss_result:
        fig.add_trace(go.Scatter(x=maturities, y=nss_result["fitted_values"], mode="lines", name="NSS", line=dict(color=COLORS["warning"], width=2.2, dash="dash")))
    return create_chart_layout(fig, "Current Treasury Curve and Model Fits", "Yield (%)", 440)


# =============================================================================
# UI HELPERS
# =============================================================================

def render_api_gate() -> None:
    st.markdown(
        f"""
        <div class="section-card" style="max-width:560px; margin:40px auto; text-align:center;">
            <div class="section-title">FRED API Key Required</div>
            <div style="color:{COLORS['muted']}; font-size:0.85rem;">
                This institutional dashboard uses live U.S. Treasury data from FRED.<br>
                Get a free API key from the official FRED documentation page.
            </div>
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
            <div class="main-title">Yield Curve Analytics | Institutional v34 Refactor</div>
            <div class="main-sub">Executive reporting • research diagnostics • lighter professional design • systematic section redesign</div>
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

    if not st.session_state.data_fetched:
        with st.spinner("Fetching Treasury and recession data from FRED..."):
            yield_df = fetch_all_yield_data(st.session_state.api_key)
            recession_series = fetch_recession_data(st.session_state.api_key)
        if yield_df is None:
            st.error("Failed to fetch FRED data.")
            st.stop()
        st.session_state.yield_data = yield_df
        st.session_state.recession_data = recession_series
        st.session_state.data_fetched = True

    yield_df = st.session_state.yield_data.copy()
    recession_series = st.session_state.recession_data.copy() if st.session_state.recession_data is not None else None
    selected_cols = [c for c in yield_df.columns if c in MATURITY_MAP]
    maturities = np.array([MATURITY_MAP[c] for c in selected_cols], dtype=float)
    latest_curve = yield_df.iloc[-1][selected_cols].values.astype(float)

    spreads = compute_spreads(yield_df)
    recessions = identify_recessions(recession_series)
    inversions = calculate_inversion_periods(spreads)
    lead_times = calculate_lead_times(inversions, recessions)
    regime, regime_text = classify_regime(spreads, yield_df)
    recession_prob = recession_probability_proxy(spreads, yield_df)

    current_2y = yield_df["2Y"].iloc[-1] if "2Y" in yield_df.columns else np.nan
    current_10y = yield_df["10Y"].iloc[-1] if "10Y" in yield_df.columns else np.nan
    current_30y = yield_df["30Y"].iloc[-1] if "30Y" in yield_df.columns else np.nan
    current_spread = spreads["10Y-2Y"].iloc[-1] if "10Y-2Y" in spreads.columns else np.nan

    # compute once
    ns_result = NelsonSiegelModel.fit_ns(maturities, latest_curve)
    nss_result = NelsonSiegelModel.fit_nss(maturities, latest_curve)
    dynamic_params = rolling_ns_parameters(yield_df[selected_cols], maturities, selected_cols, rolling_years)
    factor_df = factor_contributions(yield_df)
    pca_risk = pca_risk_decomp(yield_df[selected_cols])
    forecast_df = forecast_curve(yield_df[selected_cols], forecast_horizon)
    arb = arbitrage_diagnostics(yield_df[selected_cols], maturities)

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

    executive_tabs = st.tabs(["Executive View", "Research View", "Risk & Recession", "Forecast & Export"])

    with executive_tabs[0]:
        left, right = st.columns([1.15, 1])
        with left:
            st.plotly_chart(chart_current_curve(maturities, latest_curve, ns_result, nss_result), use_container_width=True)
        with right:
            summary_df = pd.DataFrame({
                "Metric": ["Regime", "2Y", "10Y", "30Y", "10Y-2Y", "10Y-3M", "Recession Probability"],
                "Value": [
                    regime,
                    f"{current_2y:.2f}%",
                    f"{current_10y:.2f}%",
                    f"{current_30y:.2f}%" if np.isfinite(current_30y) else "N/A",
                    f"{current_spread:.1f} bps" if np.isfinite(current_spread) else "N/A",
                    f"{spreads['10Y-3M'].iloc[-1]:.1f} bps" if '10Y-3M' in spreads.columns else 'N/A',
                    f"{100 * recession_prob:.1f}%",
                ],
                "Interpretation": [
                    regime_text,
                    "Short-end benchmark",
                    "Long-end benchmark",
                    "Ultra-long reference",
                    "Primary slope signal",
                    "Alternative recession signal",
                    "Internal probability proxy",
                ],
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        lower_left, lower_right = st.columns(2)
        with lower_left:
            fig2 = chart_2y(yield_df)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
        with lower_right:
            fig10 = chart_10y(yield_df)
            if fig10:
                st.plotly_chart(fig10, use_container_width=True)
        st.plotly_chart(chart_spreads(spreads, recessions), use_container_width=True)

    with executive_tabs[1]:
        research_tabs = st.tabs(["Data Table", "NS Model", "NSS Model", "Dynamic Analysis", "Factor & PCA"])
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
                        f"{ns_result['params'][0]:.4f}", f"{ns_result['params'][1]:.4f}", f"{ns_result['params'][2]:.4f}", f"{ns_result['params'][3]:.4f}",
                        f"{ns_result['rmse']:.4f}", f"{ns_result['mae']:.4f}", f"{ns_result['r_squared']:.4f}"
                    ]
                }), use_container_width=True, hide_index=True)
        with research_tabs[2]:
            if nss_result:
                st.dataframe(pd.DataFrame({
                    "Parameter": ["β0", "β1", "β2", "β3", "λ1", "λ2", "RMSE", "MAE", "R²"],
                    "Value": [
                        f"{nss_result['params'][0]:.4f}", f"{nss_result['params'][1]:.4f}", f"{nss_result['params'][2]:.4f}", f"{nss_result['params'][3]:.4f}",
                        f"{nss_result['params'][4]:.4f}", f"{nss_result['params'][5]:.4f}", f"{nss_result['rmse']:.4f}", f"{nss_result['mae']:.4f}", f"{nss_result['r_squared']:.4f}"
                    ]
                }), use_container_width=True, hide_index=True)
        with research_tabs[3]:
            if dynamic_params is not None and not dynamic_params.empty:
                fig = make_subplots(rows=2, cols=2, subplot_titles=("β0 Level", "β1 Slope", "β2 Curvature", "RMSE"))
                fig.add_trace(go.Scatter(x=dynamic_params['date'], y=dynamic_params['beta0'], mode='lines', name='β0', line=dict(color=COLORS['positive'])), row=1, col=1)
                fig.add_trace(go.Scatter(x=dynamic_params['date'], y=dynamic_params['beta1'], mode='lines', name='β1', line=dict(color=COLORS['accent'])), row=1, col=2)
                fig.add_trace(go.Scatter(x=dynamic_params['date'], y=dynamic_params['beta2'], mode='lines', name='β2', line=dict(color=COLORS['warning'])), row=2, col=1)
                fig.add_trace(go.Scatter(x=dynamic_params['date'], y=dynamic_params['rmse'], mode='lines', name='RMSE', line=dict(color=COLORS['muted'])), row=2, col=2)
                st.plotly_chart(create_chart_layout(fig, 'Rolling Nelson-Siegel Parameters', height=620), use_container_width=True)
        with research_tabs[4]:
            col_a, col_b = st.columns(2)
            with col_a:
                if factor_df is not None and not factor_df.empty:
                    fig = go.Figure()
                    palette = [COLORS['accent'], COLORS['warning'], COLORS['positive']]
                    for i, col in enumerate(factor_df.columns):
                        fig.add_trace(go.Scatter(x=factor_df.index, y=factor_df[col], mode='lines', name=col, line=dict(color=palette[i % len(palette)], width=1.8)))
                    st.plotly_chart(create_chart_layout(fig, 'Historical Factor Contributions', 'Value', 430), use_container_width=True)
            with col_b:
                if pca_risk:
                    fig = go.Figure()
                    ev = pca_risk['explained_variance'] * 100
                    fig.add_trace(go.Bar(x=[f'PC{i+1}' for i in range(len(ev))], y=ev, marker_color=COLORS['accent']))
                    st.plotly_chart(create_chart_layout(fig, 'PCA Variance Explained', 'Percent', 430), use_container_width=True)
                    st.dataframe(pca_risk['loadings'].round(4), use_container_width=True)

    with executive_tabs[2]:
        rr_tabs = st.tabs(["Risk Metrics", "Recession", "Lead Times", "Arbitrage"])
        with rr_tabs[0]:
            if '10Y' in yield_df.columns:
                risk = calculate_var_metrics(yield_df['10Y'].pct_change(), confidence_level, 10)
                if risk:
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric('Historical VaR', f"{risk['VaR_Historical']:.4f}")
                    r2.metric('Parametric VaR', f"{risk['VaR_Parametric']:.4f}")
                    r3.metric('Cornish-Fisher VaR', f"{risk['VaR_CornishFisher']:.4f}")
                    r4.metric('CVaR', f"{risk['CVaR']:.4f}")
        with rr_tabs[1]:
            fig = go.Figure()
            if '10Y-2Y' in spreads.columns:
                fig.add_trace(go.Scatter(x=spreads.index, y=spreads['10Y-2Y'], mode='lines', name='10Y-2Y', line=dict(color=COLORS['negative'], width=2)))
            fig.add_hline(y=0, line_dash='dash', line_color=COLORS['muted'])
            add_recession_bands(fig, recessions)
            st.plotly_chart(create_chart_layout(fig, '10Y-2Y vs NBER Recessions', 'bps', 500), use_container_width=True)
            if recessions:
                st.dataframe(pd.DataFrame(recessions), use_container_width=True, hide_index=True)
        with rr_tabs[2]:
            if inversions:
                st.dataframe(pd.DataFrame(inversions), use_container_width=True, hide_index=True)
            if lead_times:
                st.dataframe(pd.DataFrame(lead_times), use_container_width=True, hide_index=True)
        with rr_tabs[3]:
            if arb:
                a1, a2, a3 = st.columns(3)
                a1.metric('Mean Abs Error', f"{arb['mean_abs_error']:.4f}")
                a2.metric('Max Abs Error', f"{arb['max_abs_error']:.4f}")
                a3.metric('Mispriced Tenors', f"{arb['mispriced_count']}")
                if not arb['mispriced_table'].empty:
                    st.dataframe(arb['mispriced_table'].round(4), use_container_width=True)
                else:
                    st.info('No material mispricing detected under current threshold.')

    with executive_tabs[3]:
        fx_tabs = st.tabs(["Forecasting", "Export"])
        with fx_tabs[0]:
            if forecast_df is not None and not forecast_df.empty:
                selected = st.multiselect('Select tenors', selected_cols, default=['2Y', '10Y'] if {'2Y', '10Y'}.issubset(set(selected_cols)) else selected_cols[:2])
                fig = go.Figure()
                for col in selected:
                    fig.add_trace(go.Scatter(x=yield_df.index[-200:], y=yield_df[col].tail(200), mode='lines', name=f'{col} history', line=dict(width=1.7)))
                    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[col], mode='lines', name=f'{col} forecast', line=dict(width=2, dash='dash')))
                st.plotly_chart(create_chart_layout(fig, 'Linear Trend Yield Forecast', 'Yield (%)', 500), use_container_width=True)
                st.dataframe(forecast_df.head(10).round(3), use_container_width=True)
            else:
                st.warning('Insufficient data for forecasting.')
        with fx_tabs[1]:
            st.download_button('Download Yield Data (CSV)', yield_df.to_csv().encode('utf-8'), f"yield_data_{datetime.now():%Y%m%d_%H%M%S}.csv", 'text/csv')
            st.download_button('Download Spread Data (CSV)', spreads.to_csv().encode('utf-8'), f"spreads_{datetime.now():%Y%m%d_%H%M%S}.csv", 'text/csv')
            if dynamic_params is not None and not dynamic_params.empty:
                st.download_button('Download Dynamic Parameters', dynamic_params.to_csv(index=False).encode('utf-8'), f"dynamic_params_{datetime.now():%Y%m%d_%H%M%S}.csv", 'text/csv')
            if pca_risk:
                st.download_button('Download PCA Loadings', pca_risk['loadings'].to_csv().encode('utf-8'), f"pca_loadings_{datetime.now():%Y%m%d_%H%M%S}.csv", 'text/csv')
            if lead_times:
                st.download_button('Download Lead Times', pd.DataFrame(lead_times).to_csv(index=False).encode('utf-8'), f"lead_times_{datetime.now():%Y%m%d_%H%M%S}.csv", 'text/csv')

    st.markdown('---')
    st.markdown(
        f"<div style='text-align:center; color:{COLORS['muted']}; font-size:0.72rem;'>Institutional v34 refactor • FRED data • executive and research layers separated • generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
