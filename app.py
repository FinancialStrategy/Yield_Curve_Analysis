# Institutional merged Streamlit code placeholder
# The previous attempt generated a full merged script but the environment reset before persisting.
# Rewriting the merged file now.

# =============================================================================
# HEDGE FUND YIELD CURVE ANALYTICS PLATFORM
# INSTITUTIONAL MERGED EDITION
# =============================================================================
# Version: 30.0 | Institutional Merge of v19 + v27
# =============================================================================

import time
import warnings
from datetime import datetime
from typing import List, Optional

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

st.set_page_config(page_title="Yield Curve Analytics | Institutional Hedge Fund Platform",
                   page_icon="📈", layout="wide", initial_sidebar_state="expanded")

COLORS = {
    "bg": "#07111f", "surface": "#101826", "surface_2": "#172132", "grid": "#2a3447",
    "text": "#e8edf5", "muted": "#aab6c7", "line": "#8ea3b9", "accent": "#d4af37",
    "positive": "#6aa381", "negative": "#b56b6b", "warning": "#c49a5a",
    "recession": "rgba(180, 190, 205, 0.18)"
}

FRED_SERIES = {"1M":"DGS1MO","3M":"DGS3MO","6M":"DGS6MO","1Y":"DGS1","2Y":"DGS2","3Y":"DGS3","5Y":"DGS5","7Y":"DGS7","10Y":"DGS10","20Y":"DGS20","30Y":"DGS30"}
MATURITY_MAP = {"1M":1/12,"3M":0.25,"6M":0.5,"1Y":1,"2Y":2,"3Y":3,"5Y":5,"7Y":7,"10Y":10,"20Y":20,"30Y":30}
YAHOO_TICKERS = {"^TNX":"10Y Treasury Yield Index","^FVX":"5Y Treasury Yield Index","^IRX":"13W T-Bill Index","TLT":"20+Y Treasury Bond ETF","IEF":"7-10Y Treasury Bond ETF","SHY":"1-3Y Treasury Bond ETF"}

for key, default in {
    "api_key_validated": False, "yield_data": None, "recession_data": None, "ohlc_data": None,
    "data_fetched": False, "ns_results": None, "nss_results": None, "dynamic_params": None,
    "factor_contributions": None, "pca_risk": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown(f'''
<style>
[data-testid="stAppViewContainer"] {{
  background: linear-gradient(180deg, {COLORS["bg"]} 0%, #0b1320 100%);
}}
.hf-header {{
  background: linear-gradient(90deg, {COLORS["surface"]} 0%, {COLORS["surface_2"]} 100%);
  border: 1px solid {COLORS["grid"]};
  border-left: 4px solid {COLORS["accent"]};
  border-radius: 14px; padding: 1.25rem 1.4rem; margin-bottom: 1rem;
}}
.metric-card {{
  background: linear-gradient(180deg, {COLORS["surface"]}, {COLORS["surface_2"]});
  border: 1px solid {COLORS["grid"]}; border-radius: 12px; padding: 0.95rem; min-height: 108px;
}}
.metric-label {{ color: {COLORS["muted"]}; font-size: 0.68rem; text-transform: uppercase; font-weight: 700; }}
.metric-value {{ color: {COLORS["text"]}; font-size: 1.5rem; font-weight: 800; margin-top: 0.42rem; font-family: "Courier New", monospace; }}
.metric-sub {{ color: {COLORS["muted"]}; font-size: 0.73rem; margin-top: 0.3rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 0rem; background-color: {COLORS["surface"]}; border-bottom: 1px solid {COLORS["grid"]}; }}
.stTabs [data-baseweb="tab"] {{ color: {COLORS["muted"]}; font-size: 0.70rem; font-weight: 700; text-transform: uppercase; padding: 0.65rem 0.95rem; }}
.stTabs [aria-selected="true"] {{ color: {COLORS["accent"]}; border-bottom: 2px solid {COLORS["accent"]}; }}
#MainMenu, header, footer {{ visibility: hidden; }}
</style>
''', unsafe_allow_html=True)

def fred_request(api_key: str, series_id: str) -> Optional[pd.Series]:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json", "observation_start": "1990-01-01", "sort_order": "asc"}
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=25)
            r.raise_for_status()
            payload = r.json()
            obs = payload.get("observations", [])
            dates, vals = [], []
            for item in obs:
                val = item.get("value")
                if val not in (".", None):
                    dates.append(pd.to_datetime(item["date"]))
                    vals.append(float(val))
            return pd.Series(vals, index=dates, name=series_id) if dates else None
        except Exception:
            if attempt == 2:
                return None
            time.sleep(0.7 * (attempt + 1))
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def validate_fred_api_key(api_key: str) -> bool:
    if not api_key or len(api_key) < 10:
        return False
    s = fred_request(api_key, "DGS10")
    return s is not None and len(s) > 10

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_yield_data(api_key: str) -> Optional[pd.DataFrame]:
    data = {}
    for name, sid in FRED_SERIES.items():
        s = fred_request(api_key, sid)
        if s is not None:
            data[name] = s
    return pd.DataFrame(data).dropna() if data else None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recession_data(api_key: str) -> Optional[pd.Series]:
    return fred_request(api_key, "USREC")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlc_data(ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
        return df if df is not None and not df.empty else None
    except Exception:
        return None

def sma(x, n): return x.rolling(n).mean()
def ema(x, n): return x.ewm(span=n, adjust=False).mean()
def rsi(x, n=14):
    d = x.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    rs = g / l
    return 100 - (100 / (1 + rs))
def macd(x):
    m = ema(x, 12) - ema(x, 26)
    s = ema(m, 9)
    return m, s, m - s
def bb(x, n=20, k=2):
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
    out["BB_Upper"], out["BB_Middle"], out["BB_Lower"] = bb(out["Close"])
    return out

class NelsonSiegelModel:
    @staticmethod
    def nelson_siegel(tau, beta0, beta1, beta2, lambda1):
        tau = np.asarray(tau, dtype=float)
        ts = np.where(tau == 0, 1e-8, tau)
        x = lambda1 * ts
        t1 = (1 - np.exp(-x)) / x
        t2 = t1 - np.exp(-x)
        out = beta0 + beta1 * t1 + beta2 * t2
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
    def fit_nelson_siegel(maturities, yields):
        def objective(p):
            f = NelsonSiegelModel.nelson_siegel(maturities, *p)
            return np.sum((yields - f) ** 2)
        bounds = [(yields.min() - 2, yields.max() + 2), (-15, 15), (-15, 15), (0.01, 5)]
        best, best_fun = None, np.inf
        for _ in range(8):
            x0 = [np.random.uniform(a, b) for a, b in bounds]
            res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
            if res.success and res.fun < best_fun:
                best, best_fun = res, res.fun
        if best is None:
            return None
        fitted = NelsonSiegelModel.nelson_siegel(maturities, *best.x)
        sse = np.sum((yields - fitted) ** 2)
        sst = np.sum((yields - np.mean(yields)) ** 2)
        return {"params": best.x, "fitted_values": fitted, "rmse": float(np.sqrt(np.mean((yields - fitted) ** 2))), "mae": float(np.mean(np.abs(yields - fitted))), "r_squared": float(1 - sse / sst) if sst > 0 else np.nan}

    @staticmethod
    def fit_svensson(maturities, yields):
        def objective(p):
            f = NelsonSiegelModel.nss(maturities, *p)
            w = 1 / (maturities + 0.25)
            return np.sum(w * (yields - f) ** 2)
        bounds = [(yields.min() - 2, yields.max() + 2), (-20, 20), (-20, 20), (-20, 20), (0.01, 10), (0.01, 10)]
        res = differential_evolution(objective, bounds=bounds, maxiter=180, popsize=10, polish=True, seed=42)
        if not res.success:
            return None
        fitted = NelsonSiegelModel.nss(maturities, *res.x)
        sse = np.sum((yields - fitted) ** 2)
        sst = np.sum((yields - np.mean(yields)) ** 2)
        return {"params": res.x, "fitted_values": fitted, "rmse": float(np.sqrt(np.mean((yields - fitted) ** 2))), "mae": float(np.mean(np.abs(yields - fitted))), "r_squared": float(1 - sse / sst) if sst > 0 else np.nan}

class DynamicParameterAnalysis:
    @staticmethod
    def calibrate_rolling_window(yield_df, maturities, selected_cols, window_years=5):
        window_size = window_years * 252
        if len(yield_df) <= window_size + 5:
            return pd.DataFrame()
        out = []
        for i in range(window_size, len(yield_df), 21):
            curve = yield_df.iloc[i][selected_cols].values
            res = NelsonSiegelModel.fit_nelson_siegel(maturities, curve)
            if res:
                out.append({"date": yield_df.index[i], "beta0": res["params"][0], "beta1": res["params"][1], "beta2": res["params"][2], "lambda": res["params"][3], "rmse": res["rmse"]})
        return pd.DataFrame(out)

    @staticmethod
    def calculate_factor_contributions(yield_df):
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

class AdvancedRiskMetrics:
    @staticmethod
    def calculate_pca_risk(yield_df, n_components=3):
        returns = yield_df.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if len(returns) < 10 or returns.shape[1] < 2:
            return None
        scaler = StandardScaler()
        x = scaler.fit_transform(returns)
        k = min(n_components, len(returns.columns), len(returns) - 1)
        pca = PCA(n_components=k)
        pcs = pca.fit_transform(x)
        loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(k)], index=yield_df.columns)
        factors = pd.DataFrame(pcs, index=returns.index, columns=[f"PC{i+1}" for i in range(k)])
        return {"explained_variance": pca.explained_variance_ratio_, "cumulative_variance": np.cumsum(pca.explained_variance_ratio_), "loadings": loadings, "factors": factors}

    @staticmethod
    def calculate_var_metrics(returns, confidence=0.95, horizon=10):
        returns = returns.dropna()
        if len(returns) < 20:
            return None
        var_hist = np.percentile(returns, (1 - confidence) * 100)
        cvar = returns[returns <= var_hist].mean()
        var_param = norm.ppf(1 - confidence) * returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()
        z = norm.ppf(1 - confidence)
        zcf = z + (z**2 - 1) * skew / 6 + (z**3 - 3 * z) * kurt / 24 - (2 * z**3 - 5 * z) * skew**2 / 36
        return {"VaR_Historical": float(var_hist * np.sqrt(horizon)), "VaR_Parametric": float(var_param * np.sqrt(horizon)), "VaR_CornishFisher": float(zcf * returns.std() * np.sqrt(horizon)), "CVaR": float(cvar * np.sqrt(horizon)), "Skewness": float(skew), "Kurtosis": float(kurt)}

class YieldCurveForecasting:
    @staticmethod
    def forecast_linear(yield_df, horizon=20):
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

class ArbitrageDetection:
    @staticmethod
    def detect_arbitrage_opportunities(yield_df, maturities):
        latest_yields = yield_df.iloc[-1].values[:len(maturities)]
        nss = NelsonSiegelModel.fit_svensson(maturities, latest_yields)
        if nss is None:
            return None
        theo = nss["fitted_values"]
        resid = latest_yields - theo
        rows = []
        for i, (m, r) in enumerate(zip(maturities, resid)):
            if abs(r) > 0.10:
                rows.append({"Tenor": m, "Actual": latest_yields[i], "Theoretical": theo[i], "Difference": r, "Signal": "Undervalued" if r > 0 else "Overvalued"})
        return {"mean_abs_error": float(np.mean(np.abs(resid))), "max_abs_error": float(np.max(np.abs(resid))), "std_error": float(np.std(resid)), "mispriced_count": len(rows), "mispriced_table": pd.DataFrame(rows) if rows else pd.DataFrame()}

class NBERRecessionAnalysis:
    @staticmethod
    def identify_recessions(recession_series):
        if recession_series is None or len(recession_series) == 0:
            return []
        recs, in_rec, start = [], False, None
        for date, value in recession_series.dropna().items():
            if value == 1 and not in_rec:
                in_rec, start = True, date
            elif value == 0 and in_rec:
                recs.append({"start": start, "end": date, "duration_days": (date - start).days, "duration_months": (date - start).days / 30.44})
                in_rec = False
        return recs

    @staticmethod
    def calculate_inversion_periods(spreads):
        if "10Y-2Y" not in spreads.columns:
            return []
        s = spreads["10Y-2Y"].dropna()
        out, in_inv, start = [], False, None
        for date, value in s.items():
            if value < 0 and not in_inv:
                in_inv, start = True, date
            elif value >= 0 and in_inv:
                seg = s.loc[start:date]
                out.append({"start": start, "end": date, "depth": seg.min(), "duration_days": (date - start).days, "duration_months": (date - start).days / 30.44})
                in_inv = False
        return out

    @staticmethod
    def calculate_lead_times(inversion_periods, recessions):
        rows = []
        for inv in inversion_periods:
            for rec in recessions:
                if inv["start"] < rec["start"]:
                    lead_days = (rec["start"] - inv["start"]).days
                    rows.append({"inversion_start": inv["start"], "recession_start": rec["start"], "lead_days": lead_days, "lead_months": lead_days / 30.44, "inversion_depth": inv["depth"]})
                    break
        return rows

def compute_spreads(yield_df):
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

def classify_regime(spreads, yield_df):
    spread = spreads["10Y-2Y"].iloc[-1] if "10Y-2Y" in spreads.columns else np.nan
    y10 = yield_df["10Y"].iloc[-1] if "10Y" in yield_df.columns else np.nan
    if np.isfinite(spread) and spread < 0:
        return "Risk-off", "Curve inversion implies a defensive macro regime."
    if np.isfinite(spread) and spread < 50:
        return "Neutral / Late Cycle", "Curve flattening suggests late-cycle caution."
    if np.isfinite(y10) and y10 > 5.5:
        return "Neutral / Restrictive", "Positive slope but restrictive long-end rates."
    return "Risk-on", "Positive slope supports pro-risk macro positioning."

def recession_probability_proxy(spreads, yield_df):
    score = 0.0
    if "10Y-2Y" in spreads.columns:
        score += np.clip((-spreads["10Y-2Y"].iloc[-1]) / 100, 0, 1.5)
    if "10Y-3M" in spreads.columns:
        score += np.clip((-spreads["10Y-3M"].iloc[-1]) / 100, 0, 1.5)
    if "10Y" in yield_df.columns:
        score += np.clip((yield_df["10Y"].iloc[-1] - 4.5) / 3, 0, 1.0)
    return float(1 / (1 + np.exp(-2.2 * (score - 0.8))))

def add_recession_bands(fig, recessions):
    for rec in recessions:
        fig.add_vrect(x0=rec["start"], x1=rec["end"], fillcolor=COLORS["recession"], opacity=0.35, layer="below", line_width=0)
    return fig

def create_institutional_layout(fig, title, y_title=None, height=500):
    fig.update_layout(template="plotly_dark", paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["surface"],
                      font=dict(family="Arial", size=11, color=COLORS["muted"]),
                      title=dict(text=title, x=0.01, xanchor="left", font=dict(size=15, color=COLORS["text"])),
                      margin=dict(l=50, r=25, t=70, b=40), hovermode="x unified", height=height,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    if y_title:
        fig.update_yaxes(title_text=y_title)
    return fig

def plot_2y_yield_chart(yield_df):
    if "2Y" not in yield_df.columns:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yield_df.index, y=yield_df["2Y"], mode="lines", name="2Y", line=dict(color=COLORS["warning"], width=2.4)))
    fig.update_layout(xaxis=dict(rangeselector=dict(buttons=[
        dict(count=1, label="1M", step="month", stepmode="backward"),
        dict(count=3, label="3M", step="month", stepmode="backward"),
        dict(count=6, label="6M", step="month", stepmode="backward"),
        dict(count=1, label="1Y", step="year", stepmode="backward"),
        dict(step="all", label="ALL"),
    ]), rangeslider=dict(visible=True), type="date"))
    return create_institutional_layout(fig, "2-Year Treasury Yield", "Yield (%)", 420)

def plot_10y_yield_chart(yield_df):
    if "10Y" not in yield_df.columns:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yield_df.index, y=yield_df["10Y"], mode="lines", name="10Y", line=dict(color=COLORS["line"], width=2.4)))
    fig.update_layout(xaxis=dict(rangeselector=dict(buttons=[
        dict(count=1, label="1M", step="month", stepmode="backward"),
        dict(count=3, label="3M", step="month", stepmode="backward"),
        dict(count=6, label="6M", step="month", stepmode="backward"),
        dict(count=1, label="1Y", step="year", stepmode="backward"),
        dict(step="all", label="ALL"),
    ]), rangeslider=dict(visible=True), type="date"))
    return create_institutional_layout(fig, "10-Year Treasury Yield", "Yield (%)", 420)

def create_ohlc_candlestick_chart(ohlc_data, ticker, title):
    if ohlc_data is None or ticker not in ohlc_data:
        return None
    data = ohlc_data[ticker]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
                                 increasing=dict(line=dict(color=COLORS["positive"]), fillcolor=COLORS["positive"]),
                                 decreasing=dict(line=dict(color=COLORS["negative"]), fillcolor=COLORS["negative"]),
                                 showlegend=False))
    if "SMA_20" in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA_20"], mode="lines", name="SMA 20", line=dict(color=COLORS["line"], width=1.2)))
    if "SMA_50" in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA_50"], mode="lines", name="SMA 50", line=dict(color=COLORS["accent"], width=1.2)))
    fig.update_layout(xaxis=dict(rangeselector=dict(buttons=[
        dict(count=1, label="1M", step="month", stepmode="backward"),
        dict(count=3, label="3M", step="month", stepmode="backward"),
        dict(count=6, label="6M", step="month", stepmode="backward"),
        dict(count=1, label="1Y", step="year", stepmode="backward"),
        dict(step="all", label="ALL"),
    ]), rangeslider=dict(visible=True), type="date"))
    return create_institutional_layout(fig, title, "Price", 520)

def plot_ohlc_comparison_chart(ohlc_data, tickers_to_compare):
    fig = go.Figure()
    palette = [COLORS["line"], COLORS["accent"], COLORS["positive"], COLORS["warning"], COLORS["negative"], COLORS["muted"]]
    j = 0
    for ticker in tickers_to_compare:
        if ticker in ohlc_data and not ohlc_data[ticker].empty:
            data = ohlc_data[ticker]
            normed = (data["Close"] / data["Close"].iloc[0] - 1) * 100
            fig.add_trace(go.Scatter(x=data.index, y=normed, mode="lines", name=ticker, line=dict(color=palette[j % len(palette)], width=2)))
            j += 1
    fig.update_layout(xaxis=dict(rangeselector=dict(buttons=[
        dict(count=1, label="1M", step="month", stepmode="backward"),
        dict(count=6, label="6M", step="month", stepmode="backward"),
        dict(count=1, label="1Y", step="year", stepmode="backward"),
        dict(step="all", label="ALL"),
    ]), rangeslider=dict(visible=True), type="date"))
    return create_institutional_layout(fig, "OHLC Instrument Comparison (Normalized)", "Return (%)", 480)

def create_technical_indicators_chart(ohlc_data, ticker, title):
    if ohlc_data is None or ticker not in ohlc_data:
        return None
    data = ohlc_data[ticker]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25], subplot_titles=(title, "RSI (14)", "MACD"))
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close", line=dict(color=COLORS["line"], width=2)), row=1, col=1)
    if "SMA_20" in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA_20"], mode="lines", name="SMA 20", line=dict(color=COLORS["accent"], width=1.2)), row=1, col=1)
    if "SMA_50" in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA_50"], mode="lines", name="SMA 50", line=dict(color=COLORS["warning"], width=1.2)), row=1, col=1)
    if "RSI" in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], mode="lines", name="RSI", line=dict(color=COLORS["line"], width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=COLORS["negative"], row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=COLORS["positive"], row=2, col=1)
    if "MACD" in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], mode="lines", name="MACD", line=dict(color=COLORS["positive"], width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["MACD_Signal"], mode="lines", name="Signal", line=dict(color=COLORS["negative"], width=1.3)), row=3, col=1)
    return create_institutional_layout(fig, title, height=760)

def render_api_key_input():
    st.markdown(f'''
    <div style="background:{COLORS["surface"]}; border:1px solid {COLORS["grid"]}; border-radius:14px; padding:20px; max-width:560px; margin:40px auto; text-align:center;">
        <h3 style="color:{COLORS["text"]};">FRED API Key Required</h3>
        <p style="color:{COLORS["muted"]};">This institutional dashboard uses live U.S. Treasury data from FRED.</p>
        <p><a href="https://fred.stlouisfed.org/docs/api/api_key.html" target="_blank">Get a free FRED API key</a></p>
    </div>
    ''', unsafe_allow_html=True)
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
            st.success("API key validated successfully.")
            time.sleep(0.6)
            st.rerun()
        else:
            st.error("Invalid API key. Please check and try again.")
    st.stop()

def main():
    st.markdown('''
    <div class="hf-header">
        <div style="color:white; font-weight:700; font-size:1.28rem; margin:0;">Yield Curve Analytics | Institutional Hedge Fund Platform</div>
        <div style="color:#aab6c7; margin-top:6px; font-size:0.76rem; text-transform:uppercase;">FRED Data | Nelson-Siegel Family | Executive Summary | OHLC | PCA | Arbitrage | Recession</div>
    </div>
    ''', unsafe_allow_html=True)

    if not st.session_state.api_key_validated:
        render_api_key_input()

    with st.sidebar:
        st.markdown("### Control Tower")
        st.caption("Institutional fixed-income macro monitoring")
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            for key, val in {"yield_data": None, "recession_data": None, "ohlc_data": None, "data_fetched": False, "ns_results": None, "nss_results": None, "dynamic_params": None, "factor_contributions": None, "pca_risk": None}.items():
                st.session_state[key] = val
            st.rerun()
        window_years = st.slider("Rolling window (years)", 2, 10, 5)
        forecast_horizon = st.slider("Forecast horizon (business days)", 5, 60, 20)
        confidence_level = st.slider("VaR confidence", 0.90, 0.99, 0.95, 0.01)
        ohlc_period = st.selectbox("OHLC period", ["6mo", "1y", "2y", "5y"], index=2)

    if not st.session_state.data_fetched:
        with st.spinner("Fetching Treasury, recession, and OHLC data..."):
            yield_df = fetch_all_yield_data(st.session_state.api_key)
            recession_series = fetch_recession_data(st.session_state.api_key)
            ohlc_raw = {}
            for ticker in YAHOO_TICKERS.keys():
                df = fetch_ohlc_data(ticker, ohlc_period)
                if df is not None and not df.empty:
                    ohlc_raw[ticker] = add_technical_indicators(df)
        if yield_df is None:
            st.error("Failed to fetch FRED data.")
            st.stop()
        st.session_state.yield_data = yield_df
        st.session_state.recession_data = recession_series
        st.session_state.ohlc_data = ohlc_raw
        st.session_state.data_fetched = True

    yield_df = st.session_state.yield_data.copy()
    recession_series = st.session_state.recession_data.copy() if st.session_state.recession_data is not None else None
    ohlc_data = st.session_state.ohlc_data if st.session_state.ohlc_data is not None else {}

    available_cols = [c for c in yield_df.columns if c in MATURITY_MAP]
    maturities = np.array([MATURITY_MAP[c] for c in available_cols], dtype=float)
    yield_values = yield_df.iloc[-1][available_cols].values.astype(float)

    spreads = compute_spreads(yield_df)
    recessions = NBERRecessionAnalysis.identify_recessions(recession_series)
    inversion_periods = NBERRecessionAnalysis.calculate_inversion_periods(spreads)
    lead_times = NBERRecessionAnalysis.calculate_lead_times(inversion_periods, recessions)

    current_10y = yield_df["10Y"].iloc[-1] if "10Y" in yield_df.columns else np.nan
    current_2y = yield_df["2Y"].iloc[-1] if "2Y" in yield_df.columns else np.nan
    current_30y = yield_df["30Y"].iloc[-1] if "30Y" in yield_df.columns else np.nan
    current_1m = yield_df["1M"].iloc[-1] if "1M" in yield_df.columns else np.nan
    current_spread = spreads["10Y-2Y"].iloc[-1] if "10Y-2Y" in spreads.columns else np.nan

    regime, regime_text = classify_regime(spreads, yield_df)
    recession_prob = recession_probability_proxy(spreads, yield_df)

    with st.spinner("Calibrating Nelson-Siegel family models..."):
        ns_result = NelsonSiegelModel.fit_nelson_siegel(maturities, yield_values)
        nss_result = NelsonSiegelModel.fit_svensson(maturities, yield_values)

    with st.spinner("Running dynamic / factor / PCA analysis..."):
        dynamic_params = DynamicParameterAnalysis.calibrate_rolling_window(yield_df[available_cols], maturities, available_cols, window_years)
        factors = DynamicParameterAnalysis.calculate_factor_contributions(yield_df)
        pca_risk = AdvancedRiskMetrics.calculate_pca_risk(yield_df[available_cols])

    with st.spinner("Detecting arbitrage and generating forecast..."):
        arbitrage_stats = ArbitrageDetection.detect_arbitrage_opportunities(yield_df[available_cols], maturities)
        forecast_result = YieldCurveForecasting.forecast_linear(yield_df[available_cols], forecast_horizon)

    cols = st.columns(6)
    payload = [
        ("Regime", regime, regime_text),
        ("10Y Yield", f"{current_10y:.2f}%", "Long-end benchmark"),
        ("2Y Yield", f"{current_2y:.2f}%", "Policy-sensitive node"),
        ("10Y-2Y Spread", f"{current_spread:.1f} bps", "Primary recession signal"),
        ("1M Yield", f"{current_1m:.2f}%", "Money-market anchor"),
        ("Recession Probability", f"{100*recession_prob:.1f}%", "Institutional proxy"),
    ]
    for c, (label, value, sub) in zip(cols, payload):
        with c:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    tabs = st.tabs([
        "Executive Summary", "Data Table", "2Y & 10Y Dynamic Charts", "OHLC Analysis", "Spread Dynamics",
        "NS Model Fit", "NSS Model Fit", "Model Comparison", "Dynamic Analysis", "Factor Analysis",
        "Risk Metrics", "Arbitrage", "NBER Recession", "Forecasting", "Data Export"
    ])

    with tabs[0]:
        st.markdown("### Executive Summary")
        a, b = st.columns([1.2, 1])
        with a:
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(x=maturities, y=yield_values, mode="markers+lines", name="Current Curve", marker=dict(size=10, color=COLORS["accent"]), line=dict(color=COLORS["line"], width=2.4)))
            fig_curve = create_institutional_layout(fig_curve, "Current U.S. Treasury Curve", "Yield (%)", 420)
            st.plotly_chart(fig_curve, use_container_width=True)
        with b:
            summary_df = pd.DataFrame({
                "Metric": ["Regime", "10Y", "2Y", "30Y", "10Y-2Y", "10Y-3M", "Recession Probability"],
                "Value": [regime, f"{current_10y:.2f}%", f"{current_2y:.2f}%", f"{current_30y:.2f}%" if np.isfinite(current_30y) else "N/A", f"{current_spread:.1f} bps" if np.isfinite(current_spread) else "N/A", f"{spreads['10Y-3M'].iloc[-1]:.1f} bps" if "10Y-3M" in spreads.columns else "N/A", f"{100*recession_prob:.1f}%"],
                "Interpretation": [regime_text, "Long-end benchmark", "Short-end policy anchor", "Long duration reference", "Primary slope signal", "Alternative recession signal", "Composite internal proxy"]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        fig_spread = go.Figure()
        if "10Y-2Y" in spreads.columns:
            fig_spread.add_trace(go.Scatter(x=spreads.index, y=spreads["10Y-2Y"], mode="lines", name="10Y-2Y", line=dict(color=COLORS["negative"], width=2)))
            fig_spread.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"])
            add_recession_bands(fig_spread, recessions)
            fig_spread = create_institutional_layout(fig_spread, "10Y-2Y Spread with NBER Recessions", "bps", 430)
            st.plotly_chart(fig_spread, use_container_width=True)

    with tabs[1]:
        st.markdown("### Historical Yield Data (Latest to Earliest)")
        display_df = yield_df.iloc[::-1].reset_index()
        display_df.columns = ["Date"] + list(yield_df.columns)
        for col in display_df.columns:
            if col != "Date":
                display_df[col] = display_df[col].map(lambda x: f"{x:.2f}%")
        display_df["Date"] = pd.to_datetime(display_df["Date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(display_df, use_container_width=True, height=430)

    with tabs[2]:
        st.markdown("### 2-Year and 10-Year Treasury Yield Dynamics")
        left, right = st.columns(2)
        with left:
            f2 = plot_2y_yield_chart(yield_df)
            if f2: st.plotly_chart(f2, use_container_width=True)
        with right:
            f10 = plot_10y_yield_chart(yield_df)
            if f10: st.plotly_chart(f10, use_container_width=True)

    with tabs[3]:
        st.markdown("### OHLC Analysis")
        selected_ticker = st.selectbox("Select OHLC instrument", list(YAHOO_TICKERS.keys()), format_func=lambda x: f"{x} | {YAHOO_TICKERS[x]}")
        l, r = st.columns([1.2, 1])
        with l:
            fig_ohlc = create_ohlc_candlestick_chart(ohlc_data, selected_ticker, f"{selected_ticker} Candlestick")
            if fig_ohlc: st.plotly_chart(fig_ohlc, use_container_width=True)
        with r:
            fig_ta = create_technical_indicators_chart(ohlc_data, selected_ticker, f"{selected_ticker} Technicals")
            if fig_ta: st.plotly_chart(fig_ta, use_container_width=True)
        comp = st.multiselect("Compare OHLC instruments", list(YAHOO_TICKERS.keys()), default=list(YAHOO_TICKERS.keys())[:3])
        if comp:
            fig_cmp = plot_ohlc_comparison_chart(ohlc_data, comp)
            st.plotly_chart(fig_cmp, use_container_width=True)

    with tabs[4]:
        st.markdown("### Spread Dynamics")
        fig = go.Figure()
        palette = [COLORS["negative"], COLORS["line"], COLORS["warning"], COLORS["positive"]]
        for i, col in enumerate(spreads.columns):
            fig.add_trace(go.Scatter(x=spreads.index, y=spreads[col], mode="lines", name=col, line=dict(color=palette[i % len(palette)], width=2)))
        fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"])
        add_recession_bands(fig, recessions)
        fig = create_institutional_layout(fig, "Treasury Spread Dashboard", "bps", 520)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:
        st.markdown("### Nelson-Siegel Model Fit")
        if ns_result:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=maturities, y=yield_values, mode="markers", name="Actual", marker=dict(size=11, color=COLORS["accent"])))
            fig.add_trace(go.Scatter(x=maturities, y=ns_result["fitted_values"], mode="lines", name="NS Fit", line=dict(color=COLORS["positive"], width=2.6)))
            fig = create_institutional_layout(fig, "NS Curve Fit", "Yield (%)", 430)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[6]:
        st.markdown("### Nelson-Siegel-Svensson Model Fit")
        if nss_result:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=maturities, y=yield_values, mode="markers", name="Actual", marker=dict(size=11, color=COLORS["accent"])))
            fig.add_trace(go.Scatter(x=maturities, y=nss_result["fitted_values"], mode="lines", name="NSS Fit", line=dict(color=COLORS["warning"], width=2.6)))
            fig = create_institutional_layout(fig, "NSS Curve Fit", "Yield (%)", 430)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[7]:
        st.markdown("### Model Comparison")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=maturities, y=yield_values, mode="markers", name="Actual", marker=dict(size=11, color=COLORS["accent"])))
        if ns_result:
            fig.add_trace(go.Scatter(x=maturities, y=ns_result["fitted_values"], mode="lines", name="NS", line=dict(color=COLORS["positive"], width=2.5)))
        if nss_result:
            fig.add_trace(go.Scatter(x=maturities, y=nss_result["fitted_values"], mode="lines", name="NSS", line=dict(color=COLORS["warning"], width=2.5, dash="dash")))
        fig = create_institutional_layout(fig, "NS vs NSS Comparison", "Yield (%)", 470)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[8]:
        st.markdown("### Dynamic Analysis")
        if dynamic_params is not None and not dynamic_params.empty:
            fig = make_subplots(rows=2, cols=2, subplot_titles=("β0 Level", "β1 Slope", "β2 Curvature", "RMSE"))
            fig.add_trace(go.Scatter(x=dynamic_params["date"], y=dynamic_params["beta0"], mode="lines", name="β0", line=dict(color=COLORS["positive"])), row=1, col=1)
            fig.add_trace(go.Scatter(x=dynamic_params["date"], y=dynamic_params["beta1"], mode="lines", name="β1", line=dict(color=COLORS["line"])), row=1, col=2)
            fig.add_trace(go.Scatter(x=dynamic_params["date"], y=dynamic_params["beta2"], mode="lines", name="β2", line=dict(color=COLORS["warning"])), row=2, col=1)
            fig.add_trace(go.Scatter(x=dynamic_params["date"], y=dynamic_params["rmse"], mode="lines", name="RMSE", line=dict(color=COLORS["muted"])), row=2, col=2)
            fig = create_institutional_layout(fig, "Rolling Nelson-Siegel Parameters", height=620)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[9]:
        st.markdown("### Factor Analysis")
        left, right = st.columns(2)
        with left:
            if factors is not None and not factors.empty:
                fig = go.Figure()
                palette = [COLORS["line"], COLORS["warning"], COLORS["positive"], COLORS["accent"]]
                for i, col in enumerate(factors.columns):
                    fig.add_trace(go.Scatter(x=factors.index, y=factors[col], mode="lines", name=col, line=dict(color=palette[i % len(palette)], width=1.8)))
                fig = create_institutional_layout(fig, "Historical Factor Contributions", "Value", 450)
                st.plotly_chart(fig, use_container_width=True)
        with right:
            if pca_risk:
                fig = go.Figure()
                ev = pca_risk["explained_variance"] * 100
                fig.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(ev))], y=ev, marker_color=COLORS["accent"]))
                fig = create_institutional_layout(fig, "PCA Variance Explained", "Percent", 450)
                st.plotly_chart(fig, use_container_width=True)
        if pca_risk:
            st.dataframe(pca_risk["loadings"].round(4), use_container_width=True)

    with tabs[10]:
        st.markdown("### Risk Metrics")
        if "10Y" in yield_df.columns:
            returns_10y = yield_df["10Y"].pct_change()
            risk = AdvancedRiskMetrics.calculate_var_metrics(returns_10y, confidence=confidence_level, horizon=10)
            if risk:
                a, b, c, d = st.columns(4)
                a.metric("Historical VaR", f"{risk['VaR_Historical']:.4f}")
                b.metric("Parametric VaR", f"{risk['VaR_Parametric']:.4f}")
                c.metric("Cornish-Fisher VaR", f"{risk['VaR_CornishFisher']:.4f}")
                d.metric("CVaR", f"{risk['CVaR']:.4f}")

    with tabs[11]:
        st.markdown("### Arbitrage / Mispricing Diagnostics")
        if arbitrage_stats:
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean Abs Error", f"{arbitrage_stats['mean_abs_error']:.4f}")
            c2.metric("Max Abs Error", f"{arbitrage_stats['max_abs_error']:.4f}")
            c3.metric("Mispriced Tenors", f"{arbitrage_stats['mispriced_count']}")
            if not arbitrage_stats["mispriced_table"].empty:
                st.dataframe(arbitrage_stats["mispriced_table"].round(4), use_container_width=True)

    with tabs[12]:
        st.markdown("### NBER Recession Details")
        fig = go.Figure()
        if "10Y-2Y" in spreads.columns:
            fig.add_trace(go.Scatter(x=spreads.index, y=spreads["10Y-2Y"], mode="lines", name="10Y-2Y", line=dict(color=COLORS["negative"], width=2)))
        fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"])
        add_recession_bands(fig, recessions)
        fig = create_institutional_layout(fig, "10Y-2Y vs NBER Recessions", "bps", 500)
        st.plotly_chart(fig, use_container_width=True)
        if recessions:
            st.dataframe(pd.DataFrame(recessions), use_container_width=True, hide_index=True)
        if inversion_periods:
            st.dataframe(pd.DataFrame(inversion_periods), use_container_width=True, hide_index=True)
        if lead_times:
            st.dataframe(pd.DataFrame(lead_times), use_container_width=True, hide_index=True)

    with tabs[13]:
        st.markdown("### Forecasting")
        if forecast_result is not None and not forecast_result.empty:
            chosen = st.multiselect("Select tenors", available_cols, default=["2Y", "10Y"] if {"2Y", "10Y"}.issubset(set(available_cols)) else available_cols[:2])
            fig = go.Figure()
            for col in chosen:
                fig.add_trace(go.Scatter(x=yield_df.index[-200:], y=yield_df[col].tail(200), mode="lines", name=f"{col} history", line=dict(width=1.8)))
                fig.add_trace(go.Scatter(x=forecast_result.index, y=forecast_result[col], mode="lines", name=f"{col} forecast", line=dict(width=2.2, dash="dash")))
            fig = create_institutional_layout(fig, "Linear Trend Yield Forecast", "Yield (%)", 500)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(forecast_result.head(10).round(3), use_container_width=True)

    with tabs[14]:
        st.markdown("### Data Export")
        st.download_button("Download Yield Data (CSV)", yield_df.to_csv().encode("utf-8"), f"yield_data_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
        st.download_button("Download Spread Data (CSV)", spreads.to_csv().encode("utf-8"), f"spread_data_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
        if dynamic_params is not None and not dynamic_params.empty:
            st.download_button("Download Dynamic Parameters", dynamic_params.to_csv(index=False).encode("utf-8"), f"dynamic_params_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
        if pca_risk:
            st.download_button("Download PCA Loadings", pca_risk["loadings"].to_csv().encode("utf-8"), f"pca_loadings_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
        if lead_times:
            st.download_button("Download Lead Times", pd.DataFrame(lead_times).to_csv(index=False).encode("utf-8"), f"lead_times_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")

if __name__ == "__main__":
    main()
