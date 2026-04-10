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
# PRODUCTION READY VERSION - ALL FEATURES IMPLEMENTED
# Includes: Monte Carlo, ML Forecasting, Backtesting, Volatility Analytics, 
# Cross-Asset Correlations, Nelson-Siegel Model, Technical Analysis
# =============================================================================

st.set_page_config(
    page_title="Dynamic Quantitative Analysis Platform | Institutional Fixed-Income",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# COLOR SCHEME - PROFESSIONAL INSTITUTIONAL DESIGN
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
    "^VXN": "Nasdaq Volatility Index",
}

CORRELATION_TICKERS = {
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ Composite",
    "QQQ": "Nasdaq 100 ETF",
    "DXY": "US Dollar Index",
    "GC=F": "Gold Futures",
    "CL=F": "Crude Oil Futures",
}

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

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
# CUSTOM CSS STYLING
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
    .success-box {{
        background: #e6f4ea;
        border: 1px solid #b7e0c1;
        border-left: 4px solid {COLORS['positive']};
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
# CONFIGURATION CLASS
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
# DATA LAYER - WITH PROPER ERROR HANDLING
# =============================================================================

def fred_request(api_key: str, series_id: str) -> Optional[pd.Series]:
    """Fetch data from FRED API with retry logic"""
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
    """Validate FRED API key by testing with DGS10 series"""
    if not api_key or len(api_key) < 10:
        return False
    s = fred_request(api_key, "DGS10")
    return s is not None and len(s) > 10

@st.cache_data(ttl=CFG.cache_ttl_sec, show_spinner=False)
def fetch_all_yield_data(api_key: str) -> Optional[pd.DataFrame]:
    """Fetch all Treasury yield curve data from FRED"""
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
    """Fetch NBER recession indicator data"""
    return fred_request(api_key, "USREC")

@st.cache_data(ttl=CFG.cache_ttl_sec, show_spinner=False)
def fetch_ohlc_data(ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """Fetch OHLC data from Yahoo Finance"""
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None

@st.cache_data(ttl=CFG.cache_ttl_sec, show_spinner=False)
def fetch_volatility_data() -> Optional[pd.DataFrame]:
    """Fetch volatility indices (VIX, VXN) with proper error handling"""
    data = {}
    for ticker in VOLATILITY_TICKERS:
        try:
            df = yf.download(ticker, period="2y", progress=False)
            if df is not None and not df.empty and 'Close' in df.columns:
                data[ticker] = df['Close']
        except Exception:
            continue
    
    if not data:
        return None
    
    df_result = pd.DataFrame(data)
    if not df_result.empty:
        return df_result.dropna()
    return None

@st.cache_data(ttl=CFG.cache_ttl_sec, show_spinner=False)
def fetch_correlation_data() -> Optional[pd.DataFrame]:
    """Fetch correlation assets (S&P 500, NASDAQ, Gold, Oil) with proper error handling"""
    data = {}
    for ticker, name in CORRELATION_TICKERS.items():
        try:
            df = yf.download(ticker, period="2y", progress=False)
            if df is not None and not df.empty and 'Close' in df.columns:
                data[name] = df['Close']
        except Exception:
            continue
    
    if not data:
        return None
    
    df_result = pd.DataFrame(data)
    if not df_result.empty:
        return df_result.dropna()
    return None

# =============================================================================
# MONTE CARLO SIMULATIONS
# =============================================================================

class MonteCarloSimulator:
    """Advanced Monte Carlo simulation for yield curve scenarios"""
    
    @staticmethod
    def simulate_geometric_brownian_motion(initial_yield: float, mu: float, sigma: float, 
                                           days: int, simulations: int = 1000) -> np.ndarray:
        """
        Simulate yield paths using Geometric Brownian Motion
        Formula: dS = μS dt + σS dW
        """
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
        """
        Simulate interest rates using Vasicek mean-reverting model
        Formula: dr = κ(θ - r)dt + σdW
        """
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
        """Calculate Value at Risk from simulated paths"""
        final_values = paths[:, -1]
        return np.percentile(final_values, (1 - confidence) * 100)
    
    @staticmethod
    def calculate_cvar_from_paths(paths: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        final_values = paths[:, -1]
        var = np.percentile(final_values, (1 - confidence) * 100)
        return final_values[final_values <= var].mean()

# =============================================================================
# MACHINE LEARNING FORECASTING
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
        
        if not X:
            return np.array([]), np.array([])
        return np.array(X), np.array(y)
    
    @staticmethod
    def train_random_forest(X: np.ndarray, y: np.ndarray, 
                            test_size: float = 0.2) -> Dict:
        """Train Random Forest model for yield forecasting"""
        if len(X) == 0:
            return {}
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        model = RandomForestRegressor(n_estimators=50, max_depth=8, 
                                      random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        feature_importance = pd.DataFrame({
            'feature': [f'Lag_{i}' for i in range(X.shape[1])],
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
        """Generate rolling forecasts using trained model"""
        forecasts = []
        current_window = last_observations.copy()
        
        for _ in range(horizon):
            pred = model.predict(current_window.reshape(1, -1))[0]
            forecasts.append(pred)
            current_window = np.roll(current_window, -n_features)
            current_window[-n_features:] = pred
        
        return np.array(forecasts)

# =============================================================================
# BACKTESTING FRAMEWORK
# =============================================================================

class BacktestEngine:
    """Professional backtesting framework for trading strategies"""
    
    @staticmethod
    def backtest_curve_strategy(yield_df: pd.DataFrame, spreads: pd.DataFrame,
                                strategy_type: str = "curve_inversion") -> Dict:
        """Backtest curve-based trading strategies"""
        
        if strategy_type == "curve_inversion":
            if "10Y-2Y" not in spreads.columns or "10Y" not in yield_df.columns:
                return {}
            
            signals = spreads["10Y-2Y"] < 0
            returns = yield_df["10Y"].pct_change().shift(-1)
            
            strategy_returns = signals.shift(1) * returns
            buy_hold_returns = returns
            
            cumulative_strategy = (1 + strategy_returns.fillna(0)).cumprod()
            cumulative_bh = (1 + buy_hold_returns.fillna(0)).cumprod()
            
            if strategy_returns.std() == 0:
                sharpe_strategy = 0
            else:
                sharpe_strategy = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            
            if buy_hold_returns.std() == 0:
                sharpe_bh = 0
            else:
                sharpe_bh = buy_hold_returns.mean() / buy_hold_returns.std() * np.sqrt(252)
            
            max_drawdown = (cumulative_strategy / cumulative_strategy.cummax() - 1).min()
            
            # Calculate additional metrics
            winning_trades = strategy_returns[strategy_returns > 0]
            losing_trades = strategy_returns[strategy_returns < 0]
            
            return {
                "strategy_name": "Curve Inversion Strategy (Long Duration on Inversion)",
                "cumulative_returns": cumulative_strategy,
                "buy_hold_returns": cumulative_bh,
                "sharpe_ratio_strategy": sharpe_strategy,
                "sharpe_ratio_bh": sharpe_bh,
                "max_drawdown": max_drawdown,
                "total_return_strategy": cumulative_strategy.iloc[-1] - 1,
                "total_return_bh": cumulative_bh.iloc[-1] - 1,
                "win_rate": (strategy_returns[strategy_returns != 0] > 0).mean() if len(strategy_returns[strategy_returns != 0]) > 0 else 0,
                "avg_win": winning_trades.mean() if len(winning_trades) > 0 else 0,
                "avg_loss": losing_trades.mean() if len(losing_trades) > 0 else 0,
                "profit_factor": abs(winning_trades.sum() / losing_trades.sum()) if losing_trades.sum() != 0 else np.inf,
            }
        
        elif strategy_type == "momentum":
            if "10Y" not in yield_df.columns:
                return {}
            
            sma_20 = yield_df["10Y"].rolling(20).mean()
            sma_50 = yield_df["10Y"].rolling(50).mean()
            signals = sma_20 > sma_50
            returns = yield_df["10Y"].pct_change().shift(-1)
            
            strategy_returns = signals.shift(1) * returns
            cumulative_strategy = (1 + strategy_returns.fillna(0)).cumprod()
            
            if strategy_returns.std() == 0:
                sharpe = 0
            else:
                sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            
            return {
                "strategy_name": "Moving Average Crossover (20/50)",
                "cumulative_returns": cumulative_strategy,
                "sharpe_ratio": sharpe,
                "total_return": cumulative_strategy.iloc[-1] - 1,
                "win_rate": (strategy_returns[strategy_returns != 0] > 0).mean() if len(strategy_returns[strategy_returns != 0]) > 0 else 0,
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
# VOLATILITY ANALYZER
# =============================================================================

class VolatilityAnalyzer:
    """Analyze options-implied volatility indices"""
    
    @staticmethod
    def calculate_volatility_regime(vix: pd.Series) -> Dict:
        """Determine volatility regime based on VIX levels"""
        if vix is None or len(vix) == 0:
            return {"current_vix": 0, "regime": "N/A", "outlook": "Data unavailable"}
        
        current_vix = vix.iloc[-1]
        
        if current_vix < 12:
            regime = "EXTREME COMPLACENCY"
            outlook = "High risk of volatility spike - consider tail hedging"
            color = "green"
        elif current_vix < 15:
            regime = "LOW VOLATILITY"
            outlook = "Normal complacent market conditions"
            color = "lightgreen"
        elif current_vix < 20:
            regime = "NORMAL VOLATILITY"
            outlook = "Typical market conditions"
            color = "yellow"
        elif current_vix < 25:
            regime = "ELEVATED VOLATILITY"
            outlook = "Increased uncertainty - reduce leverage"
            color = "orange"
        elif current_vix < 35:
            regime = "HIGH VOLATILITY"
            outlook = "Market stress, consider active hedging"
            color = "red"
        else:
            regime = "EXTREME VOLATILITY"
            outlook = "Crisis conditions - defensive positioning recommended"
            color = "darkred"
        
        percentile = (vix < current_vix).mean() if len(vix) > 0 else 0.5
        
        # Calculate VIX term structure proxy
        vix_ma_20 = vix.rolling(20).mean()
        term_structure = "Normal" if current_vix > vix_ma_20.iloc[-1] else "Inverted"
        
        return {
            "current_vix": current_vix,
            "regime": regime,
            "outlook": outlook,
            "color": color,
            "percentile": percentile,
            "mean_vix": vix.mean(),
            "median_vix": vix.median(),
            "max_vix": vix.max(),
            "min_vix": vix.min(),
            "vix_percentile": f"{percentile * 100:.1f}%",
            "term_structure": term_structure,
        }
    
    @staticmethod
    def calculate_vol_of_vol(vix: pd.Series, window: int = 20) -> pd.Series:
        """Calculate volatility of volatility (vol of vol)"""
        if vix is None or len(vix) < window:
            return pd.Series()
        return vix.pct_change().rolling(window).std() * np.sqrt(252)
    
    @staticmethod
    def identify_volatility_spikes(vix: pd.Series, threshold: float = 1.5) -> pd.DataFrame:
        """Identify volatility spike events using z-score"""
        if vix is None or len(vix) < 20:
            return pd.DataFrame()
        
        vix_ma = vix.rolling(20).mean()
        vix_std = vix.rolling(20).std()
        z_score = (vix - vix_ma) / vix_std
        
        spikes = vix[z_score > threshold]
        
        spike_events = []
        in_spike = False
        start = None
        
        for date, value in spikes.items():
            if not in_spike:
                in_spike = True
                start = date
            elif date not in spikes.index and in_spike:
                in_spike = False
                spike_events.append({
                    "start": start,
                    "end": date,
                    "peak_vix": spikes.loc[start:date].max() if start in spikes.index else value,
                    "duration_days": (date - start).days if start is not None else 0,
                })
        
        return pd.DataFrame(spike_events)

# =============================================================================
# CORRELATION ANALYZER
# =============================================================================

class CorrelationAnalyzer:
    """Cross-asset correlation analysis"""
    
    @staticmethod
    def calculate_rolling_correlations(yield_series: pd.Series, 
                                       other_assets: pd.DataFrame,
                                       window: int = 60) -> pd.DataFrame:
        """Calculate rolling correlations between yields and other assets"""
        if other_assets is None or other_assets.empty:
            return pd.DataFrame()
        
        correlations = pd.DataFrame(index=other_assets.index)
        for col in other_assets.columns:
            if len(yield_series) == len(other_assets):
                correlations[f"Correlation_10Y_{col}"] = yield_series.rolling(window).corr(other_assets[col])
        return correlations
    
    @staticmethod
    def calculate_correlation_matrix(assets_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate full correlation matrix"""
        if assets_df is None or assets_df.empty:
            return pd.DataFrame()
        returns = assets_df.pct_change().dropna()
        return returns.corr()
    
    @staticmethod
    def identify_regime_shifts(correlation_series: pd.Series, 
                               threshold: float = 0.5) -> List[Dict]:
        """Identify significant correlation regime shifts"""
        if correlation_series is None or len(correlation_series) < 2:
            return []
        
        changes = correlation_series.diff()
        significant_changes = changes[abs(changes) > threshold]
        
        shifts = []
        for date, change in significant_changes.items():
            idx = correlation_series.index.get_loc(date)
            if idx > 0:
                shifts.append({
                    "date": date,
                    "change": change,
                    "direction": "Positive regime shift" if change > 0 else "Negative regime shift",
                    "from_value": correlation_series.iloc[idx - 1],
                    "to_value": correlation_series.iloc[idx],
                })
        
        return shifts
    
    @staticmethod
    def calculate_beta(yield_series: pd.Series, market_series: pd.Series, 
                       window: int = 60) -> pd.Series:
        """Calculate rolling beta relative to market"""
        if yield_series is None or market_series is None:
            return pd.Series()
        
        yield_returns = yield_series.pct_change()
        market_returns = market_series.pct_change()
        
        covariance = yield_returns.rolling(window).cov(market_returns)
        variance = market_returns.rolling(window).var()
        
        return covariance / variance

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
    """Relative Strength Index"""
    delta = x.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(x: pd.Series):
    """MACD Indicator"""
    m = ema(x, 12) - ema(x, 26)
    s = ema(m, 9)
    return m, s, m - s

def bb(x: pd.Series, n: int = 20, k: float = 2.0):
    """Bollinger Bands"""
    mid = sma(x, n)
    sd = x.rolling(n).std()
    return mid + k * sd, mid, mid - k * sd

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators to DataFrame"""
    if df is None or df.empty:
        return df
    out = df.copy()
    out["SMA_20"] = sma(out["Close"], 20)
    out["SMA_50"] = sma(out["Close"], 50)
    out["SMA_200"] = sma(out["Close"], 200)
    out["EMA_12"] = ema(out["Close"], 12)
    out["RSI"] = rsi(out["Close"], 14)
    out["MACD"], out["MACD_Signal"], out["MACD_Hist"] = macd(out["Close"])
    out["BB_Upper"], out["BB_Middle"], out["BB_Lower"] = bb(out["Close"])
    return out

# =============================================================================
# ANALYTICS FUNCTIONS
# =============================================================================

def compute_spreads(yield_df: pd.DataFrame) -> pd.DataFrame:
    """Compute key yield spreads in basis points"""
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
    """Classify current macro regime based on curve shape"""
    if "10Y-2Y" not in spreads.columns or spreads.empty:
        return "Data Loading", "Please wait for data to load"
    
    spread = spreads["10Y-2Y"].iloc[-1]
    y10 = yield_df["10Y"].iloc[-1] if "10Y" in yield_df.columns else 0
    
    if np.isfinite(spread) and spread < 0:
        return "Risk-off / Recession Watch", "Curve inversion signals defensive macro regime and elevated recession risk"
    elif np.isfinite(spread) and spread < 50:
        return "Neutral / Late Cycle", "Curve flattening suggests late-cycle caution and potential policy inflection"
    elif np.isfinite(y10) and y10 > 5.5:
        return "Neutral / Restrictive", "Elevated long-end rates indicate restrictive financial conditions"
    return "Risk-on / Expansion", "Positive slope supports pro-risk positioning and cyclical growth"

def recession_probability_proxy(spreads: pd.DataFrame, yield_df: pd.DataFrame) -> float:
    """Calculate recession probability proxy using logistic regression on spreads"""
    if "10Y-2Y" not in spreads.columns or "10Y" not in yield_df.columns:
        return 0.5
    
    score = 0.0
    score += np.clip((-spreads["10Y-2Y"].iloc[-1]) / 100, 0, 1.5)
    score += np.clip((yield_df["10Y"].iloc[-1] - 4.5) / 3.0, 0, 1.0)
    return float(1 / (1 + np.exp(-2.2 * (score - 0.8))))

def identify_recessions(recession_series: Optional[pd.Series]) -> List[dict]:
    """Identify NBER recession periods from indicator series"""
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

# =============================================================================
# NELSON-SIEGEL MODEL
# =============================================================================

class NelsonSiegelModel:
    """Nelson-Siegel yield curve fitting model"""
    
    @staticmethod
    def nelson_siegel(tau, beta0, beta1, beta2, lambda1):
        """Nelson-Siegel yield curve function"""
        tau = np.asarray(tau, dtype=float)
        x = lambda1 * tau
        term1 = (1 - np.exp(-x)) / x
        term2 = term1 - np.exp(-x)
        return beta0 + beta1 * term1 + beta2 * term2
    
    @staticmethod
    def fit_ns(maturities: np.ndarray, yields_: np.ndarray):
        """Fit Nelson-Siegel model to observed yields"""
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
            "fitted_values": fitted,
            "rmse": float(np.sqrt(np.mean((yields_ - fitted) ** 2))),
            "mae": float(np.mean(np.abs(yields_ - fitted))),
            "r_squared": float(1 - sse / sst) if sst > 0 else np.nan,
            "residuals": yields_ - fitted,
        }

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_chart_layout(fig: go.Figure, title: str, y_title: str = None, 
                        height: int = 460, x_title: str = "Date") -> go.Figure:
    """Apply professional chart styling"""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(size=12, color=COLORS["text"]),
        title=dict(text=title, x=0.01, font=dict(size=16, weight="bold")),
        margin=dict(l=60, r=30, t=80, b=50),
        hovermode="x unified",
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title=x_title, gridcolor=COLORS["grid"], showgrid=True, tickfont=dict(size=11)),
        yaxis=dict(title=y_title, gridcolor=COLORS["grid"], showgrid=True, tickfont=dict(size=11)),
    )
    return fig

def add_recession_bands(fig: go.Figure, recessions: List[dict]) -> go.Figure:
    """Add NBER recession bands to chart"""
    for rec in recessions:
        fig.add_vrect(x0=rec["start"], x1=rec["end"], 
                      fillcolor=COLORS["recession"], opacity=0.35, 
                      layer="below", line_width=0)
    return fig

def chart_current_curve(maturities: np.ndarray, yields_: np.ndarray, 
                        ns_result: Optional[dict]) -> go.Figure:
    """Display current yield curve with model fit"""
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
            mode="lines", name="Nelson-Siegel Fit",
            line=dict(color=COLORS["warning"], width=2.5, dash="dot")
        ))
    return create_chart_layout(fig, "Current Treasury Yield Curve", 
                               "Yield (%)", 480, "Maturity (Years)")

def chart_monte_carlo(initial_yield: float, simulation_results: Dict, 
                      horizon_days: int) -> go.Figure:
    """Visualize Monte Carlo simulation results"""
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
        x=x_axis, y=simulation_results["median"],
        mode='lines', name='Median Path',
        line=dict(color=COLORS["warning"], width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=[0], y=[initial_yield],
        mode='markers', name='Current Yield',
        marker=dict(size=14, color=COLORS["positive"], symbol='star')
    ))
    
    return create_chart_layout(fig, "Monte Carlo Simulation - 10Y Yield Paths", 
                               "Yield (%)", 500, "Trading Days Ahead")

def chart_backtest_results(backtest_results: Dict) -> go.Figure:
    """Visualize backtesting results"""
    fig = go.Figure()
    
    if "cumulative_returns" in backtest_results and backtest_results["cumulative_returns"] is not None:
        fig.add_trace(go.Scatter(
            x=backtest_results["cumulative_returns"].index,
            y=backtest_results["cumulative_returns"].values,
            mode='lines',
            name=backtest_results.get("strategy_name", "Strategy"),
            line=dict(color=COLORS["accent"], width=2.5)
        ))
    
    if "buy_hold_returns" in backtest_results and backtest_results["buy_hold_returns"] is not None:
        fig.add_trace(go.Scatter(
            x=backtest_results["buy_hold_returns"].index,
            y=backtest_results["buy_hold_returns"].values,
            mode='lines',
            name='Buy & Hold',
            line=dict(color=COLORS["muted"], width=2, dash='dash')
        ))
    
    return create_chart_layout(fig, "Backtest Performance Comparison", 
                               "Cumulative Return", 500)

def chart_volatility_dashboard(vix_data: pd.Series, vol_regime: Dict) -> go.Figure:
    """Create volatility dashboard with VIX and regimes"""
    if vix_data is None or len(vix_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Volatility data unavailable", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("VIX - CBOE Volatility Index", 
                                       "Volatility of Volatility (20-day)"),
                        vertical_spacing=0.15,
                        row_heights=[0.6, 0.4])
    
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
    fig.add_hline(y=12, line_dash="dot", line_color="lightgreen",
                  annotation_text="Complacent", row=1, col=1)
    
    vol_of_vol = VolatilityAnalyzer.calculate_vol_of_vol(vix_data)
    if len(vol_of_vol) > 0:
        fig.add_trace(go.Scatter(
            x=vix_data.index, y=vol_of_vol,
            mode='lines', name='Vol of Vol',
            line=dict(color=COLORS["accent"], width=2)
        ), row=2, col=1)
    
    fig.update_yaxes(title_text="VIX", row=1, col=1)
    fig.update_yaxes(title_text="Vol of Vol", row=2, col=1)
    
    title = f"Volatility Dashboard | Current VIX: {vol_regime.get('current_vix', 0):.2f} | Regime: {vol_regime.get('regime', 'N/A')}"
    return create_chart_layout(fig, title, height=600)

def chart_correlation_heatmap(correlation_matrix: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap"""
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
    
    fig.update_layout(title="Cross-Asset Correlation Matrix", height=550, width=750)
    return fig

def chart_yield_history(yield_df: pd.DataFrame, tenor: str, color: str, title: str) -> Optional[go.Figure]:
    """Display historical yield chart"""
    if tenor not in yield_df.columns:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yield_df.index, y=yield_df[tenor], 
        mode="lines", name=tenor, 
        line=dict(color=color, width=2.5)
    ))
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))
    return create_chart_layout(fig, title, "Yield (%)", 420)

# =============================================================================
# UI HELPERS
# =============================================================================

def render_api_gate() -> None:
    """Display API key input gate"""
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
    """Display professional KPI card"""
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main() -> None:
    """Main application entry point"""
    
    # Header
    st.markdown(
        """
        <div class="main-title-card">
            <div class="main-title">Dynamic Quantitative Analysis Model</div>
            <div class="main-subtitle">Institutional Fixed-Income Platform | Monte Carlo | ML Forecasting | Backtesting | Volatility Analytics | Cross-Asset Correlations</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # API Key Validation
    if not st.session_state.api_key_validated:
        render_api_gate()

    # Sidebar Controls
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
        rolling_years = st.slider("Rolling window (years)", 2, 10, 5, 
                                  help="Number of years for rolling parameter estimation")
        forecast_horizon = st.slider("Forecast horizon (days)", 5, 60, 20,
                                      help="Forward yield curve projection period")
        confidence_level = st.slider("Confidence level", 0.90, 0.99, 0.95, 0.01,
                                      help="Statistical confidence for risk metrics")
        mc_simulations = st.slider("Monte Carlo simulations", 500, 3000, 1000, 500,
                                    help="Number of simulation paths")
        ml_lags = st.slider("ML model lags", 3, 10, 5,
                           help="Number of lag periods for feature engineering")

    # Data Fetching
    if not st.session_state.data_fetched:
        with st.spinner("Fetching all data (yields, volatility, correlations, market data)..."):
            yield_df = fetch_all_yield_data(st.session_state.api_key)
            recession_series = fetch_recession_data(st.session_state.api_key)
            volatility_df = fetch_volatility_data()
            correlation_df = fetch_correlation_data()
            
            ohlc_data = {}
            for ticker in YAHOO_TICKERS.keys():
                df = fetch_ohlc_data(ticker, "2y")
                if df is not None:
                    ohlc_data[ticker] = add_technical_indicators(df)
        
        if yield_df is None:
            st.error("Failed to fetch FRED data. Please check your API key and try again.")
            st.stop()
        
        st.session_state.yield_data = yield_df
        st.session_state.recession_data = recession_series
        st.session_state.volatility_data = volatility_df
        st.session_state.correlation_data = correlation_df
        st.session_state.ohlc_data = ohlc_data
        st.session_state.data_fetched = True

    # Load data from session state
    yield_df = st.session_state.yield_data.copy()
    recession_series = st.session_state.recession_data
    volatility_df = st.session_state.volatility_data
    correlation_df = st.session_state.correlation_data

    # Prepare data for analysis
    selected_cols = [c for c in yield_df.columns if c in MATURITY_MAP][:6]
    maturities = np.array([MATURITY_MAP[c] for c in selected_cols])
    latest_curve = yield_df.iloc[-1][selected_cols].values.astype(float)

    spreads = compute_spreads(yield_df)
    recessions = identify_recessions(recession_series)
    regime, regime_text = classify_regime(spreads, yield_df)
    recession_prob = recession_probability_proxy(spreads, yield_df)

    current_2y = yield_df["2Y"].iloc[-1] if "2Y" in yield_df.columns else np.nan
    current_10y = yield_df["10Y"].iloc[-1] if "10Y" in yield_df.columns else np.nan
    current_30y = yield_df["30Y"].iloc[-1] if "30Y" in yield_df.columns else np.nan
    current_spread = spreads["10Y-2Y"].iloc[-1] if "10Y-2Y" in spreads.columns else np.nan

    # Run Nelson-Siegel model
    ns_result = NelsonSiegelModel.fit_ns(maturities, latest_curve)

    # KPI Row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        kpi_card("📊 Macro Regime", regime, regime_text[:40] + "...")
    with col2:
        kpi_card("🏦 2Y Yield", f"{current_2y:.2f}%" if not np.isnan(current_2y) else "N/A", "Policy anchor")
    with col3:
        kpi_card("📈 10Y Yield", f"{current_10y:.2f}%" if not np.isnan(current_10y) else "N/A", "Benchmark")
    with col4:
        kpi_card("🏛️ 30Y Yield", f"{current_30y:.2f}%" if not np.isnan(current_30y) else "N/A", "Long duration")
    with col5:
        kpi_card("🔄 10Y-2Y Spread", f"{current_spread:.1f} bps" if not np.isnan(current_spread) else "N/A", "Recession signal")
    with col6:
        kpi_card("⚠️ Recession Prob", f"{100 * recession_prob:.1f}%", "Proxy estimate")

    # Main Tabs
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

    # ========================================================================
    # TAB 0: EXECUTIVE VIEW
    # ========================================================================
    with main_tabs[0]:
        st.plotly_chart(chart_current_curve(maturities, latest_curve, ns_result), 
                       use_container_width=True)
        
        exec_col1, exec_col2 = st.columns(2)
        with exec_col1:
            fig_2y = chart_yield_history(yield_df, "2Y", COLORS["warning"], 
                                         "2-Year Treasury Yield - Policy Expectations")
            if fig_2y:
                st.plotly_chart(fig_2y, use_container_width=True)
        with exec_col2:
            fig_10y = chart_yield_history(yield_df, "10Y", COLORS["accent"], 
                                          "10-Year Treasury Yield - Growth & Inflation")
            if fig_10y:
                st.plotly_chart(fig_10y, use_container_width=True)
        
        # Spread chart with recession bands
        fig_spread = go.Figure()
        if "10Y-2Y" in spreads.columns:
            fig_spread.add_trace(go.Scatter(
                x=spreads.index, y=spreads["10Y-2Y"], 
                mode="lines", name="10Y-2Y Spread",
                line=dict(color=COLORS["negative"], width=2.5)
            ))
        fig_spread.add_hline(y=0, line_dash="dash", line_color=COLORS["text_secondary"], line_width=2)
        add_recession_bands(fig_spread, recessions)
        st.plotly_chart(create_chart_layout(fig_spread, "Treasury Spread Dashboard - 10Y-2Y with Recessions", 
                                           "Basis Points (bps)", 500), use_container_width=True)

    # ========================================================================
    # TAB 1: MONTE CARLO SIMULATION
    # ========================================================================
    with main_tabs[1]:
        st.subheader("🎲 Monte Carlo Simulation - Yield Curve Scenarios")
        st.markdown("""
        <div class="note-box">
        <b>Monte Carlo Simulation Framework</b><br>
        Simulates 1,000+ potential paths for 10Y Treasury yields using Geometric Brownian Motion.
        Provides probabilistic forecasts, VaR estimates, and confidence intervals for risk management.
        </div>
        """, unsafe_allow_html=True)
        
        mc_col1, mc_col2 = st.columns(2)
        with mc_col1:
            simulation_model = st.selectbox("Simulation Model", 
                                           ["Geometric Brownian Motion", "Vasicek Mean-Reverting"])
            simulation_days = st.slider("Simulation Horizon (Days)", 30, 252, 126, 30)
        with mc_col2:
            show_individual_paths = st.checkbox("Show Sample Paths (50)", value=False)
        
        if st.button("🎲 Run Monte Carlo Simulation", use_container_width=True):
            with st.spinner(f"Running {mc_simulations} simulations..."):
                initial_yield = current_10y if not np.isnan(current_10y) else 4.0
                mu = yield_df["10Y"].pct_change().mean() * 252 if len(yield_df) > 1 else 0
                sigma = yield_df["10Y"].pct_change().std() * np.sqrt(252) if len(yield_df) > 1 else 0.1
                
                if simulation_model == "Geometric Brownian Motion":
                    paths = MonteCarloSimulator.simulate_geometric_brownian_motion(
                        initial_yield, mu, sigma, simulation_days, mc_simulations
                    )
                else:
                    kappa = 0.5
                    theta = yield_df["10Y"].mean()
                    paths = MonteCarloSimulator.simulate_vasicek(
                        initial_yield, kappa, theta, sigma, simulation_days, mc_simulations
                    )
                
                sim_results = MonteCarloSimulator.calculate_confidence_intervals(paths, confidence_level)
                var_estimate = MonteCarloSimulator.calculate_var_from_paths(paths, confidence_level)
                cvar_estimate = MonteCarloSimulator.calculate_cvar_from_paths(paths, confidence_level)
                
                # Display metrics
                mc_metrics = st.columns(5)
                mc_metrics[0].metric("Expected Yield (Mean)", f"{sim_results['mean'][-1]:.2f}%")
                mc_metrics[1].metric("Median Yield", f"{sim_results['median'][-1]:.2f}%")
                mc_metrics[2].metric(f"{int(confidence_level*100)}% VaR", f"{var_estimate:.2f}%")
                mc_metrics[3].metric(f"{int(confidence_level*100)}% CVaR", f"{cvar_estimate:.2f}%")
                mc_metrics[4].metric("Uncertainty Range", f"±{sim_results['std'][-1]:.2f}%")
                
                # Plot results
                fig_mc = chart_monte_carlo(initial_yield, sim_results, simulation_days)
                
                if show_individual_paths:
                    for i in range(min(50, mc_simulations)):
                        fig_mc.add_trace(go.Scatter(
                            x=np.arange(simulation_days), y=paths[i],
                            mode='lines', line=dict(width=0.5, color='rgba(100,100,100,0.2)'),
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

    # ========================================================================
    # TAB 2: MACHINE LEARNING FORECASTING
    # ========================================================================
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
                X, y = MLForecastModel.prepare_features(yield_df[selected_cols], lags=ml_lags)
                
                if len(X) > 50:
                    ml_results = MLForecastModel.train_random_forest(X, y, test_size=0.2)
                    
                    ml_metrics = st.columns(4)
                    ml_metrics[0].metric("RMSE", f"{ml_results['rmse']*100:.2f} bps")
                    ml_metrics[1].metric("R² Score", f"{ml_results['r2']:.3f}")
                    ml_metrics[2].metric("Test Samples", len(ml_results['actual']))
                    ml_metrics[3].metric("Features", X.shape[1])
                    
                    # Feature importance chart
                    fig_importance = go.Figure()
                    top_features = ml_results['feature_importance'].head(10)
                    fig_importance.add_trace(go.Bar(
                        x=top_features['importance'].values,
                        y=top_features['feature'].values,
                        orientation='h',
                        marker_color=COLORS["warning"]
                    ))
                    fig_importance.update_layout(title="Top 10 Feature Importances",
                                                xaxis_title="Importance", height=400)
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Actual vs Predicted scatter
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(
                        x=ml_results['actual'][:, 0], 
                        y=ml_results['predictions'][:, 0],
                        mode='markers',
                        marker=dict(color=COLORS["accent"], size=8)
                    ))
                    fig_pred.add_trace(go.Scatter(
                        x=[ml_results['actual'][:, 0].min(), ml_results['actual'][:, 0].max()],
                        y=[ml_results['actual'][:, 0].min(), ml_results['actual'][:, 0].max()],
                        mode='lines', name='Perfect Fit',
                        line=dict(color='red', dash='dash')
                    ))
                    fig_pred.update_layout(title="Actual vs Predicted 10Y Yield (Test Set)",
                                          xaxis_title="Actual Yield (%)",
                                          yaxis_title="Predicted Yield (%)",
                                          height=400)
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    st.success(f"✓ Model trained successfully on {len(X)} samples")
                else:
                    st.warning(f"Insufficient data for ML training. Need at least 50 samples, have {len(X)}.")

    # ========================================================================
    # TAB 3: BACKTESTING
    # ========================================================================
    with main_tabs[3]:
        st.subheader("📊 Strategy Backtesting Framework")
        st.markdown("""
        <div class="note-box">
        <b>Backtesting Engine</b><br>
        Test quantitative trading strategies against historical data. Includes curve inversion and momentum strategies.
        </div>
        """, unsafe_allow_html=True)
        
        strategy_type = st.selectbox("Select Strategy", 
                                     ["curve_inversion", "momentum"],
                                     format_func=lambda x: "Curve Inversion Strategy" if x == "curve_inversion" else "Moving Average Crossover")
        
        if st.button("📈 Run Backtest", use_container_width=True):
            with st.spinner("Running backtest..."):
                backtest_results = BacktestEngine.backtest_curve_strategy(yield_df, spreads, strategy_type)
                
                if backtest_results:
                    bt_metrics = st.columns(5)
                    bt_metrics[0].metric("Strategy Return", f"{backtest_results.get('total_return_strategy', backtest_results.get('total_return', 0))*100:.1f}%")
                    if "total_return_bh" in backtest_results:
                        bt_metrics[1].metric("Buy & Hold Return", f"{backtest_results.get('total_return_bh', 0)*100:.1f}%")
                    bt_metrics[2].metric("Sharpe Ratio", f"{backtest_results.get('sharpe_ratio_strategy', backtest_results.get('sharpe_ratio', 0)):.2f}")
                    bt_metrics[3].metric("Win Rate", f"{backtest_results.get('win_rate', 0)*100:.1f}%")
                    if "profit_factor" in backtest_results:
                        bt_metrics[4].metric("Profit Factor", f"{backtest_results.get('profit_factor', 0):.2f}")
                    
                    fig_bt = chart_backtest_results(backtest_results)
                    st.plotly_chart(fig_bt, use_container_width=True)
                else:
                    st.warning("Insufficient data for backtesting")

    # ========================================================================
    # TAB 4: VOLATILITY ANALYTICS
    # ========================================================================
    with main_tabs[4]:
        st.subheader("📉 Options-Implied Volatility Analysis")
        st.markdown("""
        <div class="note-box">
        <b>Volatility Analytics Suite</b><br>
        Real-time analysis of VIX (S&P 500 volatility). Includes regime detection, spike identification, and vol-of-vol calculations.
        </div>
        """, unsafe_allow_html=True)
        
        if volatility_df is not None and not volatility_df.empty and "^VIX" in volatility_df.columns:
            vix_analysis = VolatilityAnalyzer.calculate_volatility_regime(volatility_df["^VIX"])
            
            vix_metrics = st.columns(5)
            vix_metrics[0].metric("Current VIX", f"{vix_analysis['current_vix']:.2f}")
            vix_metrics[1].metric("Regime", vix_analysis['regime'])
            vix_metrics[2].metric("Percentile", vix_analysis['vix_percentile'])
            vix_metrics[3].metric("Mean VIX", f"{vix_analysis['mean_vix']:.2f}")
            vix_metrics[4].metric("Max VIX (2Y)", f"{vix_analysis['max_vix']:.2f}")
            
            st.markdown(f"""
            <div class="warning-box" style="border-left-color: {vix_analysis.get('color', 'orange')}">
            <b>📊 Volatility Outlook:</b> {vix_analysis['outlook']}<br>
            <b>Term Structure:</b> {vix_analysis.get('term_structure', 'N/A')} (Current vs 20-day MA)
            </div>
            """, unsafe_allow_html=True)
            
            fig_vix = chart_volatility_dashboard(volatility_df["^VIX"], vix_analysis)
            st.plotly_chart(fig_vix, use_container_width=True)
            
            # Volatility spike detection
            spikes = VolatilityAnalyzer.identify_volatility_spikes(volatility_df["^VIX"])
            if not spikes.empty:
                st.subheader("⚠️ Recent Volatility Spike Events")
                st.dataframe(spikes.tail(10), use_container_width=True)
        else:
            st.info("Volatility data unavailable. VIX data will appear when market data is accessible.")

    # ========================================================================
    # TAB 5: CORRELATIONS
    # ========================================================================
    with main_tabs[5]:
        st.subheader("🔄 Cross-Asset Correlation Analysis")
        st.markdown("""
        <div class="note-box">
        <b>Cross-Asset Correlation Framework</b><br>
        Analyze relationships between Treasury yields and major asset classes (equities, FX, commodities).
        </div>
        """, unsafe_allow_html=True)
        
        if correlation_df is not None and not correlation_df.empty:
            all_assets = pd.concat([yield_df["10Y"], correlation_df], axis=1).dropna()
            all_assets.columns = ["10Y Yield"] + list(correlation_df.columns)
            corr_matrix = CorrelationAnalyzer.calculate_correlation_matrix(all_assets)
            
            if not corr_matrix.empty:
                fig_corr = chart_correlation_heatmap(corr_matrix)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Rolling correlations
                st.subheader("Rolling Correlations with 10Y Yield (60-day window)")
                rolling_corr = CorrelationAnalyzer.calculate_rolling_correlations(
                    yield_df["10Y"], correlation_df, window=60
                )
                
                if not rolling_corr.empty:
                    fig_rolling = go.Figure()
                    for col in rolling_corr.columns:
                        fig_rolling.add_trace(go.Scatter(
                            x=rolling_corr.index, y=rolling_corr[col],
                            mode='lines', name=col.replace("Correlation_10Y_", ""),
                            line=dict(width=1.5)
                        ))
                    fig_rolling.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_rolling.add_hline(y=0.5, line_dash="dash", line_color="green")
                    fig_rolling.add_hline(y=-0.5, line_dash="dash", line_color="red")
                    st.plotly_chart(create_chart_layout(fig_rolling, "Rolling Correlations with 10Y Yield", 
                                                       "Correlation Coefficient", 500), use_container_width=True)
        else:
            st.info("Correlation data unavailable. Will appear when market data is accessible.")

    # ========================================================================
    # TAB 6: RESEARCH
    # ========================================================================
    with main_tabs[6]:
        st.subheader("📐 Nelson-Siegel Model Parameters")
        
        if ns_result:
            st.dataframe(pd.DataFrame({
                "Parameter": ["β₀ (Level Factor)", "β₁ (Slope Factor)", "β₂ (Curvature Factor)", 
                             "λ (Decay Rate)", "RMSE (bps)", "MAE (bps)", "R²"],
                "Value": [f"{ns_result['params'][0]:.4f}", f"{ns_result['params'][1]:.4f}",
                         f"{ns_result['params'][2]:.4f}", f"{ns_result['params'][3]:.4f}",
                         f"{ns_result['rmse']*100:.2f}", f"{ns_result['mae']*100:.2f}",
                         f"{ns_result['r_squared']:.4f}"]
            }), use_container_width=True, hide_index=True)
            
            st.markdown("""
            <div class="note-box">
            <b>📐 Nelson-Siegel Model Parameter Interpretation</b><br><br>
            <b>β₀ (Level Factor):</b> Represents the long-term equilibrium interest rate. Higher values indicate expectations of higher future rates.<br><br>
            <b>β₁ (Slope Factor):</b> Captures the difference between short and long-term rates. Negative values indicate an inverted curve (recession signal).<br><br>
            <b>β₂ (Curvature Factor):</b> Measures the hump or dip in the medium-term section of the curve.<br><br>
            <b>λ (Decay Rate):</b> Determines where the slope and curvature factors have maximum impact.<br><br>
            <b>RMSE & MAE:</b> Model fit quality metrics. Values below 5bps indicate excellent fit.
            </div>
            """, unsafe_allow_html=True)
            
            # Residual chart
            fig_resid = go.Figure()
            fig_resid.add_trace(go.Bar(
                x=selected_cols, y=ns_result['residuals'] * 100,
                name="Residuals", marker_color=COLORS["accent"], opacity=0.7
            ))
            fig_resid.add_hline(y=0, line_dash="dash", line_color=COLORS["text_secondary"])
            fig_resid.add_hrect(y0=-5, y1=5, line_width=0, fillcolor="green", opacity=0.1)
            st.plotly_chart(create_chart_layout(fig_resid, "Model Residuals by Maturity", 
                                               "Residual (bps)", 450, "Tenor"), use_container_width=True)

    # ========================================================================
    # TAB 7: EXPORT
    # ========================================================================
    with main_tabs[7]:
        st.subheader("💾 Export Data")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        with export_col1:
            st.download_button("📊 Yield Data (CSV)", 
                              yield_df.to_csv().encode("utf-8"),
                              f"yield_data_{datetime.now():%Y%m%d_%H%M%S}.csv")
            st.download_button("📈 Spreads (CSV)", 
                              spreads.to_csv().encode("utf-8"),
                              f"spreads_{datetime.now():%Y%m%d_%H%M%S}.csv")
        with export_col2:
            if ns_result:
                st.download_button("📐 Model Parameters", 
                                  pd.DataFrame(ns_result['params']).to_csv().encode("utf-8"),
                                  f"ns_params_{datetime.now():%Y%m%d_%H%M%S}.csv")
            if volatility_df is not None:
                st.download_button("📉 Volatility Data", 
                                  volatility_df.to_csv().encode("utf-8"),
                                  f"volatility_{datetime.now():%Y%m%d_%H%M%S}.csv")
        with export_col3:
            if correlation_df is not None:
                st.download_button("🔄 Correlation Data", 
                                  correlation_df.to_csv().encode("utf-8"),
                                  f"correlations_{datetime.now():%Y%m%d_%H%M%S}.csv")
        
        st.markdown("---")
        st.markdown("""
        <div class="note-box">
        <b>📦 Deployment Requirements</b><br><br>
        Create a <code>requirements.txt</code> file with:
        <pre>
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
requests>=2.31.0
yfinance>=0.2.28
scipy>=1.10.0
scikit-learn>=1.3.0
        </pre>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#667085; font-size:0.75rem; font-family: monospace;'>"
        "Institutional Quantitative Platform | Monte Carlo | ML Forecasting | Backtesting | Volatility Analytics | Cross-Asset Correlations<br>"
        "MK Istanbul Fintech LabGEN © 2026 | All Rights Reserved"
        "</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
