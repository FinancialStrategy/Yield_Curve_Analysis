import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# FRED API CONFIGURATION
# =============================================================================

FRED_SERIES = {
    '3M': 'DGS3MO',
    '2Y': 'DGS2',
    '5Y': 'DGS5',
    '10Y': 'DGS10',
    '30Y': 'DGS30'
}

BOND_ETFS = ['TLT', 'IEF', 'SHY', 'BND', 'GOVT']

# =============================================================================
# FRED API FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600)
def fetch_fred_series(api_key, series_id, start_date, end_date):
    """Fetch data from FRED API"""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date.strftime('%Y-%m-%d'),
        'observation_end': end_date.strftime('%Y-%m-%d'),
        'sort_order': 'asc'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        observations = data.get('observations', [])
        dates = []
        values = []
        for obs in observations:
            value = obs.get('value')
            if value not in ('.', None):
                dates.append(pd.to_datetime(obs['date']))
                values.append(float(value))
        
        if dates:
            return pd.Series(values, index=dates, name=series_id)
        return pd.Series(dtype='float64')
    except Exception:
        return pd.Series(dtype='float64')

@st.cache_data(ttl=3600)
def fetch_all_yield_data(api_key, start_date, end_date):
    """Fetch all yield data from FRED"""
    data = {}
    for name, series_id in FRED_SERIES.items():
        series = fetch_fred_series(api_key, series_id, start_date, end_date)
        if not series.empty:
            data[name] = series
    
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    df = df.ffill().bfill()
    return df

def validate_fred_api_key(api_key):
    """Validate FRED API key"""
    if not api_key or len(api_key) < 10:
        return False
    
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {'series_id': 'DGS10', 'api_key': api_key, 'file_type': 'json', 'limit': 1}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return 'observations' in data
    except:
        return False

# =============================================================================
# YAHOO FINANCE FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600)
def fetch_yahoo_data(ticker, start_date, end_date):
    """Fetch data from Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if not data.empty:
            if 'Adj Close' in data.columns:
                return data['Adj Close'].tz_localize(None)
            elif 'Close' in data.columns:
                return data['Close'].tz_localize(None)
    except:
        pass
    return pd.Series(dtype='float64')

# =============================================================================
# SAFE VALUE EXTRACTION HELPER
# =============================================================================

def safe_float(series, default=0.0):
    """Safely extract float value from Series or scalar"""
    if series is None:
        return default
    if isinstance(series, pd.Series):
        if series.empty:
            return default
        val = series.iloc[-1]
        if pd.isna(val):
            return default
        return float(val)
    if isinstance(series, (int, float)):
        return float(series)
    return default

def safe_series_val(series, default=0.0):
    """Safely get last value from Series"""
    if series is None or series.empty:
        return default
    val = series.iloc[-1]
    if pd.isna(val):
        return default
    return float(val)

# =============================================================================
# MODULE 1: YIELD CURVE ANALYSIS
# =============================================================================

def plot_yield_curve(df, selected_date):
    """Create yield curve plot"""
    if df.empty:
        return None
    
    available_dates = df.index
    closest_date = available_dates[available_dates <= pd.Timestamp(selected_date)].max()
    if pd.isnull(closest_date):
        return None
    
    maturities = {'3M': 0.25, '2Y': 2, '5Y': 5, '10Y': 10, '30Y': 30}
    
    fig = go.Figure()
    
    # Current curve
    current = df.loc[closest_date]
    x_vals, y_vals = [], []
    for mat, year in maturities.items():
        if mat in current.index and pd.notna(current[mat]):
            x_vals.append(year)
            y_vals.append(current[mat])
    
    if x_vals:
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', 
                                 name=closest_date.strftime('%Y-%m-%d'),
                                 line=dict(color='#2c5f8a', width=3), marker=dict(size=10)))
    
    # 1 year ago
    year_ago = available_dates[available_dates <= (closest_date - pd.DateOffset(years=1))].max()
    if pd.notna(year_ago) and year_ago in df.index:
        hist = df.loc[year_ago]
        x_h, y_h = [], []
        for mat, year in maturities.items():
            if mat in hist.index and pd.notna(hist[mat]):
                x_h.append(year)
                y_h.append(hist[mat])
        if x_h:
            fig.add_trace(go.Scatter(x=x_h, y=y_h, mode='lines+markers', name='1 Year Ago',
                                     line=dict(color='#c17f3a', width=2, dash='dash'), marker=dict(size=8)))
    
    fig.update_layout(title='U.S. Treasury Yield Curve', xaxis_title='Years to Maturity',
                     yaxis_title='Yield (%)', template='plotly_white', height=500)
    return fig

# =============================================================================
# MODULE 2: SPREAD ANALYSIS
# =============================================================================

def calculate_spreads(df):
    """Calculate yield spreads"""
    if df.empty:
        return pd.DataFrame()
    
    spreads = pd.DataFrame(index=df.index)
    if '2Y' in df and '10Y' in df:
        spreads['2s10s'] = df['10Y'] - df['2Y']
    if '3M' in df and '10Y' in df:
        spreads['3m10y'] = df['10Y'] - df['3M']
    if '5Y' in df and '30Y' in df:
        spreads['5s30s'] = df['30Y'] - df['5Y']
    return spreads

def plot_spreads(spreads):
    """Plot spreads"""
    if spreads.empty:
        return None
    
    fig = go.Figure()
    for col in spreads.columns:
        fig.add_trace(go.Scatter(x=spreads.index, y=spreads[col], name=col.upper(),
                                 line=dict(width=2), fill='tozeroy', opacity=0.3))
    fig.add_hline(y=0, line_dash='dash', line_color='red')
    fig.update_layout(title='Treasury Yield Spreads', xaxis_title='Date', 
                     yaxis_title='Spread (%)', template='plotly_white', height=500)
    return fig

def get_recession_probability(spreads):
    """Calculate recession probability from 2s10s spread"""
    if spreads.empty or '2s10s' not in spreads:
        return 0.5
    current = safe_series_val(spreads['2s10s'], 0.0)
    prob = 1 / (1 + np.exp(-(-current * 2 - 0.5)))
    return min(max(prob, 0.01), 0.99)

# =============================================================================
# MODULE 3: TRADING STRATEGY (STEP 5)
# =============================================================================

def create_trading_strategy(df):
    """Create trading strategy based on 10Y-3M spread"""
    if df.empty or '10Y' not in df or '3M' not in df:
        return None
    
    data = df.copy()
    data['Spread'] = data['10Y'] - data['3M']
    data['Signal'] = 0
    data.loc[data['Spread'] < 0, 'Signal'] = 1
    data.loc[data['Spread'] > 0, 'Signal'] = -1
    
    return data[['Spread', 'Signal']]

def plot_trading_signals(strategy_data):
    """Plot trading signals"""
    if strategy_data is None or strategy_data.empty:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strategy_data.index, y=strategy_data['Signal'],
                             mode='lines', name='Signal', line=dict(color='#2c5f8a', width=2)))
    fig.add_hline(y=1, line_dash='dash', line_color='green', annotation_text="BUY")
    fig.add_hline(y=-1, line_dash='dash', line_color='red', annotation_text="SELL")
    fig.update_layout(title='Trading Signals (10Y-3M Spread)', xaxis_title='Date',
                     yaxis_title='Signal (1=Buy, -1=Sell)', template='plotly_white', height=500)
    return fig

# =============================================================================
# MODULE 4: BACKTEST (STEP 6)
# =============================================================================

def backtest_strategy(df, etf_prices):
    """Backtest the trading strategy"""
    if df.empty or etf_prices.empty:
        return None
    
    strategy = create_trading_strategy(df)
    if strategy is None:
        return None
    
    common_idx = etf_prices.index.intersection(strategy.index)
    if len(common_idx) == 0:
        return None
    
    signals = strategy['Signal'].reindex(common_idx)
    returns = etf_prices.pct_change().reindex(common_idx)
    
    strategy_returns = signals.shift(1) * returns
    strategy_returns = strategy_returns.fillna(0)
    
    cumulative_strategy = (1 + strategy_returns).cumprod()
    cumulative_benchmark = (1 + returns.fillna(0)).cumprod()
    
    total_return_strategy = safe_series_val(cumulative_strategy - 1, 0.0)
    total_return_benchmark = safe_series_val(cumulative_benchmark - 1, 0.0)
    
    sharpe = float((strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)) if strategy_returns.std() > 0 else 0.0
    
    rolling_max = cumulative_strategy.expanding().max()
    drawdown = (cumulative_strategy - rolling_max) / rolling_max
    max_drawdown = safe_series_val(drawdown, 0.0)
    
    non_zero = strategy_returns[strategy_returns != 0]
    win_rate = float((non_zero > 0).sum() / len(non_zero)) if len(non_zero) > 0 else 0.0
    
    return {
        'cumulative_strategy': cumulative_strategy,
        'cumulative_benchmark': cumulative_benchmark,
        'total_return_strategy': total_return_strategy,
        'total_return_benchmark': total_return_benchmark,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }

def plot_backtest_results(results):
    """Plot backtest results"""
    if results is None:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results['cumulative_strategy'].index, y=results['cumulative_strategy'].values,
                             name='Strategy', line=dict(color='#2c5f8a', width=2)))
    fig.add_trace(go.Scatter(x=results['cumulative_benchmark'].index, y=results['cumulative_benchmark'].values,
                             name='Buy & Hold', line=dict(color='#c17f3a', width=2, dash='dash')))
    fig.update_layout(title='Cumulative Returns', xaxis_title='Date', yaxis_title='Return',
                     template='plotly_white', height=500)
    return fig

# =============================================================================
# MODULE 5: NELSON-SIEGEL MODEL
# =============================================================================

def nelson_siegel(tau, beta0, beta1, beta2, lambda1):
    """Nelson-Siegel function"""
    tau = np.asarray(tau, dtype=float)
    x = lambda1 * np.where(tau == 0, 1e-8, tau)
    term1 = (1 - np.exp(-x)) / x
    term2 = term1 - np.exp(-x)
    return beta0 + beta1 * term1 + beta2 * term2

def fit_nelson_siegel(maturities, yields):
    """Fit Nelson-Siegel model"""
    if len(maturities) == 0:
        return None
    
    def objective(params):
        fitted = nelson_siegel(maturities, *params)
        return np.sum((yields - fitted) ** 2)
    
    bounds = [(yields.min() - 2, yields.max() + 2), (-15, 15), (-15, 15), (0.01, 5)]
    
    best_result = None
    best_fun = np.inf
    
    for _ in range(5):
        x0 = [np.random.uniform(a, b) for a, b in bounds]
        result = minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B')
        if result.success and result.fun < best_fun:
            best_result = result
            best_fun = result.fun
    
    if best_result is None:
        return None
    
    fitted = nelson_siegel(maturities, *best_result.x)
    rmse = np.sqrt(np.mean((yields - fitted) ** 2))
    r2 = 1 - np.sum((yields - fitted) ** 2) / np.sum((yields - np.mean(yields)) ** 2)
    
    return {'params': best_result.x, 'rmse': rmse, 'r2': r2, 'fitted': fitted}

# =============================================================================
# MODULE 6: MONTE CARLO SIMULATION
# =============================================================================

def run_monte_carlo(initial_yield, mu, sigma, days, simulations=1000):
    """Run Monte Carlo simulation"""
    dt = 1 / 252
    paths = np.zeros((simulations, days))
    paths[:, 0] = initial_yield
    
    for i in range(1, days):
        z = np.random.standard_normal(simulations)
        paths[:, i] = paths[:, i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    mean = np.mean(paths, axis=0)
    lower_ci = np.percentile(paths, 2.5, axis=0)
    upper_ci = np.percentile(paths, 97.5, axis=0)
    
    return {'mean': mean, 'lower_ci': lower_ci, 'upper_ci': upper_ci, 'paths': paths}

def plot_monte_carlo(results, initial_yield, days):
    """Plot Monte Carlo results"""
    fig = go.Figure()
    x_axis = np.arange(days)
    
    fig.add_trace(go.Scatter(x=x_axis, y=results['upper_ci'], fill=None, mode='lines',
                             line=dict(color='rgba(0,0,0,0)'), showlegend=False))
    fig.add_trace(go.Scatter(x=x_axis, y=results['lower_ci'], fill='tonexty', mode='lines',
                             fillcolor='rgba(44, 95, 138, 0.2)', line=dict(color='rgba(0,0,0,0)'),
                             name='95% Confidence Interval'))
    fig.add_trace(go.Scatter(x=x_axis, y=results['mean'], mode='lines', name='Mean Path',
                             line=dict(color='#2c5f8a', width=2.5)))
    fig.add_trace(go.Scatter(x=[0], y=[initial_yield], mode='markers', name='Current',
                             marker=dict(size=12, color='#4a7c59', symbol='star')))
    
    fig.update_layout(title='Monte Carlo Simulation - 10Y Yield', xaxis_title='Trading Days',
                     yaxis_title='Yield (%)', template='plotly_white', height=500)
    return fig

# =============================================================================
# MODULE 7: MACHINE LEARNING
# =============================================================================

def prepare_ml_data(df, lags=5):
    """Prepare data for ML model"""
    if df.empty or '10Y' not in df:
        return None, None, None
    
    X, y = [], []
    for i in range(lags, len(df) - 1):
        features = []
        for col in df.columns:
            features.extend(df[col].iloc[i-lags:i].values)
        X.append(features)
        y.append(df['10Y'].iloc[i + 1])
    
    if not X:
        return None, None, None
    
    X_arr, y_arr = np.array(X), np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)
    
    return X_scaled, y_arr, scaler

def train_ml_model(X, y, model_type='Random Forest'):
    """Train ML model"""
    if X is None or len(X) < 50:
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'Gradient Boosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return {'rmse': rmse, 'r2': r2, 'model': model}

# =============================================================================
# MODULE 8: VOLATILITY ANALYSIS
# =============================================================================

def analyze_volatility(vix_series):
    """Analyze VIX volatility regime"""
    current = safe_series_val(vix_series, 0.0)
    
    if current <= 0:
        return {'regime': 'N/A', 'current_vix': 0.0}
    
    if current < 15:
        regime = "LOW VOLATILITY"
    elif current < 20:
        regime = "NORMAL VOLATILITY"
    elif current < 30:
        regime = "HIGH VOLATILITY"
    else:
        regime = "EXTREME VOLATILITY"
    
    return {'regime': regime, 'current_vix': current}

def plot_vix(vix_series):
    """Plot VIX chart"""
    if vix_series.empty:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vix_series.index, y=vix_series.values, fill='tozeroy',
                            line=dict(color='#c17f3a', width=2)))
    fig.add_hline(y=20, line_dash='dash', line_color='red')
    fig.update_layout(title='VIX Historical Chart', xaxis_title='Date',
                     yaxis_title='VIX', template='plotly_white', height=400)
    return fig

# =============================================================================
# MODULE 9: TECHNICAL ANALYSIS
# =============================================================================

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def plot_technical(price_data, ticker):
    """Plot technical indicators"""
    if price_data.empty:
        return None
    
    rsi = calculate_rsi(price_data)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        row_heights=[0.6, 0.4], subplot_titles=(f'{ticker} Price', 'RSI (14)'))
    
    fig.add_trace(go.Scatter(x=price_data.index, y=price_data.values, name='Price',
                             line=dict(color='#2c5f8a', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi.values, name='RSI',
                             line=dict(color='#c17f3a', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='red', row=2, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='green', row=2, col=1)
    
    fig.update_layout(title=f'Technical Analysis - {ticker}', height=600, template='plotly_white')
    return fig

# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="Yield Curve Analysis Platform", page_icon="📈", layout="wide")

if 'api_key_validated' not in st.session_state:
    st.session_state.api_key_validated = False

st.title("📈 Yield Curve Analysis Platform")
st.markdown("*FRED API | 9 Professional Modules*")

# API Key Input
if not st.session_state.api_key_validated:
    st.markdown("""
    ### 🔑 FRED API Key Required
    
    Get your free API key from [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)
    """)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        api_key = st.text_input("FRED API Key", type="password")
        if st.button("Validate & Connect", use_container_width=True):
            if validate_fred_api_key(api_key):
                st.session_state.api_key = api_key
                st.session_state.api_key_validated = True
                st.rerun()
            else:
                st.error("Invalid API key")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    start = st.date_input("Start Date", start_date, max_value=end_date)
    end = st.date_input("End Date", end_date, max_value=end_date)
    
    st.markdown("---")
    st.header("📊 Modules")
    
    m1 = st.checkbox("1. Yield Curve Analysis", value=True)
    m2 = st.checkbox("2. Spread Analysis", value=True)
    m3 = st.checkbox("3. Trading Strategy (Step 5)", value=True)
    m4 = st.checkbox("4. Strategy Backtest (Step 6)", value=True)
    m5 = st.checkbox("5. Nelson-Siegel Model", value=True)
    m6 = st.checkbox("6. Monte Carlo Simulation", value=True)
    m7 = st.checkbox("7. Machine Learning Forecast", value=True)
    m8 = st.checkbox("8. Volatility Analysis", value=True)
    m9 = st.checkbox("9. Technical Analysis", value=True)
    
    st.markdown("---")
    
    if m4:
        st.subheader("Backtest Settings")
        selected_etf = st.selectbox("Select ETF", BOND_ETFS, index=0)
    
    if m6:
        st.subheader("Monte Carlo Settings")
        mc_sims = st.slider("Simulations", 500, 5000, 1000, 500)
        mc_days = st.slider("Horizon (days)", 50, 500, 252, 50)
    
    run = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Main
if run:
    with st.spinner("Loading data from FRED..."):
        
        # Fetch yield data
        df = fetch_all_yield_data(st.session_state.api_key, start, end)
        
        if df.empty:
            st.error("Failed to fetch data. Check API key and date range.")
            if st.button("Reset API Key"):
                st.session_state.api_key_validated = False
                st.rerun()
            st.stop()
        
        # Fetch market data
        vix_data = fetch_yahoo_data('^VIX', start, end)
        etf_data = fetch_yahoo_data(selected_etf if m4 else 'TLT', start, end)
        
        # Calculate spreads
        spreads = calculate_spreads(df)
        recession_prob = get_recession_probability(spreads)
        
        # SAFE CURRENT METRICS EXTRACTION - FIXED!
        current_10y = safe_series_val(df['10Y'], 0.0) if '10Y' in df else 0.0
        spread_val = safe_series_val(spreads['2s10s'], 0.0) if '2s10s' in spreads else 0.0
        vix_val = safe_series_val(vix_data, 0.0)
        
        # Display KPIs
        st.subheader("📊 Market Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("10Y Yield", f"{current_10y:.2f}%")
        with col2:
            st.metric("2s10s Spread", f"{spread_val:.2f}%")
        with col3:
            st.metric("Recession Probability", f"{recession_prob:.1%}")
        with col4:
            st.metric("VIX", f"{vix_val:.2f}")
        
        if spread_val < 0:
            st.warning("⚠️ YIELD CURVE IS INVERTED! Historically signals recession within 6-18 months.")
        
        st.success(f"✅ Data loaded: {len(df)} trading days | {start} to {end}")
        
        # Module 1: Yield Curve
        if m1:
            with st.expander("📈 Module 1: Yield Curve Analysis", expanded=True):
                fig = plot_yield_curve(df, end)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    steepness = safe_series_val(df['10Y'] - df['2Y'], 0.0) if '2Y' in df else 0.0
                    st.metric("Curve Steepness (10Y-2Y)", f"{steepness:.2f}%")
                    st.metric("Short End (3M)", f"{safe_series_val(df['3M'], 0.0):.2f}%" if '3M' in df else "N/A")
                with col2:
                    st.metric("Long End (30Y)", f"{safe_series_val(df['30Y'], 0.0):.2f}%" if '30Y' in df else "N/A")
                    if len(df) > 1:
                        daily_change = safe_series_val(df['10Y'], 0.0) - safe_series_val(df['10Y'].shift(1), 0.0)
                        st.metric("10Y Daily Change", f"{daily_change:+.2f}%")
        
        # Module 2: Spread Analysis
        if m2 and not spreads.empty:
            with st.expander("📊 Module 2: Spread Analysis", expanded=True):
                fig = plot_spreads(spreads)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if '2s10s' in spreads:
                        st.metric("Current 2s10s", f"{safe_series_val(spreads['2s10s'], 0.0):.2f}%")
                    if '3m10y' in spreads:
                        st.metric("Current 3m10y", f"{safe_series_val(spreads['3m10y'], 0.0):.2f}%")
                with col2:
                    if '5s30s' in spreads:
                        st.metric("Current 5s30s", f"{safe_series_val(spreads['5s30s'], 0.0):.2f}%")
                    st.metric("Recession Probability", f"{recession_prob:.1%}")
        
        # Module 3: Trading Strategy (Step 5)
        if m3:
            with st.expander("🎯 Module 3: Trading Strategy (Step 5)", expanded=True):
                st.markdown("**Strategy Logic:**")
                st.markdown("- **BUY (Signal = 1)**: When 10Y-3M Spread < 0 (Inverted Curve)")
                st.markdown("- **SELL (Signal = -1)**: When 10Y-3M Spread > 0 (Normal Curve)")
                
                strategy = create_trading_strategy(df)
                if strategy is not None:
                    st.markdown("**Recent Signals (Last 10 Days)**")
                    display_df = strategy.tail(10).copy()
                    display_df['Spread'] = display_df['Spread'].apply(lambda x: f"{x:.2f}%")
                    st.dataframe(display_df, use_container_width=True)
                    
                    fig = plot_trading_signals(strategy)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        
        # Module 4: Strategy Backtest (Step 6)
        if m4 and not etf_data.empty:
            with st.expander("💰 Module 4: Strategy Backtest (Step 6)", expanded=True):
                st.markdown(f"**Backtest Results - {selected_etf} ETF**")
                
                backtest_results = backtest_strategy(df, etf_data)
                if backtest_results:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Strategy Return", f"{backtest_results['total_return_strategy']:.2%}")
                        st.metric("Sharpe Ratio", f"{backtest_results['sharpe_ratio']:.2f}")
                    with col2:
                        st.metric("Benchmark Return", f"{backtest_results['total_return_benchmark']:.2%}")
                        st.metric("Max Drawdown", f"{backtest_results['max_drawdown']:.2%}")
                    with col3:
                        excess = backtest_results['total_return_strategy'] - backtest_results['total_return_benchmark']
                        st.metric("Excess Return", f"{excess:.2%}")
                        st.metric("Win Rate", f"{backtest_results['win_rate']:.2%}")
                    
                    fig = plot_backtest_results(backtest_results)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Backtest failed. Try different date range.")
        
        # Module 5: Nelson-Siegel Model
        if m5:
            with st.expander("📐 Module 5: Nelson-Siegel Model", expanded=True):
                maturities = np.array([0.25, 2, 5, 10, 30])
                yields_list = []
                for m in ['3M', '2Y', '5Y', '10Y', '30Y']:
                    if m in df:
                        yields_list.append(safe_series_val(df[m], 0.0))
                    else:
                        yields_list.append(0.0)
                yields_arr = np.array(yields_list)
                
                ns_result = fit_nelson_siegel(maturities, yields_arr)
                
                if ns_result:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Model RMSE", f"{ns_result['rmse']*100:.2f} bps")
                        st.metric("Model R²", f"{ns_result['r2']:.4f}")
                    with col2:
                        params_df = pd.DataFrame({
                            'Parameter': ['β₀ (Level)', 'β₁ (Slope)', 'β₂ (Curvature)', 'λ (Decay)'],
                            'Value': [f"{ns_result['params'][0]:.4f}", f"{ns_result['params'][1]:.4f}",
                                     f"{ns_result['params'][2]:.4f}", f"{ns_result['params'][3]:.4f}"]
                        })
                        st.dataframe(params_df, hide_index=True, use_container_width=True)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=maturities, y=yields_arr, mode='markers', name='Actual Data',
                                            marker=dict(size=12, color='#2c5f8a')))
                    fig.add_trace(go.Scatter(x=np.linspace(0.25, 30, 100), y=ns_result['fitted'],
                                            mode='lines', name='NS Fit', line=dict(color='#c17f3a', width=2)))
                    fig.update_layout(title='Nelson-Siegel Model Fit', xaxis_title='Maturity (Years)',
                                     yaxis_title='Yield (%)', template='plotly_white', height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not fit Nelson-Siegel model")
        
        # Module 6: Monte Carlo Simulation
        if m6:
            with st.expander("🎲 Module 6: Monte Carlo Simulation", expanded=True):
                if st.button("Run Monte Carlo Simulation", use_container_width=True):
                    with st.spinner(f"Running {mc_sims} simulations..."):
                        returns = df['10Y'].pct_change().dropna()
                        mu = float(returns.mean() * 252) if len(returns) > 0 else 0.0
                        sigma = float(returns.std() * np.sqrt(252)) if len(returns) > 0 else 0.1
                        
                        mc_results = run_monte_carlo(current_10y, mu, sigma, mc_days, mc_sims)
                        fig = plot_monte_carlo(mc_results, current_10y, mc_days)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        terminal = mc_results['paths'][:, -1]
                        var_95 = float(np.percentile(terminal, 5))
                        expected = float(mc_results['mean'][-1])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Initial Yield", f"{current_10y:.2f}%")
                            st.metric("Expected Terminal", f"{expected:.2f}%")
                        with col2:
                            st.metric("Drift (μ)", f"{mu:.4f}")
                            st.metric("Volatility (σ)", f"{sigma:.4f}")
                        with col3:
                            st.metric("95% VaR", f"{var_95:.2f}%")
                            st.metric("Simulations", f"{mc_sims:,}")
        
        # Module 7: Machine Learning Forecast
        if m7:
            with st.expander("🤖 Module 7: Machine Learning Forecast", expanded=True):
                if st.button("Train ML Model", use_container_width=True):
                    with st.spinner("Training model..."):
                        X, y, _ = prepare_ml_data(df, 5)
                        if X is not None and len(X) > 50:
                            ml_result = train_ml_model(X, y, "Random Forest")
                            if ml_result:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("RMSE", f"{ml_result['rmse']*100:.2f} bps")
                                with col2:
                                    st.metric("R²", f"{ml_result['r2']:.3f}")
                                st.success(f"Model trained on {len(X)} samples")
                            else:
                                st.warning("Model training failed")
                        else:
                            st.warning(f"Insufficient data. Need >50 samples, have {len(X) if X is not None else 0}")
        
        # Module 8: Volatility Analysis
        if m8 and not vix_data.empty:
            with st.expander("⚡ Module 8: Volatility Analysis", expanded=True):
                vol_analysis = analyze_volatility(vix_data)
                st.metric("Current VIX", f"{vol_analysis['current_vix']:.2f}")
                st.info(f"📊 Regime: {vol_analysis['regime']}")
                
                fig = plot_vix(vix_data)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        # Module 9: Technical Analysis
        if m9:
            with st.expander("🛠 Module 9: Technical Analysis", expanded=True):
                tech_ticker = st.selectbox("Select Asset", ['TLT', 'IEF', 'SHY', 'SPY', 'QQQ'], key="tech_ticker")
                tech_data = fetch_yahoo_data(tech_ticker, start, end)
                
                if not tech_data.empty:
                    fig = plot_technical(tech_data, tech_ticker)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    rsi_vals = calculate_rsi(tech_data)
                    rsi_val = safe_series_val(rsi_vals, 50.0)
                    rsi_signal = "Oversold" if rsi_val < 30 else "Overbought" if rsi_val > 70 else "Neutral"
                    st.metric("Current RSI", f"{rsi_val:.1f}", delta=rsi_signal)
                else:
                    st.warning(f"Could not fetch {tech_ticker} data")

else:
    st.info("👈 Configure parameters and click 'Run Analysis'")
    
    with st.expander("📖 9 Professional Modules", expanded=True):
        st.markdown("""
        ### 🚀 Complete Fixed-Income Analytics Platform
        
        | # | Module | Description |
        |---|--------|-------------|
        | 1 | **Yield Curve Analysis** | Interactive U.S. Treasury yield curve visualization |
        | 2 | **Spread Analysis** | 2s10s, 3m10y, 5s30s spreads and forward rates |
        | 3 | **Trading Strategy (Step 5)** | Buy/Sell signals based on 10Y-3M spread |
        | 4 | **Strategy Backtest (Step 6)** | Backtest with TLT/IEF/SHY/BND ETFs |
        | 5 | **Nelson-Siegel Model** | Parametric yield curve modeling |
        | 6 | **Monte Carlo Simulation** | GBM simulation, VaR, confidence intervals |
        | 7 | **Machine Learning Forecast** | Random Forest yield prediction |
        | 8 | **Volatility Analysis** | VIX analysis and regime detection |
        | 9 | **Technical Analysis** | RSI indicators for ETFs |
        
        ### 🔑 Getting a FRED API Key
        
        1. Visit [FRED API website](https://fred.stlouisfed.org/docs/api/api_key.html)
        2. Click "Request API Key"
        3. Register for a free account
        4. Your API key will be emailed instantly
        5. Enter it in the sidebar to start
        
        ### ⚠️ Risk Warning
        Past performance does not guarantee future results. This platform is for educational purposes only.
        """)

st.markdown("---")
st.markdown(f"*Data source: FRED (Federal Reserve Economic Data) | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
