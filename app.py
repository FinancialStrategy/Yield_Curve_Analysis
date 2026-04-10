import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import yfinance as yf
from datetime import datetime, timedelta, date
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

RECESSION_SERIES = 'USREC'

BOND_ETFS = {
    'TLT': 'TLT',
    'IEF': 'IEF',
    'SHY': 'SHY',
    'BND': 'BND',
    'GOVT': 'GOVT'
}

VOLATILITY_TICKERS = {
    '^VIX': 'CBOE Volatility Index'
}

CORRELATION_TICKERS = {
    '^GSPC': 'S&P 500',
    '^IXIC': 'Nasdaq Composite',
    'GLD': 'Gold',
    'UUP': 'US Dollar Index'
}

# =============================================================================
# FRED API FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_series(api_key, series_id, start_date, end_date):
    """Fetch official data from FRED API"""
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
        if not observations:
            return pd.Series(dtype='float64')
        
        dates = []
        values = []
        for obs in observations:
            value = obs.get('value')
            if value != '.' and value is not None:
                dates.append(pd.to_datetime(obs['date']))
                values.append(float(value))
        
        if dates:
            return pd.Series(values, index=dates, name=series_id)
        return pd.Series(dtype='float64')
    
    except Exception as e:
        return pd.Series(dtype='float64')

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_yield_data(api_key, start_date, end_date):
    """Fetch all yield curve data from FRED"""
    data = {}
    
    for name, series_id in FRED_SERIES.items():
        series_data = fetch_fred_series(api_key, series_id, start_date, end_date)
        if not series_data.empty:
            data[name] = series_data
    
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

def validate_fred_api_key(api_key):
    """Validate FRED API key"""
    if not api_key or len(api_key) < 10:
        return False
    
    test_url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': 'DGS10',
        'api_key': api_key,
        'file_type': 'json',
        'limit': 1
    }
    
    try:
        response = requests.get(test_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return 'observations' in data
    except:
        return False

# =============================================================================
# YAHOO FINANCE FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_data(ticker, start_date, end_date):
    """Fetch price data from Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if not data.empty:
            if 'Adj Close' in data.columns:
                series = data['Adj Close']
            elif 'Close' in data.columns:
                series = data['Close']
            else:
                series = data.iloc[:, 0]
            series.index = series.index.tz_localize(None)
            return series
    except Exception as e:
        pass
    return pd.Series(dtype='float64')

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_tlt_data(start_date, end_date):
    """Fetch TLT price data specifically for backtesting"""
    try:
        data = yf.download('TLT', start=start_date, end=end_date, progress=False)
        if not data.empty:
            if 'Adj Close' in data.columns:
                series = data['Adj Close']
            elif 'Close' in data.columns:
                series = data['Close']
            else:
                series = data.iloc[:, 0]
            series.index = series.index.tz_localize(None)
            return series
    except Exception as e:
        pass
    return pd.Series(dtype='float64')

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_bundle(start_date, end_date):
    """Fetch market data for volatility and correlation analysis"""
    vol_data = {}
    for ticker, name in VOLATILITY_TICKERS.items():
        series = fetch_yahoo_data(ticker, start_date, end_date)
        if not series.empty:
            vol_data[name] = series
    
    corr_data = {}
    for ticker, name in CORRELATION_TICKERS.items():
        series = fetch_yahoo_data(ticker, start_date, end_date)
        if not series.empty:
            corr_data[name] = series
    
    return pd.DataFrame(vol_data), pd.DataFrame(corr_data)

# =============================================================================
# MODULE 1: YIELD CURVE ANALYSIS
# =============================================================================

def plot_yield_curve(df, selected_date):
    """Create interactive yield curve plot"""
    if df.empty:
        return None
    
    available_dates = df.index
    closest_date = available_dates[available_dates <= pd.Timestamp(selected_date)].max()
    
    if pd.isnull(closest_date):
        return None
    
    maturities = {'3M': 0.25, '2Y': 2, '5Y': 5, '10Y': 10, '30Y': 30}
    
    fig = go.Figure()
    
    current_yields = df.loc[closest_date]
    valid_maturities = []
    valid_yields = []
    
    for mat, year in maturities.items():
        if mat in current_yields.index and pd.notna(current_yields[mat]):
            valid_maturities.append(year)
            valid_yields.append(current_yields[mat])
    
    if valid_maturities:
        fig.add_trace(go.Scatter(
            x=valid_maturities,
            y=valid_yields,
            name=closest_date.strftime('%Y-%m-%d'),
            line=dict(color='#2c5f8a', width=3),
            mode='lines+markers',
            marker=dict(size=10)
        ))
    
    year_ago = closest_date - pd.DateOffset(years=1)
    year_ago = available_dates[available_dates <= year_ago].max()
    
    if pd.notna(year_ago) and year_ago in df.index:
        historical_yields = df.loc[year_ago]
        valid_hist = []
        valid_hist_yields = []
        
        for mat, year in maturities.items():
            if mat in historical_yields.index and pd.notna(historical_yields[mat]):
                valid_hist.append(year)
                valid_hist_yields.append(historical_yields[mat])
        
        if valid_hist:
            fig.add_trace(go.Scatter(
                x=valid_hist,
                y=valid_hist_yields,
                name='1 Year Ago',
                line=dict(color='#c17f3a', width=2, dash='dash'),
                mode='lines+markers',
                marker=dict(size=8)
            ))
    
    fig.update_layout(
        title='U.S. Treasury Yield Curve (Official FRED Data)',
        xaxis_title='Years to Maturity',
        yaxis_title='Yield (%)',
        template='plotly_white',
        showlegend=True,
        hovermode='x unified',
        height=500
    )
    
    return fig

# =============================================================================
# MODULE 2: SPREAD ANALYSIS
# =============================================================================

def calculate_spreads(df):
    """Calculate various yield spreads"""
    if df.empty:
        return pd.DataFrame()
    
    spreads = pd.DataFrame(index=df.index)
    
    if '2Y' in df.columns and '10Y' in df.columns:
        spreads['2s10s'] = df['10Y'] - df['2Y']
    
    if '3M' in df.columns and '10Y' in df.columns:
        spreads['3m10y'] = df['10Y'] - df['3M']
    
    if '5Y' in df.columns and '30Y' in df.columns:
        spreads['5s30s'] = df['30Y'] - df['5Y']
    
    return spreads

def plot_spreads(spreads):
    """Create spread visualization"""
    if spreads.empty:
        return None
    
    fig = go.Figure()
    colors = {'2s10s': '#2c5f8a', '3m10y': '#c17f3a', '5s30s': '#4a7c59'}
    
    for col in spreads.columns:
        fig.add_trace(go.Scatter(
            x=spreads.index,
            y=spreads[col],
            name=col.upper(),
            line=dict(color=colors.get(col, '#666'), width=2),
            fill='tozeroy',
            fillcolor='rgba(44, 95, 138, 0.1)'
        ))
    
    fig.add_hline(y=0, line_color='red', line_dash='dash')
    fig.add_hline(y=-0.5, line_color='orange', line_dash='dot')
    
    fig.update_layout(
        title='Treasury Yield Spreads',
        xaxis_title='Date',
        yaxis_title='Spread (%)',
        template='plotly_white',
        height=500
    )
    
    return fig

def calculate_forward_rates(df):
    """Calculate implied forward rates"""
    if df.empty:
        return pd.DataFrame()
    
    forwards = pd.DataFrame(index=df.index)
    maturities = {'3M': 0.25, '2Y': 2, '5Y': 5, '10Y': 10, '30Y': 30}
    
    pairs = [('3M', '2Y'), ('2Y', '5Y'), ('5Y', '10Y'), ('10Y', '30Y')]
    
    for short_term, long_term in pairs:
        if short_term in df.columns and long_term in df.columns:
            r1 = maturities[short_term]
            r2 = maturities[long_term]
            
            try:
                forward = (((1 + df[long_term] / 100) ** r2 / 
                           (1 + df[short_term] / 100) ** r1) ** 
                          (1 / (r2 - r1)) - 1) * 100
                forwards[f'{short_term}→{long_term}'] = forward
            except:
                pass
    
    return forwards

def get_recession_probability(spreads_df):
    """Calculate recession probability based on 2s10s spread"""
    if spreads_df.empty or '2s10s' not in spreads_df.columns:
        return 0.5
    
    current_spread = spreads_df['2s10s'].iloc[-1]
    if pd.isna(current_spread):
        return 0.5
    
    prob = 1 / (1 + np.exp(-(-current_spread * 2 - 0.5)))
    return min(max(prob, 0.01), 0.99)

# =============================================================================
# MODULE 3: YIELD SPREAD TRADING STRATEGY (STEP 5)
# =============================================================================

def develop_yield_spread_trading_strategy(yield_df):
    """Develop a simple trading strategy based on 10Y-3M spread"""
    if yield_df.empty:
        return None, None
    
    data = yield_df.copy()
    
    if '10Y' in data.columns and '3M' in data.columns:
        data['10Y-3M Spread'] = data['10Y'] - data['3M']
    else:
        return None, None
    
    data['Signal'] = 0
    data.loc[data['10Y-3M Spread'] < 0, 'Signal'] = 1
    data.loc[data['10Y-3M Spread'] > 0, 'Signal'] = -1
    
    return data[['10Y-3M Spread', 'Signal']], data

def plot_trading_signals(strategy_data):
    """Plot trading signals visualization"""
    if strategy_data is None or strategy_data.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=strategy_data.index,
        y=strategy_data['Signal'],
        mode='lines',
        name='Trading Signal',
        line=dict(color='#2c5f8a', width=2)
    ))
    
    fig.add_hline(y=1, line_dash='dash', line_color='green', 
                  annotation_text="BUY Signal (Inverted Curve)")
    fig.add_hline(y=-1, line_dash='dash', line_color='red', 
                  annotation_text="SELL Signal (Normal Curve)")
    fig.add_hline(y=0, line_dash='dot', line_color='gray')
    
    fig.update_layout(
        title='Trading Signals Based on Yield Spread (10Y-3M)',
        xaxis_title='Date',
        yaxis_title='Signal (1 = Buy, -1 = Sell, 0 = Neutral)',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    return fig

# =============================================================================
# MODULE 4: STRATEGY BACKTEST WITH TLT (STEP 6)
# =============================================================================

def backtest_yield_spread_strategy(yield_df, start_date, end_date, selected_etf='TLT'):
    """Backtest the yield spread strategy using ETF data"""
    if yield_df.empty:
        return None, None, None
    
    data = yield_df.copy()
    
    if '10Y' in data.columns and '3M' in data.columns:
        data['10Y-3M Spread'] = data['10Y'] - data['3M']
    else:
        return None, None, None
    
    data['Signal'] = 0
    data.loc[data['10Y-3M Spread'] < 0, 'Signal'] = 1
    data.loc[data['10Y-3M Spread'] > 0, 'Signal'] = -1
    
    # Fetch ETF data
    if selected_etf == 'TLT':
        etf_data = fetch_tlt_data(start_date, end_date)
    else:
        etf_data = fetch_yahoo_data(selected_etf, start_date, end_date)
    
    if etf_data.empty:
        return None, None, None
    
    etf_df = pd.DataFrame(etf_data)
    etf_df.columns = ['Price']
    etf_df['Returns'] = etf_df['Price'].pct_change()
    
    common_index = data.index.intersection(etf_df.index)
    if len(common_index) == 0:
        return None, None, None
    
    signals_aligned = data['Signal'].reindex(common_index)
    returns_aligned = etf_df['Returns'].reindex(common_index)
    
    strategy_returns = signals_aligned.shift(1) * returns_aligned
    strategy_returns = strategy_returns.fillna(0)
    
    cumulative_returns = (1 + strategy_returns).cumprod()
    benchmark_returns = (1 + returns_aligned.fillna(0)).cumprod()
    
    metrics = calculate_backtest_metrics(strategy_returns, returns_aligned.fillna(0), 
                                         cumulative_returns, benchmark_returns)
    
    backtest_results = {
        'strategy_returns': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'benchmark_returns': benchmark_returns,
        'signals': signals_aligned,
        'metrics': metrics,
        'spread_data': data['10Y-3M Spread'].reindex(common_index),
        'signal_data': signals_aligned
    }
    
    return backtest_results, data, etf_df

def calculate_backtest_metrics(strategy_returns, benchmark_returns, cum_strategy, cum_benchmark):
    """Calculate comprehensive backtest metrics"""
    if strategy_returns.empty:
        return {}
    
    strategy_clean = strategy_returns[strategy_returns != 0]
    ann_factor = np.sqrt(252)
    
    # Convert to float to avoid pandas Series issues
    total_return_strategy = float(cum_strategy.iloc[-1] - 1) if len(cum_strategy) > 0 else 0.0
    total_return_benchmark = float(cum_benchmark.iloc[-1] - 1) if len(cum_benchmark) > 0 else 0.0
    
    sharpe_strategy = float((strategy_returns.mean() / strategy_returns.std()) * ann_factor) if strategy_returns.std() > 0 else 0.0
    sharpe_benchmark = float((benchmark_returns.mean() / benchmark_returns.std()) * ann_factor) if benchmark_returns.std() > 0 else 0.0
    
    max_drawdown_strategy = float(calculate_max_drawdown(cum_strategy))
    max_drawdown_benchmark = float(calculate_max_drawdown(cum_benchmark))
    
    total_trades = len(strategy_clean)
    win_rate = float((strategy_clean > 0).sum() / total_trades) if total_trades > 0 else 0.0
    
    gross_profits = float(strategy_clean[strategy_clean > 0].sum()) if len(strategy_clean[strategy_clean > 0]) > 0 else 0.0
    gross_losses = float(abs(strategy_clean[strategy_clean < 0].sum())) if len(strategy_clean[strategy_clean < 0]) > 0 else 0.0
    profit_factor = float(gross_profits / gross_losses) if gross_losses > 0 else 0.0
    
    volatility = float(strategy_returns.std() * ann_factor)
    calmar_ratio = float(total_return_strategy / abs(max_drawdown_strategy)) if max_drawdown_strategy != 0 else 0.0
    
    return {
        'Total Return Strategy': total_return_strategy,
        'Total Return Benchmark': total_return_benchmark,
        'Excess Return': total_return_strategy - total_return_benchmark,
        'Sharpe Ratio Strategy': sharpe_strategy,
        'Sharpe Ratio Benchmark': sharpe_benchmark,
        'Max Drawdown Strategy': max_drawdown_strategy,
        'Max Drawdown Benchmark': max_drawdown_benchmark,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'Volatility (Annualized)': volatility,
        'Calmar Ratio': calmar_ratio,
        'Number of Trades': total_trades
    }

def calculate_max_drawdown(cumulative_returns):
    """Calculate maximum drawdown"""
    if cumulative_returns.empty or len(cumulative_returns) == 0:
        return 0.0
    
    cumulative_clean = cumulative_returns.fillna(1)
    rolling_max = cumulative_clean.expanding().max()
    drawdown = (cumulative_clean - rolling_max) / rolling_max
    return float(drawdown.min())

def plot_cumulative_returns(backtest_results):
    """Plot cumulative returns comparison"""
    if backtest_results is None:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=backtest_results['cumulative_returns'].index,
        y=backtest_results['cumulative_returns'].values,
        mode='lines',
        name='Yield Spread Strategy',
        line=dict(color='#2c5f8a', width=2.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=backtest_results['benchmark_returns'].index,
        y=backtest_results['benchmark_returns'].values,
        mode='lines',
        name='Buy & Hold',
        line=dict(color='#c17f3a', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Cumulative Returns of Yield Spread Strategy vs Buy & Hold',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        template='plotly_white',
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

# =============================================================================
# MODULE 5: NELSON-SIEGEL MODEL
# =============================================================================

class NelsonSiegelModel:
    @staticmethod
    def nelson_siegel(tau, beta0, beta1, beta2, lambda1):
        tau = np.asarray(tau, dtype=float)
        tau_safe = np.where(tau == 0, 1e-8, tau)
        x = lambda1 * tau_safe
        term1 = (1 - np.exp(-x)) / x
        term2 = term1 - np.exp(-x)
        return beta0 + beta1 * term1 + beta2 * term2
    
    @staticmethod
    def fit_curve(maturities, yields):
        if len(maturities) == 0 or len(yields) == 0:
            return None
        
        def objective(params):
            fitted = NelsonSiegelModel.nelson_siegel(maturities, *params)
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
        
        fitted = NelsonSiegelModel.nelson_siegel(maturities, *best_result.x)
        rmse = np.sqrt(np.mean((yields - fitted) ** 2))
        r2 = 1 - np.sum((yields - fitted) ** 2) / np.sum((yields - np.mean(yields)) ** 2)
        
        return {'params': best_result.x, 'fitted': fitted, 'rmse': rmse, 'r2': r2}

# =============================================================================
# MODULE 6: PCA AND FACTOR ANALYSIS
# =============================================================================

def calculate_factor_contributions(yield_df):
    """Calculate Level, Slope, Curvature factors"""
    if yield_df.empty:
        return pd.DataFrame()
    
    factors = pd.DataFrame(index=yield_df.index)
    
    if '10Y' in yield_df.columns:
        factors['Level'] = yield_df['10Y']
    
    if '10Y' in yield_df.columns and '3M' in yield_df.columns:
        factors['Slope'] = (yield_df['10Y'] - yield_df['3M']) * 100
    
    if all(x in yield_df.columns for x in ['3M', '5Y', '10Y']):
        factors['Curvature'] = (2 * yield_df['5Y'] - (yield_df['3M'] + yield_df['10Y'])) * 100
    
    return factors

def perform_pca_analysis(yield_df, n_components=3):
    """Perform PCA for risk factor analysis"""
    if yield_df.empty or len(yield_df) < 20:
        return None
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    returns = yield_df.pct_change().dropna()
    if returns.empty or returns.shape[1] < 2:
        return None
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(returns)
    
    n_comp = min(n_components, returns.shape[1], returns.shape[0] - 1)
    pca = PCA(n_components=n_comp)
    pca.fit(scaled_data)
    
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_comp)],
        index=returns.columns
    )
    
    return {
        'explained_variance': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'loadings': loadings,
        'n_components': n_comp
    }

# =============================================================================
# MODULE 7: MONTE CARLO SIMULATION
# =============================================================================

class MonteCarloSimulator:
    @staticmethod
    def simulate_gbm(initial_price, mu, sigma, days, simulations=1000):
        dt = 1 / 252
        paths = np.zeros((simulations, days))
        paths[:, 0] = initial_price
        
        for i in range(1, days):
            z = np.random.standard_normal(simulations)
            paths[:, i] = paths[:, i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
        return paths
    
    @staticmethod
    def calculate_confidence_intervals(paths, confidence=0.95):
        lower_percentile = (1 - confidence) / 2 * 100
        upper_percentile = (1 + confidence) / 2 * 100
        
        return {
            'mean': np.mean(paths, axis=0),
            'median': np.percentile(paths, 50, axis=0),
            'lower_ci': np.percentile(paths, lower_percentile, axis=0),
            'upper_ci': np.percentile(paths, upper_percentile, axis=0),
            'std': np.std(paths, axis=0)
        }
    
    @staticmethod
    def calculate_var(paths, confidence=0.95):
        return np.percentile(paths[:, -1], (1 - confidence) * 100)

def plot_monte_carlo(simulation_results, initial_value, horizon_days, model_name="GBM"):
    """Create Monte Carlo simulation chart"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Simulation Paths with Confidence Bands', 'Terminal Distribution',
                       'Mean Path with Confidence', 'Risk Metrics'),
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    x_axis = np.arange(horizon_days)
    
    # 1. Simulation paths with confidence bands
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=simulation_results['upper_ci'],
        fill=None,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=simulation_results['lower_ci'],
        fill='tonexty',
        mode='lines',
        fillcolor='rgba(44, 95, 138, 0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name='95% Confidence Interval'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=simulation_results['mean'],
        mode='lines',
        name='Mean Path',
        line=dict(color='#2c5f8a', width=3)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=[0],
        y=[initial_value],
        mode='markers',
        name='Current Value',
        marker=dict(size=12, color='#4a7c59', symbol='star')
    ), row=1, col=1)
    
    # 2. Terminal distribution
    terminal_values = simulation_results['mean'][-1] + np.random.normal(0, simulation_results['std'][-1], 1000)
    fig.add_trace(go.Histogram(
        x=terminal_values,
        nbinsx=50,
        name='Terminal Distribution',
        marker_color='#2c5f8a',
        opacity=0.7,
        showlegend=False
    ), row=1, col=2)
    
    fig.add_vline(x=simulation_results['mean'][-1], line_dash='dash', line_color='red',
                  annotation_text=f"Mean: {simulation_results['mean'][-1]:.2f}%", row=1, col=2)
    
    # 3. Mean path with confidence
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=simulation_results['mean'],
        mode='lines',
        name='Mean',
        line=dict(color='#2c5f8a', width=2)
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=simulation_results['mean'] + simulation_results['std'],
        mode='lines',
        name='+1 Std Dev',
        line=dict(color='#c17f3a', width=1, dash='dash')
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=simulation_results['mean'] - simulation_results['std'],
        mode='lines',
        name='-1 Std Dev',
        line=dict(color='#c17f3a', width=1, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(193, 127, 58, 0.1)'
    ), row=2, col=1)
    
    # 4. Risk metrics table
    var_95 = MonteCarloSimulator.calculate_var(terminal_values, 0.95)
    
    table_data = [
        ['Initial Value', f'{initial_value:.2f}%'],
        ['Expected Value', f'{simulation_results["mean"][-1]:.2f}%'],
        ['Median Value', f'{simulation_results["median"][-1]:.2f}%'],
        ['Standard Deviation', f'{simulation_results["std"][-1]:.2f}%'],
        ['95% VaR', f'{var_95:.2f}%'],
        ['Upside Probability', f'{(simulation_results["mean"][-1] > initial_value)*100:.1f}%']
    ]
    
    fig.add_trace(go.Table(
        header=dict(values=['Risk Metric', 'Value'],
                   fill_color='#2c5f8a',
                   align='center',
                   font=dict(color='white', size=12)),
        cells=dict(values=[list(zip(*table_data))[0], list(zip(*table_data))[1]],
                  fill_color='white',
                  align='center',
                  font=dict(size=11))
    ), row=2, col=2)
    
    fig.update_layout(
        title=f'<b>Monte Carlo Simulation - {model_name} Model</b><br>95% Confidence Interval | {horizon_days} Day Horizon',
        showlegend=True,
        template='plotly_white',
        height=800,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Trading Days Ahead", row=1, col=1)
    fig.update_yaxes(title_text="Yield (%)", row=1, col=1)
    fig.update_xaxes(title_text="Yield (%)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_xaxes(title_text="Trading Days Ahead", row=2, col=1)
    fig.update_yaxes(title_text="Yield (%)", row=2, col=1)
    
    return fig

# =============================================================================
# MODULE 8: MACHINE LEARNING FORECAST
# =============================================================================

class MLForecastModel:
    @staticmethod
    def prepare_features(yield_df, target_col='10Y', lags=5):
        if yield_df.empty or target_col not in yield_df.columns:
            return None, None, None
        
        X, y = [], []
        for i in range(lags, len(yield_df) - 1):
            features = []
            for col in yield_df.columns:
                features.extend(yield_df[col].iloc[i-lags:i].values)
            X.append(features)
            y.append(yield_df[target_col].iloc[i + 1])
        
        if not X:
            return None, None, None
        
        X_arr, y_arr = np.array(X), np.array(y)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)
        
        return X_scaled, y_arr, scaler
    
    @staticmethod
    def train_model(X, y, model_type='Random Forest', test_size=0.2):
        if X is None or len(X) == 0:
            return {}
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        if model_type == 'Gradient Boosting':
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': [f'Lagged_{i}' for i in range(X.shape[1])],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
        else:
            importance = pd.DataFrame()
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2, 'feature_importance': importance}

# =============================================================================
# MODULE 9: ADVANCED BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    @staticmethod
    def backtest_strategy(yield_df, spreads, etf_returns, strategy_type='Curve Inversion'):
        if yield_df.empty or spreads.empty or etf_returns.empty:
            return None
        
        if strategy_type == 'Curve Inversion':
            if '2s10s' not in spreads.columns:
                return None
            signals = spreads['2s10s'] < 0
        elif strategy_type == 'Momentum':
            if '2s10s' not in spreads.columns:
                return None
            momentum = spreads['2s10s'].diff(20)
            signals = (spreads['2s10s'] < 0) & (momentum < 0)
        elif strategy_type == 'Mean Reversion':
            if '2s10s' not in spreads.columns:
                return None
            zscore = (spreads['2s10s'] - spreads['2s10s'].rolling(252).mean()) / spreads['2s10s'].rolling(252).std()
            signals = zscore < -1.5
        else:
            return None
        
        common_idx = etf_returns.index.intersection(signals.index)
        if len(common_idx) == 0:
            return None
        
        signals_aligned = signals.reindex(common_idx).fillna(0).astype(int)
        returns_aligned = etf_returns.reindex(common_idx)
        
        strategy_returns = signals_aligned.shift(1) * returns_aligned
        strategy_returns = strategy_returns.fillna(0)
        
        cumulative_strategy = (1 + strategy_returns).cumprod()
        cumulative_benchmark = (1 + returns_aligned.fillna(0)).cumprod()
        
        metrics = BacktestEngine.calculate_metrics(strategy_returns, returns_aligned.fillna(0))
        
        return {
            'strategy_returns': strategy_returns,
            'cumulative_strategy': cumulative_strategy,
            'cumulative_benchmark': cumulative_benchmark,
            'signals': signals_aligned,
            'metrics': metrics
        }
    
    @staticmethod
    def calculate_metrics(strategy_returns, benchmark_returns):
        if strategy_returns.empty:
            return {}
        
        ann_factor = np.sqrt(252)
        
        # Convert to float to avoid pandas Series issues
        total_return_strategy = float((1 + strategy_returns).prod() - 1)
        total_return_benchmark = float((1 + benchmark_returns).prod() - 1)
        sharpe_ratio = float((strategy_returns.mean() / strategy_returns.std()) * ann_factor) if strategy_returns.std() > 0 else 0.0
        volatility = float(strategy_returns.std() * ann_factor)
        max_drawdown = float(BacktestEngine.calculate_max_drawdown(strategy_returns))
        
        non_zero_returns = strategy_returns[strategy_returns != 0]
        win_rate = float((strategy_returns > 0).sum() / len(non_zero_returns)) if len(non_zero_returns) > 0 else 0.0
        
        gross_profits = float(strategy_returns[strategy_returns > 0].sum()) if len(strategy_returns[strategy_returns > 0]) > 0 else 0.0
        gross_losses = float(abs(strategy_returns[strategy_returns < 0].sum())) if len(strategy_returns[strategy_returns < 0]) > 0 else 0.0
        profit_factor = float(gross_profits / gross_losses) if gross_losses > 0 else 0.0
        
        return {
            'Total Return Strategy': total_return_strategy,
            'Total Return Benchmark': total_return_benchmark,
            'Excess Return': total_return_strategy - total_return_benchmark,
            'Sharpe Ratio': sharpe_ratio,
            'Volatility': volatility,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor
        }
    
    @staticmethod
    def calculate_max_drawdown(returns):
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return float(drawdown.min())

# =============================================================================
# MODULE 10: VOLATILITY ANALYSIS
# =============================================================================

class VolatilityAnalyzer:
    @staticmethod
    def calculate_volatility_regime(vix_series):
        if vix_series.empty:
            return {'regime': 'N/A', 'outlook': 'Data unavailable', 'current_vix': 0}
        
        current_vix = vix_series.iloc[-1]
        
        if current_vix < 12:
            regime, outlook = "EXTREME COMPLACENCY", "High risk of volatility spike"
        elif current_vix < 15:
            regime, outlook = "LOW VOLATILITY", "Normal complacent market"
        elif current_vix < 20:
            regime, outlook = "NORMAL VOLATILITY", "Typical market conditions"
        elif current_vix < 25:
            regime, outlook = "ELEVATED VOLATILITY", "Increased uncertainty"
        elif current_vix < 35:
            regime, outlook = "HIGH VOLATILITY", "Market stress, consider hedging"
        else:
            regime, outlook = "EXTREME VOLATILITY", "Crisis conditions"
        
        return {'current_vix': current_vix, 'regime': regime, 'outlook': outlook}

# =============================================================================
# MODULE 11: TECHNICAL ANALYSIS
# =============================================================================

def calculate_technical_indicators(price_series):
    """Calculate technical indicators"""
    if price_series.empty:
        return pd.DataFrame()
    
    df = pd.DataFrame(index=price_series.index)
    df['Price'] = price_series
    
    df['SMA_20'] = price_series.rolling(20).mean()
    df['SMA_50'] = price_series.rolling(50).mean()
    df['EMA_12'] = price_series.ewm(span=12, adjust=False).mean()
    df['EMA_26'] = price_series.ewm(span=26, adjust=False).mean()
    
    delta = price_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    df['BB_Middle'] = price_series.rolling(20).mean()
    bb_std = price_series.rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    return df

def plot_technical_indicators(tech_df, ticker):
    """Create technical analysis plot"""
    if tech_df.empty:
        return None
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f'{ticker} Price with Indicators', 'RSI (14)', 'MACD')
    )
    
    fig.add_trace(go.Scatter(
        x=tech_df.index, y=tech_df['Price'],
        mode='lines', name='Price',
        line=dict(color='#2c5f8a', width=2)
    ), row=1, col=1)
    
    if 'SMA_20' in tech_df.columns:
        fig.add_trace(go.Scatter(
            x=tech_df.index, y=tech_df['SMA_20'],
            mode='lines', name='SMA 20',
            line=dict(color='#4a7c59', width=1.5)
        ), row=1, col=1)
    
    if 'SMA_50' in tech_df.columns:
        fig.add_trace(go.Scatter(
            x=tech_df.index, y=tech_df['SMA_50'],
            mode='lines', name='SMA 50',
            line=dict(color='#c17f3a', width=1.5)
        ), row=1, col=1)
    
    if 'BB_Upper' in tech_df.columns:
        fig.add_trace(go.Scatter(
            x=tech_df.index, y=tech_df['BB_Upper'],
            mode='lines', name='BB Upper',
            line=dict(color='gray', width=1, dash='dash')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=tech_df.index, y=tech_df['BB_Lower'],
            mode='lines', name='BB Lower',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty'
        ), row=1, col=1)
    
    if 'RSI' in tech_df.columns:
        fig.add_trace(go.Scatter(
            x=tech_df.index, y=tech_df['RSI'],
            mode='lines', name='RSI',
            line=dict(color='#c17f3a', width=1.5)
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash='dash', line_color='red', row=2, col=1)
        fig.add_hline(y=30, line_dash='dash', line_color='green', row=2, col=1)
    
    if 'MACD' in tech_df.columns:
        fig.add_trace(go.Scatter(
            x=tech_df.index, y=tech_df['MACD'],
            mode='lines', name='MACD',
            line=dict(color='#2c5f8a', width=1.5)
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=tech_df.index, y=tech_df['MACD_Signal'],
            mode='lines', name='Signal',
            line=dict(color='#c17f3a', width=1.5)
        ), row=3, col=1)
        
        colors = ['red' if x < 0 else 'green' for x in tech_df['MACD_Histogram']]
        fig.add_trace(go.Bar(
            x=tech_df.index, y=tech_df['MACD_Histogram'],
            name='Histogram', marker_color=colors
        ), row=3, col=1)
    
    fig.update_layout(
        title=f'Technical Analysis - {ticker}',
        height=800,
        template='plotly_white',
        showlegend=True
    )
    
    return fig

# =============================================================================
# MODULE 12: SCENARIO ANALYSIS
# =============================================================================

def generate_scenarios(yield_df):
    """Generate different market scenarios"""
    if yield_df.empty:
        return {}
    
    latest = yield_df.iloc[-1].copy()
    scenarios = {}
    
    bull = latest.copy()
    bull['3M'] = bull['3M'] - 0.15
    bull['2Y'] = bull['2Y'] - 0.20
    bull['5Y'] = bull['5Y'] - 0.25
    bull['10Y'] = bull['10Y'] - 0.30
    bull['30Y'] = bull['30Y'] - 0.25
    scenarios['Bull Steepener'] = bull
    
    bear = latest.copy()
    bear['3M'] = bear['3M'] + 0.25
    bear['2Y'] = bear['2Y'] + 0.20
    bear['5Y'] = bear['5Y'] + 0.15
    bear['10Y'] = bear['10Y'] + 0.10
    bear['30Y'] = bear['30Y'] + 0.05
    scenarios['Bear Flattener'] = bear
    
    recession = latest.copy()
    recession['3M'] = recession['3M'] - 0.50
    recession['2Y'] = recession['2Y'] - 0.60
    recession['5Y'] = recession['5Y'] - 0.70
    recession['10Y'] = recession['10Y'] - 0.80
    recession['30Y'] = recession['30Y'] - 0.60
    scenarios['Recession'] = recession
    
    inflation = latest.copy()
    inflation['3M'] = inflation['3M'] + 0.50
    inflation['2Y'] = inflation['2Y'] + 0.60
    inflation['5Y'] = inflation['5Y'] + 0.55
    inflation['10Y'] = inflation['10Y'] + 0.45
    inflation['30Y'] = inflation['30Y'] + 0.30
    scenarios['Inflation Shock'] = inflation
    
    return scenarios

def plot_scenario_comparison(current_curve, scenario_curve, scenario_name):
    """Plot scenario comparison"""
    maturities = [0.25, 2, 5, 10, 30]
    current_values = [current_curve['3M'], current_curve['2Y'], current_curve['5Y'], 
                      current_curve['10Y'], current_curve['30Y']]
    scenario_values = [scenario_curve['3M'], scenario_curve['2Y'], scenario_curve['5Y'], 
                       scenario_curve['10Y'], scenario_curve['30Y']]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=maturities, y=current_values,
        name='Current Curve',
        line=dict(color='#2c5f8a', width=3),
        mode='lines+markers',
        marker=dict(size=10)
    ))
    fig.add_trace(go.Scatter(
        x=maturities, y=scenario_values,
        name=scenario_name,
        line=dict(color='#c17f3a', width=2, dash='dash'),
        mode='lines+markers',
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=f'Scenario Analysis: {scenario_name}',
        xaxis_title='Maturity (Years)',
        yaxis_title='Yield (%)',
        template='plotly_white',
        height=500
    )
    
    return fig

# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

class CorrelationAnalyzer:
    @staticmethod
    def calculate_correlation_matrix(assets_df):
        if assets_df.empty:
            return pd.DataFrame()
        
        returns = assets_df.pct_change().dropna()
        return returns.corr()
    
    @staticmethod
    def plot_correlation_heatmap(correlation_matrix):
        if correlation_matrix.empty:
            return None
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Cross-Asset Correlation Matrix',
            height=600,
            template='plotly_white'
        )
        
        return fig

# =============================================================================
# STREAMLIT UI - MAIN APPLICATION
# =============================================================================

st.set_page_config(page_title="Bond Yield Curve Analysis Platform", page_icon="📈", layout="wide")

if 'api_key_validated' not in st.session_state:
    st.session_state.api_key_validated = False

st.title("📈 Bond Yield Curve Analysis Platform")
st.markdown("*Official FRED API | 12 Professional Analysis Modules*")

# API Key Management
if not st.session_state.api_key_validated:
    st.markdown("""
    ### 🔑 FRED API Key Required
    
    This platform uses **official Federal Reserve Economic Data (FRED)**.
    
    **Get your free API key:**
    1. Go to [FRED API website](https://fred.stlouisfed.org/docs/api/api_key.html)
    2. Click "Request API Key" and register for a free account
    3. Enter your API key below
    """)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        api_key = st.text_input("FRED API Key", type="password", placeholder="Enter your FRED API key")
        
        if st.button("🔐 Validate & Connect", use_container_width=True):
            if not api_key:
                st.error("Please enter an API key")
            else:
                with st.spinner("Validating FRED API key..."):
                    if validate_fred_api_key(api_key):
                        st.session_state.api_key = api_key
                        st.session_state.api_key_validated = True
                        st.success("✅ API key validated! Data will be fetched from FRED.")
                        st.rerun()
                    else:
                        st.error("❌ Invalid API key. Please check and try again.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    default_end = datetime.now()
    default_start = default_end - timedelta(days=365*2)
    
    start_date = st.date_input("Start Date", default_start, max_value=default_end)
    end_date = st.date_input("End Date", default_end, max_value=default_end)
    
    st.markdown("---")
    st.header("📊 12 Analysis Modules")
    
    # All 12 modules as checkboxes
    module1 = st.checkbox("1. Yield Curve Analysis", value=True)
    module2 = st.checkbox("2. Spread Analysis", value=True)
    module3 = st.checkbox("3. Yield Spread Trading Strategy", value=True)
    module4 = st.checkbox("4. Strategy Backtest with ETF", value=True)
    module5 = st.checkbox("5. Nelson-Siegel Model", value=True)
    module6 = st.checkbox("6. PCA & Factor Analysis", value=True)
    module7 = st.checkbox("7. Monte Carlo Simulation", value=True)
    module8 = st.checkbox("8. Machine Learning Forecast", value=True)
    module9 = st.checkbox("9. Advanced Backtest Engine", value=True)
    module10 = st.checkbox("10. Volatility Analysis", value=True)
    module11 = st.checkbox("11. Technical Analysis", value=True)
    module12 = st.checkbox("12. Scenario & Correlation Analysis", value=True)
    
    st.markdown("---")
    
    if module4:
        st.header("📈 Strategy Parameters")
        yield_etf = st.selectbox("Select ETF", ['TLT', 'IEF', 'SHY', 'BND'], index=0)
    
    if module7:
        st.header("🎲 Monte Carlo Parameters")
        mc_simulations = st.slider("Simulations", 500, 5000, 1000, 500)
        mc_horizon = st.slider("Horizon (days)", 5, 252, 252, 5)
    
    if module8:
        st.header("🤖 ML Parameters")
        ml_model_type = st.selectbox("Model Type", ["Random Forest", "Gradient Boosting"])
    
    if module9:
        st.header("🎯 Advanced Backtest")
        adv_etf = st.selectbox("ETF", list(BOND_ETFS.keys()), key="adv_etf")
        adv_strategy = st.selectbox("Strategy", ['Curve Inversion', 'Momentum', 'Mean Reversion'])
    
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Main content
if run_analysis:
    with st.spinner("Fetching data from FRED API..."):
        
        # Fetch data
        yield_df = fetch_all_yield_data(st.session_state.api_key, start_date, end_date)
        
        if yield_df.empty:
            st.error("Failed to fetch data. Please check your API key and date range.")
            if st.button("Reset API Key"):
                st.session_state.api_key_validated = False
                st.rerun()
            st.stop()
        
        # Calculate common data
        volatility_df, correlation_df = fetch_market_bundle(start_date, end_date)
        spreads = calculate_spreads(yield_df)
        forwards = calculate_forward_rates(yield_df)
        factors = calculate_factor_contributions(yield_df)
        
        # Current metrics
        current_10y = yield_df['10Y'].iloc[-1] if '10Y' in yield_df.columns else np.nan
        current_2y = yield_df['2Y'].iloc[-1] if '2Y' in yield_df.columns else np.nan
        current_spread = spreads['2s10s'].iloc[-1] if '2s10s' in spreads.columns else np.nan
        recession_prob = get_recession_probability(spreads)
        
        # Module 3: Yield Spread Strategy
        strategy_data, full_strategy_data = develop_yield_spread_trading_strategy(yield_df)
        
        # Module 4: Backtest
        backtest_results = None
        if module4:
            backtest_results, _, _ = backtest_yield_spread_strategy(yield_df, start_date, end_date, yield_etf)
        
        # Module 5: Nelson-Siegel
        maturities = np.array([0.25, 2, 5, 10, 30])
        current_yields = np.array([yield_df[m].iloc[-1] for m in ['3M', '2Y', '5Y', '10Y', '30Y']])
        ns_result = NelsonSiegelModel.fit_curve(maturities, current_yields) if module5 else None
        
        # Module 6: PCA
        pca_result = perform_pca_analysis(yield_df) if module6 else None
        
        # Module 9: Advanced Backtest
        adv_backtest_result = None
        if module9:
            etf_data_adv = fetch_yahoo_data(adv_etf, start_date, end_date)
            if not etf_data_adv.empty:
                etf_returns_adv = etf_data_adv.pct_change()
                adv_backtest_result = BacktestEngine.backtest_strategy(yield_df, spreads, etf_returns_adv, adv_strategy)
        
        # Module 12: Correlation Matrix
        corr_matrix = None
        if module12 and not correlation_df.empty:
            all_assets = pd.DataFrame(index=yield_df.index)
            all_assets['10Y'] = yield_df['10Y']
            for col in correlation_df.columns:
                all_assets[col] = correlation_df[col]
            all_assets = all_assets.dropna()
            if not all_assets.empty:
                corr_matrix = CorrelationAnalyzer.calculate_correlation_matrix(all_assets)
        
        # KPI Row
        st.subheader("📊 Current Market Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("10Y Yield", f"{current_10y:.2f}%" if not np.isnan(current_10y) else "N/A")
        with col2:
            st.metric("2Y Yield", f"{current_2y:.2f}%" if not np.isnan(current_2y) else "N/A")
        with col3:
            st.metric("2s10s Spread", f"{current_spread:.2f}%" if not np.isnan(current_spread) else "N/A")
        with col4:
            st.metric("Recession Probability", f"{recession_prob:.1%}")
        with col5:
            vix_current = volatility_df['CBOE Volatility Index'].iloc[-1] if not volatility_df.empty else np.nan
            st.metric("VIX", f"{vix_current:.2f}" if not np.isnan(vix_current) else "N/A")
        
        if current_spread < 0:
            st.warning("⚠️ **YIELD CURVE IS INVERTED!** Historically signals recession within 6-18 months.")
        
        st.success(f"✅ Data loaded: {start_date} to {end_date} | {len(yield_df)} trading days")
        
        # Display modules in expandable sections
        st.subheader("📊 Analysis Results")
        
        # MODULE 1: Yield Curve Analysis
        if module1:
            with st.expander("📈 MODULE 1: Yield Curve Analysis", expanded=True):
                st.markdown("**Interactive U.S. Treasury Yield Curve Visualization**")
                fig = plot_yield_curve(yield_df, end_date)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Curve Statistics**")
                    latest = yield_df.iloc[-1]
                    st.metric("Curve Steepness (10Y-2Y)", f"{latest.get('10Y', 0) - latest.get('2Y', 0):.2f}%")
                    st.metric("Short End (3M)", f"{latest.get('3M', 0):.2f}%")
                    st.metric("Long End (30Y)", f"{latest.get('30Y', 0):.2f}%")
                with col2:
                    if len(yield_df) > 1:
                        changes = yield_df.iloc[-1] - yield_df.iloc[-2]
                        st.metric("10Y Daily Change", f"{changes.get('10Y', 0):+.2f}%")
                        st.metric("2Y Daily Change", f"{changes.get('2Y', 0):+.2f}%")
        
        # MODULE 2: Spread Analysis
        if module2:
            with st.expander("📊 MODULE 2: Spread Analysis", expanded=True):
                st.markdown("**Treasury Yield Spreads and Forward Rates**")
                if not spreads.empty:
                    fig = plot_spreads(spreads)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Current Spreads**")
                        for name, value in spreads.iloc[-1].items():
                            st.metric(f"{name.upper()}", f"{value:.2f}%")
                    with col2:
                        st.markdown("**Forward Rates**")
                        if not forwards.empty:
                            st.line_chart(forwards)
                        else:
                            st.info("No forward rate data available")
        
        # MODULE 3: Yield Spread Trading Strategy
        if module3 and strategy_data is not None:
            with st.expander("🎯 MODULE 3: Yield Spread Trading Strategy", expanded=True):
                st.markdown("""
                **Strategy Logic:**
                - **BUY (Signal = 1)**: When 10Y-3M Spread < 0 (Inverted Curve)
                - **SELL (Signal = -1)**: When 10Y-3M Spread > 0 (Normal Curve)
                """)
                
                st.markdown("**Recent Signals (Last 10 Days)**")
                display_df = strategy_data[['10Y-3M Spread', 'Signal']].tail(10).copy()
                display_df['10Y-3M Spread'] = display_df['10Y-3M Spread'].apply(lambda x: f"{x:.2f}%")
                st.dataframe(display_df, use_container_width=True)
                
                fig = plot_trading_signals(strategy_data)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        # MODULE 4: Strategy Backtest
        if module4 and backtest_results is not None:
            with st.expander("💰 MODULE 4: Strategy Backtest with ETF", expanded=True):
                st.markdown(f"**Backtest Results - {yield_etf} ETF**")
                
                metrics = backtest_results['metrics']
                if metrics:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Strategy Return", f"{metrics.get('Total Return Strategy', 0):.2%}")
                        st.metric("Sharpe Ratio", f"{metrics.get('Sharpe Ratio Strategy', 0):.2f}")
                    with col2:
                        st.metric("Benchmark Return", f"{metrics.get('Total Return Benchmark', 0):.2%}")
                        st.metric("Max Drawdown", f"{metrics.get('Max Drawdown Strategy', 0):.2%}")
                    with col3:
                        st.metric("Excess Return", f"{metrics.get('Excess Return', 0):.2%}")
                        st.metric("Win Rate", f"{metrics.get('Win Rate', 0):.2%}")
                    with col4:
                        st.metric("Profit Factor", f"{metrics.get('Profit Factor', 0):.2f}")
                        st.metric("Trades", metrics.get('Number of Trades', 0))
                    
                    fig = plot_cumulative_returns(backtest_results)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        
        # MODULE 5: Nelson-Siegel Model
        if module5 and ns_result:
            with st.expander("📐 MODULE 5: Nelson-Siegel Model", expanded=True):
                st.markdown("**Parametric Yield Curve Model**")
                
                col1, col2 = st.columns(2)
                with col1:
                    params_df = pd.DataFrame({
                        'Parameter': ['β₀ (Level)', 'β₁ (Slope)', 'β₂ (Curvature)', 'λ (Decay)'],
                        'Value': [f"{ns_result['params'][0]:.4f}", f"{ns_result['params'][1]:.4f}",
                                 f"{ns_result['params'][2]:.4f}", f"{ns_result['params'][3]:.4f}"]
                    })
                    st.dataframe(params_df, hide_index=True, use_container_width=True)
                    st.metric("RMSE", f"{ns_result['rmse']*100:.2f} bps")
                    st.metric("R²", f"{ns_result['r2']:.4f}")
                
                with col2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=maturities, y=current_yields, mode='markers',
                                            name='Actual Data', marker=dict(size=12, color='#2c5f8a')))
                    fig.add_trace(go.Scatter(x=np.linspace(0.25, 30, 100),
                                            y=NelsonSiegelModel.nelson_siegel(np.linspace(0.25, 30, 100), *ns_result['params']),
                                            mode='lines', name='NS Fit', line=dict(color='#c17f3a', width=2)))
                    fig.update_layout(title='Model Fit', template='plotly_white', height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        # MODULE 6: PCA & Factor Analysis
        if module6:
            with st.expander("📉 MODULE 6: PCA & Factor Analysis", expanded=True):
                st.markdown("**Level, Slope, Curvature Factors and PCA**")
                
                col1, col2 = st.columns(2)
                with col1:
                    if not factors.empty:
                        fig = go.Figure()
                        for col in factors.columns:
                            fig.add_trace(go.Scatter(x=factors.index, y=factors[col], name=col, line=dict(width=2)))
                        fig.update_layout(title='Factor Contributions', template='plotly_white', height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if pca_result:
                        fig = go.Figure(data=go.Bar(x=[f'PC{i+1}' for i in range(pca_result['n_components'])],
                                                    y=pca_result['explained_variance'] * 100, marker_color='#2c5f8a'))
                        fig.update_layout(title='PCA Explained Variance', template='plotly_white', height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(pca_result['loadings'].round(3), use_container_width=True)
        
        # MODULE 7: Monte Carlo Simulation
        if module7:
            with st.expander("🎲 MODULE 7: Monte Carlo Simulation", expanded=True):
                st.markdown("**Stochastic Yield Path Simulation**")
                
                if st.button("Run Monte Carlo Simulation", use_container_width=True):
                    with st.spinner(f"Running {mc_simulations:,} simulations..."):
                        initial_yield = current_10y if not np.isnan(current_10y) else 4.0
                        returns = yield_df['10Y'].pct_change().dropna()
                        mu = returns.mean() * 252
                        sigma = returns.std() * np.sqrt(252)
                        
                        paths = MonteCarloSimulator.simulate_gbm(initial_yield, mu, sigma, mc_horizon, mc_simulations)
                        sim_results = MonteCarloSimulator.calculate_confidence_intervals(paths, 0.95)
                        
                        fig = plot_monte_carlo(sim_results, initial_yield, mc_horizon, "GBM")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Initial Yield", f"{initial_yield:.2f}%")
                            st.metric("Expected Terminal", f"{sim_results['mean'][-1]:.2f}%")
                        with col2:
                            st.metric("Drift (μ)", f"{mu:.4f}")
                            st.metric("Volatility (σ)", f"{sigma:.4f}")
                        with col3:
                            var_95 = MonteCarloSimulator.calculate_var(paths, 0.95)
                            st.metric("95% VaR", f"{var_95:.2f}%")
                            st.metric("Simulations", f"{mc_simulations:,}")
        
        # MODULE 8: Machine Learning Forecast
        if module8:
            with st.expander("🤖 MODULE 8: Machine Learning Forecast", expanded=True):
                st.markdown(f"**{ml_model_type} Model for Yield Prediction**")
                
                if st.button("Train & Forecast", use_container_width=True):
                    with st.spinner("Training model..."):
                        X, y, scaler = MLForecastModel.prepare_features(yield_df, '10Y', 5)
                        
                        if X is not None and len(X) > 50:
                            results = MLForecastModel.train_model(X, y, ml_model_type)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("RMSE", f"{results['rmse']*100:.2f} bps")
                            with col2:
                                st.metric("MAE", f"{results['mae']*100:.2f} bps")
                            with col3:
                                st.metric("R²", f"{results['r2']:.3f}")
                            
                            if not results['feature_importance'].empty:
                                st.markdown("**Feature Importance**")
                                st.dataframe(results['feature_importance'], use_container_width=True)
                            
                            st.success(f"Model trained on {len(X)} samples")
                        else:
                            st.warning(f"Insufficient data. Need >50 samples, have {len(X) if X else 0}")
        
        # MODULE 9: Advanced Backtest
        if module9 and adv_backtest_result:
            with st.expander("⚡ MODULE 9: Advanced Backtest Engine", expanded=True):
                st.markdown(f"**Strategy: {adv_strategy} on {adv_etf}**")
                
                metrics = adv_backtest_result['metrics']
                if metrics:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Strategy Return", f"{metrics.get('Total Return Strategy', 0):.2%}")
                        st.metric("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0):.2f}")
                    with col2:
                        st.metric("Benchmark Return", f"{metrics.get('Total Return Benchmark', 0):.2%}")
                        st.metric("Volatility", f"{metrics.get('Volatility', 0):.2%}")
                    with col3:
                        st.metric("Excess Return", f"{metrics.get('Excess Return', 0):.2%}")
                        st.metric("Max Drawdown", f"{metrics.get('Max Drawdown', 0):.2%}")
                    with col4:
                        st.metric("Win Rate", f"{metrics.get('Win Rate', 0):.2%}")
                        st.metric("Profit Factor", f"{metrics.get('Profit Factor', 0):.2f}")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=adv_backtest_result['cumulative_strategy'].index, 
                                            y=adv_backtest_result['cumulative_strategy'].values,
                                            name='Strategy', line=dict(color='#2c5f8a', width=2)))
                    fig.add_trace(go.Scatter(x=adv_backtest_result['cumulative_benchmark'].index, 
                                            y=adv_backtest_result['cumulative_benchmark'].values,
                                            name='Benchmark', line=dict(color='#c17f3a', width=2, dash='dash')))
                    fig.update_layout(title='Cumulative Returns', template='plotly_white', height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        # MODULE 10: Volatility Analysis
        if module10 and not volatility_df.empty:
            with st.expander("⚡ MODULE 10: Volatility Analysis", expanded=True):
                st.markdown("**VIX Volatility Analysis**")
                
                vix_analysis = VolatilityAnalyzer.calculate_volatility_regime(volatility_df['CBOE Volatility Index'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"### Current Regime: {vix_analysis['regime']}")
                    st.info(vix_analysis['outlook'])
                    st.metric("Current VIX", f"{vix_analysis['current_vix']:.2f}")
                
                with col2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=volatility_df.index, y=volatility_df['CBOE Volatility Index'],
                                            mode='lines', fill='tozeroy', line=dict(color='#c17f3a', width=2)))
                    fig.add_hline(y=20, line_dash='dash', line_color='red')
                    fig.update_layout(title='VIX Historical Chart', template='plotly_white', height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        # MODULE 11: Technical Analysis
        if module11:
            with st.expander("🛠 MODULE 11: Technical Analysis", expanded=True):
                st.markdown("**Technical Indicators for ETF Analysis**")
                
                tech_ticker = st.selectbox("Select Asset", ['TLT', 'IEF', 'SHY', 'SPY', 'QQQ'], key="tech_ticker")
                tech_data = fetch_yahoo_data(tech_ticker, start_date, end_date)
                
                if not tech_data.empty:
                    tech_df = calculate_technical_indicators(tech_data)
                    fig = plot_technical_indicators(tech_df, tech_ticker)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Current signals
                    st.markdown("**Current Technical Signals**")
                    latest = tech_df.iloc[-1]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        rsi = latest.get('RSI', 50)
                        rsi_signal = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
                        st.metric("RSI", f"{rsi:.1f}", delta=rsi_signal)
                    with col2:
                        macd = latest.get('MACD', 0)
                        signal = latest.get('MACD_Signal', 0)
                        macd_signal = "Bullish" if macd > signal else "Bearish"
                        st.metric("MACD", f"{macd:.4f}", delta=macd_signal)
                    with col3:
                        price = latest.get('Price', 0)
                        sma50 = latest.get('SMA_50', price)
                        trend = "Above SMA50" if price > sma50 else "Below SMA50"
                        st.metric("Trend", trend)
        
        # MODULE 12: Scenario & Correlation Analysis
        if module12:
            with st.expander("🎭 MODULE 12: Scenario & Correlation Analysis", expanded=True):
                st.markdown("**Market Scenarios and Cross-Asset Correlations**")
                
                tab_scenario, tab_corr = st.tabs(["Scenarios", "Correlation"])
                
                with tab_scenario:
                    scenarios = generate_scenarios(yield_df)
                    if scenarios:
                        selected = st.selectbox("Select Scenario", list(scenarios.keys()))
                        scenario_data = scenarios[selected]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            current = yield_df.iloc[-1]
                            changes = {m: scenario_data[m] - current[m] for m in scenario_data.index}
                            df_changes = pd.DataFrame({
                                'Maturity': changes.keys(),
                                'Current': [f"{current[m]:.2f}%" for m in changes.keys()],
                                'Scenario': [f"{scenario_data[m]:.2f}%" for m in changes.keys()],
                                'Change': [f"{changes[m]*100:+.1f}bps" for m in changes.keys()]
                            })
                            st.dataframe(df_changes, hide_index=True, use_container_width=True)
                        
                        with col2:
                            fig = plot_scenario_comparison(current, scenario_data, selected)
                            st.plotly_chart(fig, use_container_width=True)
                
                with tab_corr:
                    if corr_matrix is not None:
                        fig = CorrelationAnalyzer.plot_correlation_heatmap(corr_matrix)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Correlation data not available")

else:
    st.info("👈 Configure parameters and click 'Run Analysis' to see all 12 modules in action")
    
    with st.expander("📖 12 Professional Analysis Modules", expanded=True):
        st.markdown("""
        ### 🚀 Complete Fixed-Income Analytics Platform
        
        | # | Module | Description |
        |---|--------|-------------|
        | 1 | **Yield Curve Analysis** | Interactive curve visualization, historical comparison |
        | 2 | **Spread Analysis** | 2s10s, 3m10y, 5s30s spreads, forward rates |
        | 3 | **Yield Spread Trading Strategy** | Buy/Sell signals based on 10Y-3M spread |
        | 4 | **Strategy Backtest with ETF** | Backtest with TLT/IEF/SHY/BND ETFs |
        | 5 | **Nelson-Siegel Model** | Parametric yield curve modeling |
        | 6 | **PCA & Factor Analysis** | Level, Slope, Curvature factors |
        | 7 | **Monte Carlo Simulation** | GBM simulation, VaR, confidence intervals |
        | 8 | **Machine Learning Forecast** | Random Forest, Gradient Boosting predictions |
        | 9 | **Advanced Backtest Engine** | Curve Inversion, Momentum, Mean Reversion |
        | 10 | **Volatility Analysis** | VIX analysis, regime detection |
        | 11 | **Technical Analysis** | RSI, MACD, Bollinger Bands |
        | 12 | **Scenario & Correlation** | Bull/Bear scenarios, correlation matrix |
        
        ### 🔑 Getting FRED API Key
        1. Visit [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)
        2. Register for free account
        3. Enter API key in sidebar
        """)

st.markdown("---")
st.markdown(f"*Data: FRED (Federal Reserve Economic Data) | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
