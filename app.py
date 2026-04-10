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
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# FRED API CONFIGURATION - OFFICIAL FEDERAL RESERVE DATA
# =============================================================================

# FRED API Series IDs (Official Federal Reserve data)
FRED_SERIES = {
    '3M': 'DGS3MO',     # 3-Month Treasury Bill (Official FRED series)
    '2Y': 'DGS2',       # 2-Year Treasury Note (Official FRED series)
    '5Y': 'DGS5',       # 5-Year Treasury Note (Official FRED series)
    '10Y': 'DGS10',     # 10-Year Treasury Note (Official FRED series)
    '30Y': 'DGS30'      # 30-Year Treasury Bond (Official FRED series)
}

# Recession indicator (Official FRED series)
RECESSION_SERIES = 'USREC'

# Bond ETFs for backtesting (from Yahoo Finance)
BOND_ETFS = {
    'TLT': '20+ Year Treasury Bond ETF',
    'IEF': '7-10 Year Treasury Bond ETF',
    'SHY': '1-3 Year Treasury Bond ETF',
    'BND': 'Total Bond Market ETF',
    'GOVT': 'US Treasury Bond ETF'
}

# Volatility tickers (Yahoo Finance)
VOLATILITY_TICKERS = {
    '^VIX': 'CBOE Volatility Index',
    '^VXN': 'Nasdaq Volatility Index'
}

# Correlation assets (Yahoo Finance)
CORRELATION_TICKERS = {
    '^GSPC': 'S&P 500',
    '^IXIC': 'Nasdaq Composite',
    'GLD': 'Gold',
    'UUP': 'US Dollar Index'
}

# =============================================================================
# FRED API FUNCTIONS - DATA SOURCE IS OFFICIAL FRED
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_series(api_key, series_id, start_date, end_date):
    """
    Fetch official data from FRED API - NO SYNTHETIC DATA
    Source: Federal Reserve Economic Data (FRED)
    """
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
            st.warning(f"No data found for {series_id}")
            return pd.Series(dtype='float64')
        
        dates = []
        values = []
        for obs in observations:
            value = obs.get('value')
            if value != '.' and value is not None:
                dates.append(pd.to_datetime(obs['date']))
                values.append(float(value))
        
        if dates:
            series = pd.Series(values, index=dates, name=series_id)
            return series
        return pd.Series(dtype='float64')
    
    except Exception as e:
        st.error(f"FRED API Error for {series_id}: {str(e)}")
        return pd.Series(dtype='float64')

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_yield_data(api_key, start_date, end_date):
    """
    Fetch all yield curve data from FRED
    NO SYNTHETIC DATA - All data comes directly from official FRED
    """
    data = {}
    
    for name, series_id in FRED_SERIES.items():
        series_data = fetch_fred_series(api_key, series_id, start_date, end_date)
        if not series_data.empty:
            data[name] = series_data
            st.info(f"✓ {name} data retrieved (FRED series: {series_id})")
        else:
            st.error(f"✗ {name} data unavailable (FRED series: {series_id})")
    
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    # Only forward fill, no synthetic data creation
    df = df.fillna(method='ffill')
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recession_data(api_key, start_date, end_date):
    """Fetch recession indicator from FRED"""
    return fetch_fred_series(api_key, RECESSION_SERIES, start_date, end_date)

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
# MARKET DATA FUNCTIONS (ONLY FOR ETF PRICES - BACKTESTING)
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_data(ticker, start_date, end_date):
    """Fetch ETF price data from Yahoo Finance (only for backtesting)"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if not data.empty and 'Adj Close' in data.columns:
            series = data['Adj Close']
            series.index = series.index.tz_localize(None)
            return series
    except Exception as e:
        st.warning(f"Could not fetch {ticker}: {e}")
    return pd.Series(dtype='float64')

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_tlt_data(start_date, end_date):
    """
    Fetch TLT price data (20+ Year Treasury Bonds)
    This function is specifically for the yield spread trading strategy backtest
    """
    try:
        tlt_data = yf.download('TLT', start=start_date, end=end_date, progress=False)['Adj Close']
        # Remove timezone information from TLT data
        tlt_data.index = tlt_data.index.tz_localize(None)
        return tlt_data
    except Exception as e:
        st.error(f"Could not fetch TLT data: {e}")
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
# YIELD CURVE ANALYSIS FUNCTIONS
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
    
    # Current yield curve
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
    
    # Historical comparison (1 year ago)
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
        title='Treasury Yield Spreads (Official FRED Data)',
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
    
    # Logistic regression approximation based on historical FRED data
    prob = 1 / (1 + np.exp(-(-current_spread * 2 - 0.5)))
    return min(max(prob, 0.01), 0.99)

def identify_recessions(recession_series):
    """Identify recession periods from FRED data"""
    if recession_series.empty:
        return []
    
    recessions = []
    in_recession = False
    start_date = None
    
    for date, value in recession_series.dropna().items():
        if value == 1 and not in_recession:
            in_recession = True
            start_date = date
        elif value == 0 and in_recession:
            recessions.append({'start': start_date, 'end': date})
            in_recession = False
    
    return recessions

# =============================================================================
# YIELD SPREAD TRADING STRATEGY - STEP 5: DEVELOPING A SIMPLE YIELD CURVE TRADING STRATEGY
# =============================================================================

def develop_yield_spread_trading_strategy(yield_df):
    """
    Step 5: Developing a Simple Yield Curve Trading Strategy
    
    One basic fixed-income trading strategy based on the yield curve is to buy long-term bonds 
    when the yield curve is inverted (indicating an economic downturn) and sell when the curve 
    is normal (indicating growth).
    
    This function builds a simple trading strategy based on the 10Y-3M spread.
    """
    if yield_df.empty:
        return None, None
    
    # Create a copy of the data to avoid modifying the original
    data = yield_df.copy()
    
    # Calculate yield spread (10Y - 3M)
    if '10Y' in data.columns and '3M' in data.columns:
        data['10Y-3M Spread'] = data['10Y'] - data['3M']
    else:
        st.error("Required columns '10Y' and '3M' not found for strategy development")
        return None, None
    
    # Define a simple strategy: Buy bonds when the spread is negative (inverted yield curve), sell when positive (normal curve)
    # Signal = 1: Buy when inverted (spread < 0)
    # Signal = -1: Sell when normal (spread > 0)
    # Signal = 0: No action when spread = 0
    data['Signal'] = 0
    data.loc[data['10Y-3M Spread'] < 0, 'Signal'] = 1   # Buy when inverted
    data.loc[data['10Y-3M Spread'] > 0, 'Signal'] = -1  # Sell when normal
    
    return data[['10Y-3M Spread', 'Signal']], data

def plot_trading_signals(strategy_data):
    """
    Plot the trading signals to visualize trading opportunities
    
    The trading strategy generates a buy signal when the yield curve is inverted (negative spread) 
    and a sell signal when the curve normalizes (positive spread).
    The signal is plotted over time to help visualize trading opportunities.
    """
    if strategy_data is None or strategy_data.empty:
        return None
    
    fig = go.Figure()
    
    # Add trading signal line
    fig.add_trace(go.Scatter(
        x=strategy_data.index,
        y=strategy_data['Signal'],
        mode='lines',
        name='Trading Signal',
        line=dict(color='#2c5f8a', width=2),
        fill='tozeroy',
        fillcolor='rgba(44, 95, 138, 0.2'
    ))
    
    # Add horizontal lines for reference
    fig.add_hline(y=1, line_dash='dash', line_color='green', annotation_text="BUY Signal (Inverted Curve)", annotation_position="top right")
    fig.add_hline(y=-1, line_dash='dash', line_color='red', annotation_text="SELL Signal (Normal Curve)", annotation_position="bottom right")
    fig.add_hline(y=0, line_dash='dot', line_color='gray')
    
    # Color background based on signal
    for i in range(len(strategy_data) - 1):
        if strategy_data['Signal'].iloc[i] == 1:
            fig.add_vrect(x0=strategy_data.index[i], x1=strategy_data.index[i+1], 
                         fillcolor="green", opacity=0.1, line_width=0)
        elif strategy_data['Signal'].iloc[i] == -1:
            fig.add_vrect(x0=strategy_data.index[i], x1=strategy_data.index[i+1], 
                         fillcolor="red", opacity=0.1, line_width=0)
    
    fig.update_layout(
        title='Trading Signals Based on Yield Spread (10Y-3M)',
        xaxis_title='Date',
        yaxis_title='Signal (1 = Buy, -1 = Sell, 0 = Neutral)',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    return fig

def display_signals_table(strategy_data, num_rows=10):
    """
    Display the trading signals table showing the last N rows
    """
    if strategy_data is None or strategy_data.empty:
        return None
    
    # Display signals table
    display_df = strategy_data[['10Y-3M Spread', 'Signal']].tail(num_rows).copy()
    display_df['10Y-3M Spread'] = display_df['10Y-3M Spread'].apply(lambda x: f"{x:.2f}%")
    display_df['Signal_Interpretation'] = display_df['Signal'].apply(
        lambda x: 'BUY (Inverted Curve)' if x == 1 else 'SELL (Normal Curve)' if x == -1 else 'NEUTRAL'
    )
    
    return display_df

# =============================================================================
# BACKTESTING THE STRATEGY - STEP 6: BACKTESTING THE STRATEGY
# =============================================================================

def backtest_yield_spread_strategy(yield_df, start_date, end_date, selected_etf='TLT'):
    """
    Step 6: Backtesting the Strategy
    
    To evaluate the performance of this strategy, we backtest it using bond prices or 
    Total Return Bond ETFs like TLT (20+ Year Treasury Bonds).
    
    This function assumes we invest in TLT based on the signal.
    """
    if yield_df.empty:
        return None, None, None
    
    # Create a copy of the data
    data = yield_df.copy()
    
    # Calculate yield spread (10Y - 3M)
    if '10Y' in data.columns and '3M' in data.columns:
        data['10Y-3M Spread'] = data['10Y'] - data['3M']
    else:
        st.error("Required columns '10Y' and '3M' not found for backtesting")
        return None, None, None
    
    # Define the strategy: Buy bonds when the spread is negative (inverted yield curve), sell when positive (normal curve)
    data['Signal'] = 0
    data.loc[data['10Y-3M Spread'] < 0, 'Signal'] = 1   # Buy when inverted
    data.loc[data['10Y-3M Spread'] > 0, 'Signal'] = -1  # Sell when normal
    
    # Fetch TLT price data (20+ Year Treasury Bonds) or selected ETF
    if selected_etf == 'TLT':
        tlt_data = fetch_tlt_data(start_date, end_date)
    else:
        tlt_data = fetch_yahoo_data(selected_etf, start_date, end_date)
    
    if tlt_data.empty:
        st.error(f"Could not fetch {selected_etf} price data for backtesting")
        return None, None, None
    
    # Remove timezone information from TLT data (already done in fetch functions)
    # Calculate daily returns for TLT
    tlt_df = pd.DataFrame(tlt_data)
    tlt_df.columns = ['Price']
    tlt_df['Returns'] = tlt_df['Price'].pct_change()
    
    # Align signals with TLT data
    # Shift signal by 1 to avoid look-ahead bias (use today's signal for tomorrow's trade)
    common_index = data.index.intersection(tlt_df.index)
    signals_aligned = data['Signal'].reindex(common_index)
    returns_aligned = tlt_df['Returns'].reindex(common_index)
    
    # Strategy returns: signal shifted by 1 to use today's signal for tomorrow's return
    strategy_returns = signals_aligned.shift(1) * returns_aligned
    
    # Calculate cumulative returns
    cumulative_returns = (1 + strategy_returns.fillna(0)).cumprod()
    benchmark_returns = (1 + returns_aligned.fillna(0)).cumprod()
    
    # Calculate performance metrics
    performance_metrics = calculate_backtest_performance_metrics(strategy_returns, returns_aligned, cumulative_returns, benchmark_returns)
    
    backtest_results = {
        'strategy_returns': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'benchmark_returns': benchmark_returns,
        'signals': signals_aligned,
        'metrics': performance_metrics,
        'tlt_prices': tlt_df['Price'],
        'spread_data': data['10Y-3M Spread'].reindex(common_index),
        'signal_data': signals_aligned
    }
    
    return backtest_results, data, tlt_df

def calculate_backtest_performance_metrics(strategy_returns, benchmark_returns, cumulative_strategy, cumulative_benchmark):
    """
    Calculate comprehensive performance metrics for the backtest
    """
    if strategy_returns.empty or strategy_returns.dropna().empty:
        return {}
    
    # Remove NaN values for calculations
    strategy_clean = strategy_returns.dropna()
    benchmark_clean = benchmark_returns.dropna()
    
    if len(strategy_clean) == 0:
        return {}
    
    # Annualization factor (252 trading days per year)
    ann_factor = np.sqrt(252)
    
    # Calculate metrics
    total_return_strategy = float(cumulative_strategy.iloc[-1] - 1) if len(cumulative_strategy) > 0 else 0
    total_return_benchmark = float(cumulative_benchmark.iloc[-1] - 1) if len(cumulative_benchmark) > 0 else 0
    
    # Sharpe Ratio (assuming risk-free rate = 0 for simplicity)
    sharpe_strategy = (strategy_clean.mean() / strategy_clean.std()) * ann_factor if strategy_clean.std() > 0 else 0
    sharpe_benchmark = (benchmark_clean.mean() / benchmark_clean.std()) * ann_factor if benchmark_clean.std() > 0 else 0
    
    # Maximum Drawdown
    max_drawdown_strategy = calculate_max_drawdown(cumulative_strategy)
    max_drawdown_benchmark = calculate_max_drawdown(cumulative_benchmark)
    
    # Win Rate
    positive_returns = (strategy_clean > 0).sum()
    total_trades = len(strategy_clean[strategy_clean != 0])
    win_rate = positive_returns / total_trades if total_trades > 0 else 0
    
    # Profit Factor
    gross_profits = strategy_clean[strategy_clean > 0].sum()
    gross_losses = abs(strategy_clean[strategy_clean < 0].sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else 0
    
    # Volatility
    volatility = strategy_clean.std() * ann_factor
    
    # Calmar Ratio
    calmar_ratio = (total_return_strategy / abs(max_drawdown_strategy)) if max_drawdown_strategy != 0 else 0
    
    # Average Return per Trade
    avg_return_per_trade = strategy_clean[strategy_clean != 0].mean() if len(strategy_clean[strategy_clean != 0]) > 0 else 0
    
    metrics = {
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
        'Average Return per Trade': avg_return_per_trade,
        'Number of Trades': total_trades
    }
    
    return metrics

def calculate_max_drawdown(cumulative_returns):
    """Calculate maximum drawdown from cumulative returns series"""
    if cumulative_returns.empty or len(cumulative_returns) == 0:
        return 0
    
    cumulative_clean = cumulative_returns.fillna(1)
    rolling_max = cumulative_clean.expanding().max()
    drawdown = (cumulative_clean - rolling_max) / rolling_max
    return float(drawdown.min())

def plot_cumulative_returns(backtest_results):
    """
    Plot cumulative returns of the yield spread strategy
    
    This function creates a visualization showing the cumulative returns of the 
    trading strategy compared to a simple buy-and-hold approach.
    """
    if backtest_results is None:
        return None
    
    fig = go.Figure()
    
    # Add strategy cumulative returns
    fig.add_trace(go.Scatter(
        x=backtest_results['cumulative_returns'].index,
        y=backtest_results['cumulative_returns'].values,
        mode='lines',
        name='Yield Spread Strategy',
        line=dict(color='#2c5f8a', width=2.5)
    ))
    
    # Add benchmark cumulative returns (buy and hold)
    fig.add_trace(go.Scatter(
        x=backtest_results['benchmark_returns'].index,
        y=backtest_results['benchmark_returns'].values,
        mode='lines',
        name='Buy & Hold (TLT)',
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

def plot_strategy_performance_dashboard(backtest_results):
    """
    Create a comprehensive performance dashboard for the strategy
    """
    if backtest_results is None:
        return None
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Cumulative Returns', 'Drawdown Analysis',
            'Trading Signals', 'Signal Distribution',
            'Monthly Returns Heatmap', 'Rolling Sharpe Ratio'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    # 1. Cumulative Returns
    fig.add_trace(go.Scatter(
        x=backtest_results['cumulative_returns'].index,
        y=backtest_results['cumulative_returns'].values,
        mode='lines',
        name='Strategy',
        line=dict(color='#2c5f8a', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=backtest_results['benchmark_returns'].index,
        y=backtest_results['benchmark_returns'].values,
        mode='lines',
        name='Buy & Hold',
        line=dict(color='#c17f3a', width=2, dash='dash')
    ), row=1, col=1)
    
    # 2. Drawdown Analysis
    strategy_dd = calculate_drawdown_series(backtest_results['cumulative_returns'])
    fig.add_trace(go.Scatter(
        x=strategy_dd.index,
        y=strategy_dd.values * 100,
        mode='lines',
        name='Drawdown',
        line=dict(color='red', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.2)'
    ), row=1, col=2)
    
    # 3. Trading Signals
    fig.add_trace(go.Scatter(
        x=backtest_results['signal_data'].index,
        y=backtest_results['signal_data'].values,
        mode='lines',
        name='Signal',
        line=dict(color='#2c5f8a', width=1.5)
    ), row=2, col=1)
    
    fig.add_hline(y=1, line_dash='dash', line_color='green', row=2, col=1)
    fig.add_hline(y=-1, line_dash='dash', line_color='red', row=2, col=1)
    
    # 4. Signal Distribution
    signal_counts = backtest_results['signal_data'].value_counts()
    colors = ['green' if x == 1 else 'red' if x == -1 else 'gray' for x in signal_counts.index]
    fig.add_trace(go.Bar(
        x=signal_counts.index.astype(str),
        y=signal_counts.values,
        name='Signal Count',
        marker_color=colors
    ), row=2, col=2)
    
    # 5. Monthly Returns Heatmap
    monthly_returns = calculate_monthly_returns(backtest_results['strategy_returns'])
    if monthly_returns is not None and not monthly_returns.empty:
        fig.add_trace(go.Heatmap(
            z=monthly_returns.values,
            x=monthly_returns.columns,
            y=monthly_returns.index,
            colorscale='RdYlGn',
            zmid=0,
            name='Monthly Returns',
            showscale=True
        ), row=3, col=1)
    
    # 6. Rolling Sharpe Ratio
    rolling_sharpe = calculate_rolling_sharpe(backtest_results['strategy_returns'], window=63)
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe.values,
        mode='lines',
        name='Rolling Sharpe',
        line=dict(color='#4a7c59', width=2)
    ), row=3, col=2)
    
    fig.add_hline(y=0, line_dash='dash', line_color='gray', row=3, col=2)
    fig.add_hline(y=1, line_dash='dash', line_color='green', row=3, col=2)
    
    fig.update_layout(
        title='Strategy Performance Dashboard',
        height=1000,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def calculate_drawdown_series(cumulative_returns):
    """Calculate drawdown series from cumulative returns"""
    if cumulative_returns.empty:
        return pd.Series(dtype=float)
    
    cumulative_clean = cumulative_returns.fillna(1)
    rolling_max = cumulative_clean.expanding().max()
    drawdown = (cumulative_clean - rolling_max) / rolling_max
    return drawdown

def calculate_monthly_returns(strategy_returns):
    """Calculate monthly returns for heatmap visualization"""
    if strategy_returns.empty:
        return None
    
    monthly_returns = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_returns.index = monthly_returns.index.to_period('M')
    
    # Create pivot table for heatmap
    pivot_table = pd.pivot_table(
        pd.DataFrame({'Returns': monthly_returns, 'Year': monthly_returns.index.year, 'Month': monthly_returns.index.month}),
        values='Returns',
        index='Year',
        columns='Month'
    ) * 100  # Convert to percentage
    
    return pivot_table

def calculate_rolling_sharpe(returns, window=63):
    """Calculate rolling Sharpe ratio"""
    if returns.empty or len(returns) < window:
        return pd.Series(dtype=float)
    
    ann_factor = np.sqrt(252)
    rolling_sharpe = returns.rolling(window).apply(
        lambda x: (x.mean() / x.std()) * ann_factor if x.std() > 0 else 0
    )
    
    return rolling_sharpe

# =============================================================================
# NELSON-SIEGEL MODEL (PARAMETRIC CURVE MODELING)
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
# FACTOR ANALYSIS AND PCA
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
# MONTE CARLO SIMULATION
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

# =============================================================================
# MACHINE LEARNING FORECAST MODEL
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
                'feature': [f'Feature_{i}' for i in range(X.shape[1])],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
        else:
            importance = pd.DataFrame()
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2, 'feature_importance': importance}

# =============================================================================
# BACKTEST ENGINE (ADVANCED STRATEGIES)
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
        
        metrics = {
            'Total Return Strategy': float((1 + strategy_returns).prod() - 1),
            'Total Return Benchmark': float((1 + benchmark_returns).prod() - 1),
            'Sharpe Ratio': (strategy_returns.mean() / strategy_returns.std()) * ann_factor if strategy_returns.std() > 0 else 0,
            'Volatility': strategy_returns.std() * ann_factor,
            'Max Drawdown': BacktestEngine.calculate_max_drawdown(strategy_returns),
            'Win Rate': (strategy_returns > 0).sum() / len(strategy_returns[strategy_returns != 0]) if len(strategy_returns[strategy_returns != 0]) > 0 else 0
        }
        
        metrics['Excess Return'] = metrics['Total Return Strategy'] - metrics['Total Return Benchmark']
        
        gross_profits = strategy_returns[strategy_returns > 0].sum()
        gross_losses = abs(strategy_returns[strategy_returns < 0].sum())
        metrics['Profit Factor'] = float(gross_profits / gross_losses) if gross_losses > 0 else 0
        
        return metrics
    
    @staticmethod
    def calculate_max_drawdown(returns):
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return float(drawdown.min())

# =============================================================================
# VOLATILITY ANALYSIS
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
# TECHNICAL ANALYSIS FUNCTIONS
# =============================================================================

def calculate_technical_indicators(price_series):
    """Calculate technical indicators for a given price series"""
    if price_series.empty:
        return pd.DataFrame()
    
    df = pd.DataFrame(index=price_series.index)
    df['Price'] = price_series
    
    # Moving averages
    df['SMA_20'] = price_series.rolling(20).mean()
    df['SMA_50'] = price_series.rolling(50).mean()
    df['EMA_12'] = price_series.ewm(span=12, adjust=False).mean()
    df['EMA_26'] = price_series.ewm(span=26, adjust=False).mean()
    
    # RSI
    delta = price_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
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
    
    # Price and bands
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
    
    # RSI
    if 'RSI' in tech_df.columns:
        fig.add_trace(go.Scatter(
            x=tech_df.index, y=tech_df['RSI'],
            mode='lines', name='RSI',
            line=dict(color='#c17f3a', width=1.5)
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash='dash', line_color='red', row=2, col=1)
        fig.add_hline(y=30, line_dash='dash', line_color='green', row=2, col=1)
    
    # MACD
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
# SCENARIO ANALYSIS
# =============================================================================

def generate_scenarios(yield_df):
    """Generate different market scenarios for yield curve"""
    if yield_df.empty:
        return {}
    
    latest = yield_df.iloc[-1].copy()
    scenarios = {}
    
    # Bull Steepener
    bull = latest.copy()
    bull['3M'] = bull['3M'] - 0.15
    bull['2Y'] = bull['2Y'] - 0.20
    bull['5Y'] = bull['5Y'] - 0.25
    bull['10Y'] = bull['10Y'] - 0.30
    bull['30Y'] = bull['30Y'] - 0.25
    scenarios['Bull Steepener'] = bull
    
    # Bear Flattener
    bear = latest.copy()
    bear['3M'] = bear['3M'] + 0.25
    bear['2Y'] = bear['2Y'] + 0.20
    bear['5Y'] = bear['5Y'] + 0.15
    bear['10Y'] = bear['10Y'] + 0.10
    bear['30Y'] = bear['30Y'] + 0.05
    scenarios['Bear Flattener'] = bear
    
    # Recession
    recession = latest.copy()
    recession['3M'] = recession['3M'] - 0.50
    recession['2Y'] = recession['2Y'] - 0.60
    recession['5Y'] = recession['5Y'] - 0.70
    recession['10Y'] = recession['10Y'] - 0.80
    recession['30Y'] = recession['30Y'] - 0.60
    scenarios['Recession'] = recession
    
    # Inflation Shock
    inflation = latest.copy()
    inflation['3M'] = inflation['3M'] + 0.50
    inflation['2Y'] = inflation['2Y'] + 0.60
    inflation['5Y'] = inflation['5Y'] + 0.55
    inflation['10Y'] = inflation['10Y'] + 0.45
    inflation['30Y'] = inflation['30Y'] + 0.30
    scenarios['Inflation Shock'] = inflation
    
    return scenarios

def plot_scenario_comparison(current_curve, scenario_curve, scenario_name):
    """Plot scenario comparison with current curve"""
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
# STREAMLIT UI - MAIN APPLICATION
# =============================================================================

st.set_page_config(page_title="Bond Yield Curve Analysis Platform", page_icon="📈", layout="wide")

# Session state
if 'api_key_validated' not in st.session_state:
    st.session_state.api_key_validated = False

# Title
st.title("📈 Bond Yield Curve Analysis Platform")
st.markdown("*Official FRED API | Federal Reserve Economic Data | No Synthetic Data*")

# API Key Management
if not st.session_state.api_key_validated:
    st.markdown("""
    ### 🔑 FRED API Key Required
    
    This platform uses **official Federal Reserve Economic Data (FRED)**.
    **NO SYNTHETIC OR ESTIMATED DATA IS USED.**
    
    **Get your free API key:**
    1. Go to [FRED API website](https://fred.stlouisfed.org/docs/api/api_key.html)
    2. Click "Request API Key"
    3. Create a free account (email required)
    4. Enter your API key below
    """)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        api_key = st.text_input("FRED API Key", type="password", placeholder="e.g., abcdefghijklmnopqrstuvwxyz123456")
        
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
    
    # Date range
    default_end = datetime.now()
    default_start = default_end - timedelta(days=365*2)
    
    start_date = st.date_input("Start Date", default_start, max_value=default_end)
    end_date = st.date_input("End Date", default_end, max_value=default_end)
    
    st.markdown("---")
    st.header("📊 Analysis Modules")
    
    show_all = st.checkbox("Show All Modules", value=True)
    
    if show_all:
        show_yield_curve = show_spreads = show_ns_model = show_pca = True
        show_monte_carlo = show_ml = show_backtest = True
        show_volatility = show_correlation = show_technical = show_scenarios = True
        show_yield_spread_strategy = True
        show_strategy_backtest = True
    else:
        show_yield_curve = st.checkbox("Yield Curve Analysis", value=True)
        show_spreads = st.checkbox("Spread Analysis", value=True)
        show_ns_model = st.checkbox("Nelson-Siegel Model", value=True)
        show_pca = st.checkbox("PCA Risk Analysis", value=True)
        show_monte_carlo = st.checkbox("Monte Carlo Simulation", value=True)
        show_ml = st.checkbox("Machine Learning Forecast", value=True)
        show_backtest = st.checkbox("Advanced Strategy Backtest", value=True)
        show_volatility = st.checkbox("Volatility Analysis", value=True)
        show_correlation = st.checkbox("Correlation Analysis", value=True)
        show_technical = st.checkbox("Technical Analysis", value=True)
        show_scenarios = st.checkbox("Scenario Analysis", value=True)
        show_yield_spread_strategy = st.checkbox("Yield Spread Trading Strategy (Step 5)", value=True)
        show_strategy_backtest = st.checkbox("Strategy Backtest with TLT (Step 6)", value=True)
    
    st.markdown("---")
    
    # Backtest parameters
    if show_backtest:
        st.header("🎯 Advanced Strategy Parameters")
        selected_etf = st.selectbox("Backtest ETF", list(BOND_ETFS.keys()))
        strategy_type = st.selectbox("Strategy Type", ['Curve Inversion', 'Momentum', 'Mean Reversion'])
        transaction_cost = st.slider("Transaction Cost (%)", 0.0, 0.5, 0.1, 0.01) / 100
    
    # Yield Spread Strategy parameters
    if show_strategy_backtest:
        st.header("📈 Yield Spread Strategy Parameters")
        selected_tlt_etf = st.selectbox("Select ETF for Yield Spread Strategy", ['TLT', 'IEF', 'SHY', 'BND'], index=0)
    
    # Monte Carlo parameters
    if show_monte_carlo:
        st.header("🎲 Monte Carlo Parameters")
        mc_simulations = st.slider("Number of Simulations", 500, 5000, 1000, 500)
        mc_horizon = st.slider("Forecast Horizon (days)", 5, 252, 20, 5)
    
    # ML parameters
    if show_ml:
        st.header("🤖 ML Parameters")
        ml_model_type = st.selectbox("Model Type", ["Random Forest", "Gradient Boosting"])
    
    st.markdown("---")
    
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Main content
if run_analysis:
    with st.spinner("Fetching data from FRED API..."):
        
        # Fetch yield data from FRED
        yield_df = fetch_all_yield_data(st.session_state.api_key, start_date, end_date)
        
        if yield_df.empty:
            st.error("""
            ❌ **Failed to fetch data!**
            
            Possible reasons:
            1. FRED API key may be invalid
            2. Check your internet connection
            3. Reduce the date range
            """)
            if st.button("Reset API Key"):
                st.session_state.api_key_validated = False
                st.rerun()
            st.stop()
        
        # Fetch recession data
        recession_series = fetch_recession_data(st.session_state.api_key, start_date, end_date)
        
        # Fetch market data (only for backtesting)
        volatility_df, correlation_df = fetch_market_bundle(start_date, end_date)
        
        # Calculations
        spreads = calculate_spreads(yield_df)
        forwards = calculate_forward_rates(yield_df)
        factors = calculate_factor_contributions(yield_df)
        recessions = identify_recessions(recession_series)
        
        # Current metrics
        current_10y = yield_df['10Y'].iloc[-1] if '10Y' in yield_df.columns else np.nan
        current_2y = yield_df['2Y'].iloc[-1] if '2Y' in yield_df.columns else np.nan
        current_spread = spreads['2s10s'].iloc[-1] if '2s10s' in spreads.columns else np.nan
        recession_prob = get_recession_probability(spreads)
        
        # Step 5: Yield Spread Trading Strategy
        strategy_data, full_strategy_data = develop_yield_spread_trading_strategy(yield_df)
        
        # Step 6: Backtest the Strategy with TLT
        backtest_results = None
        if show_strategy_backtest:
            backtest_results, backtest_data, tlt_data = backtest_yield_spread_strategy(
                yield_df, start_date, end_date, selected_tlt_etf
            )
        
        # Nelson-Siegel model
        maturities = np.array([0.25, 2, 5, 10, 30])
        current_yields = np.array([yield_df[m].iloc[-1] for m in ['3M', '2Y', '5Y', '10Y', '30Y']])
        ns_result = NelsonSiegelModel.fit_curve(maturities, current_yields) if show_ns_model else None
        
        # PCA analysis
        pca_result = perform_pca_analysis(yield_df) if show_pca else None
        
        # Success message
        st.success(f"✅ Data successfully loaded from FRED! Period: {start_date} to {end_date}")
        st.info(f"📊 Available maturities: {', '.join(yield_df.columns)} | Trading days: {len(yield_df)}")
        
        # KPI Row
        st.subheader("📊 Current Market Overview (Official FRED Data)")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("10Y Treasury Yield", f"{current_10y:.2f}%" if not np.isnan(current_10y) else "N/A")
        
        with col2:
            st.metric("2Y Treasury Yield", f"{current_2y:.2f}%" if not np.isnan(current_2y) else "N/A")
        
        with col3:
            delta_color = "inverse" if current_spread < 0 else "normal"
            st.metric("2s10s Spread", f"{current_spread:.2f}%" if not np.isnan(current_spread) else "N/A",
                     delta="Inverted" if current_spread < 0 else "Normal",
                     delta_color=delta_color)
        
        with col4:
            st.metric("Recession Probability", f"{recession_prob:.1%}")
        
        with col5:
            vix_current = volatility_df['CBOE Volatility Index'].iloc[-1] if not volatility_df.empty else np.nan
            st.metric("VIX (Fear Index)", f"{vix_current:.2f}" if not np.isnan(vix_current) else "N/A")
        
        # Inversion warning
        if current_spread < 0:
            st.warning("⚠️ **YIELD CURVE IS INVERTED!** Historically signals recession within 6-18 months. Defensive positioning recommended.")
        
        # Tabs - Including new tabs for Step 5 and Step 6
        tabs = []
        if show_yield_curve:
            tabs.append("📈 Yield Curve")
        if show_spreads:
            tabs.append("📊 Spread Analysis")
        if show_yield_spread_strategy:
            tabs.append("🎯 Yield Spread Strategy (Step 5)")
        if show_strategy_backtest:
            tabs.append("💰 Strategy Backtest (Step 6)")
        if show_ns_model:
            tabs.append("📐 Nelson-Siegel")
        if show_pca:
            tabs.append("📉 PCA & Factors")
        if show_monte_carlo:
            tabs.append("🎲 Monte Carlo")
        if show_ml:
            tabs.append("🤖 ML Forecast")
        if show_backtest:
            tabs.append("🎯 Advanced Backtest")
        if show_volatility:
            tabs.append("⚡ Volatility")
        if show_correlation:
            tabs.append("🔄 Correlation")
        if show_technical:
            tabs.append("🛠 Technical")
        if show_scenarios:
            tabs.append("🎭 Scenarios")
        
        if not tabs:
            tabs = ["📈 Yield Curve"]
        
        main_tabs = st.tabs(tabs)
        tab_idx = 0
        
        # TAB 1: Yield Curve
        if show_yield_curve:
            with main_tabs[tab_idx]:
                st.subheader("Yield Curve Visualization (Official FRED Data)")
                
                fig_yield = plot_yield_curve(yield_df, end_date)
                if fig_yield:
                    st.plotly_chart(fig_yield, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Curve Statistics")
                    latest = yield_df.iloc[-1]
                    st.metric("Curve Steepness (10Y-2Y)", f"{latest.get('10Y', 0) - latest.get('2Y', 0):.2f}%")
                    st.metric("Short End (3M)", f"{latest.get('3M', 0):.2f}%")
                    st.metric("Long End (30Y)", f"{latest.get('30Y', 0):.2f}%")
                
                with col2:
                    if len(yield_df) > 1:
                        changes = yield_df.iloc[-1] - yield_df.iloc[-2]
                        st.metric("10Y Daily Change", f"{changes.get('10Y', 0):+.2f}%")
                        st.metric("2Y Daily Change", f"{changes.get('2Y', 0):+.2f}%")
            tab_idx += 1
        
        # TAB 2: Spread Analysis
        if show_spreads:
            with main_tabs[tab_idx]:
                st.subheader("Yield Spread Analysis")
                
                if not spreads.empty:
                    fig_spreads = plot_spreads(spreads)
                    if fig_spreads:
                        st.plotly_chart(fig_spreads, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Current Spreads")
                        current_spreads = spreads.iloc[-1]
                        for name, value in current_spreads.items():
                            st.metric(f"{name.upper()}", f"{value:.2f}%")
                    
                    with col2:
                        st.markdown("### Forward Rate Analysis")
                        if not forwards.empty:
                            st.line_chart(forwards)
                        else:
                            st.info("No forward rate data available")
                    
                    with st.expander("Spread Statistics"):
                        st.dataframe(spreads.describe().round(2), use_container_width=True)
                else:
                    st.info("No spread data available")
            tab_idx += 1
        
        # TAB 3: Yield Spread Trading Strategy (STEP 5)
        if show_yield_spread_strategy:
            with main_tabs[tab_idx]:
                st.subheader("Step 5: Developing a Simple Yield Curve Trading Strategy")
                
                st.markdown("""
                ### One basic fixed-income trading strategy based on the yield curve is to buy long-term bonds 
                when the yield curve is inverted (indicating an economic downturn) and sell when the curve 
                is normal (indicating growth).
                
                **Strategy Logic:**
                - **BUY Signal (1)**: When 10Y-3M Spread < 0 (Inverted Yield Curve)
                - **SELL Signal (-1)**: When 10Y-3M Spread > 0 (Normal Curve)
                - **NEUTRAL (0)**: When Spread = 0
                """)
                
                if strategy_data is not None and not strategy_data.empty:
                    # Display signals table
                    st.markdown("### Trading Signals (Last 10 Days)")
                    display_df = display_signals_table(strategy_data, 10)
                    if display_df is not None:
                        st.dataframe(display_df, use_container_width=True)
                    
                    # Plot trading signals
                    st.markdown("### Trading Signals Visualization")
                    fig_signals = plot_trading_signals(strategy_data)
                    if fig_signals:
                        st.plotly_chart(fig_signals, use_container_width=True)
                    
                    # Display full signal history in expander
                    with st.expander("View Complete Signal History"):
                        st.dataframe(strategy_data, use_container_width=True)
                    
                    # Strategy explanation
                    st.markdown("""
                    ### Strategy Interpretation
                    
                    The trading strategy generates a **BUY signal** when the yield curve is inverted (negative spread) 
                    and a **SELL signal** when the curve normalizes (positive spread).
                    
                    The signal is plotted over time to help visualize trading opportunities.
                    """)
                else:
                    st.warning("Insufficient data to generate trading strategy signals. Please check date range.")
            tab_idx += 1
        
        # TAB 4: Strategy Backtest with TLT (STEP 6)
        if show_strategy_backtest:
            with main_tabs[tab_idx]:
                st.subheader("Step 6: Backtesting the Strategy with TLT")
                
                st.markdown("""
                ### Backtesting the Yield Spread Strategy
                
                To evaluate the performance of this strategy, we backtest it using bond prices or 
                Total Return Bond ETFs like **TLT (20+ Year Treasury Bonds)**.
                
                This backtest assumes we invest in TLT based on the signal.
                """)
                
                if backtest_results is not None:
                    # Display performance metrics
                    st.markdown("### Performance Metrics")
                    metrics = backtest_results['metrics']
                    
                    if metrics:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Strategy Total Return", f"{metrics.get('Total Return Strategy', 0):.2%}")
                            st.metric("Sharpe Ratio (Strategy)", f"{metrics.get('Sharpe Ratio Strategy', 0):.2f}")
                        with col2:
                            st.metric("Benchmark Total Return", f"{metrics.get('Total Return Benchmark', 0):.2%}")
                            st.metric("Sharpe Ratio (Benchmark)", f"{metrics.get('Sharpe Ratio Benchmark', 0):.2f}")
                        with col3:
                            st.metric("Excess Return", f"{metrics.get('Excess Return', 0):.2%}")
                            st.metric("Max Drawdown (Strategy)", f"{metrics.get('Max Drawdown Strategy', 0):.2%}")
                        with col4:
                            st.metric("Win Rate", f"{metrics.get('Win Rate', 0):.2%}")
                            st.metric("Profit Factor", f"{metrics.get('Profit Factor', 0):.2f}")
                        
                        st.metric("Number of Trades", metrics.get('Number of Trades', 0))
                        st.metric("Volatility (Annualized)", f"{metrics.get('Volatility (Annualized)', 0):.2%}")
                        st.metric("Calmar Ratio", f"{metrics.get('Calmar Ratio', 0):.2f}")
                    
                    # Plot cumulative returns
                    st.markdown("### Cumulative Returns")
                    fig_cumulative = plot_cumulative_returns(backtest_results)
                    if fig_cumulative:
                        st.plotly_chart(fig_cumulative, use_container_width=True)
                    
                    # Plot comprehensive performance dashboard
                    st.markdown("### Performance Dashboard")
                    fig_dashboard = plot_strategy_performance_dashboard(backtest_results)
                    if fig_dashboard:
                        st.plotly_chart(fig_dashboard, use_container_width=True)
                    
                    # Display signal distribution
                    st.markdown("### Signal Statistics")
                    signal_counts = backtest_results['signal_data'].value_counts()
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Buy Signals", signal_counts.get(1, 0))
                    with col2:
                        st.metric("Sell Signals", signal_counts.get(-1, 0))
                    with col3:
                        st.metric("Neutral Periods", signal_counts.get(0, 0))
                    
                    # Display recent signals
                    with st.expander("View Recent Trading Signals"):
                        recent_signals = pd.DataFrame({
                            'Date': backtest_results['signal_data'].tail(20).index.strftime('%Y-%m-%d'),
                            '10Y-3M Spread (%)': backtest_results['spread_data'].tail(20).values,
                            'Signal': backtest_results['signal_data'].tail(20).values,
                            'Action': ['BUY (Inverted)' if x == 1 else 'SELL (Normal)' if x == -1 else 'NEUTRAL' 
                                      for x in backtest_results['signal_data'].tail(20).values]
                        })
                        st.dataframe(recent_signals, use_container_width=True, hide_index=True)
                    
                    # Strategy explanation
                    st.markdown("""
                    ### Backtest Conclusion
                    
                    This backtest demonstrates how profitable the yield spread strategy could be when implemented 
                    during specific periods. The strategy uses the yield spread between long-term and short-term 
                    bonds to signal when to buy and sell fixed-income assets, such as U.S. Treasury Bonds.
                    
                    **Use Cases:**
                    - **Portfolio Managers**: Yield curve analysis is critical for bond investors and portfolio managers.
                    - **Risk Management**: Traders and financial analysts can use this strategy to hedge against 
                      potential economic downturns.
                    - **Automation**: This strategy can be automated using Python to monitor yield spreads and 
                      generate real-time trading signals.
                    """)
                else:
                    st.warning("Backtest results not available. Please check date range and try again.")
            tab_idx += 1
        
        # TAB 5: Nelson-Siegel Model
        if show_ns_model and ns_result:
            with main_tabs[tab_idx]:
                st.subheader("Nelson-Siegel Parametric Curve Model")
                
                col1, col2 = st.columns(2)
                with col1:
                    params_df = pd.DataFrame({
                        'Parameter': ['β₀ (Long-Term Level)', 'β₁ (Short-Term Slope)', 'β₂ (Medium-Term Curvature)', 'λ (Decay Factor)'],
                        'Value': [f"{ns_result['params'][0]:.4f}", f"{ns_result['params'][1]:.4f}",
                                 f"{ns_result['params'][2]:.4f}", f"{ns_result['params'][3]:.4f}"]
                    })
                    st.dataframe(params_df, hide_index=True, use_container_width=True)
                    st.metric("Model RMSE", f"{ns_result['rmse']*100:.2f} bps")
                    st.metric("Model R²", f"{ns_result['r2']:.4f}")
                
                with col2:
                    fig_ns = go.Figure()
                    fig_ns.add_trace(go.Scatter(x=maturities, y=current_yields, mode='markers',
                                                name='Actual Data', marker=dict(size=12, color='#2c5f8a')))
                    fig_ns.add_trace(go.Scatter(x=np.linspace(0.25, 30, 100),
                                                y=NelsonSiegelModel.nelson_siegel(np.linspace(0.25, 30, 100), *ns_result['params']),
                                                mode='lines', name='NS Fit', line=dict(color='#c17f3a', width=2)))
                    fig_ns.update_layout(title='Nelson-Siegel Model Fit', xaxis_title='Maturity (Years)',
                                        yaxis_title='Yield (%)', template='plotly_white', height=400)
                    st.plotly_chart(fig_ns, use_container_width=True)
            tab_idx += 1
        
        # TAB 6: PCA & Factors
        if show_pca:
            with main_tabs[tab_idx]:
                st.subheader("Factor Analysis & PCA")
                
                col1, col2 = st.columns(2)
                with col1:
                    if not factors.empty:
                        fig_factors = go.Figure()
                        for col in factors.columns:
                            fig_factors.add_trace(go.Scatter(x=factors.index, y=factors[col], name=col, line=dict(width=2)))
                        fig_factors.update_layout(title='Level, Slope & Curvature Factors', 
                                                 xaxis_title='Date', yaxis_title='Value',
                                                 template='plotly_white', height=400)
                        st.plotly_chart(fig_factors, use_container_width=True)
                
                with col2:
                    if pca_result:
                        fig_pca = go.Figure(data=go.Bar(x=[f'PC{i+1}' for i in range(pca_result['n_components'])],
                                                        y=pca_result['explained_variance'] * 100, 
                                                        marker_color='#2c5f8a'))
                        fig_pca.update_layout(title='PCA Explained Variance', 
                                             xaxis_title='Principal Component',
                                             yaxis_title='Explained Variance (%)', 
                                             template='plotly_white', height=400)
                        st.plotly_chart(fig_pca, use_container_width=True)
                        st.markdown("### Factor Loadings")
                        st.dataframe(pca_result['loadings'].round(3), use_container_width=True)
            tab_idx += 1
        
        # TAB 7: Monte Carlo
        if show_monte_carlo:
            with main_tabs[tab_idx]:
                st.subheader("Monte Carlo Simulation")
                
                if st.button("Run Monte Carlo Simulation", use_container_width=True):
                    with st.spinner(f"Running {mc_simulations} simulations..."):
                        initial_yield = current_10y if not np.isnan(current_10y) else 4.0
                        returns = yield_df['10Y'].pct_change().dropna()
                        mu = returns.mean() * 252
                        sigma = returns.std() * np.sqrt(252)
                        
                        paths = MonteCarloSimulator.simulate_gbm(initial_yield, mu, sigma, mc_horizon, mc_simulations)
                        sim_results = MonteCarloSimulator.calculate_confidence_intervals(paths, 0.95)
                        var_estimate = MonteCarloSimulator.calculate_var(paths, 0.95)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Expected Terminal Value", f"{sim_results['mean'][-1]:.2f}%")
                        with col2:
                            st.metric(f"95% VaR", f"{var_estimate:.2f}%")
                        with col3:
                            st.metric("Simulation Paths", f"{mc_simulations:,}")
                        
                        fig_mc = go.Figure()
                        x_axis = np.arange(mc_horizon)
                        fig_mc.add_trace(go.Scatter(x=x_axis, y=sim_results['upper_ci'], fill=None, mode='lines', 
                                                    line=dict(color='rgba(0,0,0,0)'), showlegend=False))
                        fig_mc.add_trace(go.Scatter(x=x_axis, y=sim_results['lower_ci'], fill='tonexty', mode='lines',
                                                    fillcolor='rgba(44, 95, 138, 0.2)', line=dict(color='rgba(0,0,0,0)'), 
                                                    name='95% Confidence Interval'))
                        fig_mc.add_trace(go.Scatter(x=x_axis, y=sim_results['mean'], mode='lines', name='Mean Path', 
                                                    line=dict(color='#2c5f8a', width=2.5)))
                        fig_mc.update_layout(title='10Y Yield Simulation', xaxis_title='Trading Days', 
                                            yaxis_title='Yield (%)', template='plotly_white', height=500)
                        st.plotly_chart(fig_mc, use_container_width=True)
                        
                        fig_dist = go.Figure(data=go.Histogram(x=paths[:, -1], nbinsx=50, 
                                                              marker_color='#2c5f8a', opacity=0.7))
                        fig_dist.add_vline(x=sim_results['mean'][-1], line_color='red', line_dash='dash',
                                          annotation_text="Mean")
                        fig_dist.update_layout(title='Terminal Value Distribution', xaxis_title='Yield (%)',
                                              yaxis_title='Frequency', template='plotly_white', height=400)
                        st.plotly_chart(fig_dist, use_container_width=True)
            tab_idx += 1
        
        # TAB 8: ML Forecast
        if show_ml:
            with main_tabs[tab_idx]:
                st.subheader(f"Machine Learning Forecast - {ml_model_type}")
                
                if st.button("Train ML Model", use_container_width=True):
                    with st.spinner(f"Training {ml_model_type} model..."):
                        X, y, scaler = MLForecastModel.prepare_features(yield_df, '10Y', 5)
                        
                        if X is not None and len(X) > 50:
                            ml_results = MLForecastModel.train_model(X, y, ml_model_type)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("RMSE", f"{ml_results['rmse']*100:.2f} bps")
                            with col2:
                                st.metric("MAE", f"{ml_results['mae']*100:.2f} bps")
                            with col3:
                                st.metric("R²", f"{ml_results['r2']:.3f}")
                            
                            if not ml_results['feature_importance'].empty:
                                st.markdown("### Feature Importance")
                                st.dataframe(ml_results['feature_importance'], use_container_width=True)
                            
                            st.success(f"Model trained on {len(X)} samples")
                        else:
                            st.warning(f"Insufficient data. Need at least 50 samples, have {len(X) if X is not None else 0}")
            tab_idx += 1
        
        # TAB 9: Advanced Backtest
        if show_backtest:
            with main_tabs[tab_idx]:
                st.subheader(f"Advanced Strategy Backtest: {strategy_type} on {selected_etf}")
                
                etf_data = fetch_yahoo_data(selected_etf, start_date, end_date)
                
                if not etf_data.empty:
                    etf_returns = etf_data.pct_change()
                    backtest_result_adv = BacktestEngine.backtest_strategy(yield_df, spreads, etf_returns, strategy_type)
                    
                    if backtest_result_adv:
                        metrics_adv = backtest_result_adv['metrics']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Strategy Return", f"{metrics_adv.get('Total Return Strategy', 0):.2%}")
                            st.metric("Sharpe Ratio", f"{metrics_adv.get('Sharpe Ratio', 0):.2f}")
                        with col2:
                            st.metric("Benchmark Return", f"{metrics_adv.get('Total Return Benchmark', 0):.2%}")
                            st.metric("Volatility", f"{metrics_adv.get('Volatility', 0):.2%}")
                        with col3:
                            st.metric("Excess Return", f"{metrics_adv.get('Excess Return', 0):.2%}")
                            st.metric("Max Drawdown", f"{metrics_adv.get('Max Drawdown', 0):.2%}")
                        with col4:
                            st.metric("Win Rate", f"{metrics_adv.get('Win Rate', 0):.2%}")
                            st.metric("Profit Factor", f"{metrics_adv.get('Profit Factor', 0):.2f}")
                        
                        fig_backtest_adv = go.Figure()
                        fig_backtest_adv.add_trace(go.Scatter(x=backtest_result_adv['cumulative_strategy'].index, 
                                                              y=backtest_result_adv['cumulative_strategy'].values,
                                                              name='Strategy', line=dict(color='#2c5f8a', width=2)))
                        fig_backtest_adv.add_trace(go.Scatter(x=backtest_result_adv['cumulative_benchmark'].index, 
                                                              y=backtest_result_adv['cumulative_benchmark'].values,
                                                              name='Benchmark', line=dict(color='#c17f3a', width=2, dash='dash')))
                        fig_backtest_adv.update_layout(title='Cumulative Returns', xaxis_title='Date', 
                                                      yaxis_title='Return', template='plotly_white', height=400)
                        st.plotly_chart(fig_backtest_adv, use_container_width=True)
                        
                        signal_counts_adv = backtest_result_adv['signals'].value_counts()
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Buy Signals", signal_counts_adv.get(1, 0))
                        with col2:
                            st.metric("Sell Signals", signal_counts_adv.get(0, 0))
                        with col3:
                            st.metric("Hold Periods", signal_counts_adv.get(-1, 0))
                    else:
                        st.warning("Backtest failed. Try different parameters.")
                else:
                    st.error(f"Could not fetch {selected_etf} data")
            tab_idx += 1
        
        # TAB 10: Volatility
        if show_volatility:
            with main_tabs[tab_idx]:
                st.subheader("Volatility Analysis")
                
                if not volatility_df.empty:
                    vix_analysis = VolatilityAnalyzer.calculate_volatility_regime(volatility_df['CBOE Volatility Index'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"### Current Regime: {vix_analysis['regime']}")
                        st.info(vix_analysis['outlook'])
                        st.metric("Current VIX", f"{vix_analysis['current_vix']:.2f}")
                    
                    with col2:
                        fig_vix = go.Figure()
                        fig_vix.add_trace(go.Scatter(x=volatility_df.index, y=volatility_df['CBOE Volatility Index'],
                                                     mode='lines', name='VIX', line=dict(color='#c17f3a', width=2), fill='tozeroy'))
                        fig_vix.add_hline(y=20, line_dash='dash', line_color='red', annotation_text="Elevated")
                        fig_vix.add_hline(y=15, line_dash='dash', line_color='orange', annotation_text="Normal")
                        fig_vix.update_layout(title='VIX Historical Chart', xaxis_title='Date', 
                                             yaxis_title='VIX', template='plotly_white', height=400)
                        st.plotly_chart(fig_vix, use_container_width=True)
                else:
                    st.info("Volatility data not available")
            tab_idx += 1
        
        # TAB 11: Correlation
        if show_correlation:
            with main_tabs[tab_idx]:
                st.subheader("Correlation Analysis")
                
                if not correlation_df.empty and not yield_df.empty:
                    all_assets = pd.DataFrame(index=yield_df.index)
                    all_assets['10Y Yield'] = yield_df['10Y']
                    for col in correlation_df.columns:
                        all_assets[col] = correlation_df[col]
                    all_assets = all_assets.dropna()
                    
                    if not all_assets.empty:
                        corr_matrix = CorrelationAnalyzer.calculate_correlation_matrix(all_assets)
                        fig_corr = CorrelationAnalyzer.plot_correlation_heatmap(corr_matrix)
                        if fig_corr:
                            st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Correlation data not available")
            tab_idx += 1
        
        # TAB 12: Technical Analysis
        if show_technical:
            with main_tabs[tab_idx]:
                st.subheader("Technical Analysis")
                
                tech_ticker = st.selectbox("Select Asset for Technical Analysis", ['TLT', 'IEF', 'SHY', 'SPY', 'QQQ'])
                
                tech_data = fetch_yahoo_data(tech_ticker, start_date, end_date)
                
                if not tech_data.empty:
                    tech_df = calculate_technical_indicators(tech_data)
                    fig_tech = plot_technical_indicators(tech_df, tech_ticker)
                    if fig_tech:
                        st.plotly_chart(fig_tech, use_container_width=True)
                    
                    st.subheader("Current Technical Signals")
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
                else:
                    st.error(f"Could not fetch {tech_ticker} data")
            tab_idx += 1
        
        # TAB 13: Scenarios
        if show_scenarios:
            with main_tabs[tab_idx]:
                st.subheader("Scenario Analysis")
                
                scenarios = generate_scenarios(yield_df)
                
                if scenarios:
                    selected_scenario = st.selectbox("Select Scenario", list(scenarios.keys()))
                    scenario_data = scenarios[selected_scenario]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Scenario Impact")
                        current = yield_df.iloc[-1]
                        changes = {}
                        for maturity in scenario_data.index:
                            changes[maturity] = scenario_data[maturity] - current[maturity]
                        
                        changes_df = pd.DataFrame({
                            'Maturity': changes.keys(),
                            'Current (%)': [current[m] for m in changes.keys()],
                            'Scenario (%)': [scenario_data[m] for m in changes.keys()],
                            'Change (bps)': [changes[m]*100 for m in changes.keys()]
                        })
                        st.dataframe(changes_df, use_container_width=True, hide_index=True)
                    
                    with col2:
                        fig_scenario = plot_scenario_comparison(current, scenario_data, selected_scenario)
                        st.plotly_chart(fig_scenario, use_container_width=True)
                    
                    st.markdown("### Scenario Interpretation")
                    if selected_scenario == "Bull Steepener":
                        st.info("📈 **Bull Steepener**: Long-term rates falling faster than short-term. Typically occurs during Fed easing cycles. Positive for long-duration bonds.")
                    elif selected_scenario == "Bear Flattener":
                        st.warning("📊 **Bear Flattener**: Short-term rates rising faster than long-term. Often precedes curve inversion. Caution for duration exposure.")
                    elif selected_scenario == "Recession":
                        st.error("⚠️ **Recession Scenario**: Significant curve inversion and lower rates across all maturities. Defensive positioning recommended.")
                    elif selected_scenario == "Inflation Shock":
                        st.warning("🔥 **Inflation Shock**: Rising rates across the curve. Short-term impact more severe. Reduce duration exposure.")

else:
    st.info("👈 Configure your analysis parameters and click 'Run Analysis' to start")
    
    with st.expander("📖 Platform Features - Complete Suite", expanded=True):
        st.markdown("""
        ### 🚀 Complete Feature Suite
        
        This institutional-grade platform includes **13 comprehensive analysis modules**:
        
        | # | Module | Description |
        |---|--------|-------------|
        | 1 | **Yield Curve Analysis** | Interactive curve visualization, historical comparison |
        | 2 | **Spread Analysis** | 2s10s, 3m10y, 5s30s spreads, forward rates |
        | 3 | **Yield Spread Trading Strategy (Step 5)** | Simple strategy based on 10Y-3M spread |
        | 4 | **Strategy Backtest with TLT (Step 6)** | Backtest using TLT ETF, comprehensive metrics |
        | 5 | **Nelson-Siegel Model** | Parametric yield curve modeling |
        | 6 | **PCA & Factor Analysis** | Level, Slope, Curvature factors |
        | 7 | **Monte Carlo Simulation** | GBM stochastic simulation, VaR calculation |
        | 8 | **Machine Learning Forecast** | Random Forest, Gradient Boosting predictions |
        | 9 | **Advanced Strategy Backtest** | Curve Inversion, Momentum, Mean Reversion |
        | 10 | **Volatility Analysis** | VIX analysis, volatility regime detection |
        | 11 | **Correlation Analysis** | Cross-asset correlation matrix |
        | 12 | **Technical Analysis** | RSI, MACD, Bollinger Bands, moving averages |
        | 13 | **Scenario Analysis** | Bull/Bear, Recession, Inflation scenarios |
        
        ### 📊 Data Source
        - **FRED API**: Federal Reserve Economic Data (Official)
        - **NO SYNTHETIC DATA** - All yield data comes directly from FRED
        - Yahoo Finance used only for ETF prices (backtesting) and market indicators
        
        ### 🔑 Getting a FRED API Key
        1. Visit [FRED API website](https://fred.stlouisfed.org/docs/api/api_key.html)
        2. Click "Request API Key"
        3. Register for a free account
        4. Your API key will be emailed instantly
        5. Enter it in the sidebar to start
        
        ### ⚠️ Risk Warning
        Past performance does not guarantee future results. This platform is for educational and research purposes only.
        """)

# Footer
st.markdown("---")
st.markdown(f"*Data source: FRED (Federal Reserve Economic Data) | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
