import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta, date
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# FRED API KONFIGÜRASYONU
# =============================================================================

# FRED API Series IDs for Treasury yields
FRED_SERIES = {
    '3M': 'DGS3MO',     # 3-Month Treasury Bill
    '2Y': 'DGS2',       # 2-Year Treasury Note  
    '5Y': 'DGS5',       # 5-Year Treasury Note
    '10Y': 'DGS10',     # 10-Year Treasury Note
    '30Y': 'DGS30'      # 30-Year Treasury Bond
}

# Recession indicator
RECESSION_SERIES = 'USREC'

# Bond ETFs for backtesting
BOND_ETFS = {
    'TLT': '20+ Year Treasury Bond ETF',
    'IEF': '7-10 Year Treasury Bond ETF', 
    'SHY': '1-3 Year Treasury Bond ETF',
    'BND': 'Total Bond Market ETF',
    'GOVT': 'US Treasury Bond ETF'
}

# =============================================================================
# FRED API FONKSİYONLARI
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
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
        st.warning(f"Error fetching {series_id}: {str(e)}")
        return pd.Series(dtype='float64')

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_yield_data(api_key, start_date, end_date):
    """Fetch all yield curve data from FRED"""
    data = {}
    
    for name, series_id in FRED_SERIES.items():
        series_data = fetch_fred_series(api_key, series_id, start_date, end_date)
        if not series_data.empty:
            data[name] = series_data
        else:
            st.warning(f"Could not fetch {name} data (Series: {series_id})")
    
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # Forward fill missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recession_data(api_key, start_date, end_date):
    """Fetch recession indicator data"""
    series = fetch_fred_series(api_key, RECESSION_SERIES, start_date, end_date)
    return series

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
# ALTERNATİF YFINANCE FONKSİYONLARI (FALLBACK)
# =============================================================================

def fetch_yfinance_alternative(start_date, end_date):
    """Fallback to yfinance if FRED fails"""
    import yfinance as yf
    
    # Available yfinance tickers
    tickers = {
        '3M': '^IRX',
        '5Y': '^FVX', 
        '10Y': '^TNX',
        '30Y': '^TYX'
    }
    
    data = {}
    
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not df.empty and 'Adj Close' in df.columns:
                series = df['Adj Close']
                series.index = series.index.tz_localize(None)
                data[name] = series
                
                # Convert IRX from discount rate to yield
                if name == '3M':
                    data[name] = (100 - data[name]) * 4
        except Exception as e:
            st.warning(f"Could not fetch {name} from yfinance: {e}")
    
    # Create synthetic 2Y yield (interpolation between 3M and 5Y)
    if '3M' in data and '5Y' in data and not data['3M'].empty and not data['5Y'].empty:
        data['2Y'] = (data['3M'] + data['5Y']) / 2
    else:
        data['2Y'] = pd.Series(dtype='float64')
    
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

# =============================================================================
# YIELD CURVE ANALİZ FONKSİYONLARI
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
        valid_hist_maturities = []
        valid_hist_yields = []
        
        for mat, year in maturities.items():
            if mat in historical_yields.index and pd.notna(historical_yields[mat]):
                valid_hist_maturities.append(year)
                valid_hist_yields.append(historical_yields[mat])
        
        if valid_hist_maturities:
            fig.add_trace(go.Scatter(
                x=valid_hist_maturities,
                y=valid_hist_yields,
                name='1 Year Ago',
                line=dict(color='#c17f3a', width=2, dash='dash'),
                mode='lines+markers',
                marker=dict(size=8)
            ))
    
    fig.update_layout(
        title='U.S. Treasury Yield Curve',
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
        if not spreads[col].isna().all():
            fig.add_trace(go.Scatter(
                x=spreads.index,
                y=spreads[col],
                name=col.upper(),
                line=dict(color=colors.get(col, '#666'), width=2),
                fill='tozeroy',
                fillcolor='rgba(44, 95, 138, 0.1)'
            ))
    
    fig.add_hline(y=0, line_color='red', line_dash='dash', line_width=1.5)
    fig.add_hline(y=-0.5, line_color='orange', line_dash='dot', line_width=1)
    
    fig.update_layout(
        title='Treasury Yield Spreads',
        xaxis_title='Date',
        yaxis_title='Spread (%)',
        template='plotly_white',
        hovermode='x unified',
        height=500
    )
    
    return fig

def calculate_forward_rates(df):
    """Calculate implied forward rates"""
    forwards = pd.DataFrame(index=df.index)
    maturities = {'3M': 0.25, '2Y': 2, '5Y': 5, '10Y': 10, '30Y': 30}
    
    pairs = []
    available_maturities = [m for m in maturities.keys() if m in df.columns]
    
    for i in range(len(available_maturities) - 1):
        pairs.append((available_maturities[i], available_maturities[i + 1]))
    
    for short_term, long_term in pairs:
        if short_term in df.columns and long_term in df.columns:
            r1 = maturities[short_term]
            r2 = maturities[long_term]
            
            try:
                forward_rate = (((1 + df[long_term] / 100) ** r2 / (1 + df[short_term] / 100) ** r1) ** (1 / (r2 - r1)) - 1) * 100
                forwards[f'{short_term}→{long_term}'] = forward_rate
            except:
                pass
    
    return forwards

# =============================================================================
# TRADING STRATEJİSİ
# =============================================================================

class YieldCurveTradingStrategy:
    """Trading strategy based on yield curve signals"""
    
    def __init__(self, yields_df, spreads_df):
        self.yields = yields_df
        self.spreads = spreads_df
        self.signals = None
        self.results = None
    
    def generate_signals(self):
        """Generate trading signals"""
        signals = pd.DataFrame(index=self.spreads.index)
        
        # Classic 2s10s inversion
        signals['Classic'] = 0
        if '2s10s' in self.spreads.columns:
            signals.loc[self.spreads['2s10s'] < 0, 'Classic'] = 1
            signals.loc[self.spreads['2s10s'] > 0.5, 'Classic'] = -1
        
        # 3m10y signal
        signals['3m10y'] = 0
        if '3m10y' in self.spreads.columns:
            signals.loc[self.spreads['3m10y'] < 0, '3m10y'] = 1
            signals.loc[self.spreads['3m10y'] > 0.5, '3m10y'] = -1
        
        # Composite
        signals['Composite'] = (signals['Classic'] + signals['3m10y']) / 2
        signals['Composite'] = signals['Composite'].clip(-1, 1)
        
        self.signals = signals
        return self
    
    def backtest(self, etf_prices, signal_col='Composite', transaction_cost=0.001):
        """Backtest the strategy"""
        import yfinance as yf
        
        if self.signals is None:
            self.generate_signals()
        
        if signal_col not in self.signals.columns:
            return None
        
        # Fetch ETF data if not provided
        if etf_prices is None or etf_prices.empty:
            try:
                etf_data = yf.download('TLT', start=self.yields.index[0], end=self.yields.index[-1], progress=False)
                if not etf_data.empty and 'Adj Close' in etf_data.columns:
                    etf_prices = etf_data['Adj Close']
                    etf_prices.index = etf_prices.index.tz_localize(None)
                else:
                    return None
            except:
                return None
        
        signals = self.signals[signal_col].copy()
        
        # Align signals with ETF prices
        common_idx = etf_prices.index.intersection(signals.index)
        if len(common_idx) == 0:
            return None
        
        signals_aligned = signals.reindex(common_idx)
        prices_aligned = etf_prices.reindex(common_idx)
        
        # Calculate returns
        returns = prices_aligned.pct_change()
        strategy_returns = signals_aligned.shift(1) * returns
        
        # Apply transaction costs
        signal_changes = signals_aligned.diff().abs()
        transaction_costs = signal_changes * transaction_cost
        strategy_returns = strategy_returns - transaction_costs
        
        strategy_returns = strategy_returns.fillna(0)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        buy_hold_returns = (1 + returns.fillna(0)).cumprod()
        
        # Calculate metrics
        metrics = self.calculate_metrics(strategy_returns, returns.fillna(0), 
                                        cumulative_returns, buy_hold_returns)
        
        self.results = {
            'strategy_returns': strategy_returns,
            'cumulative_strategy': cumulative_returns,
            'cumulative_bh': buy_hold_returns,
            'signals': signals_aligned,
            'metrics': metrics,
            'signal_col': signal_col
        }
        
        return self
    
    def calculate_metrics(self, strategy_returns, benchmark_returns, cum_strategy, cum_benchmark):
        """Calculate performance metrics"""
        
        strategy_clean = strategy_returns[strategy_returns != 0]
        
        if len(strategy_clean) == 0:
            return {
                'Total Return Strategy': 0,
                'Total Return Benchmark': float(cum_benchmark.iloc[-1] - 1) if len(cum_benchmark) > 0 else 0,
                'Excess Return': 0,
                'Sharpe Ratio': 0,
                'Max Drawdown': 0,
                'Win Rate': 0,
                'Profit Factor': 0
            }
        
        ann_factor = np.sqrt(252)
        
        metrics = {
            'Total Return Strategy': float(cum_strategy.iloc[-1] - 1),
            'Total Return Benchmark': float(cum_benchmark.iloc[-1] - 1),
            'Sharpe Ratio': (strategy_returns.mean() / strategy_returns.std()) * ann_factor if strategy_returns.std() > 0 else 0,
            'Max Drawdown': self.calculate_max_drawdown(cum_strategy),
            'Win Rate': float((strategy_returns > 0).sum() / len(strategy_returns[strategy_returns != 0])),
            'Profit Factor': 0
        }
        
        metrics['Excess Return'] = metrics['Total Return Strategy'] - metrics['Total Return Benchmark']
        
        gross_profits = strategy_returns[strategy_returns > 0].sum()
        gross_losses = abs(strategy_returns[strategy_returns < 0].sum())
        metrics['Profit Factor'] = float(gross_profits / gross_losses) if gross_losses > 0 else 0
        
        return metrics
    
    @staticmethod
    def calculate_max_drawdown(cumulative_returns):
        if len(cumulative_returns) == 0:
            return 0
        cum_clean = cumulative_returns.fillna(1)
        rolling_max = cum_clean.expanding().max()
        drawdown = (cum_clean - rolling_max) / rolling_max
        return float(drawdown.min())

def get_recession_probability(spreads_df):
    """Calculate recession probability"""
    if '2s10s' not in spreads_df.columns or spreads_df.empty:
        return 0.5
    
    current_spread = spreads_df['2s10s'].iloc[-1]
    if pd.isna(current_spread):
        return 0.5
    
    prob = 1 / (1 + np.exp(-(-current_spread * 2 - 0.5)))
    return min(max(prob, 0.01), 0.99)

# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(
    page_title="Bond Yield Curve Analysis Platform",
    page_icon="📈",
    layout="wide"
)

# Session state initialization
if 'api_key_validated' not in st.session_state:
    st.session_state.api_key_validated = False
if 'yield_data' not in st.session_state:
    st.session_state.yield_data = None

# Title
st.title("📈 Bond Yield Curve Analysis Platform")
st.markdown("*Institutional-Grade Fixed Income Analytics with FRED Data*")

# API Key Management
if not st.session_state.api_key_validated:
    st.markdown("""
    ### 🔑 FRED API Key Required
    
    This platform uses FRED (Federal Reserve Economic Data) for accurate Treasury yield data.
    
    **Get your free API key:**
    1. Go to [FRED API website](https://fred.stlouisfed.org/docs/api/api_key.html)
    2. Register for a free account
    3. Request an API key (instant, free)
    4. Enter your key below
    """)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        api_key = st.text_input("Enter your FRED API Key", type="password", placeholder="abcdefghijklmnopqrstuvwxyz123456")
        
        if st.button("🔐 Validate & Connect", use_container_width=True):
            if not api_key:
                st.error("Please enter an API key")
            else:
                with st.spinner("Validating API key..."):
                    if validate_fred_api_key(api_key):
                        st.session_state.api_key = api_key
                        st.session_state.api_key_validated = True
                        st.success("✅ API key validated successfully!")
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
    
    data_source = st.radio("Data Source", ["FRED API (Recommended)", "Yahoo Finance (Fallback)"])
    
    st.markdown("---")
    st.header("📊 Strategy Parameters")
    
    selected_etf = st.selectbox("ETF for Backtest", list(BOND_ETFS.keys()))
    selected_strategy = st.selectbox("Trading Strategy", ['Composite', 'Classic', '3m10y'])
    transaction_cost = st.slider("Transaction Cost (%)", 0.0, 0.5, 0.1, 0.01) / 100
    
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Main content
if run_analysis:
    with st.spinner("Fetching yield curve data from FRED..."):
        
        # Fetch data based on source
        if data_source == "FRED API (Recommended)":
            df = fetch_all_yield_data(st.session_state.api_key, start_date, end_date)
        else:
            df = fetch_yfinance_alternative(start_date, end_date)
        
        if df.empty:
            st.error("""
            ❌ **Failed to fetch yield data**
            
            Possible solutions:
            1. Try switching to "Yahoo Finance (Fallback)" in the sidebar
            2. Reduce the date range
            3. Check your internet connection
            4. Your FRED API key might be invalid - please re-enter it
            """)
            
            # Reset API key option
            if st.button("Reset API Key"):
                st.session_state.api_key_validated = False
                st.rerun()
            st.stop()
        
        # Fetch recession data
        recession_data = fetch_recession_data(st.session_state.api_key, start_date, end_date)
        
        # Calculate metrics
        spreads = calculate_spreads(df)
        forwards = calculate_forward_rates(df)
        recession_prob = get_recession_probability(spreads)
        
        # Display success
        st.success(f"✅ Data loaded successfully! Period: {start_date} to {end_date}")
        st.info(f"📊 Available maturities: {', '.join(df.columns)}")
        
        # KPI Row
        st.subheader("📊 Current Market Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            current_10y = df['10Y'].iloc[-1] if '10Y' in df.columns else np.nan
            st.metric("10Y Yield", f"{current_10y:.2f}%" if not np.isnan(current_10y) else "N/A")
        
        with col2:
            current_2y = df['2Y'].iloc[-1] if '2Y' in df.columns else np.nan
            st.metric("2Y Yield", f"{current_2y:.2f}%" if not np.isnan(current_2y) else "N/A")
        
        with col3:
            current_spread = spreads['2s10s'].iloc[-1] if '2s10s' in spreads.columns else np.nan
            delta_color = "inverse" if current_spread < 0 else "normal"
            st.metric("2s10s Spread", f"{current_spread:.2f}%" if not np.isnan(current_spread) else "N/A",
                     delta="Inverted" if current_spread < 0 else "Normal",
                     delta_color=delta_color)
        
        with col4:
            st.metric("Recession Probability", f"{recession_prob:.1%}")
        
        with col5:
            st.metric("Data Source", "FRED" if data_source == "FRED API (Recommended)" else "Yahoo Finance")
        
        # Recession warning
        if current_spread < 0:
            st.warning("⚠️ **YIELD CURVE IS INVERTED!** Historically, this signals a recession within 6-18 months. Consider defensive positioning.")
        elif current_spread < 0.5:
            st.info("📊 **Yield curve is flattening.** Monitor closely for potential inversion signals.")
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Yield Curve", 
            "📊 Spread Analysis", 
            "🎯 Trading Strategy",
            "📋 Data Explorer"
        ])
        
        # Tab 1: Yield Curve
        with tab1:
            fig_yield = plot_yield_curve(df, end_date)
            if fig_yield:
                st.plotly_chart(fig_yield, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Curve Statistics")
                if not df.empty:
                    latest = df.iloc[-1]
                    st.metric("Curve Steepness (10Y-2Y)", f"{latest.get('10Y', 0) - latest.get('2Y', 0):.2f}%")
                    st.metric("Short End (3M)", f"{latest.get('3M', 0):.2f}%")
                    st.metric("Long End (30Y)", f"{latest.get('30Y', 0):.2f}%")
            
            with col2:
                st.markdown("### Recent Changes (1 Day)")
                if len(df) > 1:
                    changes = df.iloc[-1] - df.iloc[-2]
                    st.metric("10Y Change", f"{changes.get('10Y', 0):+.2f}%")
                    st.metric("2Y Change", f"{changes.get('2Y', 0):+.2f}%")
        
        # Tab 2: Spread Analysis
        with tab2:
            if not spreads.empty:
                fig_spreads = plot_spreads(spreads)
                if fig_spreads:
                    st.plotly_chart(fig_spreads, use_container_width=True)
                
                st.markdown("### Current Spreads")
                current_spreads = spreads.iloc[-1]
                cols = st.columns(len(current_spreads))
                for idx, (name, value) in enumerate(current_spreads.items()):
                    if pd.notna(value):
                        cols[idx].metric(f"{name.upper()}", f"{value:.2f}%")
                
                # Historical spread statistics
                with st.expander("Spread Statistics"):
                    st.dataframe(spreads.describe().round(2), use_container_width=True)
        
        # Tab 3: Trading Strategy
        with tab3:
            st.subheader(f"Strategy: {selected_strategy} on {selected_etf}")
            
            # Import yfinance for ETF data
            import yfinance as yf
            
            # Fetch ETF data
            etf_data = yf.download(selected_etf, start=start_date, end=end_date, progress=False)
            if not etf_data.empty and 'Adj Close' in etf_data.columns:
                etf_prices = etf_data['Adj Close']
                etf_prices.index = etf_prices.index.tz_localize(None)
                
                # Run backtest
                strategy = YieldCurveTradingStrategy(df, spreads)
                strategy.generate_signals()
                backtest_results = strategy.backtest(etf_prices, selected_strategy, transaction_cost)
                
                if backtest_results and backtest_results.results:
                    results = backtest_results.results
                    metrics = results['metrics']
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Strategy Return", f"{metrics['Total Return Strategy']:.2%}")
                        st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
                    
                    with col2:
                        st.metric("Benchmark Return", f"{metrics['Total Return Benchmark']:.2%}")
                        st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
                    
                    with col3:
                        st.metric("Excess Return", f"{metrics['Excess Return']:.2%}")
                        st.metric("Win Rate", f"{metrics['Win Rate']:.2%}")
                    
                    with col4:
                        st.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")
                    
                    # Cumulative returns chart
                    fig_returns = go.Figure()
                    fig_returns.add_trace(go.Scatter(
                        x=results['cumulative_strategy'].index,
                        y=results['cumulative_strategy'].values,
                        name='Strategy',
                        line=dict(color='#2c5f8a', width=2)
                    ))
                    fig_returns.add_trace(go.Scatter(
                        x=results['cumulative_bh'].index,
                        y=results['cumulative_bh'].values,
                        name='Buy & Hold',
                        line=dict(color='#c17f3a', width=2, dash='dash')
                    ))
                    fig_returns.update_layout(
                        title='Cumulative Returns',
                        xaxis_title='Date',
                        yaxis_title='Return',
                        template='plotly_white',
                        height=400
                    )
                    st.plotly_chart(fig_returns, use_container_width=True)
                    
                    # Recent signals
                    with st.expander("Recent Trading Signals"):
                        recent = pd.DataFrame({
                            'Date': results['signals'].tail(20).index.strftime('%Y-%m-%d'),
                            'Signal': results['signals'].tail(20).values,
                            'Action': ['BUY' if x > 0 else 'SELL' if x < 0 else 'HOLD' 
                                      for x in results['signals'].tail(20).values]
                        })
                        st.dataframe(recent, use_container_width=True, hide_index=True)
                else:
                    st.warning("Backtest failed. Try a different date range or ETF.")
            else:
                st.error(f"Could not fetch {selected_etf} data")
        
        # Tab 4: Data Explorer
        with tab4:
            st.subheader("Historical Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Summary Statistics")
                st.dataframe(df.describe().round(2), use_container_width=True)
            
            with col2:
                st.markdown("### Correlation Matrix")
                if len(df.columns) > 1:
                    st.dataframe(df.corr().round(2), use_container_width=True)
            
            st.markdown("### Raw Data (Last 50 rows)")
            st.dataframe(df.tail(50), use_container_width=True)
            
            # Download
            csv = df.to_csv()
            st.download_button("📥 Download Yield Data (CSV)", csv, f"yield_data_{start_date}_{end_date}.csv", "text/csv")

else:
    st.info("👈 Configure your parameters and click 'Run Analysis' to start")
    
    with st.expander("📖 About This Platform", expanded=True):
        st.markdown("""
        ### Features
        
        **Data Sources:**
        - **FRED API**: Official Federal Reserve Economic Data (recommended)
        - **Yahoo Finance**: Fallback data source
        
        **Analysis Modules:**
        
        1. **Yield Curve Analysis**
           - Interactive curve visualization
           - Historical comparison (1 year ago)
           - Curve statistics and daily changes
        
        2. **Spread Analysis**
           - 2s10s (2-year vs 10-year) - Key recession indicator
           - 3m10y (3-month vs 10-year)
           - Historical spread visualization
        
        3. **Trading Strategy**
           - Multiple strategy types (Classic, 3m10y, Composite)
           - Backtesting with transaction costs
           - Performance metrics (Sharpe, Drawdown, Win Rate)
        
        4. **Data Export**
           - Download yield data as CSV
           - Export for further analysis
        
        ### Getting a FRED API Key
        
        1. Visit [FRED API website](https://fred.stlouisfed.org/docs/api/api_key.html)
        2. Click "Request API Key"
        3. Register for a free account
        4. Your API key will be emailed instantly
        5. Enter it in the sidebar
        
        ### Risk Warning
        
        Past performance does not guarantee future results. This platform is for educational purposes only.
        """)

# Footer
st.markdown("---")
st.markdown(f"*Data source: FRED | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
