import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta, date
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CORRECTED TREASURY SYMBOLS
# =============================================================================

# Valid yfinance symbols for Treasury yields
TREASURY_SYMBOLS = {
    '3M': '^IRX',   # 13-week Treasury Bill (works)
    '2Y': '^IRX',   # Using 3M as proxy (most reliable) - OR use '^FVX' for 5Y as proxy
    '5Y': '^FVX',   # 5-year Treasury Note (works)
    '10Y': '^TNX',  # 10-year Treasury Note (works)
    '30Y': '^TYX'   # 30-year Treasury Bond (works)
}

# Alternative: Use FRED data via yfinance (more reliable)
FRED_TREASURY_SYMBOLS = {
    '3M': 'DGS3MO',     # 3-Month Treasury Bill
    '2Y': 'DGS2',       # 2-Year Treasury Note
    '5Y': 'DGS5',       # 5-Year Treasury Note
    '10Y': 'DGS10',     # 10-Year Treasury Note
    '30Y': 'DGS30'      # 30-Year Treasury Bond
}

# Bond ETFs for backtesting
BOND_ETFS = {
    'TLT': '20+ Year Treasury Bond ETF',
    'IEF': '7-10 Year Treasury Bond ETF',
    'SHY': '1-3 Year Treasury Bond ETF',
    'BND': 'Total Bond Market ETF',
    'GOVT': 'US Treasury Bond ETF'
}

# =============================================================================
# IMPROVED DATA FETCHING FUNCTIONS
# =============================================================================

def fetch_yield_data_fred(start_date=None, end_date=None):
    """
    Fetch yield curve data using FRED symbols via yfinance
    This is more reliable than market tickers
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365*2)
    if end_date is None:
        end_date = datetime.now()
    
    yields_data = {}
    
    for maturity, symbol in FRED_TREASURY_SYMBOLS.items():
        try:
            # Add FRED prefix for yfinance
            fred_symbol = f"{symbol}=X"
            data = yf.download(fred_symbol, start=start_date, end=end_date, progress=False)
            
            if not data.empty and 'Adj Close' in data.columns:
                yields_data[maturity] = data['Adj Close']
            else:
                # Fallback to regular symbol without =X
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not data.empty and 'Adj Close' in data.columns:
                    yields_data[maturity] = data['Adj Close']
                else:
                    print(f"Could not fetch data for {maturity}")
                    yields_data[maturity] = pd.Series(dtype='float64')
        except Exception as e:
            print(f"Error fetching {maturity}: {e}")
            yields_data[maturity] = pd.Series(dtype='float64')
    
    df = pd.DataFrame(yields_data)
    
    # Fill missing values
    if not df.empty:
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def fetch_yield_data_alternative(start_date=None, end_date=None):
    """
    Alternative method using market tickers with proxy for 2Y
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365*2)
    if end_date is None:
        end_date = datetime.now()
    
    yields_data = {}
    
    # Use available tickers
    available_symbols = {
        '3M': '^IRX',
        '5Y': '^FVX',
        '10Y': '^TNX',
        '30Y': '^TYX'
    }
    
    for maturity, symbol in available_symbols.items():
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not data.empty and 'Adj Close' in data.columns:
                yields_data[maturity] = data['Adj Close']
            else:
                yields_data[maturity] = pd.Series(dtype='float64')
        except Exception as e:
            print(f"Error fetching {maturity}: {e}")
            yields_data[maturity] = pd.Series(dtype='float64')
    
    # Create synthetic 2Y yield (interpolation between 3M and 5Y)
    if '3M' in yields_data and '5Y' in yields_data:
        if not yields_data['3M'].empty and not yields_data['5Y'].empty:
            # Simple interpolation: 2Y = (3M + 5Y) / 2
            yields_data['2Y'] = (yields_data['3M'] + yields_data['5Y']) / 2
        else:
            yields_data['2Y'] = pd.Series(dtype='float64')
    else:
        yields_data['2Y'] = pd.Series(dtype='float64')
    
    df = pd.DataFrame(yields_data)
    
    if not df.empty:
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def fetch_yield_data_combined(start_date=None, end_date=None):
    """
    Combined approach: Try FRED first, fallback to market data
    """
    # First try FRED data
    df_fred = fetch_yield_data_fred(start_date, end_date)
    
    if not df_fred.empty and len(df_fred) > 10:
        return df_fred
    
    # Fallback to alternative market data
    df_market = fetch_yield_data_alternative(start_date, end_date)
    
    if not df_market.empty:
        return df_market
    
    # If all fails, return empty dataframe
    return pd.DataFrame()

def fetch_etf_data(etf_symbol, start_date, end_date):
    """Fetch ETF price data for backtesting"""
    try:
        data = yf.download(etf_symbol, start=start_date, end=end_date, progress=False)
        if not data.empty and 'Adj Close' in data.columns:
            return data['Adj Close']
    except Exception as e:
        print(f"Error fetching {etf_symbol}: {e}")
    return pd.Series(dtype='float64')

def fetch_vix_data(start_date, end_date):
    """Fetch VIX data for volatility analysis"""
    try:
        data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
        if not data.empty and 'Adj Close' in data.columns:
            return data['Adj Close']
    except Exception as e:
        print(f"Error fetching VIX: {e}")
    return pd.Series(dtype='float64')

def fetch_economic_indicators(start_date, end_date):
    """Fetch additional economic indicators"""
    indicators = {}
    
    # Try to fetch various indicators
    symbols = {
        'SPY': 'S&P 500',
        'QQQ': 'Nasdaq',
        'GLD': 'Gold',
        'UUP': 'Dollar Index'
    }
    
    for symbol, name in symbols.items():
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not data.empty and 'Adj Close' in data.columns:
                indicators[name] = data['Adj Close']
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
    
    return pd.DataFrame(indicators) if indicators else pd.DataFrame()

# =============================================================================
# YIELD CURVE ANALYSIS FUNCTIONS
# =============================================================================

def plot_yield_curve(df, selected_date):
    """
    Create interactive yield curve plot
    """
    if df.empty:
        return None
    
    available_dates = df.index
    closest_date = available_dates[available_dates <= pd.Timestamp(selected_date)].max()
    
    if pd.isnull(closest_date):
        return None
    
    # Define maturities in years
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
    
    # Add historical comparison (1 year ago)
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
    """
    Calculate various yield spreads
    """
    spreads = pd.DataFrame(index=df.index)
    
    if '2Y' in df.columns and '10Y' in df.columns:
        spreads['2s10s'] = df['10Y'] - df['2Y']
    
    if '3M' in df.columns and '10Y' in df.columns:
        spreads['3m10y'] = df['10Y'] - df['3M']
    
    if '5Y' in df.columns and '30Y' in df.columns:
        spreads['5s30s'] = df['30Y'] - df['5Y']
    
    if '2Y' in df.columns and '5Y' in df.columns:
        spreads['2s5s'] = df['5Y'] - df['2Y']
    
    return spreads

def plot_spreads(spreads):
    """
    Create spread visualization
    """
    if spreads.empty:
        return None
    
    fig = go.Figure()
    
    colors = {'2s10s': '#2c5f8a', '3m10y': '#c17f3a', '5s30s': '#4a7c59', '2s5s': '#8b5cf6'}
    
    for col in spreads.columns:
        if not spreads[col].isna().all():
            fig.add_trace(go.Scatter(
                x=spreads.index,
                y=spreads[col],
                name=col.upper(),
                line=dict(color=colors.get(col, '#666'), width=2),
                fill='tozeroy',
                fillcolor=f'rgba(44, 95, 138, 0.1)'
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
    """
    Calculate implied forward rates
    """
    forwards = pd.DataFrame(index=df.index)
    maturities = {'3M': 0.25, '2Y': 2, '5Y': 5, '10Y': 10, '30Y': 30}
    
    pairs = []
    if '3M' in df.columns and '2Y' in df.columns:
        pairs.append(('3M', '2Y'))
    if '2Y' in df.columns and '5Y' in df.columns:
        pairs.append(('2Y', '5Y'))
    if '5Y' in df.columns and '10Y' in df.columns:
        pairs.append(('5Y', '10Y'))
    if '10Y' in df.columns and '30Y' in df.columns:
        pairs.append(('10Y', '30Y'))
    
    for short_term, long_term in pairs:
        r1 = maturities[short_term]
        r2 = maturities[long_term]
        
        try:
            forward_rate = (((1 + df[long_term] / 100) ** r2 / (1 + df[short_term] / 100) ** r1) ** (1 / (r2 - r1)) - 1) * 100
            forwards[f'{short_term}→{long_term}'] = forward_rate
        except Exception as e:
            print(f"Could not calculate forward rate for {short_term}-{long_term}: {e}")
    
    return forwards

# =============================================================================
# TRADING STRATEGY FUNCTIONS
# =============================================================================

class YieldCurveTradingStrategy:
    """Trading strategy based on yield curve signals"""
    
    def __init__(self, yields_df, spreads_df):
        self.yields = yields_df
        self.spreads = spreads_df
        self.signals = None
        self.results = None
    
    def generate_signals(self, strategy_type='composite'):
        """Generate trading signals based on yield curve"""
        
        signals = pd.DataFrame(index=self.spreads.index)
        
        # Strategy 1: Classic 2s10s inversion
        signals['Classic'] = 0
        if '2s10s' in self.spreads.columns:
            signals.loc[self.spreads['2s10s'] < 0, 'Classic'] = 1
            signals.loc[self.spreads['2s10s'] > 0.5, 'Classic'] = -1
        
        # Strategy 2: 3m10y signal
        signals['3m10y'] = 0
        if '3m10y' in self.spreads.columns:
            signals.loc[self.spreads['3m10y'] < 0, '3m10y'] = 1
            signals.loc[self.spreads['3m10y'] > 0.5, '3m10y'] = -1
        
        # Strategy 3: Momentum-enhanced
        signals['Momentum'] = 0
        if '2s10s' in self.spreads.columns:
            spread_change = self.spreads['2s10s'].diff(20)
            signals.loc[(self.spreads['2s10s'] < 0) & (spread_change < 0), 'Momentum'] = 1
            signals.loc[(self.spreads['2s10s'] > 0) & (spread_change > 0), 'Momentum'] = -1
        
        # Strategy 4: Composite
        signals['Composite'] = (signals['Classic'] + signals['3m10y'] + signals['Momentum']) / 3
        signals['Composite'] = signals['Composite'].clip(-1, 1)
        
        self.signals = signals
        return self
    
    def backtest(self, etf_prices, signal_col='Composite', transaction_cost=0.001):
        """Backtest the strategy"""
        
        if self.signals is None:
            self.generate_signals()
        
        if signal_col not in self.signals.columns:
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
            return self.get_empty_metrics(cum_benchmark)
        
        ann_factor = np.sqrt(252)
        
        metrics = {
            'Total Return Strategy': float(cum_strategy.iloc[-1] - 1) if len(cum_strategy) > 0 else 0,
            'Total Return Benchmark': float(cum_benchmark.iloc[-1] - 1) if len(cum_benchmark) > 0 else 0,
            'Sharpe Ratio': (strategy_returns.mean() / strategy_returns.std()) * ann_factor if strategy_returns.std() > 0 else 0,
            'Benchmark Sharpe': (benchmark_returns.mean() / benchmark_returns.std()) * ann_factor if benchmark_returns.std() > 0 else 0,
            'Max Drawdown': self.calculate_max_drawdown(cum_strategy),
            'Benchmark Drawdown': self.calculate_max_drawdown(cum_benchmark),
            'Win Rate': float((strategy_returns > 0).sum() / len(strategy_returns[strategy_returns != 0])) if len(strategy_returns[strategy_returns != 0]) > 0 else 0,
            'Profit Factor': 0,
            'Calmar Ratio': 0
        }
        
        metrics['Excess Return'] = metrics['Total Return Strategy'] - metrics['Total Return Benchmark']
        
        gross_profits = strategy_returns[strategy_returns > 0].sum()
        gross_losses = abs(strategy_returns[strategy_returns < 0].sum())
        metrics['Profit Factor'] = float(gross_profits / gross_losses) if gross_losses > 0 else 0
        
        if metrics['Max Drawdown'] != 0 and len(cum_strategy) > 0:
            years = len(cum_strategy) / 252
            if years > 0 and cum_strategy.iloc[-1] > 0:
                annual_return = (cum_strategy.iloc[-1] ** (1/years) - 1)
                metrics['Calmar Ratio'] = float(annual_return / abs(metrics['Max Drawdown']))
        
        return metrics
    
    @staticmethod
    def calculate_max_drawdown(cumulative_returns):
        """Calculate maximum drawdown"""
        if len(cumulative_returns) == 0:
            return 0
        
        cum_clean = cumulative_returns.fillna(1)
        rolling_max = cum_clean.expanding().max()
        drawdown = (cum_clean - rolling_max) / rolling_max
        return float(drawdown.min())
    
    @staticmethod
    def get_empty_metrics(cum_benchmark):
        """Return empty metrics when no data available"""
        return {
            'Total Return Strategy': 0,
            'Total Return Benchmark': float(cum_benchmark.iloc[-1] - 1) if len(cum_benchmark) > 0 else 0,
            'Excess Return': 0,
            'Sharpe Ratio': 0,
            'Benchmark Sharpe': 0,
            'Max Drawdown': 0,
            'Benchmark Drawdown': 0,
            'Win Rate': 0,
            'Profit Factor': 0,
            'Calmar Ratio': 0
        }

def get_recession_probability(spreads_df):
    """Calculate recession probability based on yield curve"""
    if '2s10s' not in spreads_df.columns or spreads_df.empty:
        return 0.5
    
    current_spread = spreads_df['2s10s'].iloc[-1]
    if pd.isna(current_spread):
        return 0.5
    
    # Logistic function based on historical relationship
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

# Title
st.title("📈 Bond Yield Curve Analysis Platform")
st.markdown("*Institutional-Grade Fixed Income Analytics*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Date range selection
    default_end = datetime.now()
    default_start = default_end - timedelta(days=365*2)
    
    start_date = st.date_input("Start Date", default_start, max_value=default_end)
    end_date = st.date_input("End Date", default_end, max_value=default_end)
    
    data_source = st.selectbox("Data Source", ["FRED (Recommended)", "Market Data (Alternative)"])
    
    st.markdown("---")
    st.header("📊 Strategy Parameters")
    
    selected_etf = st.selectbox("Select ETF for Backtest", list(BOND_ETFS.keys()))
    selected_strategy = st.selectbox("Trading Strategy", 
                                     ['Composite', 'Classic', '3m10y', 'Momentum'])
    transaction_cost = st.slider("Transaction Cost (%)", 0.0, 0.5, 0.1, 0.01) / 100
    
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Main content
if run_analysis:
    with st.spinner("Fetching market data..."):
        # Fetch data based on selected source
        if data_source == "FRED (Recommended)":
            df = fetch_yield_data_combined(start_date, end_date)
        else:
            df = fetch_yield_data_alternative(start_date, end_date)
        
        if df.empty:
            st.error("""
            Failed to fetch yield data. Possible solutions:
            1. Try selecting "Market Data (Alternative)" in the sidebar
            2. Reduce the date range
            3. Check your internet connection
            4. The data source might be temporarily unavailable
            """)
            st.stop()
        
        # Display data info
        st.info(f"✅ Data loaded successfully. Available maturities: {', '.join(df.columns)}")
        
        # Calculate spreads and forwards
        spreads = calculate_spreads(df)
        forwards = calculate_forward_rates(df)
        
        # Fetch ETF data for backtest
        etf_prices = fetch_etf_data(selected_etf, start_date, end_date)
        
        # Fetch VIX data
        vix_data = fetch_vix_data(start_date, end_date)
        
        # Run trading strategy
        strategy = YieldCurveTradingStrategy(df, spreads)
        strategy.generate_signals()
        backtest_results = strategy.backtest(etf_prices, selected_strategy, transaction_cost)
        
        # Calculate recession probability
        recession_prob = get_recession_probability(spreads)
        
        # Display KPIs
        st.subheader("📊 Market Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            current_10y = df['10Y'].iloc[-1] if '10Y' in df.columns and not df.empty else np.nan
            st.metric("10Y Yield", f"{current_10y:.2f}%" if not np.isnan(current_10y) else "N/A")
        
        with col2:
            current_2y = df['2Y'].iloc[-1] if '2Y' in df.columns and not df.empty else np.nan
            st.metric("2Y Yield", f"{current_2y:.2f}%" if not np.isnan(current_2y) else "N/A")
        
        with col3:
            current_spread = spreads['2s10s'].iloc[-1] if '2s10s' in spreads.columns and not spreads.empty else np.nan
            st.metric("2s10s Spread", f"{current_spread:.2f}%" if not np.isnan(current_spread) else "N/A")
        
        with col4:
            st.metric("Recession Probability", f"{recession_prob:.1%}")
        
        with col5:
            current_vix = vix_data.iloc[-1] if not vix_data.empty else np.nan
            st.metric("VIX", f"{current_vix:.2f}" if not np.isnan(current_vix) else "N/A")
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Yield Curve", 
            "📊 Spread Analysis", 
            "🎯 Trading Strategy",
            "📋 Data Explorer"
        ])
        
        # Tab 1: Yield Curve
        with tab1:
            st.subheader("Yield Curve Visualization")
            
            fig_yield = plot_yield_curve(df, end_date)
            if fig_yield:
                st.plotly_chart(fig_yield, use_container_width=True)
            else:
                st.warning("No yield curve data available for the selected date")
            
            # Additional curve statistics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Curve Statistics")
                if not df.empty:
                    latest = df.iloc[-1]
                    st.metric("Curve Steepness (10Y-2Y)", 
                             f"{latest.get('10Y', 0) - latest.get('2Y', 0):.2f}%")
                    st.metric("Short End (3M)", f"{latest.get('3M', 0):.2f}%")
                    st.metric("Long End (30Y)", f"{latest.get('30Y', 0):.2f}%")
            
            with col2:
                st.markdown("### Recent Changes")
                if len(df) > 5:
                    changes = df.diff().iloc[-1]
                    st.metric("10Y 1-Day Change", f"{changes.get('10Y', 0):+.2f}%")
                    st.metric("2Y 1-Day Change", f"{changes.get('2Y', 0):+.2f}%")
        
        # Tab 2: Spread Analysis
        with tab2:
            st.subheader("Yield Spread Analysis")
            
            if not spreads.empty:
                fig_spreads = plot_spreads(spreads)
                if fig_spreads:
                    st.plotly_chart(fig_spreads, use_container_width=True)
                
                st.markdown("### Current Spreads")
                current_spreads = spreads.iloc[-1]
                cols = st.columns(len(current_spreads))
                for idx, (spread_name, value) in enumerate(current_spreads.items()):
                    if pd.notna(value):
                        color = "🟢" if value > 0 else "🔴" if value < -0.5 else "🟡"
                        cols[idx].metric(f"{color} {spread_name.upper()}", f"{value:.2f}%")
                
                # Interpretation
                st.markdown("### Market Interpretation")
                if '2s10s' in spreads.columns:
                    latest_2s10s = spreads['2s10s'].iloc[-1]
                    if latest_2s10s < 0:
                        st.warning("⚠️ **INVERTED YIELD CURVE** - Historically signals recession within 6-18 months. Consider defensive positioning.")
                    elif latest_2s10s < 0.5:
                        st.info("📊 **Flattening Curve** - Monitor for potential inversion. Economic expansion may be late-cycle.")
                    else:
                        st.success("✅ **Normal/Steep Curve** - Economic expansion likely continuing. Risk-on environment.")
            else:
                st.info("No spread data available")
        
        # Tab 3: Trading Strategy
        with tab3:
            st.subheader(f"Trading Strategy: {selected_strategy}")
            
            if backtest_results and backtest_results.results:
                results = backtest_results.results
                metrics = results['metrics']
                
                # Performance metrics
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
                    st.metric("Calmar Ratio", f"{metrics['Calmar Ratio']:.2f}")
                
                # Cumulative returns chart
                st.markdown("### Cumulative Returns")
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
                    title='Strategy vs Benchmark',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Return',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig_returns, use_container_width=True)
                
                # Recent signals
                with st.expander("Recent Trading Signals"):
                    recent_signals = pd.DataFrame({
                        'Date': results['signals'].tail(20).index.strftime('%Y-%m-%d'),
                        'Signal': results['signals'].tail(20).values,
                        'Action': ['BUY' if x > 0 else 'SELL' if x < 0 else 'HOLD' 
                                  for x in results['signals'].tail(20).values]
                    })
                    st.dataframe(recent_signals, use_container_width=True, hide_index=True)
            else:
                st.warning("Backtest results not available. Try a different date range or ETF.")
        
        # Tab 4: Data Explorer
        with tab4:
            st.subheader("Historical Yield Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Summary Statistics")
                st.dataframe(df.describe().round(2), use_container_width=True)
            
            with col2:
                st.markdown("### Correlation Matrix")
                if len(df.columns) > 1:
                    st.dataframe(df.corr().round(2), use_container_width=True)
            
            st.markdown("### Raw Data")
            st.dataframe(df.tail(50), use_container_width=True)
            
            # Download buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                csv_yields = df.to_csv()
                st.download_button("📥 Download Yield Data", csv_yields, "yield_data.csv", "text/csv")
            
            with col2:
                if not spreads.empty:
                    csv_spreads = spreads.to_csv()
                    st.download_button("📥 Download Spreads", csv_spreads, "spreads.csv", "text/csv")
            
            with col3:
                if backtest_results and backtest_results.results:
                    results_df = pd.DataFrame({
                        'Date': backtest_results.results['cumulative_strategy'].index,
                        'Strategy_Returns': backtest_results.results['cumulative_strategy'].values,
                        'Benchmark_Returns': backtest_results.results['cumulative_bh'].values
                    })
                    csv_results = results_df.to_csv()
                    st.download_button("📥 Download Backtest Results", csv_results, "backtest_results.csv", "text/csv")
    
else:
    st.info("👈 Configure your analysis parameters in the sidebar and click 'Run Analysis' to begin")
    
    with st.expander("📖 Platform Features", expanded=True):
        st.markdown("""
        ### Features Included:
        
        **1. Yield Curve Analysis**
        - Interactive curve visualization
        - Historical comparison (1 year ago)
        - Curve statistics and recent changes
        
        **2. Spread Analysis**
        - 2s10s (2-year vs 10-year) - Key recession indicator
        - 3m10y (3-month vs 10-year)
        - 5s30s (5-year vs 30-year)
        - Market interpretation and warnings
        
        **3. Trading Strategies**
        - Multiple strategy types (Classic, Momentum, Composite)
        - Full backtesting with transaction costs
        - Performance metrics (Sharpe, Drawdown, Win Rate)
        
        **4. Data Export**
        - Download yield data
        - Export spreads
        - Save backtest results
        """)

# Footer
st.markdown("---")
st.markdown(f"*Data as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data source: FRED via yfinance*")
