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
# CONFIGURATION
# =============================================================================

# Treasury symbols
TREASURY_SYMBOLS = {
    '3M': '^IRX',   # 13-week Treasury Bill
    '2Y': '^GB2Y',  # 2-year Treasury Note
    '5Y': '^FVX',   # 5-year Treasury Note
    '10Y': '^TNX',  # 10-year Treasury Note
    '30Y': '^TYX'   # 30-year Treasury Bond
}

# Bond ETFs for backtesting
BOND_ETFS = {
    'TLT': '20+ Year Treasury Bond ETF',
    'IEF': '7-10 Year Treasury Bond ETF',
    'SHY': '1-3 Year Treasury Bond ETF',
    'BND': 'Total Bond Market ETF'
}

# =============================================================================
# DATA FETCHING FUNCTIONS
# =============================================================================

def fetch_yield_data(start_date=None, end_date=None):
    """
    Fetch yield curve data using yfinance
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    if end_date is None:
        end_date = datetime.now()
    
    yields_data = {}
    
    for maturity, symbol in TREASURY_SYMBOLS.items():
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not data.empty and 'Adj Close' in data.columns:
                yields_data[maturity] = data['Adj Close']
                
                # Convert IRX (3M) from discount rate to yield
                if maturity == '3M':
                    yields_data[maturity] = yields_data[maturity].apply(lambda x: (100 - x) * 4 if pd.notna(x) else x)
            else:
                yields_data[maturity] = pd.Series(dtype='float64')
        except Exception as e:
            print(f"Could not fetch data for {maturity}: {e}")
            yields_data[maturity] = pd.Series(dtype='float64')
    
    df = pd.DataFrame(yields_data)
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

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

# =============================================================================
# YIELD CURVE ANALYSIS FUNCTIONS
# =============================================================================

def plot_yield_curve(df, selected_date):
    """
    Create interactive yield curve plot
    """
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
    
    fig.add_trace(go.Scatter(
        x=valid_maturities,
        y=valid_yields,
        name=closest_date.strftime('%Y-%m-%d'),
        line=dict(color='#2c5f8a', width=3),
        mode='lines+markers',
        marker=dict(size=8)
    ))
    
    # Add historical comparison (1 month ago)
    month_ago = closest_date - pd.DateOffset(months=1)
    month_ago = available_dates[available_dates <= month_ago].max()
    
    if pd.notna(month_ago) and month_ago in df.index:
        historical_yields = df.loc[month_ago]
        valid_hist_maturities = []
        valid_hist_yields = []
        
        for mat, year in maturities.items():
            if mat in historical_yields.index and pd.notna(historical_yields[mat]):
                valid_hist_maturities.append(year)
                valid_hist_yields.append(historical_yields[mat])
        
        fig.add_trace(go.Scatter(
            x=valid_hist_maturities,
            y=valid_hist_yields,
            name='1 Month Ago',
            line=dict(color='#c17f3a', width=2, dash='dash'),
            mode='lines+markers',
            marker=dict(size=6)
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
    
    return spreads

def plot_spreads(spreads):
    """
    Create spread visualization
    """
    fig = go.Figure()
    
    colors = {'2s10s': '#2c5f8a', '3m10y': '#c17f3a', '5s30s': '#4a7c59'}
    
    for col in spreads.columns:
        if not spreads[col].isna().all():
            fig.add_trace(go.Scatter(
                x=spreads.index,
                y=spreads[col],
                name=col,
                line=dict(color=colors.get(col, '#666'), width=2),
                fill='tozeroy',
                fillcolor=f'rgba({",".join(map(str, colors.get(col, "#666").lstrip("#")[:6]))},0.1)' if col in colors else None
            ))
    
    fig.add_hline(y=0, line_color='red', line_dash='dash', line_width=1.5)
    fig.add_hline(y=-0.5, line_color='orange', line_dash='dot', line_width=1, 
                  annotation_text="Inversion Warning", annotation_position="bottom right")
    
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
    
    for short_term, long_term in [('3M', '2Y'), ('2Y', '5Y'), ('5Y', '10Y'), ('10Y', '30Y')]:
        if short_term in df.columns and long_term in df.columns:
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
            signals.loc[self.spreads['2s10s'] < 0, 'Classic'] = 1  # Buy on inversion
            signals.loc[self.spreads['2s10s'] > 0.5, 'Classic'] = -1  # Sell when steep
        
        # Strategy 2: 3m10y signal (more reliable recession indicator)
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
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_forward_rates(forwards_df):
    """Plot forward rates"""
    if forwards_df.empty:
        return None
    
    fig = go.Figure()
    
    for col in forwards_df.columns:
        fig.add_trace(go.Scatter(
            x=forwards_df.index,
            y=forwards_df[col],
            name=col,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title='Implied Forward Rates',
        xaxis_title='Date',
        yaxis_title='Forward Rate (%)',
        template='plotly_white',
        hovermode='x unified',
        height=500
    )
    
    return fig

def plot_backtest_results(results):
    """Plot backtest results"""
    if results is None:
        return None
    
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=('Cumulative Returns', 'Signal Distribution'),
                        vertical_spacing=0.15,
                        row_heights=[0.6, 0.4])
    
    # Cumulative returns
    fig.add_trace(go.Scatter(
        x=results['cumulative_strategy'].index,
        y=results['cumulative_strategy'].values,
        name=f'Strategy ({results["signal_col"]})',
        line=dict(color='#2c5f8a', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=results['cumulative_bh'].index,
        y=results['cumulative_bh'].values,
        name='Buy & Hold',
        line=dict(color='#c17f3a', width=2, dash='dash')
    ), row=1, col=1)
    
    # Signal distribution
    signal_counts = results['signals'].value_counts()
    colors = ['#4a7c59' if x > 0 else '#c05656' if x < 0 else '#999' for x in signal_counts.index]
    
    fig.add_trace(go.Bar(
        x=signal_counts.index.astype(str),
        y=signal_counts.values,
        name='Signal Distribution',
        marker_color=colors
    ), row=2, col=1)
    
    fig.update_layout(
        title='Strategy Backtest Results',
        template='plotly_white',
        height=700,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_xaxes(title_text="Signal", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    
    return fig

# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(
    page_title="Bond Yield Curve Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #eef2f7 0%, #f7f9fc 100%);
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .main-title {
        background: linear-gradient(135deg, #1a2a3a 0%, #2c4a6e 100%);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div class="main-title">
    <h1>📈 Bond Yield Curve Analysis Platform</h1>
    <p>Institutional-Grade Fixed Income Analytics | Yield Curves | Trading Strategies | Risk Metrics</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Date range selection
    default_end = datetime.now()
    default_start = default_end - timedelta(days=365*2)
    
    start_date = st.date_input("Start Date", default_start, max_value=default_end)
    end_date = st.date_input("End Date", default_end, max_value=default_end)
    
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
        # Fetch data
        df = fetch_yield_data(start_date, end_date)
        
        if df.empty:
            st.error("Failed to fetch yield data. Please check your date range and try again.")
            st.stop()
        
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
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Yield Curve", 
            "📊 Spread Analysis", 
            "🎯 Trading Strategy",
            "📉 Forward Rates",
            "📋 Data Explorer"
        ])
        
        # Tab 1: Yield Curve
        with tab1:
            st.subheader("Yield Curve Visualization")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                fig_yield = plot_yield_curve(df, end_date)
                if fig_yield:
                    st.plotly_chart(fig_yield, use_container_width=True)
                else:
                    st.warning("No yield curve data available for the selected date")
            
            with col2:
                st.markdown("### Curve Statistics")
                if not df.empty:
                    latest = df.iloc[-1]
                    st.markdown(f"""
                    - **Steepness (10Y-2Y):** {latest.get('10Y', 0) - latest.get('2Y', 0):.2f}%
                    - **Level (10Y):** {latest.get('10Y', 0):.2f}%
                    - **Short End (3M):** {latest.get('3M', 0):.2f}%
                    - **Long End (30Y):** {latest.get('30Y', 0):.2f}%
                    """)
        
        # Tab 2: Spread Analysis
        with tab2:
            st.subheader("Yield Spread Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if not spreads.empty:
                    fig_spreads = plot_spreads(spreads)
                    st.plotly_chart(fig_spreads, use_container_width=True)
                else:
                    st.info("No spread data available")
            
            with col2:
                st.markdown("### Current Spreads")
                if not spreads.empty:
                    current_spreads = spreads.iloc[-1]
                    for spread_name, value in current_spreads.items():
                        if pd.notna(value):
                            color = "🟢" if value > 0 else "🔴" if value < -0.5 else "🟡"
                            st.metric(f"{color} {spread_name.upper()} Spread", f"{value:.2f}%")
                
                st.markdown("### Interpretation")
                if '2s10s' in spreads.columns and not spreads.empty:
                    latest_2s10s = spreads['2s10s'].iloc[-1]
                    if latest_2s10s < 0:
                        st.warning("⚠️ Yield curve is INVERTED! Historically signals recession within 6-18 months.")
                    elif latest_2s10s < 0.5:
                        st.info("📊 Yield curve is flattening. Monitor for potential inversion.")
                    else:
                        st.success("✅ Yield curve is normal/steep. Economic expansion likely continuing.")
        
        # Tab 3: Trading Strategy
        with tab3:
            st.subheader(f"Trading Strategy: {selected_strategy}")
            
            if backtest_results and backtest_results.results:
                results = backtest_results.results
                metrics = results['metrics']
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Return (Strategy)", f"{metrics['Total Return Strategy']:.2%}")
                    st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
                
                with col2:
                    st.metric("Total Return (Benchmark)", f"{metrics['Total Return Benchmark']:.2%}")
                    st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
                
                with col3:
                    st.metric("Excess Return", f"{metrics['Excess Return']:.2%}")
                    st.metric("Win Rate", f"{metrics['Win Rate']:.2%}")
                
                with col4:
                    st.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")
                    st.metric("Calmar Ratio", f"{metrics['Calmar Ratio']:.2f}")
                
                # Plot backtest results
                fig_backtest = plot_backtest_results(results)
                if fig_backtest:
                    st.plotly_chart(fig_backtest, use_container_width=True)
                
                # Signal distribution
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
        
        # Tab 4: Forward Rates
        with tab4:
            st.subheader("Implied Forward Rate Analysis")
            
            if not forwards.empty:
                fig_forwards = plot_forward_rates(forwards)
                if fig_forwards:
                    st.plotly_chart(fig_forwards, use_container_width=True)
                
                # Latest forward rates
                st.markdown("### Current Forward Rates")
                latest_forwards = forwards.iloc[-1]
                cols = st.columns(len(latest_forwards))
                for idx, (period, rate) in enumerate(latest_forwards.items()):
                    if pd.notna(rate):
                        cols[idx].metric(period, f"{rate:.2f}%")
            else:
                st.info("Forward rate data not available")
        
        # Tab 5: Data Explorer
        with tab5:
            st.subheader("Historical Yield Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Summary Statistics")
                st.dataframe(df.describe().round(2), use_container_width=True)
            
            with col2:
                st.markdown("### Correlation Matrix")
                st.dataframe(df.corr().round(2), use_container_width=True)
            
            st.markdown("### Raw Data")
            st.dataframe(df.tail(50), use_container_width=True)
            
            # Download buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                csv_yields = df.to_csv()
                st.download_button("📥 Download Yield Data (CSV)", csv_yields, "yield_data.csv", "text/csv")
            
            with col2:
                if not spreads.empty:
                    csv_spreads = spreads.to_csv()
                    st.download_button("📥 Download Spreads (CSV)", csv_spreads, "spreads.csv", "text/csv")
            
            with col3:
                if backtest_results and backtest_results.results:
                    results_df = pd.DataFrame({
                        'Date': backtest_results.results['cumulative_strategy'].index,
                        'Strategy_Returns': backtest_results.results['cumulative_strategy'].values,
                        'Benchmark_Returns': backtest_results.results['cumulative_bh'].values
                    })
                    csv_results = results_df.to_csv()
                    st.download_button("📥 Download Backtest Results (CSV)", csv_results, "backtest_results.csv", "text/csv")
    
else:
    # Initial state - show instructions
    st.info("👈 Configure your analysis parameters in the sidebar and click 'Run Analysis' to begin")
    
    st.markdown("""
    ### 📖 Platform Features
    
    This institutional-grade fixed income analytics platform provides:
    
    #### 1. Yield Curve Analysis
    - Interactive yield curve visualization
    - Historical curve comparison
    - Steepness, level, and curvature metrics
    
    #### 2. Spread Analysis  
    - 2s10s (2-year vs 10-year) - Key recession indicator
    - 3m10y (3-month vs 10-year) - Alternative inversion measure
    - 5s30s (5-year vs 30-year) - Long-term steepness
    
    #### 3. Trading Strategies
    - Classic inversion-based signals
    - Momentum-enhanced strategies
    - Composite ensemble signals
    - Full backtesting with transaction costs
    
    #### 4. Forward Rates
    - Implied forward rate curves
    - Market expectations for future rates
    - Term structure analysis
    
    ### 🎯 Use Cases
    
    - **Portfolio Management**: Generate tactical trading signals
    - **Risk Management**: Monitor recession indicators
    - **Research**: Analyze historical yield curve behavior
    - **Strategy Development**: Backtest quantitative strategies
    
    ### ⚠️ Risk Warning
    
    Past performance does not guarantee future results. This platform is for educational and research purposes only.
    """)

# Footer
st.markdown("---")
st.markdown(f"*Data as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Built with yfinance, Plotly, and Streamlit*")
