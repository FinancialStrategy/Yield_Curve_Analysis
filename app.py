import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

def safe_get_adj_close(data, ticker=None):
    """Safely extract Adj Close from yfinance data regardless of structure"""
    if data is None or data.empty:
        return None
    
    if 'Adj Close' in data.columns:
        if ticker and isinstance(data['Adj Close'], pd.DataFrame) and ticker in data['Adj Close'].columns:
            return data['Adj Close'][ticker]
        elif not isinstance(data['Adj Close'], pd.DataFrame):
            return data['Adj Close']
        else:
            return data['Adj Close'].iloc[:, 0] if len(data['Adj Close'].columns) > 0 else None
    
    if 'Close' in data.columns:
        if isinstance(data['Close'], pd.DataFrame):
            return data['Close'].iloc[:, 0] if len(data['Close'].columns) > 0 else None
        return data['Close']
    
    if len(data.columns) == 1:
        return data.iloc[:, 0]
    
    return None

class YieldCurveStrategy:
    """Advanced yield curve trading strategy with multiple signals"""
    
    def __init__(self, start_date='2010-01-01', end_date=None):
        self.start_date = start_date
        # Set end date to yesterday if not specified
        if end_date is None:
            self.end_date = (date.today() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            self.end_date = end_date
        self.data = None
        self.signals = None
        self.results = None
        
    def fetch_data(self):
        """Fetch treasury yields and bond ETF data"""
        
        # Treasury yields
        tickers = {
            '^IRX': '3M',
            '^FVX': '5Y', 
            '^TNX': '10Y',
            '^TYX': '30Y'
        }
        
        yields_data = {}
        
        for ticker, name in tickers.items():
            try:
                df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                price_series = safe_get_adj_close(df, ticker)
                
                if price_series is not None:
                    price_series.index = pd.to_datetime(price_series.index)
                    price_series.index = price_series.index.tz_localize(None)
                    yields_data[name] = price_series
            except Exception as e:
                st.warning(f"Could not fetch data for {ticker}: {str(e)}")
        
        if not yields_data:
            raise Exception("No yield data could be fetched")
        
        yields = pd.DataFrame(yields_data)
        
        # Bond ETFs
        etfs = ['TLT', 'IEF', 'SHY', 'BND']
        bond_data = {}
        
        for ticker in etfs:
            try:
                df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                price_series = safe_get_adj_close(df, ticker)
                
                if price_series is not None:
                    price_series.index = pd.to_datetime(price_series.index)
                    price_series.index = price_series.index.tz_localize(None)
                    bond_data[ticker] = price_series
            except Exception as e:
                st.warning(f"Error fetching {ticker}: {str(e)}")
        
        if not bond_data:
            raise Exception("No bond ETF data could be fetched")
        
        bond_prices = pd.DataFrame(bond_data)
        
        # VIX for volatility regime
        try:
            vix_df = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False)
            vix_series = safe_get_adj_close(vix_df, '^VIX')
            if vix_series is not None:
                vix_series.index = pd.to_datetime(vix_series.index)
                vix_series.index = vix_series.index.tz_localize(None)
                common_idx = bond_prices.index.intersection(vix_series.index)
                bond_prices['VIX'] = vix_series.reindex(common_idx)
        except Exception as e:
            print(f"VIX data unavailable: {e}")
            
        self.data = {'yields': yields, 'bond_prices': bond_prices}
        return self
    
    def calculate_spreads(self):
        """Calculate various yield spreads"""
        yields = self.data['yields']
        
        if yields.empty:
            raise Exception("No yield data available for spread calculation")
        
        spreads = pd.DataFrame(index=yields.index)
        
        if '10Y' in yields.columns and '3M' in yields.columns:
            common_idx = yields['10Y'].index.intersection(yields['3M'].index)
            spreads['10Y-3M'] = yields['10Y'].loc[common_idx] - yields['3M'].loc[common_idx]
        
        if '10Y' in yields.columns and '5Y' in yields.columns:
            common_idx = yields['10Y'].index.intersection(yields['5Y'].index)
            spreads['10Y-5Y'] = yields['10Y'].loc[common_idx] - yields['5Y'].loc[common_idx]
        
        if '30Y' in yields.columns and '10Y' in yields.columns:
            common_idx = yields['30Y'].index.intersection(yields['10Y'].index)
            spreads['30Y-10Y'] = yields['30Y'].loc[common_idx] - yields['10Y'].loc[common_idx]
        
        if '10Y-3M' in spreads.columns:
            spreads['Slope'] = spreads['10Y-3M'].copy()
            rolling_mean = spreads['10Y-3M'].rolling(252, min_periods=50).mean()
            rolling_std = spreads['10Y-3M'].rolling(252, min_periods=50).std()
            spreads['10Y-3M_Zscore'] = (spreads['10Y-3M'] - rolling_mean) / rolling_std
        
        self.data['spreads'] = spreads
        return self
    
    def generate_signals(self, strategy='composite'):
        """Generate trading signals using multiple approaches"""
        
        spreads = self.data['spreads']
        
        if spreads.empty or '10Y-3M' not in spreads.columns:
            st.error("Required spread data not available")
            return self
        
        signals = pd.DataFrame(index=spreads.index)
        
        # Strategy 1: Classic inversion signal
        signals['Classic'] = 0
        signals.loc[spreads['10Y-3M'] < 0, 'Classic'] = 1
        signals.loc[spreads['10Y-3M'] > 0.5, 'Classic'] = -1
        
        # Strategy 2: Momentum-enhanced signal
        signals['Momentum'] = 0
        spread_change = spreads['10Y-3M'].diff(20)
        signals.loc[(spreads['10Y-3M'] < 0) & (spread_change < 0), 'Momentum'] = 1
        signals.loc[(spreads['10Y-3M'] > 0) & (spread_change > 0), 'Momentum'] = -1
        
        # Strategy 3: Z-score based mean reversion
        signals['ZScore'] = 0
        if '10Y-3M_Zscore' in spreads.columns:
            zscore = spreads['10Y-3M_Zscore']
            signals.loc[zscore < -1.5, 'ZScore'] = 1
            signals.loc[zscore > 1.5, 'ZScore'] = -1
        
        # Strategy 4: Composite
        signals['Composite'] = (signals['Classic'] + signals['Momentum'] + signals['ZScore']) / 3
        signals['Composite'] = signals['Composite'].clip(-1, 1)
        
        # Strategy 5: Adaptive threshold
        if 'VIX' in self.data['bond_prices'].columns:
            vix_aligned = self.data['bond_prices']['VIX'].reindex(signals.index).fillna(method='ffill')
            vol = vix_aligned / 20
            vol = vol.clip(0, 1)
            adaptive_threshold = 0.5 + vol
            signals['Adaptive'] = 0
            signals.loc[spreads['10Y-3M'] < -adaptive_threshold, 'Adaptive'] = 1
            signals.loc[spreads['10Y-3M'] > adaptive_threshold, 'Adaptive'] = -1
        else:
            signals['Adaptive'] = signals['Classic']
        
        self.signals = signals
        return self
    
    def backtest(self, etf='TLT', signal_col='Composite', transaction_cost=0.001):
        """Backtest the strategy"""
        
        bond_prices = self.data['bond_prices']
        
        if etf not in bond_prices.columns:
            raise Exception(f"ETF {etf} not available in data")
        
        signals = self.signals[signal_col].copy()
        
        common_idx = bond_prices.index.intersection(signals.index)
        if len(common_idx) == 0:
            raise Exception("No overlapping dates between signals and bond data")
        
        signals_aligned = signals.reindex(common_idx)
        prices_aligned = bond_prices[etf].reindex(common_idx)
        
        returns = prices_aligned.pct_change()
        strategy_returns = signals_aligned.shift(1) * returns
        
        signal_changes = signals_aligned.diff().abs()
        transaction_costs = signal_changes * transaction_cost
        strategy_returns = strategy_returns - transaction_costs
        
        strategy_returns = strategy_returns.fillna(0)
        
        cumulative_returns = (1 + strategy_returns).cumprod()
        buy_hold_returns = (1 + returns.fillna(0)).cumprod()
        
        metrics = self.calculate_metrics(strategy_returns, returns.fillna(0), cumulative_returns, buy_hold_returns)
        
        self.results = {
            'strategy_returns': strategy_returns,
            'cumulative_strategy': cumulative_returns,
            'cumulative_bh': buy_hold_returns,
            'signals': signals_aligned,
            'metrics': metrics,
            'etf': etf,
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
                'Benchmark Sharpe': 0,
                'Max Drawdown': 0,
                'Benchmark Drawdown': 0,
                'Win Rate': 0,
                'Profit Factor': 0,
                'Calmar Ratio': 0
            }
        
        ann_factor = np.sqrt(252)
        
        strategy_std = strategy_returns.std()
        benchmark_std = benchmark_returns.std()
        
        metrics = {
            'Total Return Strategy': float(cum_strategy.iloc[-1] - 1) if len(cum_strategy) > 0 else 0,
            'Total Return Benchmark': float(cum_benchmark.iloc[-1] - 1) if len(cum_benchmark) > 0 else 0,
            'Excess Return': 0,
            'Sharpe Ratio': (strategy_returns.mean() / strategy_std) * ann_factor if strategy_std > 0 else 0,
            'Benchmark Sharpe': (benchmark_returns.mean() / benchmark_std) * ann_factor if benchmark_std > 0 else 0,
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
    
    def get_current_metrics(self):
        """Get current market metrics for KPI display"""
        if self.data is None:
            return {}
        
        yields = self.data['yields']
        spreads = self.data['spreads']
        bond_prices = self.data['bond_prices']
        
        metrics = {}
        
        # Current yields
        if '10Y' in yields.columns and len(yields) > 0:
            metrics['current_10y'] = yields['10Y'].iloc[-1]
        else:
            metrics['current_10y'] = np.nan
            
        if '2Y' in yields.columns and len(yields) > 0:
            metrics['current_2y'] = yields['2Y'].iloc[-1]
        else:
            metrics['current_2y'] = np.nan
            
        if '3M' in yields.columns and len(yields) > 0:
            metrics['current_3m'] = yields['3M'].iloc[-1]
        else:
            metrics['current_3m'] = np.nan
        
        # Current spread
        if spreads is not None and '10Y-3M' in spreads.columns and len(spreads) > 0:
            metrics['current_spread'] = spreads['10Y-3M'].iloc[-1]
        else:
            metrics['current_spread'] = np.nan
        
        # Current VIX
        if 'VIX' in bond_prices.columns and len(bond_prices) > 0:
            metrics['current_vix'] = bond_prices['VIX'].iloc[-1]
        else:
            metrics['current_vix'] = np.nan
        
        # Regime classification
        if 'current_spread' in metrics and not np.isnan(metrics['current_spread']):
            if metrics['current_spread'] < 0:
                metrics['regime'] = "Risk-off / Recession Watch"
                metrics['regime_text'] = "Curve inversion signals defensive positioning"
            elif metrics['current_spread'] < 0.5:
                metrics['regime'] = "Neutral / Late Cycle"
                metrics['regime_text'] = "Curve flattening suggests caution"
            else:
                metrics['regime'] = "Risk-on / Expansion"
                metrics['regime_text'] = "Positive slope supports risk-taking"
        else:
            metrics['regime'] = "Data Loading"
            metrics['regime_text'] = "Please wait for data"
        
        # Recession probability proxy
        if 'current_spread' in metrics and not np.isnan(metrics['current_spread']):
            prob = max(0, min(1, (-metrics['current_spread'] / 2) + 0.3))
            metrics['recession_prob'] = prob
        else:
            metrics['recession_prob'] = 0.5
        
        return metrics
    
    def create_visualizations(self):
        """Create Plotly visualizations"""
        if self.results is None:
            return None
        
        charts = {}
        
        # Chart 1: Cumulative Returns
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=self.results['cumulative_strategy'].index,
            y=self.results['cumulative_strategy'].values,
            mode='lines',
            name=f'Strategy ({self.results["signal_col"]})',
            line=dict(color='#2c5f8a', width=2)
        ))
        fig1.add_trace(go.Scatter(
            x=self.results['cumulative_bh'].index,
            y=self.results['cumulative_bh'].values,
            mode='lines',
            name='Buy & Hold',
            line=dict(color='#c17f3a', width=2, dash='dash')
        ))
        fig1.update_layout(
            title='Cumulative Returns Comparison',
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        charts['returns'] = fig1
        
        # Chart 2: Yield Spread History
        if self.data['spreads'] is not None and '10Y-3M' in self.data['spreads'].columns:
            fig2 = go.Figure()
            spreads_aligned = self.data['spreads']['10Y-3M']
            fig2.add_trace(go.Scatter(
                x=spreads_aligned.index,
                y=spreads_aligned.values,
                mode='lines',
                name='10Y-3M Spread',
                line=dict(color='blue', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,0,255,0.1)'
            ))
            fig2.add_hline(y=0, line_dash="dash", line_color="red", 
                          annotation_text="Inversion Threshold")
            fig2.add_hline(y=0.5, line_dash="dash", line_color="orange",
                          annotation_text="Steep Threshold")
            fig2.update_layout(
                title='10Y-3M Yield Spread History',
                xaxis_title='Date',
                yaxis_title='Spread (%)',
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            charts['spread'] = fig2
        
        return charts

# =============================================================================
# STREAMLIT UI WITH DYNAMIC UPDATES
# =============================================================================

st.set_page_config(page_title="Yield Curve Strategy Backtest", layout="wide")

st.title("📈 Yield Curve Trading Strategy Backtest")
st.markdown("Advanced fixed-income strategy using yield curve inversion signals")

# Initialize session state for tracking
if 'last_run_params' not in st.session_state:
    st.session_state.last_run_params = None
if 'strategy_results' not in st.session_state:
    st.session_state.strategy_results = None

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Strategy Configuration")
    
    # Date selection with automatic end date
    today = date.today()
    default_end = today - pd.Timedelta(days=1)
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date", 
            pd.to_datetime('2015-01-01'),
            help="Beginning of backtest period"
        )
    with col2:
        end_date = st.date_input(
            "End Date", 
            default_end,
            help=f"Defaults to yesterday ({default_end.strftime('%Y-%m-%d')})"
        )
    
    # Ensure end date is not in the future
    if end_date > today:
        st.warning(f"End date adjusted to yesterday ({default_end})")
        end_date = default_end
    
    etf_choice = st.selectbox(
        "Bond ETF", 
        ['TLT', 'IEF', 'SHY', 'BND'], 
        help="TLT: Long-term, IEF: Intermediate, SHY: Short-term"
    )
    
    signal_choice = st.selectbox(
        "Signal Type", 
        ['Composite', 'Classic', 'Momentum', 'ZScore', 'Adaptive']
    )
    
    transaction_cost = st.slider("Transaction Cost (%)", 0.0, 0.5, 0.1, 0.01) / 100
    
    st.markdown("---")
    
    # Run button with clear indication that it triggers a refresh
    run_backtest = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
    
    # Show current parameters
    st.markdown("### 📅 Current Parameters")
    st.info(f"""
    **Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
    **ETF:** {etf_choice}
    **Signal:** {signal_choice}
    """)

# Main content
if run_backtest:
    # Clear cache to force fresh data fetch
    st.cache_data.clear()
    
    with st.spinner("Fetching data and running backtest..."):
        try:
            # Create unique key for this run
            run_key = f"{start_date}_{end_date}_{etf_choice}_{signal_choice}"
            
            # Initialize strategy with current dates
            strategy = YieldCurveStrategy(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            # Fetch data
            status_text = st.empty()
            status_text.text("📡 Fetching yield curve data...")
            strategy.fetch_data()
            
            status_text.text("📐 Calculating spreads and signals...")
            strategy.calculate_spreads()
            strategy.generate_signals()
            
            status_text.text("💹 Running backtest...")
            strategy.backtest(
                etf=etf_choice, 
                signal_col=signal_choice,
                transaction_cost=transaction_cost
            )
            
            status_text.empty()
            
            # Store results in session state
            st.session_state.strategy_results = strategy
            st.session_state.last_run_params = run_key
            
            # Display success
            st.success(f"✅ Backtest completed for period {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")
            st.info("""
            **Troubleshooting tips:**
            1. Check your internet connection
            2. Try a different date range (e.g., 2020-01-01 to 2023-12-31)
            3. The yfinance API might be temporarily rate-limited - wait a few minutes
            4. Try refreshing the page and running again
            """)
            st.stop()

# Display results if available
if st.session_state.strategy_results is not None:
    strategy = st.session_state.strategy_results
    
    # Get current market metrics for KPI display
    current_metrics = strategy.get_current_metrics()
    
    # KPI ROW - These will update when new data is fetched
    st.subheader("📊 Current Market Metrics")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            "📊 Macro Regime", 
            current_metrics.get('regime', 'N/A'),
            delta=current_metrics.get('regime_text', '')[:30]
        )
    
    with col2:
        current_10y = current_metrics.get('current_10y', np.nan)
        st.metric(
            "🏦 10Y Yield", 
            f"{current_10y:.2f}%" if not np.isnan(current_10y) else "N/A",
            help="Current 10-year Treasury yield"
        )
    
    with col3:
        current_3m = current_metrics.get('current_3m', np.nan)
        st.metric(
            "📝 3M Yield", 
            f"{current_3m:.2f}%" if not np.isnan(current_3m) else "N/A",
            help="Current 3-month Treasury yield"
        )
    
    with col4:
        current_spread = current_metrics.get('current_spread', np.nan)
        spread_color = "normal" if current_spread >= 0 else "inverse"
        st.metric(
            "🔄 10Y-3M Spread", 
            f"{current_spread:.2f}%" if not np.isnan(current_spread) else "N/A",
            delta=f"{'Inverted' if current_spread < 0 else 'Normal'}",
            delta_color=spread_color
        )
    
    with col5:
        recession_prob = current_metrics.get('recession_prob', 0.5)
        st.metric(
            "⚠️ Recession Prob", 
            f"{recession_prob:.1%}",
            help="Estimated probability based on spread"
        )
    
    with col6:
        current_vix = current_metrics.get('current_vix', np.nan)
        st.metric(
            "📉 VIX", 
            f"{current_vix:.2f}" if not np.isnan(current_vix) else "N/A",
            help="CBOE Volatility Index"
        )
    
    # Performance Metrics
    st.subheader("📊 Strategy Performance Metrics")
    
    metrics = strategy.results['metrics']
    
    col1, col2, col3, col4 = st.columns(4)
    col5, col6, col7, col8 = st.columns(4)
    
    with col1:
        st.metric("Total Return (Strategy)", f"{metrics['Total Return Strategy']:.2%}")
    with col2:
        st.metric("Total Return (Benchmark)", f"{metrics['Total Return Benchmark']:.2%}")
    with col3:
        st.metric("Excess Return", f"{metrics['Excess Return']:.2%}")
    with col4:
        st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
    
    with col5:
        st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
    with col6:
        st.metric("Win Rate", f"{metrics['Win Rate']:.2%}")
    with col7:
        st.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")
    with col8:
        st.metric("Calmar Ratio", f"{metrics['Calmar Ratio']:.2f}")
    
    # Charts
    st.subheader("📈 Strategy Visualization")
    charts = strategy.create_visualizations()
    
    if charts:
        if 'returns' in charts:
            st.plotly_chart(charts['returns'], use_container_width=True)
        if 'spread' in charts:
            st.plotly_chart(charts['spread'], use_container_width=True)
    
    # Signal Statistics
    st.subheader("🎯 Signal Statistics")
    signals = strategy.results['signals']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trading Days", len(signals))
    with col2:
        buy_signals = (signals > 0).sum()
        st.metric("Buy Signals", buy_signals, delta=f"{buy_signals/len(signals)*100:.1f}%")
    with col3:
        sell_signals = (signals < 0).sum()
        st.metric("Sell Signals", sell_signals, delta=f"{sell_signals/len(signals)*100:.1f}%")
    with col4:
        neutral = (signals == 0).sum()
        st.metric("Neutral", neutral, delta=f"{neutral/len(signals)*100:.1f}%")
    
    # Recent signals
    with st.expander("View Recent Signals"):
        recent_signals = pd.DataFrame({
            'Date': signals.tail(20).index.strftime('%Y-%m-%d'),
            'Signal': signals.tail(20).values,
            'Action': ['BUY' if x > 0 else 'SELL' if x < 0 else 'NEUTRAL' for x in signals.tail(20).values]
        })
        st.dataframe(recent_signals, use_container_width=True, hide_index=True)
    
    # Download results
    st.subheader("💾 Export Results")
    
    results_df = pd.DataFrame({
        'Date': strategy.results['cumulative_strategy'].index.strftime('%Y-%m-%d'),
        'Strategy_Returns': strategy.results['cumulative_strategy'].values,
        'Benchmark_Returns': strategy.results['cumulative_bh'].values,
        'Signal': strategy.results['signals'].values
    })
    
    csv = results_df.to_csv(index=False)
    st.download_button(
        "📥 Download Backtest Results (CSV)", 
        csv, 
        f"backtest_results_{start_date}_{end_date}.csv", 
        "text/csv",
        use_container_width=True
    )

else:
    st.info("👈 Configure your strategy parameters and click 'Run Backtest' to start")
    
    # Show explanation
    with st.expander("📖 How This Strategy Works", expanded=True):
        st.markdown("""
        ### The Yield Curve Inversion Strategy
        
        This strategy is based on a well-documented market anomaly where an inverted yield curve (short-term rates > long-term rates) historically precedes economic recessions and bond market rallies.
        
        #### Signal Logic
        - **BUY Signal (1)**: When 10Y-3M spread < 0 (inverted curve)
          - Historically precedes economic recessions
          - Bond prices tend to rally as yields fall
          
        - **SELL Signal (-1)**: When 10Y-3M spread > 0.5% (steep curve)
          - Typically during economic expansion
          - Rising yields pressure bond prices lower
        
        #### Strategy Variants
        | Strategy | Description | Best For |
        |----------|-------------|----------|
        | **Classic** | Simple inversion-based signal | Clear recession signals |
        | **Momentum** | Incorporates spread momentum (20-day change) | Trend-following |
        | **ZScore** | Statistical z-score mean reversion | Mean reversion trades |
        | **Adaptive** | Adjusts thresholds based on VIX volatility | Volatile markets |
        | **Composite** | Ensemble average of all signals (recommended) | All market conditions |
        
        ### ⚠️ Risk Warning
        Past performance does not guarantee future results. This strategy is for educational purposes only.
        Always conduct your own research and consider your risk tolerance before making investment decisions.
        """)

# Footer
st.markdown("---")
st.markdown(f"*Data as of {date.today().strftime('%Y-%m-%d')} | Built with yfinance, Plotly, and Streamlit | Educational purposes only*")
