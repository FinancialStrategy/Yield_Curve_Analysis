import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def safe_get_adj_close(data, ticker=None):
    """Safely extract Adj Close from yfinance data regardless of structure"""
    if data is None or data.empty:
        return None
    
    # Case 1: Direct 'Adj Close' column
    if 'Adj Close' in data.columns:
        if ticker and isinstance(data['Adj Close'], pd.DataFrame) and ticker in data['Adj Close'].columns:
            return data['Adj Close'][ticker]
        elif not isinstance(data['Adj Close'], pd.DataFrame):
            return data['Adj Close']
        else:
            return data['Adj Close'].iloc[:, 0] if len(data['Adj Close'].columns) > 0 else None
    
    # Case 2: 'Close' column (for ETFs)
    if 'Close' in data.columns:
        if isinstance(data['Close'], pd.DataFrame):
            return data['Close'].iloc[:, 0] if len(data['Close'].columns) > 0 else None
        return data['Close']
    
    # Case 3: Direct price data
    if len(data.columns) == 1:
        return data.iloc[:, 0]
    
    return None

class YieldCurveStrategy:
    """Advanced yield curve trading strategy with multiple signals"""
    
    def __init__(self, start_date='2010-01-01', end_date='2023-12-31'):
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.signals = None
        self.results = None
        
    def fetch_data(self):
        """Fetch treasury yields and bond ETF data"""
        
        # Treasury yields - download individually
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
                # Align VIX with bond prices index
                common_idx = bond_prices.index.intersection(vix_series.index)
                bond_prices['VIX'] = vix_series.reindex(common_idx)
        except Exception as e:
            print(f"VIX data unavailable: {e}")
            
        self.data = {'yields': yields, 'bond_prices': bond_prices}
        return self
    
    def calculate_spreads(self):
        """Calculate various yield spreads"""
        yields = self.data['yields']
        
        # Ensure we have data
        if yields.empty:
            raise Exception("No yield data available for spread calculation")
        
        spreads = pd.DataFrame(index=yields.index)
        
        # Calculate spreads with proper alignment
        if '10Y' in yields.columns and '3M' in yields.columns:
            # Ensure both series have the same index
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
            
            # Calculate Z-score with rolling window
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
            # Align VIX with signals index
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
        
        # Ensure all indices are aligned
        common_idx = bond_prices.index.intersection(signals.index)
        if len(common_idx) == 0:
            raise Exception("No overlapping dates between signals and bond data")
        
        # Reindex everything to common index
        signals_aligned = signals.reindex(common_idx)
        prices_aligned = bond_prices[etf].reindex(common_idx)
        
        # Calculate returns
        returns = prices_aligned.pct_change()
        
        # Generate strategy returns (signal shifted to avoid look-ahead bias)
        strategy_returns = signals_aligned.shift(1) * returns
        
        # Apply transaction costs when signals change
        signal_changes = signals_aligned.diff().abs()
        transaction_costs = signal_changes * transaction_cost
        strategy_returns = strategy_returns - transaction_costs
        
        # Fill NaN values
        strategy_returns = strategy_returns.fillna(0)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        buy_hold_returns = (1 + returns.fillna(0)).cumprod()
        
        # Performance metrics
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
        
        # Remove NaN values from strategy returns for calculations
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
        
        # Annualization factor
        ann_factor = np.sqrt(252)
        
        # Calculate metrics
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
        
        # Calculate profit factor
        gross_profits = strategy_returns[strategy_returns > 0].sum()
        gross_losses = abs(strategy_returns[strategy_returns < 0].sum())
        metrics['Profit Factor'] = float(gross_profits / gross_losses) if gross_losses > 0 else 0
        
        # Calculate Calmar ratio
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
        
        # Ensure we have positive values for calculation
        cum_clean = cumulative_returns.fillna(1)
        rolling_max = cum_clean.expanding().max()
        drawdown = (cum_clean - rolling_max) / rolling_max
        return float(drawdown.min())
    
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
        
        # Chart 2: Yield Spread and Signals
        if self.data['spreads'] is not None and '10Y-3M' in self.data['spreads'].columns:
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Align spread with signals index
            spreads_aligned = self.data['spreads']['10Y-3M'].reindex(self.results['signals'].index)
            
            fig2.add_trace(
                go.Scatter(x=spreads_aligned.index, y=spreads_aligned.values, 
                          name='10Y-3M Spread', line=dict(color='blue', width=2)),
                secondary_y=False
            )
            fig2.add_trace(
                go.Scatter(x=self.results['signals'].index, y=self.results['signals'].values, 
                          name='Trading Signal', line=dict(color='green', width=1.5)),
                secondary_y=True
            )
            fig2.add_hline(y=0, line_dash="dash", line_color="red", secondary_y=False)
            fig2.update_layout(title='Yield Spread and Trading Signals', template='plotly_white', height=500)
            fig2.update_yaxes(title_text="Spread (bps)", secondary_y=False)
            fig2.update_yaxes(title_text="Signal (-1 to 1)", secondary_y=True)
            charts['spread'] = fig2
        
        return charts

# Streamlit UI
st.set_page_config(page_title="Yield Curve Strategy Backtest", layout="wide")

st.title("📈 Yield Curve Trading Strategy Backtest")
st.markdown("Advanced fixed-income strategy using yield curve inversion signals")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Strategy Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", pd.to_datetime('2015-01-01'))
    with col2:
        end_date = st.date_input("End Date", pd.to_datetime('2023-12-31'))
    
    etf_choice = st.selectbox("Bond ETF", ['TLT', 'IEF', 'SHY', 'BND'], 
                               help="TLT: Long-term, IEF: Intermediate, SHY: Short-term")
    
    signal_choice = st.selectbox("Signal Type", ['Composite', 'Classic', 'Momentum', 'ZScore', 'Adaptive'])
    
    transaction_cost = st.slider("Transaction Cost (%)", 0.0, 0.5, 0.1, 0.01) / 100
    
    st.markdown("---")
    st.markdown("### 📊 Strategy Logic")
    st.info("""
    - **BUY**: When yield curve inverts (10Y < 3M)
    - **SELL**: When curve steepens (>0.5%)
    - **Composite**: Ensemble of multiple signals
    """)
    
    run_backtest = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

# Main content
if run_backtest:
    with st.spinner("Fetching data and running backtest..."):
        try:
            # Initialize and run strategy
            strategy = YieldCurveStrategy(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            # Fetch data with progress indicators
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
            
            # Display metrics
            st.subheader("📊 Performance Metrics")
            
            metrics = strategy.results['metrics']
            
            # Create 2 rows of metrics
            row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
            row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
            
            with row1_col1:
                st.metric("Total Return (Strategy)", f"{metrics['Total Return Strategy']:.2%}")
            with row1_col2:
                st.metric("Total Return (Benchmark)", f"{metrics['Total Return Benchmark']:.2%}")
            with row1_col3:
                delta_color = "normal" if metrics['Excess Return'] >= 0 else "inverse"
                st.metric("Excess Return", f"{metrics['Excess Return']:.2%}", 
                         delta=f"{'+' if metrics['Excess Return'] >= 0 else ''}{metrics['Excess Return']:.2%}")
            with row1_col4:
                st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
            
            with row2_col1:
                st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
            with row2_col2:
                st.metric("Win Rate", f"{metrics['Win Rate']:.2%}")
            with row2_col3:
                st.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")
            with row2_col4:
                st.metric("Calmar Ratio", f"{metrics['Calmar Ratio']:.2f}")
            
            # Display charts
            st.subheader("📈 Strategy Visualization")
            charts = strategy.create_visualizations()
            
            if charts:
                if 'returns' in charts:
                    st.plotly_chart(charts['returns'], use_container_width=True)
                if 'spread' in charts:
                    st.plotly_chart(charts['spread'], use_container_width=True)
            
            # Display signal statistics
            st.subheader("🎯 Signal Statistics")
            signals = strategy.results['signals']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Trading Days", len(signals))
            with col2:
                st.metric("Buy Signals", (signals > 0).sum())
            with col3:
                st.metric("Sell Signals", (signals < 0).sum())
            with col4:
                st.metric("Neutral", (signals == 0).sum())
            
            # Show sample of signals
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
            st.download_button("📥 Download Backtest Results (CSV)", csv, "backtest_results.csv", "text/csv", use_container_width=True)
            
            # Show success message
            st.success("✅ Backtest completed successfully!")
            
        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")
            st.info("""
            **Troubleshooting tips:**
            1. Check your internet connection
            2. Try a different date range (e.g., 2020-01-01 to 2023-12-31)
            3. The yfinance API might be temporarily rate-limited - wait a few minutes
            4. Try refreshing the page and running again
            """)

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
        
        #### Performance Metrics Explained
        - **Sharpe Ratio**: Risk-adjusted return (>1.0 is good, >2.0 is excellent)
        - **Max Drawdown**: Largest peak-to-trough decline (lower is better)
        - **Win Rate**: Percentage of profitable trades (>50% is good)
        - **Profit Factor**: Gross profits / gross losses (>1.5 is excellent)
        - **Calmar Ratio**: Return / max drawdown (>1.0 is good)
        
        ### ⚠️ Risk Warning
        Past performance does not guarantee future results. This strategy is for educational purposes only.
        Always conduct your own research and consider your risk tolerance before making investment decisions.
        """)

# Footer
st.markdown("---")
st.markdown("*Built with yfinance, Plotly, and Streamlit | Educational purposes only*")
