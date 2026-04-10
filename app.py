import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
        
        # Treasury yields - FIXED: Handle MultiIndex columns
        tickers = ['^IRX', '^FVX', '^TNX', '^TYX']  # 3M, 5Y, 10Y, 30Y
        df = yf.download(tickers, start=self.start_date, end=self.end_date)
        
        # Check if we have MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            # Extract 'Adj Close' for all tickers
            yields = df['Adj Close'].copy()
        else:
            # Fallback for single ticker or different structure
            yields = df[['Adj Close']] if 'Adj Close' in df.columns else df
        
        yields.columns = ['3M', '5Y', '10Y', '30Y']
        yields.index = yields.index.tz_localize(None)
        
        # Bond ETFs
        etfs = {
            'TLT': '20+ Year Treasury',
            'IEF': '7-10 Year Treasury', 
            'SHY': '1-3 Year Treasury',
            'BND': 'Total Bond Market'
        }
        
        bond_data = {}
        for ticker, name in etfs.items():
            df_etf = yf.download(ticker, start=self.start_date, end=self.end_date)
            
            # Handle MultiIndex for ETF data
            if isinstance(df_etf.columns, pd.MultiIndex):
                price = df_etf['Adj Close'].copy()
            else:
                price = df_etf['Adj Close'] if 'Adj Close' in df_etf.columns else df_etf['Close']
            
            price.index = price.index.tz_localize(None)
            bond_data[ticker] = price
        
        bond_prices = pd.DataFrame(bond_data)
        
        # Economic indicators
        try:
            vix = yf.download('^VIX', start=self.start_date, end=self.end_date)
            if isinstance(vix.columns, pd.MultiIndex):
                vix_series = vix['Adj Close'].copy()
            else:
                vix_series = vix['Adj Close'] if 'Adj Close' in vix.columns else vix['Close']
            vix_series.index = vix_series.index.tz_localize(None)
            bond_prices['VIX'] = vix_series
        except Exception as e:
            print(f"VIX data unavailable: {e}")
            
        self.data = {'yields': yields, 'bond_prices': bond_prices}
        return self
    
    def calculate_spreads(self):
        """Calculate various yield spreads"""
        yields = self.data['yields']
        
        spreads = pd.DataFrame(index=yields.index)
        spreads['10Y-3M'] = yields['10Y'] - yields['3M']
        spreads['10Y-5Y'] = yields['10Y'] - yields['5Y']
        spreads['30Y-10Y'] = yields['30Y'] - yields['10Y']
        spreads['Slope'] = spreads['10Y-3M']
        
        # Normalize spreads for comparison
        for col in spreads.columns:
            if col not in ['10Y-5Y', '30Y-10Y', 'Slope']:
                spreads[f'{col}_Zscore'] = (spreads[col] - spreads[col].rolling(252).mean()) / spreads[col].rolling(252).std()
            
        self.data['spreads'] = spreads
        return self
    
    def generate_signals(self, strategy='composite'):
        """Generate trading signals using multiple approaches"""
        
        spreads = self.data['spreads']
        signals = pd.DataFrame(index=spreads.index)
        
        # Strategy 1: Classic inversion signal
        signals['Classic'] = 0
        signals.loc[spreads['10Y-3M'] < 0, 'Classic'] = 1  # Buy on inversion
        signals.loc[spreads['10Y-3M'] > 0.5, 'Classic'] = -1  # Sell when steep
        
        # Strategy 2: Momentum-enhanced signal
        signals['Momentum'] = 0
        spread_change = spreads['10Y-3M'].diff(20)  # 1-month change
        signals.loc[(spreads['10Y-3M'] < 0) & (spread_change < 0), 'Momentum'] = 1
        signals.loc[(spreads['10Y-3M'] > 0) & (spread_change > 0), 'Momentum'] = -1
        
        # Strategy 3: Z-score based mean reversion
        signals['ZScore'] = 0
        if '10Y-3M_Zscore' in spreads.columns:
            zscore = spreads['10Y-3M_Zscore']
            signals.loc[zscore < -1.5, 'ZScore'] = 1
            signals.loc[zscore > 1.5, 'ZScore'] = -1
        
        # Strategy 4: Composite (average of all signals)
        signals['Composite'] = (signals['Classic'] + signals['Momentum'] + signals['ZScore']) / 3
        signals['Composite'] = signals['Composite'].clip(-1, 1)
        
        # Strategy 5: Adaptive threshold based on volatility
        if 'VIX' in self.data['bond_prices'].columns:
            vol = self.data['bond_prices']['VIX'] / 20  # Normalize VIX
            adaptive_threshold = 0.5 + vol.clip(0, 1)
            signals['Adaptive'] = 0
            signals.loc[spreads['10Y-3M'] < -adaptive_threshold, 'Adaptive'] = 1
            signals.loc[spreads['10Y-3M'] > adaptive_threshold, 'Adaptive'] = -1
        else:
            signals['Adaptive'] = signals['Classic']
        
        self.signals = signals
        return self
    
    def backtest(self, etf='TLT', signal_col='Composite', transaction_cost=0.001):
        """Backtest the strategy with realistic assumptions"""
        
        bond_prices = self.data['bond_prices']
        signals = self.signals[signal_col].copy()
        
        # Align signals with bond data
        common_idx = bond_prices.index.intersection(signals.index)
        signals = signals.loc[common_idx]
        prices = bond_prices[etf].loc[common_idx]
        
        # Calculate returns
        returns = prices.pct_change()
        
        # Generate strategy returns (signal shifted to avoid look-ahead bias)
        strategy_returns = signals.shift(1) * returns
        
        # Apply transaction costs when signals change
        signal_changes = signals.diff().abs()
        transaction_costs = signal_changes * transaction_cost
        strategy_returns = strategy_returns - transaction_costs
        
        # Calculate metrics
        cumulative_returns = (1 + strategy_returns).cumprod()
        buy_hold_returns = (1 + returns).cumprod()
        
        # Performance metrics
        metrics = self.calculate_metrics(strategy_returns, returns, cumulative_returns, buy_hold_returns)
        
        self.results = {
            'strategy_returns': strategy_returns,
            'cumulative_strategy': cumulative_returns,
            'cumulative_bh': buy_hold_returns,
            'signals': signals,
            'metrics': metrics,
            'etf': etf,
            'signal_col': signal_col
        }
        
        return self
    
    def calculate_metrics(self, strategy_returns, benchmark_returns, cum_strategy, cum_benchmark):
        """Calculate comprehensive performance metrics"""
        
        # Remove NaN values
        strategy_returns = strategy_returns.dropna()
        benchmark_returns = benchmark_returns.loc[strategy_returns.index]
        
        if len(strategy_returns) == 0 or strategy_returns.std() == 0:
            return {'Error': 'Insufficient data for metrics calculation'}
        
        # Annualization factor
        ann_factor = np.sqrt(252)
        
        metrics = {
            'Total Return Strategy': float(cum_strategy.iloc[-1] - 1) if len(cum_strategy) > 0 else 0,
            'Total Return Benchmark': float(cum_benchmark.iloc[-1] - 1) if len(cum_benchmark) > 0 else 0,
            'Excess Return': 0,
            'Sharpe Ratio': (strategy_returns.mean() / strategy_returns.std()) * ann_factor,
            'Benchmark Sharpe': (benchmark_returns.mean() / benchmark_returns.std()) * ann_factor if benchmark_returns.std() > 0 else 0,
            'Max Drawdown': self.calculate_max_drawdown(cum_strategy),
            'Benchmark Drawdown': self.calculate_max_drawdown(cum_benchmark),
            'Win Rate': float((strategy_returns[strategy_returns > 0].count() / strategy_returns.count())) if strategy_returns.count() > 0 else 0,
            'Average Win': float(strategy_returns[strategy_returns > 0].mean()) if len(strategy_returns[strategy_returns > 0]) > 0 else 0,
            'Average Loss': float(strategy_returns[strategy_returns < 0].mean()) if len(strategy_returns[strategy_returns < 0]) > 0 else 0,
            'Profit Factor': 0,
            'Calmar Ratio': 0
        }
        
        metrics['Excess Return'] = metrics['Total Return Strategy'] - metrics['Total Return Benchmark']
        
        if metrics['Average Loss'] != 0:
            metrics['Profit Factor'] = abs(metrics['Average Win'] / metrics['Average Loss'])
        
        if metrics['Max Drawdown'] != 0:
            annual_return = (cum_strategy.iloc[-1] ** (252/len(cum_strategy)) - 1) if len(cum_strategy) > 0 else 0
            metrics['Calmar Ratio'] = annual_return / abs(metrics['Max Drawdown'])
        
        return metrics
    
    @staticmethod
    def calculate_max_drawdown(cumulative_returns):
        """Calculate maximum drawdown"""
        if len(cumulative_returns) == 0:
            return 0
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return float(drawdown.min())
    
    def get_plotly_charts(self):
        """Create interactive Plotly charts for Streamlit"""
        if self.results is None:
            return None
        
        # Chart 1: Cumulative Returns
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=self.results['cumulative_strategy'].index,
            y=self.results['cumulative_strategy'].values,
            mode='lines',
            name='Strategy',
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
            title=f'Cumulative Returns - {self.results["signal_col"]} Strategy',
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Chart 2: Yield Spread and Signals
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        
        spreads = self.data['spreads']['10Y-3M'].loc[self.results['signals'].index]
        fig2.add_trace(
            go.Scatter(x=spreads.index, y=spreads.values, name='10Y-3M Spread', line=dict(color='blue')),
            secondary_y=False
        )
        fig2.add_trace(
            go.Scatter(x=self.results['signals'].index, y=self.results['signals'].values, 
                      name='Signal', line=dict(color='green', width=1)),
            secondary_y=True
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="red", secondary_y=False)
        fig2.update_layout(title='Yield Spread and Trading Signals', template='plotly_white')
        fig2.update_yaxes(title_text="Spread (bps)", secondary_y=False)
        fig2.update_yaxes(title_text="Signal (-1 to 1)", secondary_y=True)
        
        # Chart 3: Drawdown
        fig3 = go.Figure()
        strategy_dd = self.calculate_drawdown_series(self.results['cumulative_strategy'])
        benchmark_dd = self.calculate_drawdown_series(self.results['cumulative_bh'])
        
        fig3.add_trace(go.Scatter(x=strategy_dd.index, y=strategy_dd.values, name='Strategy', 
                                  fill='tozeroy', line=dict(color='red')))
        fig3.add_trace(go.Scatter(x=benchmark_dd.index, y=benchmark_dd.values, name='Benchmark', 
                                  line=dict(color='orange', dash='dash')))
        fig3.update_layout(title='Drawdown Analysis', xaxis_title='Date', 
                          yaxis_title='Drawdown (%)', template='plotly_white')
        
        return {'returns': fig1, 'spread': fig2, 'drawdown': fig3}
    
    @staticmethod
    def calculate_drawdown_series(cumulative_returns):
        """Calculate drawdown series"""
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown * 100

# Streamlit UI
st.set_page_config(page_title="Yield Curve Strategy Backtest", layout="wide")

st.title("📈 Yield Curve Trading Strategy Backtest")
st.markdown("Advanced fixed-income strategy using yield curve inversion signals")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Strategy Configuration")
    
    start_date = st.date_input("Start Date", pd.to_datetime('2015-01-01'))
    end_date = st.date_input("End Date", pd.to_datetime('2023-12-31'))
    
    etf_choice = st.selectbox("Bond ETF", ['TLT', 'IEF', 'SHY', 'BND'])
    
    signal_choice = st.selectbox("Signal Type", ['Composite', 'Classic', 'Momentum', 'ZScore', 'Adaptive'])
    
    transaction_cost = st.slider("Transaction Cost (%)", 0.0, 0.5, 0.1, 0.01) / 100
    
    run_backtest = st.button("🚀 Run Backtest", type="primary")

# Main content
if run_backtest:
    with st.spinner("Fetching data and running backtest..."):
        try:
            # Initialize and run strategy
            strategy = YieldCurveStrategy(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            strategy.fetch_data().calculate_spreads()
            strategy.generate_signals().backtest(
                etf=etf_choice, 
                signal_col=signal_choice,
                transaction_cost=transaction_cost
            )
            
            # Display metrics
            st.subheader("📊 Performance Metrics")
            
            metrics = strategy.results['metrics']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Return", f"{metrics['Total Return Strategy']:.2%}")
                st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
            
            with col2:
                st.metric("Benchmark Return", f"{metrics['Total Return Benchmark']:.2%}")
                st.metric("Benchmark Sharpe", f"{metrics['Benchmark Sharpe']:.2f}")
            
            with col3:
                st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
                st.metric("Win Rate", f"{metrics['Win Rate']:.2%}")
            
            with col4:
                st.metric("Excess Return", f"{metrics['Excess Return']:.2%}")
                st.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")
            
            # Display charts
            st.subheader("📈 Strategy Visualization")
            charts = strategy.get_plotly_charts()
            
            if charts:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(charts['returns'], use_container_width=True)
                with col2:
                    st.plotly_chart(charts['spread'], use_container_width=True)
                
                st.plotly_chart(charts['drawdown'], use_container_width=True)
            
            # Display signal distribution
            st.subheader("🎯 Signal Distribution")
            signal_counts = strategy.results['signals'].value_counts()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Buy Signals", int(signal_counts.get(1, 0)))
            with col2:
                st.metric("Sell Signals", int(signal_counts.get(-1, 0)))
            with col3:
                st.metric("Neutral", int(signal_counts.get(0, 0)))
            
        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")
            st.info("Try adjusting the date range or check your internet connection.")

else:
    st.info("👈 Configure your strategy parameters and click 'Run Backtest' to start")
    
    # Show explanation
    st.markdown("""
    ### How This Strategy Works
    
    The **Yield Curve Inversion Strategy** is based on a well-documented market anomaly:
    
    - **BUY Signal**: When the yield curve inverts (10Y - 3M < 0), historically this precedes economic recessions and bond rallies
    - **SELL Signal**: When the curve is steep (10Y - 3M > 0.5%), economic expansion typically leads to rising yields
    
    ### Multiple Signal Variants
    
    - **Classic**: Simple inversion-based signal
    - **Momentum**: Incorporates spread momentum
    - **ZScore**: Mean reversion using statistical z-scores  
    - **Adaptive**: Adjusts thresholds based on market volatility (VIX)
    - **Composite**: Ensemble average of all signals
    
    ### Risk Warning
    
    Past performance does not guarantee future results. This strategy is for educational purposes only.
    """)
