import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# ENHANCED YIELD CURVE STRATEGY
# =============================================================================

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
        print("Fetching market data...")
        
        # Treasury yields
        tickers = ['^IRX', '^FVX', '^TNX', '^TYX']  # 3M, 5Y, 10Y, 30Y
        yields = yf.download(tickers, start=self.start_date, end=self.end_date)['Adj Close']
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
            price = yf.download(ticker, start=self.start_date, end=self.end_date)['Adj Close']
            price.index = price.index.tz_localize(None)
            bond_data[ticker] = price
        
        bond_prices = pd.DataFrame(bond_data)
        
        # Economic indicators for context
        try:
            # VIX for risk sentiment
            vix = yf.download('^VIX', start=self.start_date, end=self.end_date)['Adj Close']
            vix.index = vix.index.tz_localize(None)
            bond_prices['VIX'] = vix
        except:
            print("VIX data unavailable")
            
        self.data = {'yields': yields, 'bond_prices': bond_prices}
        return self
    
    def calculate_spreads(self):
        """Calculate various yield spreads"""
        yields = self.data['yields']
        
        spreads = pd.DataFrame(index=yields.index)
        spreads['10Y-3M'] = yields['10Y'] - yields['3M']
        spreads['10Y-2Y'] = yields['10Y'] - yields['5Y']  # Using 5Y as proxy for 2Y
        spreads['30Y-10Y'] = yields['30Y'] - yields['10Y']
        spreads['Slope'] = spreads['10Y-3M']
        
        # Normalize spreads for comparison
        for col in spreads.columns:
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
        signals.loc[(spreads['10Y-3M'] < 0) & (spread_change < 0), 'Momentum'] = 1  # Inverting further
        signals.loc[(spreads['10Y-3M'] > 0) & (spread_change > 0), 'Momentum'] = -1  # Steepening further
        
        # Strategy 3: Z-score based mean reversion
        signals['ZScore'] = 0
        zscore = spreads['10Y-3M_Zscore']
        signals.loc[zscore < -1.5, 'ZScore'] = 1  # Extreme inversion
        signals.loc[zscore > 1.5, 'ZScore'] = -1  # Extreme steepening
        
        # Strategy 4: Composite (average of all signals)
        signals['Composite'] = (signals['Classic'] + signals['Momentum'] + signals['ZScore']) / 3
        signals['Composite'] = signals['Composite'].clip(-1, 1)
        
        # Strategy 5: Adaptive threshold based on volatility
        vol = self.data['bond_prices']['TLT'].pct_change().rolling(20).std()
        adaptive_threshold = 0.5 + vol * 10
        signals['Adaptive'] = 0
        signals.loc[spreads['10Y-3M'] < -adaptive_threshold, 'Adaptive'] = 1
        signals.loc[spreads['10Y-3M'] > adaptive_threshold, 'Adaptive'] = -1
        
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
        
        # Annualization factor
        ann_factor = np.sqrt(252)
        
        metrics = {
            'Total Return Strategy': cum_strategy.iloc[-1] - 1,
            'Total Return Benchmark': cum_benchmark.iloc[-1] - 1,
            'Excess Return': (cum_strategy.iloc[-1] - 1) - (cum_benchmark.iloc[-1] - 1),
            'Sharpe Ratio': (strategy_returns.mean() / strategy_returns.std()) * ann_factor,
            'Benchmark Sharpe': (benchmark_returns.mean() / benchmark_returns.std()) * ann_factor,
            'Max Drawdown': self.calculate_max_drawdown(cum_strategy),
            'Benchmark Drawdown': self.calculate_max_drawdown(cum_benchmark),
            'Win Rate': (strategy_returns[strategy_returns > 0].count() / strategy_returns.count()),
            'Average Win': strategy_returns[strategy_returns > 0].mean(),
            'Average Loss': strategy_returns[strategy_returns < 0].mean(),
            'Profit Factor': abs(strategy_returns[strategy_returns > 0].sum() / strategy_returns[strategy_returns < 0].sum()),
            'Calmar Ratio': ((cum_strategy.iloc[-1] ** (252/len(cum_strategy)) - 1) / 
                           abs(self.calculate_max_drawdown(cum_strategy))) if self.calculate_max_drawdown(cum_strategy) != 0 else 0
        }
        
        # Add beta and alpha
        if len(strategy_returns) > 0 and len(benchmark_returns) > 0:
            covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
            variance = np.var(benchmark_returns)
            metrics['Beta'] = covariance / variance if variance != 0 else 0
            metrics['Alpha'] = (strategy_returns.mean() - metrics['Beta'] * benchmark_returns.mean()) * ann_factor
        
        return metrics
    
    @staticmethod
    def calculate_max_drawdown(cumulative_returns):
        """Calculate maximum drawdown"""
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown.min()
    
    def plot_results(self):
        """Create comprehensive visualization"""
        if self.results is None:
            print("No results to plot. Run backtest first.")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Yield Curve Strategy Backtest - {self.results["signal_col"]} Signal on {self.results["etf"]}', 
                     fontsize=14, fontweight='bold')
        
        # 1. Cumulative returns comparison
        ax1 = axes[0, 0]
        self.results['cumulative_strategy'].plot(ax=ax1, label='Strategy', linewidth=2)
        self.results['cumulative_bh'].plot(ax=ax1, label='Buy & Hold', linewidth=2, alpha=0.7)
        ax1.set_title('Cumulative Returns Comparison')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Yield spread and signals
        ax2 = axes[0, 1]
        spreads = self.data['spreads']['10Y-3M'].loc[self.results['signals'].index]
        ax2.plot(spreads.index, spreads, label='10Y-3M Spread', color='blue', linewidth=1.5)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2_twin = ax2.twinx()
        self.results['signals'].plot(ax=ax2_twin, label='Signal', color='green', alpha=0.7, linewidth=1)
        ax2.set_title('Yield Spread and Trading Signals')
        ax2.set_ylabel('Spread (%)')
        ax2_twin.set_ylabel('Signal')
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown analysis
        ax3 = axes[1, 0]
        strategy_dd = self.calculate_drawdown_series(self.results['cumulative_strategy'])
        benchmark_dd = self.calculate_drawdown_series(self.results['cumulative_bh'])
        strategy_dd.plot(ax=ax3, label='Strategy', color='red', linewidth=1.5)
        benchmark_dd.plot(ax=ax3, label='Benchmark', color='orange', alpha=0.7)
        ax3.set_title('Drawdown Analysis')
        ax3.set_ylabel('Drawdown (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.fill_between(strategy_dd.index, 0, strategy_dd.values, alpha=0.3, color='red')
        
        # 4. Rolling Sharpe ratio
        ax4 = axes[1, 1]
        rolling_sharpe = self.results['strategy_returns'].rolling(252).apply(
            lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() != 0 else 0
        )
        rolling_sharpe.plot(ax=ax4, color='purple', linewidth=1.5)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Good')
        ax4.axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Poor')
        ax4.set_title('Rolling 1-Year Sharpe Ratio')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance metrics table
        ax5 = axes[2, 0]
        ax5.axis('tight')
        ax5.axis('off')
        metrics_df = pd.DataFrame([self.results['metrics']]).T
        metrics_df.columns = ['Value']
        metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f'{x:.3f}' if isinstance(x, float) else x)
        table = ax5.table(cellText=metrics_df.values, rowLabels=metrics_df.index, 
                          colLabels=['Value'], cellLoc='right', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        ax5.set_title('Performance Metrics', fontsize=10, pad=20)
        
        # 6. Signal distribution
        ax6 = axes[2, 1]
        signal_counts = self.results['signals'].value_counts()
        colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in signal_counts.index]
        ax6.bar(signal_counts.index.astype(str), signal_counts.values, color=colors, alpha=0.7)
        ax6.set_title('Signal Distribution')
        ax6.set_xlabel('Signal')
        ax6.set_ylabel('Frequency')
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    @staticmethod
    def calculate_drawdown_series(cumulative_returns):
        """Calculate drawdown series"""
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown * 100
    
    def print_summary(self):
        """Print performance summary"""
        if self.results is None:
            print("No results to display. Run backtest first.")
            return
        
        metrics = self.results['metrics']
        
        print("\n" + "="*80)
        print(f"YIELD CURVE STRATEGY BACKTEST RESULTS")
        print(f"Signal: {self.results['signal_col']} | ETF: {self.results['etf']}")
        print("="*80)
        
        print(f"\n📊 RETURNS:")
        print(f"   Strategy Total Return:  {metrics['Total Return Strategy']:.2%}")
        print(f"   Benchmark Total Return: {metrics['Total Return Benchmark']:.2%}")
        print(f"   Excess Return:          {metrics['Excess Return']:.2%}")
        
        print(f"\n📈 RISK METRICS:")
        print(f"   Strategy Sharpe:        {metrics['Sharpe Ratio']:.2f}")
        print(f"   Benchmark Sharpe:       {metrics['Benchmark Sharpe']:.2f}")
        print(f"   Strategy Max DD:        {metrics['Max Drawdown']:.2%}")
        print(f"   Benchmark Max DD:       {metrics['Benchmark Drawdown']:.2%}")
        print(f"   Calmar Ratio:           {metrics['Calmar Ratio']:.2f}")
        
        print(f"\n🎯 TRADE STATISTICS:")
        print(f"   Win Rate:               {metrics['Win Rate']:.2%}")
        print(f"   Profit Factor:          {metrics['Profit Factor']:.2f}")
        print(f"   Avg Win / Avg Loss:     {abs(metrics['Average Win'] / metrics['Average Loss']):.2f}")
        
        if 'Beta' in metrics:
            print(f"\n📐 MARKET EXPOSURE:")
            print(f"   Beta:                   {metrics['Beta']:.2f}")
            print(f"   Alpha (annualized):     {metrics['Alpha']:.2%}")
        
        print("\n" + "="*80)

# =============================================================================
# COMPARATIVE ANALYSIS
# =============================================================================

def compare_strategies():
    """Compare multiple strategy variations"""
    
    strategy = YieldCurveStrategy(start_date='2015-01-01', end_date='2023-12-31')
    strategy.fetch_data().calculate_spreads()
    
    results = {}
    
    # Test different signal types
    signal_types = ['Classic', 'Momentum', 'ZScore', 'Composite', 'Adaptive']
    
    for signal in signal_types:
        print(f"\nTesting {signal} strategy...")
        strategy.generate_signals().backtest(etf='TLT', signal_col=signal)
        results[signal] = strategy.results['metrics']
    
    # Create comparison dataframe
    comparison = pd.DataFrame(results).T
    
    # Sort by Sharpe ratio
    comparison = comparison.sort_values('Sharpe Ratio', ascending=False)
    
    print("\n" + "="*80)
    print("STRATEGY COMPARISON (TLT, 2015-2023)")
    print("="*80)
    
    display_cols = ['Total Return Strategy', 'Sharpe Ratio', 'Max Drawdown', 
                    'Win Rate', 'Profit Factor', 'Calmar Ratio']
    
    for col in display_cols:
        comparison[col] = comparison[col].apply(lambda x: f'{x:.3f}' if isinstance(x, float) else x)
    
    print(comparison[display_cols].to_string())
    
    return comparison

# =============================================================================
# REAL-TIME SIGNAL GENERATION
# =============================================================================

def get_current_signal():
    """Generate current trading signal based on latest data"""
    
    try:
        # Fetch latest yields
        yields = yf.download(['^IRX', '^TNX'], period='5d')['Adj Close']
        yields.columns = ['3M', '10Y']
        yields.index = yields.index.tz_localize(None)
        
        latest_3m = yields['3M'].iloc[-1]
        latest_10y = yields['10Y'].iloc[-1]
        spread = latest_10y - latest_3m
        
        # Calculate historical context (using last 252 days)
        hist_yields = yf.download(['^IRX', '^TNX'], period='1y')['Adj Close']
        hist_yields.columns = ['3M', '10Y']
        hist_spread = hist_yields['10Y'] - hist_yields['3M']
        
        zscore = (spread - hist_spread.mean()) / hist_spread.std()
        
        # Generate signal
        if spread < 0 and zscore < -1:
            signal = "STRONG BUY"
            confidence = "High"
            rationale = "Deep inversion with extreme negative z-score"
        elif spread < 0:
            signal = "BUY"
            confidence = "Medium"
            rationale = "Yield curve inverted, historically precedes bond rallies"
        elif spread > 1.5:
            signal = "SELL"
            confidence = "Medium"
            rationale = "Very steep curve, bond yields may rise further"
        elif spread > 0.5:
            signal = "REDUCE"
            confidence = "Low"
            rationale = "Moderately steep curve, caution advised"
        else:
            signal = "NEUTRAL"
            confidence = "Low"
            rationale = "Normal curve shape, no strong signal"
        
        print("\n" + "="*60)
        print("CURRENT YIELD CURVE SIGNAL")
        print("="*60)
        print(f"10Y Yield:        {latest_10y:.2f}%")
        print(f"3M Yield:         {latest_3m:.2f}%")
        print(f"10Y-3M Spread:    {spread:.2f}%")
        print(f"Z-Score:          {zscore:.2f}")
        print(f"\n🚦 SIGNAL:        {signal}")
        print(f"📊 Confidence:    {confidence}")
        print(f"💡 Rationale:     {rationale}")
        print("="*60)
        
        return {
            'signal': signal,
            'spread': spread,
            'zscore': zscore,
            'confidence': confidence,
            'rationale': rationale
        }
        
    except Exception as e:
        print(f"Error fetching current data: {e}")
        return None

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    # Run backtest
    print("🚀 INITIALIZING YIELD CURVE STRATEGY BACKTEST")
    print("-" * 60)
    
    # Test on TLT (long-term bonds)
    strategy = YieldCurveStrategy(start_date='2010-01-01', end_date='2023-12-31')
    strategy.fetch_data().calculate_spreads()
    strategy.generate_signals().backtest(etf='TLT', signal_col='Composite')
    
    # Display results
    strategy.print_summary()
    strategy.plot_results()
    
    # Compare different strategies
    print("\n" + "🔄 RUNNING STRATEGY COMPARISON")
    comparison = compare_strategies()
    
    # Get current market signal
    print("\n" + "📡 GENERATING CURRENT MARKET SIGNAL")
    current_signal = get_current_signal()
    
    # Additional analysis: Different bond maturities
    print("\n" + "🏦 TESTING ACROSS DIFFERENT MATURITIES")
    
    for etf in ['TLT', 'IEF', 'SHY']:
        strategy.backtest(etf=etf, signal_col='Composite')
        print(f"\n{etf} Results:")
        print(f"  Total Return: {strategy.results['metrics']['Total Return Strategy']:.2%}")
        print(f"  Sharpe Ratio: {strategy.results['metrics']['Sharpe Ratio']:.2f}")
        print(f"  Max Drawdown: {strategy.results['metrics']['Max Drawdown']:.2%}")
