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
# FRED API KONFIGÜRASYONU
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
    'TLT': '20+ Year Treasury Bond ETF',
    'IEF': '7-10 Year Treasury Bond ETF',
    'SHY': '1-3 Year Treasury Bond ETF',
    'BND': 'Total Bond Market ETF',
    'GOVT': 'US Treasury Bond ETF'
}

VOLATILITY_TICKERS = {
    '^VIX': 'CBOE Volatility Index',
    '^VXN': 'Nasdaq Volatility Index',
    '^VXD': 'DJIA Volatility Index'
}

CORRELATION_TICKERS = {
    '^GSPC': 'S&P 500',
    '^IXIC': 'Nasdaq Composite',
    'GLD': 'Gold',
    'UUP': 'US Dollar Index',
    '^TNX': '10Y Treasury Yield'
}

# =============================================================================
# FRED API FONKSİYONLARI
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_series(api_key, series_id, start_date, end_date):
    """FRED API'den veri çekme"""
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
    """Tüm getiri eğrisi verilerini FRED'den çekme"""
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

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recession_data(api_key, start_date, end_date):
    """Resesyon göstergesi verilerini çekme"""
    return fetch_fred_series(api_key, RECESSION_SERIES, start_date, end_date)

def validate_fred_api_key(api_key):
    """FRED API key doğrulama"""
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
# MARKET DATA FONKSİYONLARI (YAHOO FINANCE)
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_data(ticker, start_date, end_date):
    """Yahoo Finance'den veri çekme"""
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
def fetch_market_bundle(start_date, end_date):
    """Tüm market verilerini çekme"""
    # Volatilite verileri
    vol_data = {}
    for ticker, name in VOLATILITY_TICKERS.items():
        series = fetch_yahoo_data(ticker, start_date, end_date)
        if not series.empty:
            vol_data[name] = series
    
    # Korelasyon verileri
    corr_data = {}
    for ticker, name in CORRELATION_TICKERS.items():
        series = fetch_yahoo_data(ticker, start_date, end_date)
        if not series.empty:
            corr_data[name] = series
    
    return pd.DataFrame(vol_data), pd.DataFrame(corr_data)

# =============================================================================
# TEKNİK GÖSTERGELER
# =============================================================================

def calculate_technical_indicators(price_series):
    """Teknik gösterge hesaplama"""
    if price_series.empty:
        return pd.DataFrame()
    
    df = pd.DataFrame(index=price_series.index)
    df['Price'] = price_series
    
    # Hareketli ortalamalar
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
    
    # Bollinger Bantları
    df['BB_Middle'] = price_series.rolling(20).mean()
    bb_std = price_series.rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # ATR (Average True Range)
    high_low = price_series - price_series.shift()
    df['ATR'] = high_low.abs().rolling(14).mean()
    
    return df

# =============================================================================
# NELSON-SIEGEL MODELİ
# =============================================================================

class NelsonSiegelModel:
    """Nelson-Siegel getiri eğrisi modellemesi"""
    
    @staticmethod
    def nelson_siegel(tau, beta0, beta1, beta2, lambda1):
        """Nelson-Siegel fonksiyonu"""
        tau = np.asarray(tau, dtype=float)
        tau_safe = np.where(tau == 0, 1e-8, tau)
        x = lambda1 * tau_safe
        term1 = (1 - np.exp(-x)) / x
        term2 = term1 - np.exp(-x)
        return beta0 + beta1 * term1 + beta2 * term2
    
    @staticmethod
    def fit_curve(maturities, yields):
        """Nelson-Siegel parametrelerini optimize etme"""
        if len(maturities) == 0 or len(yields) == 0:
            return None
        
        def objective(params):
            fitted = NelsonSiegelModel.nelson_siegel(maturities, *params)
            return np.sum((yields - fitted) ** 2)
        
        bounds = [
            (yields.min() - 2, yields.max() + 2),
            (-15, 15), (-15, 15), (0.01, 5)
        ]
        
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
        
        return {
            'params': best_result.x,
            'fitted': fitted,
            'rmse': rmse,
            'r2': r2,
            'residuals': yields - fitted
        }

# =============================================================================
# FAKTÖR ANALİZİ ve PCA
# =============================================================================

def calculate_factor_contributions(yield_df):
    """Faktör katkılarını hesaplama (Level, Slope, Curvature)"""
    if yield_df.empty:
        return pd.DataFrame()
    
    factors = pd.DataFrame(index=yield_df.index)
    
    # Level (10Y yield)
    if '10Y' in yield_df.columns:
        factors['Level'] = yield_df['10Y']
    
    # Slope (10Y - 3M)
    if '10Y' in yield_df.columns and '3M' in yield_df.columns:
        factors['Slope'] = (yield_df['10Y'] - yield_df['3M']) * 100
    
    # Curvature (2*5Y - (3M + 10Y))
    if all(x in yield_df.columns for x in ['3M', '5Y', '10Y']):
        factors['Curvature'] = (2 * yield_df['5Y'] - (yield_df['3M'] + yield_df['10Y'])) * 100
    
    return factors

def perform_pca_analysis(yield_df, n_components=3):
    """PCA ile risk faktörü analizi"""
    if yield_df.empty or len(yield_df) < 20:
        return None
    
    returns = yield_df.pct_change().dropna()
    if returns.empty or returns.shape[1] < 2:
        return None
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
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
# MONTE CARLO SİMÜLASYONU
# =============================================================================

class MonteCarloSimulator:
    """Monte Carlo simülasyonu için sınıf"""
    
    @staticmethod
    def simulate_gbm(initial_price, mu, sigma, days, simulations=1000):
        """Geometric Brownian Motion simülasyonu"""
        dt = 1 / 252
        paths = np.zeros((simulations, days))
        paths[:, 0] = initial_price
        
        for i in range(1, days):
            z = np.random.standard_normal(simulations)
            paths[:, i] = paths[:, i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
        return paths
    
    @staticmethod
    def simulate_vasicek(initial_rate, kappa, theta, sigma, days, simulations=1000):
        """Vasicek mean-reverting model simülasyonu"""
        dt = 1 / 252
        paths = np.zeros((simulations, days))
        paths[:, 0] = initial_rate
        
        for i in range(1, days):
            z = np.random.standard_normal(simulations)
            dr = kappa * (theta - paths[:, i-1]) * dt + sigma * np.sqrt(dt) * z
            paths[:, i] = paths[:, i-1] + dr
        
        return paths
    
    @staticmethod
    def calculate_confidence_intervals(paths, confidence=0.95):
        """Güven aralıklarını hesaplama"""
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
        """Value at Risk hesaplama"""
        return np.percentile(paths[:, -1], (1 - confidence) * 100)

# =============================================================================
# MAKİNE ÖĞRENMESİ MODELLERİ
# =============================================================================

class MLForecastModel:
    """Makine öğrenmesi ile tahminleme"""
    
    @staticmethod
    def prepare_features(yield_df, target_col='10Y', lags=5):
        """Özellik mühendisliği"""
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
        """Model eğitimi"""
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
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': [f'Feature_{i}' for i in range(X.shape[1])],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
        else:
            importance = pd.DataFrame()
        
        return {
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'feature_importance': importance
        }

# =============================================================================
# BACKTEST MOTORU
# =============================================================================

class BacktestEngine:
    """Strateji backtest motoru"""
    
    @staticmethod
    def backtest_strategy(yield_df, spreads, etf_returns, strategy_type='Curve Inversion'):
        """Strateji backtest'i"""
        if yield_df.empty or spreads.empty or etf_returns.empty:
            return None
        
        # Sinyal üretimi
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
        
        # Sinyalleri hizalama
        common_idx = etf_returns.index.intersection(signals.index)
        if len(common_idx) == 0:
            return None
        
        signals_aligned = signals.reindex(common_idx).fillna(0).astype(int)
        returns_aligned = etf_returns.reindex(common_idx)
        
        # Strateji getirileri
        strategy_returns = signals_aligned.shift(1) * returns_aligned
        strategy_returns = strategy_returns.fillna(0)
        
        # Kümülatif getiriler
        cumulative_strategy = (1 + strategy_returns).cumprod()
        cumulative_benchmark = (1 + returns_aligned.fillna(0)).cumprod()
        
        # Performans metrikleri
        metrics = BacktestEngine.calculate_metrics(strategy_returns, returns_aligned.fillna(0))
        
        return {
            'strategy_returns': strategy_returns,
            'cumulative_strategy': cumulative_strategy,
            'cumulative_benchmark': cumulative_benchmark,
            'signals': signals_aligned,
            'metrics': metrics,
            'strategy_type': strategy_type
        }
    
    @staticmethod
    def calculate_metrics(strategy_returns, benchmark_returns):
        """Performans metrikleri hesaplama"""
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
        
        # Profit Factor
        gross_profits = strategy_returns[strategy_returns > 0].sum()
        gross_losses = abs(strategy_returns[strategy_returns < 0].sum())
        metrics['Profit Factor'] = float(gross_profits / gross_losses) if gross_losses > 0 else 0
        
        return metrics
    
    @staticmethod
    def calculate_max_drawdown(returns):
        """Maximum drawdown hesaplama"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return float(drawdown.min())

# =============================================================================
# VOLATİLİTE ve KORELASYON ANALİZİ
# =============================================================================

class VolatilityAnalyzer:
    """Volatilite analizi sınıfı"""
    
    @staticmethod
    def calculate_volatility_regime(vix_series):
        """VIX'e göre volatilite rejimi belirleme"""
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
        
        return {
            'current_vix': current_vix,
            'regime': regime,
            'outlook': outlook,
            'percentile': (vix_series < current_vix).mean()
        }
    
    @staticmethod
    def calculate_vol_of_vol(vix_series, window=20):
        """Volatilitenin volatilitesi"""
        if vix_series.empty or len(vix_series) < window:
            return pd.Series(dtype=float)
        return vix_series.pct_change().rolling(window).std() * np.sqrt(252)

class CorrelationAnalyzer:
    """Korelasyon analizi sınıfı"""
    
    @staticmethod
    def calculate_correlation_matrix(assets_df):
        """Korelasyon matrisi hesaplama"""
        if assets_df.empty:
            return pd.DataFrame()
        
        returns = assets_df.pct_change().dropna()
        return returns.corr()
    
    @staticmethod
    def calculate_rolling_correlation(asset1, asset2, window=60):
        """Rolling korelasyon hesaplama"""
        if asset1.empty or asset2.empty:
            return pd.Series(dtype=float)
        
        returns1 = asset1.pct_change()
        returns2 = asset2.pct_change()
        
        return returns1.rolling(window).corr(returns2)

# =============================================================================
# SCENARIO ANALİZİ
# =============================================================================

def generate_scenarios(yield_df):
    """Farklı senaryolar için getiri eğrisi tahminleri"""
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

# =============================================================================
# GÖSTERGELER ve METRİKLER
# =============================================================================

def calculate_spreads(yield_df):
    """Yield spread'lerini hesaplama"""
    if yield_df.empty:
        return pd.DataFrame()
    
    spreads = pd.DataFrame(index=yield_df.index)
    
    if '2Y' in yield_df.columns and '10Y' in yield_df.columns:
        spreads['2s10s'] = yield_df['10Y'] - yield_df['2Y']
    
    if '3M' in yield_df.columns and '10Y' in yield_df.columns:
        spreads['3m10y'] = yield_df['10Y'] - yield_df['3M']
    
    if '5Y' in yield_df.columns and '30Y' in yield_df.columns:
        spreads['5s30s'] = yield_df['30Y'] - yield_df['5Y']
    
    return spreads

def calculate_forward_rates(yield_df):
    """Forward rate hesaplama"""
    if yield_df.empty:
        return pd.DataFrame()
    
    forwards = pd.DataFrame(index=yield_df.index)
    maturities = {'3M': 0.25, '2Y': 2, '5Y': 5, '10Y': 10, '30Y': 30}
    
    pairs = [('3M', '2Y'), ('2Y', '5Y'), ('5Y', '10Y'), ('10Y', '30Y')]
    
    for short_term, long_term in pairs:
        if short_term in yield_df.columns and long_term in yield_df.columns:
            r1 = maturities[short_term]
            r2 = maturities[long_term]
            
            try:
                forward = (((1 + yield_df[long_term] / 100) ** r2 / 
                           (1 + yield_df[short_term] / 100) ** r1) ** 
                          (1 / (r2 - r1)) - 1) * 100
                forwards[f'{short_term}→{long_term}'] = forward
            except:
                pass
    
    return forwards

def get_recession_probability(spreads_df):
    """Resesyon olasılığı hesaplama (2s10s spread'e göre)"""
    if spreads_df.empty or '2s10s' not in spreads_df.columns:
        return 0.5
    
    current_spread = spreads_df['2s10s'].iloc[-1]
    if pd.isna(current_spread):
        return 0.5
    
    # Logistic regression approximation
    prob = 1 / (1 + np.exp(-(-current_spread * 2 - 0.5)))
    return min(max(prob, 0.01), 0.99)

def identify_recessions(recession_series):
    """Resesyon dönemlerini belirleme"""
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
# GÖRSELLEŞTİRME FONKSİYONLARI
# =============================================================================

def plot_yield_curve(df, selected_date):
    """Interaktif getiri eğrisi grafiği"""
    if df.empty:
        return None
    
    available_dates = df.index
    closest_date = available_dates[available_dates <= pd.Timestamp(selected_date)].max()
    
    if pd.isnull(closest_date):
        return None
    
    maturities = {'3M': 0.25, '2Y': 2, '5Y': 5, '10Y': 10, '30Y': 30}
    
    fig = go.Figure()
    
    # Current curve
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
        title='U.S. Treasury Yield Curve',
        xaxis_title='Years to Maturity',
        yaxis_title='Yield (%)',
        template='plotly_white',
        showlegend=True,
        hovermode='x unified',
        height=500
    )
    
    return fig

def plot_spreads(spreads):
    """Spread grafiği"""
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
    fig.update_layout(
        title='Treasury Yield Spreads',
        xaxis_title='Date',
        yaxis_title='Spread (%)',
        template='plotly_white',
        height=500
    )
    
    return fig

def plot_monte_carlo(simulation_results, initial_value, horizon_days, title="Monte Carlo Simulation"):
    """Monte Carlo simülasyon grafiği"""
    fig = go.Figure()
    x_axis = np.arange(horizon_days)
    
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=simulation_results['upper_ci'],
        fill=None,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=simulation_results['lower_ci'],
        fill='tonexty',
        mode='lines',
        fillcolor='rgba(44, 95, 138, 0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name='95% Confidence Interval'
    ))
    
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=simulation_results['mean'],
        mode='lines',
        name='Mean Path',
        line=dict(color='#2c5f8a', width=2.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0],
        y=[initial_value],
        mode='markers',
        name='Current',
        marker=dict(size=12, color='#4a7c59', symbol='star')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Trading Days Ahead',
        yaxis_title='Yield (%)',
        template='plotly_white',
        height=500
    )
    
    return fig

def plot_correlation_heatmap(correlation_matrix):
    """Korelasyon ısı haritası"""
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

def plot_technical_indicators(tech_df, ticker):
    """Teknik gösterge grafikleri"""
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
        
        # MACD histogram
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
# STREAMLIT UI - TAM ENTEGRE
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
if 'recession_data' not in st.session_state:
    st.session_state.recession_data = None
if 'volatility_data' not in st.session_state:
    st.session_state.volatility_data = None
if 'correlation_data' not in st.session_state:
    st.session_state.correlation_data = None

# Title
st.title("📈 Bond Yield Curve Analysis Platform")
st.markdown("*Institutional-Grade Fixed Income Analytics | Full Feature Suite*")

# API Key Management
if not st.session_state.api_key_validated:
    st.markdown("""
    ### 🔑 FRED API Key Required
    
    Bu platform FRED (Federal Reserve Economic Data) kullanarak doğru Treasury verileri sağlar.
    
    **Ücretsiz API Key almak için:**
    1. [FRED API website](https://fred.stlouisfed.org/docs/api/api_key.html) adresine gidin
    2. "Request API Key" butonuna tıklayın
    3. Ücretsiz hesap oluşturun
    4. Email'inize gelen API key'i girin
    """)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        api_key = st.text_input("FRED API Key", type="password", placeholder="Enter your API key")
        
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
    
    st.markdown("---")
    
    # Analysis modules selection
    st.header("📊 Analysis Modules")
    show_yield_curve = st.checkbox("Yield Curve Analysis", value=True)
    show_spreads = st.checkbox("Spread Analysis", value=True)
    show_ns_model = st.checkbox("Nelson-Siegel Model", value=True)
    show_pca = st.checkbox("PCA Risk Analysis", value=True)
    show_monte_carlo = st.checkbox("Monte Carlo Simulation", value=True)
    show_ml = st.checkbox("Machine Learning Forecast", value=True)
    show_backtest = st.checkbox("Strategy Backtest", value=True)
    show_volatility = st.checkbox("Volatility Analysis", value=True)
    show_correlation = st.checkbox("Correlation Analysis", value=True)
    show_technical = st.checkbox("Technical Analysis", value=True)
    show_scenarios = st.checkbox("Scenario Analysis", value=True)
    
    st.markdown("---")
    
    # Strategy parameters
    if show_backtest:
        st.header("🎯 Strategy Parameters")
        selected_etf = st.selectbox("ETF for Backtest", list(BOND_ETFS.keys()))
        strategy_type = st.selectbox("Strategy Type", 
                                     ['Curve Inversion', 'Momentum', 'Mean Reversion'])
        transaction_cost = st.slider("Transaction Cost (%)", 0.0, 0.5, 0.1, 0.01) / 100
    
    if show_monte_carlo:
        st.header("🎲 Monte Carlo Parameters")
        mc_simulations = st.slider("Number of Simulations", 500, 5000, 1000, 500)
        mc_horizon = st.slider("Forecast Horizon (days)", 5, 252, 20, 5)
        mc_confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
    
    if show_ml:
        st.header("🤖 ML Parameters")
        ml_model = st.selectbox("Model Type", ["Random Forest", "Gradient Boosting"])
        ml_lags = st.slider("Number of Lags", 3, 21, 5)
    
    st.markdown("---")
    
    run_analysis = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

# Main content
if run_analysis:
    with st.spinner("Fetching all market data..."):
        
        # Fetch yield data from FRED
        yield_df = fetch_all_yield_data(st.session_state.api_key, start_date, end_date)
        
        if yield_df.empty:
            st.error("Failed to fetch yield data. Please check your API key and date range.")
            if st.button("Reset API Key"):
                st.session_state.api_key_validated = False
                st.rerun()
            st.stop()
        
        # Fetch recession data
        recession_series = fetch_recession_data(st.session_state.api_key, start_date, end_date)
        
        # Fetch market data
        volatility_df, correlation_df = fetch_market_bundle(start_date, end_date)
        
        # Calculate spreads and forwards
        spreads = calculate_spreads(yield_df)
        forwards = calculate_forward_rates(yield_df)
        factors = calculate_factor_contributions(yield_df)
        recessions = identify_recessions(recession_series)
        
        # Current metrics
        current_10y = yield_df['10Y'].iloc[-1] if '10Y' in yield_df.columns else np.nan
        current_2y = yield_df['2Y'].iloc[-1] if '2Y' in yield_df.columns else np.nan
        current_spread = spreads['2s10s'].iloc[-1] if '2s10s' in spreads.columns else np.nan
        recession_prob = get_recession_probability(spreads)
        
        # Nelson-Siegel model
        maturities = np.array([0.25, 2, 5, 10, 30])
        current_yields = np.array([yield_df[m].iloc[-1] for m in ['3M', '2Y', '5Y', '10Y', '30Y']])
        ns_result = NelsonSiegelModel.fit_curve(maturities, current_yields) if show_ns_model else None
        
        # PCA analysis
        pca_result = perform_pca_analysis(yield_df) if show_pca else None
        
        # Success message
        st.success(f"✅ Data loaded successfully! Period: {start_date} to {end_date}")
        st.info(f"📊 Data points: {len(yield_df)} trading days | Maturities: {', '.join(yield_df.columns)}")
        
        # KPI Row
        st.subheader("📊 Current Market Overview")
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
        
        with kpi_col1:
            st.metric("10Y Yield", f"{current_10y:.2f}%" if not np.isnan(current_10y) else "N/A")
        
        with kpi_col2:
            st.metric("2Y Yield", f"{current_2y:.2f}%" if not np.isnan(current_2y) else "N/A")
        
        with kpi_col3:
            delta_color = "inverse" if current_spread < 0 else "normal"
            st.metric("2s10s Spread", f"{current_spread:.2f}%" if not np.isnan(current_spread) else "N/A",
                     delta="Inverted" if current_spread < 0 else "Normal",
                     delta_color=delta_color)
        
        with kpi_col4:
            st.metric("Recession Probability", f"{recession_prob:.1%}")
        
        with kpi_col5:
            vix_current = volatility_df['CBOE Volatility Index'].iloc[-1] if not volatility_df.empty else np.nan
            st.metric("VIX", f"{vix_current:.2f}" if not np.isnan(vix_current) else "N/A")
        
        # Recession warning
        if current_spread < 0:
            st.warning("⚠️ **YIELD CURVE IS INVERTED!** Historically signals recession within 6-18 months. Consider defensive positioning.")
        
        # Tabs for different analyses
        tabs_list = []
        if show_yield_curve:
            tabs_list.append("📈 Yield Curve")
        if show_spreads:
            tabs_list.append("📊 Spread Analysis")
        if show_ns_model:
            tabs_list.append("📐 Nelson-Siegel")
        if show_pca:
            tabs_list.append("📉 PCA & Factors")
        if show_monte_carlo:
            tabs_list.append("🎲 Monte Carlo")
        if show_ml:
            tabs_list.append("🤖 ML Forecast")
        if show_backtest:
            tabs_list.append("🎯 Backtest")
        if show_volatility:
            tabs_list.append("⚡ Volatility")
        if show_correlation:
            tabs_list.append("🔄 Correlation")
        if show_technical:
            tabs_list.append("🛠 Technical")
        if show_scenarios:
            tabs_list.append("🎭 Scenarios")
        
        if not tabs_list:
            tabs_list = ["📈 Yield Curve"]
        
        main_tabs = st.tabs(tabs_list)
        tab_idx = 0
        
        # Tab 1: Yield Curve
        if show_yield_curve:
            with main_tabs[tab_idx]:
                st.subheader("Yield Curve Visualization")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    fig_yield = plot_yield_curve(yield_df, end_date)
                    if fig_yield:
                        st.plotly_chart(fig_yield, use_container_width=True)
                
                with col2:
                    st.markdown("### Curve Statistics")
                    latest = yield_df.iloc[-1]
                    st.metric("Curve Steepness (10Y-2Y)", f"{latest.get('10Y', 0) - latest.get('2Y', 0):.2f}%")
                    st.metric("Short End (3M)", f"{latest.get('3M', 0):.2f}%")
                    st.metric("Long End (30Y)", f"{latest.get('30Y', 0):.2f}%")
                    
                    if len(yield_df) > 1:
                        changes = yield_df.iloc[-1] - yield_df.iloc[-2]
                        st.metric("10Y Daily Change", f"{changes.get('10Y', 0):+.2f}%")
            
            tab_idx += 1
        
        # Tab 2: Spread Analysis
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
                        st.markdown("### Spread Statistics")
                        st.dataframe(spreads.describe().round(2), use_container_width=True)
                    
                    # Forward rates
                    if not forwards.empty:
                        st.markdown("### Forward Rates")
                        st.line_chart(forwards)
                else:
                    st.info("No spread data available")
            
            tab_idx += 1
        
        # Tab 3: Nelson-Siegel Model
        if show_ns_model and ns_result:
            with main_tabs[tab_idx]:
                st.subheader("Nelson-Siegel Yield Curve Model")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Model Parameters")
                    params_df = pd.DataFrame({
                        'Parameter': ['β₀ (Level)', 'β₁ (Slope)', 'β₂ (Curvature)', 'λ (Decay)'],
                        'Value': [f"{ns_result['params'][0]:.4f}", 
                                 f"{ns_result['params'][1]:.4f}",
                                 f"{ns_result['params'][2]:.4f}",
                                 f"{ns_result['params'][3]:.4f}"]
                    })
                    st.dataframe(params_df, hide_index=True, use_container_width=True)
                    
                    st.markdown("### Fit Quality")
                    st.metric("RMSE", f"{ns_result['rmse']*100:.2f} bps")
                    st.metric("R²", f"{ns_result['r2']:.4f}")
                
                with col2:
                    # Plot fitted curve
                    fig_ns = go.Figure()
                    fig_ns.add_trace(go.Scatter(
                        x=maturities,
                        y=current_yields,
                        mode='markers',
                        name='Actual Yields',
                        marker=dict(size=12, color='#2c5f8a')
                    ))
                    fig_ns.add_trace(go.Scatter(
                        x=np.linspace(0.25, 30, 100),
                        y=NelsonSiegelModel.nelson_siegel(np.linspace(0.25, 30, 100), *ns_result['params']),
                        mode='lines',
                        name='NS Fitted Curve',
                        line=dict(color='#c17f3a', width=2)
                    ))
                    fig_ns.update_layout(
                        title='Nelson-Siegel Model Fit',
                        xaxis_title='Maturity (Years)',
                        yaxis_title='Yield (%)',
                        template='plotly_white',
                        height=400
                    )
                    st.plotly_chart(fig_ns, use_container_width=True)
            
            tab_idx += 1
        
        # Tab 4: PCA & Factors
        if show_pca:
            with main_tabs[tab_idx]:
                st.subheader("Factor Analysis & PCA")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Factor Contributions")
                    if not factors.empty:
                        fig_factors = go.Figure()
                        for col in factors.columns:
                            fig_factors.add_trace(go.Scatter(
                                x=factors.index,
                                y=factors[col],
                                name=col,
                                line=dict(width=2)
                            ))
                        fig_factors.update_layout(
                            title='Level, Slope & Curvature',
                            xaxis_title='Date',
                            yaxis_title='Value',
                            template='plotly_white',
                            height=400
                        )
                        st.plotly_chart(fig_factors, use_container_width=True)
                
                with col2:
                    if pca_result:
                        st.markdown("### PCA Explained Variance")
                        fig_pca = go.Figure(data=go.Bar(
                            x=[f'PC{i+1}' for i in range(pca_result['n_components'])],
                            y=pca_result['explained_variance'] * 100,
                            marker_color='#2c5f8a'
                        ))
                        fig_pca.update_layout(
                            title='Variance Explained by Component',
                            xaxis_title='Principal Component',
                            yaxis_title='Explained Variance (%)',
                            template='plotly_white',
                            height=400
                        )
                        st.plotly_chart(fig_pca, use_container_width=True)
                        
                        st.markdown("### Factor Loadings")
                        st.dataframe(pca_result['loadings'].round(3), use_container_width=True)
            
            tab_idx += 1
        
        # Tab 5: Monte Carlo
        if show_monte_carlo:
            with main_tabs[tab_idx]:
                st.subheader("Monte Carlo Simulation")
                
                if st.button("Run Monte Carlo Simulation", use_container_width=True):
                    with st.spinner(f"Running {mc_simulations} simulations..."):
                        initial_yield = current_10y if not np.isnan(current_10y) else 4.0
                        
                        # Calculate historical parameters
                        returns = yield_df['10Y'].pct_change().dropna()
                        mu = returns.mean() * 252
                        sigma = returns.std() * np.sqrt(252)
                        
                        # Run simulation
                        paths = MonteCarloSimulator.simulate_gbm(
                            initial_yield, mu, sigma, mc_horizon, mc_simulations
                        )
                        sim_results = MonteCarloSimulator.calculate_confidence_intervals(paths, mc_confidence)
                        var_estimate = MonteCarloSimulator.calculate_var(paths, mc_confidence)
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Expected Terminal Value", f"{sim_results['mean'][-1]:.2f}%")
                        with col2:
                            st.metric(f"VaR ({mc_confidence*100:.0f}%)", f"{var_estimate:.2f}%")
                        with col3:
                            st.metric("Simulation Paths", f"{mc_simulations:,}")
                        
                        # Plot results
                        fig_mc = plot_monte_carlo(sim_results, initial_yield, mc_horizon, 
                                                  f"10Y Yield Simulation - {mc_simulations} paths")
                        st.plotly_chart(fig_mc, use_container_width=True)
                        
                        # Distribution plot
                        fig_dist = go.Figure(data=go.Histogram(
                            x=paths[:, -1],
                            nbinsx=50,
                            marker_color='#2c5f8a',
                            opacity=0.7
                        ))
                        fig_dist.add_vline(x=sim_results['mean'][-1], line_color='red', line_dash='dash',
                                          annotation_text="Mean")
                        fig_dist.update_layout(
                            title='Terminal Value Distribution',
                            xaxis_title='Yield (%)',
                            yaxis_title='Frequency',
                            template='plotly_white',
                            height=400
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
            
            tab_idx += 1
        
        # Tab 6: ML Forecast
        if show_ml:
            with main_tabs[tab_idx]:
                st.subheader(f"Machine Learning Forecast - {ml_model}")
                
                if st.button("Train ML Model", use_container_width=True):
                    with st.spinner(f"Training {ml_model} model..."):
                        X, y, scaler = MLForecastModel.prepare_features(yield_df, '10Y', ml_lags)
                        
                        if X is not None and len(X) > 50:
                            ml_results = MLForecastModel.train_model(X, y, ml_model)
                            
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
                            st.warning(f"Insufficient data for ML training. Need at least 50 samples, have {len(X) if X is not None else 0}")
            
            tab_idx += 1
        
        # Tab 7: Backtest
        if show_backtest:
            with main_tabs[tab_idx]:
                st.subheader(f"Strategy Backtest - {strategy_type} on {selected_etf}")
                
                # Fetch ETF data
                etf_data = fetch_yahoo_data(selected_etf, start_date, end_date)
                
                if not etf_data.empty:
                    etf_returns = etf_data.pct_change()
                    
                    # Run backtest
                    backtest_result = BacktestEngine.backtest_strategy(
                        yield_df, spreads, etf_returns, strategy_type
                    )
                    
                    if backtest_result:
                        metrics = backtest_result['metrics']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Strategy Return", f"{metrics['Total Return Strategy']:.2%}")
                            st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
                        with col2:
                            st.metric("Benchmark Return", f"{metrics['Total Return Benchmark']:.2%}")
                            st.metric("Volatility", f"{metrics['Volatility']:.2%}")
                        with col3:
                            st.metric("Excess Return", f"{metrics['Excess Return']:.2%}")
                            st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
                        with col4:
                            st.metric("Win Rate", f"{metrics['Win Rate']:.2%}")
                            st.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")
                        
                        # Plot cumulative returns
                        fig_backtest = go.Figure()
                        fig_backtest.add_trace(go.Scatter(
                            x=backtest_result['cumulative_strategy'].index,
                            y=backtest_result['cumulative_strategy'].values,
                            name='Strategy',
                            line=dict(color='#2c5f8a', width=2)
                        ))
                        fig_backtest.add_trace(go.Scatter(
                            x=backtest_result['cumulative_benchmark'].index,
                            y=backtest_result['cumulative_benchmark'].values,
                            name='Benchmark',
                            line=dict(color='#c17f3a', width=2, dash='dash')
                        ))
                        fig_backtest.update_layout(
                            title='Cumulative Returns',
                            xaxis_title='Date',
                            yaxis_title='Return',
                            template='plotly_white',
                            height=400
                        )
                        st.plotly_chart(fig_backtest, use_container_width=True)
                        
                        # Signal distribution
                        st.markdown("### Signal Distribution")
                        signal_counts = backtest_result['signals'].value_counts()
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Buy Signals", signal_counts.get(1, 0))
                        with col2:
                            st.metric("Sell Signals", signal_counts.get(0, 0))
                        with col3:
                            st.metric("Hold Periods", signal_counts.get(-1, 0))
                    else:
                        st.warning("Backtest failed. Try different parameters.")
                else:
                    st.error(f"Could not fetch {selected_etf} data")
            
            tab_idx += 1
        
        # Tab 8: Volatility
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
                        st.metric("VIX Percentile", f"{vix_analysis['percentile']:.1%}")
                    
                    with col2:
                        fig_vix = go.Figure()
                        fig_vix.add_trace(go.Scatter(
                            x=volatility_df.index,
                            y=volatility_df['CBOE Volatility Index'],
                            mode='lines',
                            name='VIX',
                            line=dict(color='#c17f3a', width=2),
                            fill='tozeroy'
                        ))
                        fig_vix.add_hline(y=20, line_dash='dash', line_color='red', annotation_text="Elevated")
                        fig_vix.add_hline(y=15, line_dash='dash', line_color='orange', annotation_text="Normal")
                        fig_vix.update_layout(
                            title='VIX Historical Chart',
                            xaxis_title='Date',
                            yaxis_title='VIX',
                            template='plotly_white',
                            height=400
                        )
                        st.plotly_chart(fig_vix, use_container_width=True)
                    
                    # Vol of Vol
                    vol_of_vol = VolatilityAnalyzer.calculate_vol_of_vol(volatility_df['CBOE Volatility Index'])
                    if not vol_of_vol.empty:
                        st.subheader("Volatility of Volatility")
                        st.line_chart(vol_of_vol)
                else:
                    st.info("Volatility data not available")
            
            tab_idx += 1
        
        # Tab 9: Correlation
        if show_correlation:
            with main_tabs[tab_idx]:
                st.subheader("Cross-Asset Correlation Analysis")
                
                if not correlation_df.empty and not yield_df.empty:
                    # Combine assets
                    all_assets = pd.DataFrame(index=yield_df.index)
                    all_assets['10Y Yield'] = yield_df['10Y']
                    
                    for col in correlation_df.columns:
                        all_assets[col] = correlation_df[col]
                    
                    all_assets = all_assets.dropna()
                    
                    if not all_assets.empty:
                        corr_matrix = CorrelationAnalyzer.calculate_correlation_matrix(all_assets)
                        fig_corr = plot_correlation_heatmap(corr_matrix)
                        if fig_corr:
                            st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Rolling correlation with 10Y
                        st.subheader("Rolling Correlation with 10Y Yield")
                        fig_rolling = go.Figure()
                        for col in correlation_df.columns:
                            if not correlation_df[col].empty:
                                rolling_corr = CorrelationAnalyzer.calculate_rolling_correlation(
                                    yield_df['10Y'], correlation_df[col], 60
                                )
                                if not rolling_corr.empty:
                                    fig_rolling.add_trace(go.Scatter(
                                        x=rolling_corr.index,
                                        y=rolling_corr.values,
                                        name=col,
                                        line=dict(width=2)
                                    ))
                        
                        fig_rolling.update_layout(
                            title='60-Day Rolling Correlation with 10Y Treasury',
                            xaxis_title='Date',
                            yaxis_title='Correlation',
                            template='plotly_white',
                            height=500
                        )
                        st.plotly_chart(fig_rolling, use_container_width=True)
            
            tab_idx += 1
        
        # Tab 10: Technical Analysis
        if show_technical:
            with main_tabs[tab_idx]:
                st.subheader("Technical Analysis")
                
                selected_ticker = st.selectbox("Select Asset", ['TLT', 'IEF', 'SHY', 'SPY', 'QQQ'])
                
                price_data = fetch_yahoo_data(selected_ticker, start_date, end_date)
                
                if not price_data.empty:
                    tech_df = calculate_technical_indicators(price_data)
                    fig_tech = plot_technical_indicators(tech_df, selected_ticker)
                    if fig_tech:
                        st.plotly_chart(fig_tech, use_container_width=True)
                    
                    # Current technical signals
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
                    st.error(f"Could not fetch {selected_ticker} data")
            
            tab_idx += 1
        
        # Tab 11: Scenarios
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
                        # Visualize scenario
                        fig_scenario = go.Figure()
                        maturities_list = [0.25, 2, 5, 10, 30]
                        current_vals = [current['3M'], current['2Y'], current['5Y'], current['10Y'], current['30Y']]
                        scenario_vals = [scenario_data['3M'], scenario_data['2Y'], scenario_data['5Y'], 
                                        scenario_data['10Y'], scenario_data['30Y']]
                        
                        fig_scenario.add_trace(go.Scatter(
                            x=maturities_list,
                            y=current_vals,
                            name='Current Curve',
                            line=dict(color='#2c5f8a', width=3)
                        ))
                        fig_scenario.add_trace(go.Scatter(
                            x=maturities_list,
                            y=scenario_vals,
                            name=selected_scenario,
                            line=dict(color='#c17f3a', width=2, dash='dash')
                        ))
                        fig_scenario.update_layout(
                            title=f'Yield Curve Scenario: {selected_scenario}',
                            xaxis_title='Maturity (Years)',
                            yaxis_title='Yield (%)',
                            template='plotly_white',
                            height=400
                        )
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
    # Initial state
    st.info("👈 Configure your analysis parameters and click 'Run Full Analysis' to start")
    
    with st.expander("📖 Platform Features - Complete Suite", expanded=True):
        st.markdown("""
        ### 🚀 Full Feature Suite
        
        This institutional-grade platform includes:
        
        #### 1. 📈 **Yield Curve Analysis**
        - Interactive curve visualization
        - Historical comparison (1 year ago)
        - Daily changes and statistics
        
        #### 2. 📊 **Spread Analysis**
        - 2s10s (key recession indicator)
        - 3m10y and 5s30s spreads
        - Forward rate calculations
        
        #### 3. 📐 **Nelson-Siegel Model**
        - Parametric curve fitting
        - Level, slope, curvature parameters
        - Model fit quality metrics
        
        #### 4. 📉 **PCA & Factor Analysis**
        - Level, slope, curvature factors
        - Principal component analysis
        - Risk factor decomposition
        
        #### 5. 🎲 **Monte Carlo Simulation**
        - GBM and Vasicek models
        - Confidence intervals and VaR
        - Distribution analysis
        
        #### 6. 🤖 **Machine Learning Forecast**
        - Random Forest & Gradient Boosting
        - Feature importance analysis
        - Yield prediction models
        
        #### 7. 🎯 **Strategy Backtest**
        - Multiple strategy types
        - Transaction costs modeling
        - Performance metrics (Sharpe, Drawdown)
        
        #### 8. ⚡ **Volatility Analysis**
        - VIX regime detection
        - Volatility of volatility
        - Market stress indicators
        
        #### 9. 🔄 **Correlation Analysis**
        - Cross-asset correlation matrix
        - Rolling correlations
        - Asset relationships
        
        #### 10. 🛠 **Technical Analysis**
        - RSI, MACD, Bollinger Bands
        - Moving averages
        - Technical signals
        
        #### 11. 🎭 **Scenario Analysis**
        - Bull/Bear scenarios
        - Recession simulation
        - Inflation shock testing
        
        ### 📊 Data Sources
        - **FRED API**: Official Federal Reserve data
        - **Yahoo Finance**: Market and ETF data
        
        ### ⚠️ Risk Warning
        Past performance does not guarantee future results. This platform is for educational purposes only.
        """)

# Footer
st.markdown("---")
st.markdown(f"*Data source: FRED & Yahoo Finance | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
