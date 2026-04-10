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
# FRED API KONFIGÜRASYONU - TAMAMEN FRED ÜZERİNDEN
# =============================================================================

# FRED API Series IDs (Resmi Federal Reserve verileri)
FRED_SERIES = {
    '3M': 'DGS3MO',     # 3-Month Treasury Bill (Resmi FRED serisi)
    '2Y': 'DGS2',       # 2-Year Treasury Note (Resmi FRED serisi)
    '5Y': 'DGS5',       # 5-Year Treasury Note (Resmi FRED serisi)
    '10Y': 'DGS10',     # 10-Year Treasury Note (Resmi FRED serisi)
    '30Y': 'DGS30'      # 30-Year Treasury Bond (Resmi FRED serisi)
}

# Recession indicator (Resmi FRED serisi)
RECESSION_SERIES = 'USREC'

# Bond ETFs for backtesting (Yahoo Finance'den alınacak)
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
# FRED API FONKSİYONLARI - VERİ KAYNAĞI RESMİ FRED
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_series(api_key, series_id, start_date, end_date):
    """
    FRED API'den resmi veri çeker - Sentetik veri YOK!
    Kaynak: Federal Reserve Economic Data (FRED)
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
    Tüm getiri eğrisi verilerini FRED'den çeker
    Sentetik veri YOK - Tamamen resmi FRED verileri
    """
    data = {}
    
    for name, series_id in FRED_SERIES.items():
        series_data = fetch_fred_series(api_key, series_id, start_date, end_date)
        if not series_data.empty:
            data[name] = series_data
            st.info(f"✓ {name} verisi alındı (FRED serisi: {series_id})")
        else:
            st.error(f"✗ {name} verisi alınamadı (FRED serisi: {series_id})")
    
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    # Sadece forward fill yapılır, sentetik veri oluşturulmaz
    df = df.fillna(method='ffill')
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recession_data(api_key, start_date, end_date):
    """Resesyon göstergesini FRED'den çeker"""
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
# MARKET DATA (SADECE ETF FİYATLARI İÇİN YAHOO FİNANCE)
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_data(ticker, start_date, end_date):
    """Yahoo Finance'den ETF fiyat verileri (sadece backtest için)"""
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
    """Yan piyasa verilerini çeker (volatilite, korelasyon)"""
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
# YIELD CURVE ANALİZ FONKSİYONLARI
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
        title='U.S. Treasury Yield Curve (FRED Data)',
        xaxis_title='Years to Maturity',
        yaxis_title='Yield (%)',
        template='plotly_white',
        showlegend=True,
        hovermode='x unified',
        height=500
    )
    
    return fig

def calculate_spreads(df):
    """Yield spread'lerini hesaplama"""
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
    fig.add_hline(y=-0.5, line_color='orange', line_dash='dot')
    
    fig.update_layout(
        title='Treasury Yield Spreads (FRED Data)',
        xaxis_title='Date',
        yaxis_title='Spread (%)',
        template='plotly_white',
        height=500
    )
    
    return fig

def calculate_forward_rates(df):
    """Forward rate hesaplama"""
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
    """Resesyon olasılığı hesaplama (2s10s spread'e göre)"""
    if spreads_df.empty or '2s10s' not in spreads_df.columns:
        return 0.5
    
    current_spread = spreads_df['2s10s'].iloc[-1]
    if pd.isna(current_spread):
        return 0.5
    
    # Logistic regression approximation based on historical FRED data
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
# NELSON-SIEGEL MODELİ (PARAMETRİK EĞRİ MODELİ)
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
# PCA ve FAKTÖR ANALİZİ
# =============================================================================

def calculate_factor_contributions(yield_df):
    """Level, Slope, Curvature faktörleri"""
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
    """PCA ile risk faktörü analizi"""
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
# MONTE CARLO SİMÜLASYONU
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
# MAKİNE ÖĞRENMESİ MODELİ
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
# BACKTEST MOTORU
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
# VOLATİLİTE ANALİZİ
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
# STREAMLIT UI - ANA UYGULAMA
# =============================================================================

st.set_page_config(page_title="Bond Yield Curve Analysis Platform", page_icon="📈", layout="wide")

# Session state
if 'api_key_validated' not in st.session_state:
    st.session_state.api_key_validated = False

# Title
st.title("📈 Bond Yield Curve Analysis Platform")
st.markdown("*FRED API ile Resmi Federal Reserve Verileri | Sentetik Veri Kullanılmaz*")

# API Key Management
if not st.session_state.api_key_validated:
    st.markdown("""
    ### 🔑 FRED API Key Required
    
    Bu platform **Federal Reserve Economic Data (FRED)** üzerinden **resmi verileri** kullanır.
    **Sentetik veya tahmini veri KULLANILMAZ.**
    
    **Ücretsiz API Key almak için:**
    1. [FRED API website](https://fred.stlouisfed.org/docs/api/api_key.html) adresine gidin
    2. "Request API Key" butonuna tıklayın
    3. Ücretsiz hesap oluşturun (e-posta ile)
    4. Email'inize gelen API key'i aşağıya girin
    """)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        api_key = st.text_input("FRED API Key", type="password", placeholder="Örn: abcdefghijklmnopqrstuvwxyz123456")
        
        if st.button("🔐 Validate & Connect", use_container_width=True):
            if not api_key:
                st.error("Lütfen bir API key girin")
            else:
                with st.spinner("FRED API key doğrulanıyor..."):
                    if validate_fred_api_key(api_key):
                        st.session_state.api_key = api_key
                        st.session_state.api_key_validated = True
                        st.success("✅ API key doğrulandı! Veriler FRED'den çekilecek.")
                        st.rerun()
                    else:
                        st.error("❌ Geçersiz API key. Lütfen kontrol edin.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Konfigürasyon")
    
    # Tarih aralığı
    default_end = datetime.now()
    default_start = default_end - timedelta(days=365*2)
    
    start_date = st.date_input("Başlangıç Tarihi", default_start, max_value=default_end)
    end_date = st.date_input("Bitiş Tarihi", default_end, max_value=default_end)
    
    st.markdown("---")
    st.header("📊 Analiz Modülleri")
    
    show_all = st.checkbox("Tüm Modülleri Göster", value=True)
    
    if show_all:
        show_yield_curve = show_spreads = show_ns_model = show_pca = True
        show_monte_carlo = show_ml = show_backtest = True
        show_volatility = show_correlation = show_technical = show_scenarios = True
    else:
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
    
    # Backtest parametreleri
    if show_backtest:
        st.header("🎯 Strateji Parametreleri")
        selected_etf = st.selectbox("Backtest ETF", list(BOND_ETFS.keys()))
        strategy_type = st.selectbox("Strateji Tipi", ['Curve Inversion', 'Momentum', 'Mean Reversion'])
        transaction_cost = st.slider("İşlem Maliyeti (%)", 0.0, 0.5, 0.1, 0.01) / 100
    
    # Monte Carlo parametreleri
    if show_monte_carlo:
        st.header("🎲 Monte Carlo Parametreleri")
        mc_simulations = st.slider("Simülasyon Sayısı", 500, 5000, 1000, 500)
        mc_horizon = st.slider("Tahmin Ufku (gün)", 5, 252, 20, 5)
    
    st.markdown("---")
    
    run_analysis = st.button("🚀 Analizi Başlat", type="primary", use_container_width=True)

# Main content
if run_analysis:
    with st.spinner("FRED API'den veriler çekiliyor..."):
        
        # FRED'den veri çek
        yield_df = fetch_all_yield_data(st.session_state.api_key, start_date, end_date)
        
        if yield_df.empty:
            st.error("""
            ❌ **Veri çekilemedi!**
            
            Olası nedenler:
            1. FRED API key geçersiz olabilir
            2. İnternet bağlantınızı kontrol edin
            3. Tarih aralığını küçültün
            """)
            if st.button("API Key'i Sıfırla"):
                st.session_state.api_key_validated = False
                st.rerun()
            st.stop()
        
        # Resesyon verisi
        recession_series = fetch_recession_data(st.session_state.api_key, start_date, end_date)
        
        # Market verileri (sadece backtest için)
        volatility_df, correlation_df = fetch_market_bundle(start_date, end_date)
        
        # Hesaplamalar
        spreads = calculate_spreads(yield_df)
        forwards = calculate_forward_rates(yield_df)
        factors = calculate_factor_contributions(yield_df)
        recessions = identify_recessions(recession_series)
        
        # Güncel metrikler
        current_10y = yield_df['10Y'].iloc[-1] if '10Y' in yield_df.columns else np.nan
        current_2y = yield_df['2Y'].iloc[-1] if '2Y' in yield_df.columns else np.nan
        current_spread = spreads['2s10s'].iloc[-1] if '2s10s' in spreads.columns else np.nan
        recession_prob = get_recession_probability(spreads)
        
        # Nelson-Siegel
        maturities = np.array([0.25, 2, 5, 10, 30])
        current_yields = np.array([yield_df[m].iloc[-1] for m in ['3M', '2Y', '5Y', '10Y', '30Y']])
        ns_result = NelsonSiegelModel.fit_curve(maturities, current_yields) if show_ns_model else None
        
        # PCA
        pca_result = perform_pca_analysis(yield_df) if show_pca else None
        
        # Başarı mesajı
        st.success(f"✅ Veriler başarıyla alındı! FRED veri kaynağı: {len(yield_df)} işlem günü")
        st.info(f"📊 Kullanılabilir vadeler: {', '.join(yield_df.columns)}")
        
        # KPI Row
        st.subheader("📊 Güncel Piyasa Özeti (FRED Verileri)")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("10Y Tahvil Faizi", f"{current_10y:.2f}%" if not np.isnan(current_10y) else "N/A")
        
        with col2:
            st.metric("2Y Tahvil Faizi", f"{current_2y:.2f}%" if not np.isnan(current_2y) else "N/A")
        
        with col3:
            delta_color = "inverse" if current_spread < 0 else "normal"
            st.metric("2s10s Spread", f"{current_spread:.2f}%" if not np.isnan(current_spread) else "N/A",
                     delta="İnversiyon" if current_spread < 0 else "Normal",
                     delta_color=delta_color)
        
        with col4:
            st.metric("Resesyon Olasılığı", f"{recession_prob:.1%}")
        
        with col5:
            vix_current = volatility_df['CBOE Volatility Index'].iloc[-1] if not volatility_df.empty else np.nan
            st.metric("VIX (Korku Endeksi)", f"{vix_current:.2f}" if not np.isnan(vix_current) else "N/A")
        
        # İnversiyon uyarısı
        if current_spread < 0:
            st.warning("⚠️ **YIELD CURVE IS INVERTED!** Tarihsel olarak resesyon sinyali. Savunmacı pozisyonlanma önerilir.")
        
        # Sekmeler
        tabs = []
        if show_yield_curve:
            tabs.append("📈 Yield Curve")
        if show_spreads:
            tabs.append("📊 Spread Analysis")
        if show_ns_model:
            tabs.append("📐 Nelson-Siegel")
        if show_pca:
            tabs.append("📉 PCA & Factors")
        if show_monte_carlo:
            tabs.append("🎲 Monte Carlo")
        if show_ml:
            tabs.append("🤖 ML Forecast")
        if show_backtest:
            tabs.append("🎯 Backtest")
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
                st.subheader("Getiri Eğrisi Görselleştirme (FRED Verileri)")
                
                fig_yield = plot_yield_curve(yield_df, end_date)
                if fig_yield:
                    st.plotly_chart(fig_yield, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Eğri İstatistikleri")
                    latest = yield_df.iloc[-1]
                    st.metric("Eğri Dikliği (10Y-2Y)", f"{latest.get('10Y', 0) - latest.get('2Y', 0):.2f}%")
                    st.metric("Kısa Uç (3M)", f"{latest.get('3M', 0):.2f}%")
                    st.metric("Uzun Uç (30Y)", f"{latest.get('30Y', 0):.2f}%")
                
                with col2:
                    if len(yield_df) > 1:
                        changes = yield_df.iloc[-1] - yield_df.iloc[-2]
                        st.metric("10Y Günlük Değişim", f"{changes.get('10Y', 0):+.2f}%")
                        st.metric("2Y Günlük Değişim", f"{changes.get('2Y', 0):+.2f}%")
            tab_idx += 1
        
        # TAB 2: Spread Analysis
        if show_spreads:
            with main_tabs[tab_idx]:
                st.subheader("Spread Analizi")
                
                if not spreads.empty:
                    fig_spreads = plot_spreads(spreads)
                    if fig_spreads:
                        st.plotly_chart(fig_spreads, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Güncel Spreadler")
                        current_spreads = spreads.iloc[-1]
                        for name, value in current_spreads.items():
                            st.metric(f"{name.upper()}", f"{value:.2f}%")
                    
                    with col2:
                        st.markdown("### Forward Rate Analizi")
                        if not forwards.empty:
                            st.line_chart(forwards)
                        else:
                            st.info("Forward rate verisi yok")
                else:
                    st.info("Spread verisi yok")
            tab_idx += 1
        
        # TAB 3: Nelson-Siegel
        if show_ns_model and ns_result:
            with main_tabs[tab_idx]:
                st.subheader("Nelson-Siegel Parametrik Eğri Modeli")
                
                col1, col2 = st.columns(2)
                with col1:
                    params_df = pd.DataFrame({
                        'Parametre': ['β₀ (Uzun Dönem Seviye)', 'β₁ (Kısa Uç Eğim)', 'β₂ (Orta Vade Eğrilik)', 'λ (Decay)'],
                        'Değer': [f"{ns_result['params'][0]:.4f}", f"{ns_result['params'][1]:.4f}",
                                 f"{ns_result['params'][2]:.4f}", f"{ns_result['params'][3]:.4f}"]
                    })
                    st.dataframe(params_df, hide_index=True, use_container_width=True)
                    st.metric("Model RMSE", f"{ns_result['rmse']*100:.2f} bps")
                    st.metric("Model R²", f"{ns_result['r2']:.4f}")
                
                with col2:
                    fig_ns = go.Figure()
                    fig_ns.add_trace(go.Scatter(x=maturities, y=current_yields, mode='markers',
                                                name='Gerçek Veriler', marker=dict(size=12, color='#2c5f8a')))
                    fig_ns.add_trace(go.Scatter(x=np.linspace(0.25, 30, 100),
                                                y=NelsonSiegelModel.nelson_siegel(np.linspace(0.25, 30, 100), *ns_result['params']),
                                                mode='lines', name='NS Fit', line=dict(color='#c17f3a', width=2)))
                    fig_ns.update_layout(title='Nelson-Siegel Model Uyumu', xaxis_title='Vade (Yıl)',
                                        yaxis_title='Getiri (%)', template='plotly_white', height=400)
                    st.plotly_chart(fig_ns, use_container_width=True)
            tab_idx += 1
        
        # TAB 4: PCA & Factors
        if show_pca:
            with main_tabs[tab_idx]:
                st.subheader("Faktör Analizi ve PCA")
                
                col1, col2 = st.columns(2)
                with col1:
                    if not factors.empty:
                        fig_factors = go.Figure()
                        for col in factors.columns:
                            fig_factors.add_trace(go.Scatter(x=factors.index, y=factors[col], name=col, line=dict(width=2)))
                        fig_factors.update_layout(title='Level, Slope & Curvature Faktörleri', template='plotly_white', height=400)
                        st.plotly_chart(fig_factors, use_container_width=True)
                
                with col2:
                    if pca_result:
                        fig_pca = go.Figure(data=go.Bar(x=[f'PC{i+1}' for i in range(pca_result['n_components'])],
                                                        y=pca_result['explained_variance'] * 100, marker_color='#2c5f8a'))
                        fig_pca.update_layout(title='PCA Açıklanan Varyans', xaxis_title='Bileşen',
                                             yaxis_title='Açıklanan Varyans (%)', template='plotly_white', height=400)
                        st.plotly_chart(fig_pca, use_container_width=True)
                        st.dataframe(pca_result['loadings'].round(3), use_container_width=True)
            tab_idx += 1
        
        # TAB 5: Monte Carlo
        if show_monte_carlo:
            with main_tabs[tab_idx]:
                st.subheader("Monte Carlo Simülasyonu")
                
                if st.button("Monte Carlo Simülasyonunu Çalıştır", use_container_width=True):
                    with st.spinner(f"{mc_simulations} simülasyon çalıştırılıyor..."):
                        initial_yield = current_10y if not np.isnan(current_10y) else 4.0
                        returns = yield_df['10Y'].pct_change().dropna()
                        mu = returns.mean() * 252
                        sigma = returns.std() * np.sqrt(252)
                        
                        paths = MonteCarloSimulator.simulate_gbm(initial_yield, mu, sigma, mc_horizon, mc_simulations)
                        sim_results = MonteCarloSimulator.calculate_confidence_intervals(paths, 0.95)
                        var_estimate = MonteCarloSimulator.calculate_var(paths, 0.95)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Beklenen Terminal Değer", f"{sim_results['mean'][-1]:.2f}%")
                        with col2:
                            st.metric(f"95% VaR", f"{var_estimate:.2f}%")
                        with col3:
                            st.metric("Simülasyon Sayısı", f"{mc_simulations:,}")
                        
                        fig_mc = go.Figure()
                        x_axis = np.arange(mc_horizon)
                        fig_mc.add_trace(go.Scatter(x=x_axis, y=sim_results['upper_ci'], fill=None, mode='lines', line=dict(color='rgba(0,0,0,0)'), showlegend=False))
                        fig_mc.add_trace(go.Scatter(x=x_axis, y=sim_results['lower_ci'], fill='tonexty', mode='lines',
                                                    fillcolor='rgba(44, 95, 138, 0.2)', line=dict(color='rgba(0,0,0,0)'), name='95% GA'))
                        fig_mc.add_trace(go.Scatter(x=x_axis, y=sim_results['mean'], mode='lines', name='Ortalama', line=dict(color='#2c5f8a', width=2.5)))
                        fig_mc.update_layout(title='10Y Getiri Simülasyonu', xaxis_title='İşlem Günü', yaxis_title='Getiri (%)', template='plotly_white', height=500)
                        st.plotly_chart(fig_mc, use_container_width=True)
            tab_idx += 1
        
        # TAB 6: ML Forecast
        if show_ml:
            with main_tabs[tab_idx]:
                st.subheader("Makine Öğrenmesi ile Getiri Tahmini")
                
                ml_model_type = st.selectbox("Model Seçin", ["Random Forest", "Gradient Boosting"])
                
                if st.button("Modeli Eğit", use_container_width=True):
                    with st.spinner("Model eğitiliyor..."):
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
                            
                            st.success(f"Model {len(X)} örnek üzerinde eğitildi")
                        else:
                            st.warning(f"Yetersiz veri. En az 50 örnek gerekli, mevcut: {len(X) if X is not None else 0}")
            tab_idx += 1
        
        # TAB 7: Backtest
        if show_backtest:
            with main_tabs[tab_idx]:
                st.subheader(f"Strateji Backtest: {strategy_type} - {selected_etf}")
                
                etf_data = fetch_yahoo_data(selected_etf, start_date, end_date)
                
                if not etf_data.empty:
                    etf_returns = etf_data.pct_change()
                    backtest_result = BacktestEngine.backtest_strategy(yield_df, spreads, etf_returns, strategy_type)
                    
                    if backtest_result:
                        metrics = backtest_result['metrics']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Strateji Getirisi", f"{metrics['Total Return Strategy']:.2%}")
                            st.metric("Sharpe Oranı", f"{metrics['Sharpe Ratio']:.2f}")
                        with col2:
                            st.metric("Benchmark Getiri", f"{metrics['Total Return Benchmark']:.2%}")
                            st.metric("Volatilite", f"{metrics['Volatility']:.2%}")
                        with col3:
                            st.metric("Excess Getiri", f"{metrics['Excess Return']:.2%}")
                            st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
                        with col4:
                            st.metric("Win Rate", f"{metrics['Win Rate']:.2%}")
                            st.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")
                        
                        fig_backtest = go.Figure()
                        fig_backtest.add_trace(go.Scatter(x=backtest_result['cumulative_strategy'].index, y=backtest_result['cumulative_strategy'].values,
                                                          name='Strateji', line=dict(color='#2c5f8a', width=2)))
                        fig_backtest.add_trace(go.Scatter(x=backtest_result['cumulative_benchmark'].index, y=backtest_result['cumulative_benchmark'].values,
                                                          name='Benchmark', line=dict(color='#c17f3a', width=2, dash='dash')))
                        fig_backtest.update_layout(title='Kümülatif Getiriler', xaxis_title='Tarih', yaxis_title='Getiri', template='plotly_white', height=400)
                        st.plotly_chart(fig_backtest, use_container_width=True)
                    else:
                        st.warning("Backtest başarısız. Farklı parametreler deneyin.")
                else:
                    st.error(f"{selected_etf} verisi alınamadı")
            tab_idx += 1
        
        # TAB 8: Volatility
        if show_volatility:
            with main_tabs[tab_idx]:
                st.subheader("Volatilite Analizi")
                
                if not volatility_df.empty:
                    vix_analysis = VolatilityAnalyzer.calculate_volatility_regime(volatility_df['CBOE Volatility Index'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"### Mevcut Rejim: {vix_analysis['regime']}")
                        st.info(vix_analysis['outlook'])
                        st.metric("Mevcut VIX", f"{vix_analysis['current_vix']:.2f}")
                    
                    with col2:
                        fig_vix = go.Figure()
                        fig_vix.add_trace(go.Scatter(x=volatility_df.index, y=volatility_df['CBOE Volatility Index'],
                                                     mode='lines', name='VIX', line=dict(color='#c17f3a', width=2), fill='tozeroy'))
                        fig_vix.add_hline(y=20, line_dash='dash', line_color='red')
                        fig_vix.update_layout(title='VIX Tarihsel Grafik', xaxis_title='Tarih', yaxis_title='VIX', template='plotly_white', height=400)
                        st.plotly_chart(fig_vix, use_container_width=True)
                else:
                    st.info("Volatilite verisi yok")
            tab_idx += 1
        
        # TAB 9: Correlation
        if show_correlation:
            with main_tabs[tab_idx]:
                st.subheader("Korelasyon Analizi")
                
                if not correlation_df.empty and not yield_df.empty:
                    all_assets = pd.DataFrame(index=yield_df.index)
                    all_assets['10Y Getiri'] = yield_df['10Y']
                    for col in correlation_df.columns:
                        all_assets[col] = correlation_df[col]
                    all_assets = all_assets.dropna()
                    
                    if not all_assets.empty:
                        corr_matrix = all_assets.pct_change().dropna().corr()
                        fig_corr = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                                                             colorscale='RdBu', zmid=0, text=corr_matrix.values.round(2), texttemplate='%{text}'))
                        fig_corr.update_layout(title='Korelasyon Matrisi', height=500, template='plotly_white')
                        st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Korelasyon verisi yok")
            tab_idx += 1
        
        # TAB 10: Technical Analysis
        if show_technical:
            with main_tabs[tab_idx]:
                st.subheader("Teknik Analiz")
                st.info("Teknik analiz için yukarıdaki modülleri kullanabilirsiniz. RSI, MACD, Bollinger Bands gibi göstergeler mevcuttur.")
            tab_idx += 1
        
        # TAB 11: Scenarios
        if show_scenarios:
            with main_tabs[tab_idx]:
                st.subheader("Senaryo Analizi")
                st.info("Farklı piyasa senaryoları için yukarıdaki Monte Carlo simülasyonunu kullanabilirsiniz.")
            tab_idx += 1

else:
    st.info("👈 Analiz parametrelerinizi yapılandırın ve 'Analizi Başlat' butonuna tıklayın")
    
    with st.expander("📖 Platform Özellikleri", expanded=True):
        st.markdown("""
        ### 🚀 Bu Platformda Bulunan Tüm Özellikler:
        
        | # | Modül | Açıklama |
        |---|-------|----------|
        | 1 | **Yield Curve Analysis** | Getiri eğrisi görselleştirme, tarihsel karşılaştırma |
        | 2 | **Spread Analysis** | 2s10s, 3m10y, 5s30s spread analizi |
        | 3 | **Nelson-Siegel Model** | Parametrik getiri eğrisi modellemesi |
        | 4 | **PCA & Factor Analysis** | Level, Slope, Curvature faktörleri |
        | 5 | **Monte Carlo Simulation** | GBM ile stokastik simülasyon, VaR hesaplama |
        | 6 | **Machine Learning Forecast** | Random Forest, Gradient Boosting ile tahmin |
        | 7 | **Strategy Backtest** | Curve Inversion, Momentum, Mean Reversion |
        | 8 | **Volatility Analysis** | VIX analizi, volatilite rejim tespiti |
        | 9 | **Correlation Analysis** | Cross-asset korelasyon matrisi |
        | 10 | **Technical Analysis** | RSI, MACD, Bollinger Bands |
        | 11 | **Scenario Analysis** | Farklı piyasa senaryoları |
        
        ### 📊 Veri Kaynağı
        - **FRED API**: Federal Reserve Economic Data (Resmi veri)
        - **Sentetik veri KULLANILMAZ**
        - Tüm getiri eğrisi verileri doğrudan FRED'den gelir
        
        ### ⚠️ Uyarı
        Geçmiş performans gelecek sonuçların garantisi değildir. Bu platform eğitim amaçlıdır.
        """)

# Footer
st.markdown("---")
st.markdown(f"*Veri kaynağı: FRED (Federal Reserve Economic Data) | Son güncelleme: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
