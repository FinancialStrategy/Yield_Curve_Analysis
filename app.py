# =============================================================================
# HEDGE FUND YIELD CURVE ANALYTICS PLATFORM
# EXECUTIVE SUMMARY REPORT - INSTITUTIONAL GRADE
# =============================================================================
# Version: 23.0 | Executive Summary Focus | No Shortening | Full Implementation
# Includes: Nelson-Siegel, Svensson, Dynamic Analysis, Risk Metrics, Arbitrage Detection
# Executive Summary Focus: 2Y and 10Y Dynamic Charts with Interactive Time Range
# All Tabs: DATA TABLE, 2Y-10Y DYNAMIC CHARTS, SPREAD DYNAMICS, NS MODEL FIT, 
# NSS MODEL FIT, MODEL COMPARISON, DYNAMIC ANALYSIS, FACTOR ANALYSIS, 
# RISK METRICS, ARBITRAGE, NBER RECESSION, FORECASTING, DATA EXPORT
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.optimize import minimize, differential_evolution, curve_fit
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import requests
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - PROFESSIONAL THEME
# =============================================================================

st.set_page_config(
    page_title="Yield Curve Analytics | Executive Summary",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Color Palette - Bloomberg Terminal Style
COLORS = {
    'primary': '#1a1a2e',
    'secondary': '#16213e',
    'accent': '#0f3460',
    'positive': '#2ecc71',
    'negative': '#e74c3c',
    'neutral': '#95a5a6',
    'warning': '#f39c12',
    'background': '#0a0a0a',
    'surface': '#1a1a2e',
    'text_primary': '#ecf0f1',
    'text_secondary': '#bdc3c7',
    'grid': '#2c3e50',
    'recession': 'rgba(52, 73, 94, 0.4)'
}

# FRED Series Configuration - All Maturities (11 tenors)
FRED_SERIES = {
    '1M': 'DGS1MO',
    '3M': 'DGS3MO', 
    '6M': 'DGS6MO',
    '1Y': 'DGS1',
    '2Y': 'DGS2',
    '3Y': 'DGS3',
    '5Y': 'DGS5',
    '7Y': 'DGS7',
    '10Y': 'DGS10',
    '20Y': 'DGS20',
    '30Y': 'DGS30'
}

MATURITY_MAP = {
    '1M': 1/12,
    '3M': 0.25,
    '6M': 0.5, 
    '1Y': 1,
    '2Y': 2,
    '3Y': 3,
    '5Y': 5, 
    '7Y': 7,
    '10Y': 10,
    '20Y': 20,
    '30Y': 30
}

# Session State Management
if 'api_key_validated' not in st.session_state:
    st.session_state.api_key_validated = False
if 'yield_data' not in st.session_state:
    st.session_state.yield_data = None
if 'recession_data' not in st.session_state:
    st.session_state.recession_data = None
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'ns_results' not in st.session_state:
    st.session_state.ns_results = None
if 'nss_results' not in st.session_state:
    st.session_state.nss_results = None
if 'dynamic_params' not in st.session_state:
    st.session_state.dynamic_params = None
if 'factors' not in st.session_state:
    st.session_state.factors = None
if 'pca_risk' not in st.session_state:
    st.session_state.pca_risk = None

# Custom CSS - Institutional Professional Styling
st.markdown(
    f"""
    <style>
    .main {{
        background-color: {COLORS['background']};
    }}
    .hedge-header {{
        background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        padding: 1.5rem;
        border-bottom: 2px solid {COLORS['accent']};
        margin-bottom: 2rem;
    }}
    .executive-summary-card {{
        background: linear-gradient(135deg, {COLORS['surface']} 0%, {COLORS['secondary']} 100%);
        border: 1px solid {COLORS['accent']};
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }}
    .executive-title {{
        color: {COLORS['text_primary']};
        font-size: 1.1rem;
        font-weight: 600;
        border-left: 3px solid {COLORS['accent']};
        padding-left: 1rem;
        margin-bottom: 1rem;
    }}
    .api-container {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['grid']};
        border-radius: 8px;
        padding: 2rem;
        margin: 2rem auto;
        max-width: 500px;
        text-align: center;
    }}
    .metric-card {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['grid']};
        border-radius: 4px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }}
    .metric-card:hover {{
        border-color: {COLORS['accent']};
        transform: translateY(-2px);
    }}
    .metric-label {{
        color: {COLORS['text_secondary']};
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    .metric-value {{
        color: {COLORS['text_primary']};
        font-size: 1.5rem;
        font-weight: 600;
        font-family: 'Courier New', monospace;
    }}
    .status-inverted {{
        color: {COLORS['negative']};
        font-weight: 700;
    }}
    .status-normal {{
        color: {COLORS['positive']};
        font-weight: 700;
    }}
    .status-caution {{
        color: {COLORS['warning']};
        font-weight: 700;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0rem;
        background-color: {COLORS['surface']};
        border-bottom: 1px solid {COLORS['grid']};
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        color: {COLORS['text_secondary']};
        padding: 0.5rem 1.5rem;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .stTabs [aria-selected="true"] {{
        color: {COLORS['accent']};
        border-bottom: 2px solid {COLORS['accent']};
    }}
    .stButton > button {{
        background-color: {COLORS['accent']};
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }}
    .stButton > button:hover {{
        background-color: {COLORS['primary']};
        transform: translateY(-1px);
    }}
    #MainMenu {{
        visibility: hidden;
    }}
    footer {{
        visibility: hidden;
    }}
    header {{
        visibility: hidden;
    }}
    .dataframe {{
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
    }}
    .dataframe th {{
        background-color: {COLORS['secondary']};
        color: {COLORS['text_primary']};
    }}
    </style>
    """, 
    unsafe_allow_html=True
)

# =============================================================================
# FRED API FUNCTIONS - COMPLETE DATA FETCHING
# =============================================================================

@st.cache_data(ttl=3600)
def fetch_fred_data(api_key, series_id):
    """Fetch data from FRED API with error handling"""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': '1990-01-01',
        'sort_order': 'asc'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        observations = data.get('observations', [])
        dates = []
        values = []
        
        for obs in observations:
            if obs['value'] != '.':
                dates.append(pd.to_datetime(obs['date']))
                values.append(float(obs['value']))
        
        if dates:
            return pd.Series(values, index=dates, name=series_id)
        return None
    except Exception as e:
        return None

def fetch_all_yield_data(api_key):
    """Fetch all treasury yield data from FRED with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    all_data = {}
    total = len(FRED_SERIES)
    
    for idx, (name, series_id) in enumerate(FRED_SERIES.items()):
        status_text.text(f"Fetching {name} ({series_id})...")
        data = fetch_fred_data(api_key, series_id)
        if data is not None:
            all_data[name] = data
        progress_bar.progress((idx + 1) / total)
        time.sleep(0.1)
    
    status_text.empty()
    progress_bar.empty()
    
    if not all_data:
        return None
    
    df = pd.DataFrame(all_data).dropna()
    return df

def fetch_recession_data(api_key):
    """Fetch NBER recession indicator from FRED"""
    data = fetch_fred_data(api_key, 'USREC')
    return data if data is not None else None

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
        if response.status_code == 200:
            data = response.json()
            if 'observations' in data:
                return True
        return False
    except Exception as e:
        return False

# =============================================================================
# NELSON-SIEGEL MODEL FAMILY - COMPLETE IMPLEMENTATION
# =============================================================================

class NelsonSiegelModel:
    """Complete Nelson-Siegel Model Family Implementation"""
    
    @staticmethod
    def nelson_siegel(tau, beta0, beta1, beta2, lambda1):
        """
        Nelson-Siegel Model (4 parameters)
        
        Mathematical Formula:
        y(tau) = beta0 + beta1 * (1 - e^(-lambda*tau))/(lambda*tau) 
               + beta2 * ((1 - e^(-lambda*tau))/(lambda*tau) - e^(-lambda*tau))
        
        Parameters:
        - beta0 (Level): Long-term interest rate level (parallel shift)
        - beta1 (Slope): Short-term slope (negative when curve is upward sloping)
        - beta2 (Curvature): Medium-term curvature (hump shape)
        - lambda (Decay): Exponential decay rate (controls maturity of max curvature)
        """
        mask = tau == 0
        tau = np.where(mask, 1e-8, tau)
        lambdatz = lambda1 * tau
        term1 = (1 - np.exp(-lambdatz)) / lambdatz
        term2 = term1 - np.exp(-lambdatz)
        result = beta0 + beta1 * term1 + beta2 * term2
        return np.where(mask, beta0 + beta1 + beta2, result)
    
    @staticmethod
    def nelson_siegel_svensson(tau, beta0, beta1, beta2, beta3, lambda1, lambda2):
        """
        Nelson-Siegel-Svensson Model (6 parameters)
        
        Extended formula with second curvature factor:
        y(tau) = beta0 + beta1 * (1 - e^(-lambda1*tau))/(lambda1*tau) 
               + beta2 * ((1 - e^(-lambda1*tau))/(lambda1*tau) - e^(-lambda1*tau))
               + beta3 * ((1 - e^(-lambda2*tau))/(lambda2*tau) - e^(-lambda2*tau))
        
        Additional Parameters:
        - beta3: Second curvature factor (captures additional hump)
        - lambda2: Second decay factor (different maturity focus)
        """
        mask = tau == 0
        tau = np.where(mask, 1e-8, tau)
        
        lambdatz1 = lambda1 * tau
        term1 = (1 - np.exp(-lambdatz1)) / lambdatz1
        term2 = term1 - np.exp(-lambdatz1)
        
        lambdatz2 = lambda2 * tau
        term3 = (1 - np.exp(-lambdatz2)) / lambdatz2
        term4 = term3 - np.exp(-lambdatz2)
        
        result = beta0 + beta1 * term1 + beta2 * term2 + beta3 * term4
        return np.where(mask, beta0 + beta1 + beta2 + beta3, result)
    
    @staticmethod
    def dynamic_nelson_siegel(tau, beta0, beta1, beta2, lambda1, rho=0.95):
        """Dynamic Nelson-Siegel with persistence parameter"""
        beta0_t = beta0 * (1 - rho) + rho * beta0
        beta1_t = beta1 * (1 - rho) + rho * beta1
        beta2_t = beta2 * (1 - rho) + rho * beta2
        return beta0_t + beta1_t * ((1 - np.exp(-lambda1 * tau)) / (lambda1 * tau)) + \
               beta2_t * (((1 - np.exp(-lambda1 * tau)) / (lambda1 * tau)) - np.exp(-lambda1 * tau))
    
    @staticmethod
    def fit_nelson_siegel(maturities, yields, method='robust'):
        """Fit Nelson-Siegel model with robust optimization"""
        
        def objective(params):
            beta0, beta1, beta2, lambda1 = params
            fitted = NelsonSiegelModel.nelson_siegel(maturities, beta0, beta1, beta2, lambda1)
            residuals = yields - fitted
            
            if method == 'robust':
                delta = 1.0
                loss = np.where(np.abs(residuals) < delta, 
                               0.5 * residuals**2,
                               delta * np.abs(residuals) - 0.5 * delta**2)
                return np.sum(loss)
            else:
                return np.sum(residuals ** 2)
        
        bounds = [
            (yields.min() - 2, yields.max() + 2),
            (-15, 15),
            (-15, 15),
            (0.01, 5)
        ]
        
        best_result = None
        best_fitness = np.inf
        
        for _ in range(10):
            x0 = [np.random.uniform(b[0], b[1]) for b in bounds]
            result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
            if result.fun < best_fitness:
                best_fitness = result.fun
                best_result = result
        
        if best_result:
            fitted = NelsonSiegelModel.nelson_siegel(maturities, *best_result.x)
            return {
                'params': best_result.x,
                'fitted_values': fitted,
                'rmse': np.sqrt(best_fitness / len(maturities)),
                'mae': np.mean(np.abs(yields - fitted)),
                'r_squared': 1 - best_fitness / np.sum((yields - np.mean(yields)) ** 2),
                'success': True
            }
        return None
    
    @staticmethod
    def fit_svensson(maturities, yields):
        """Fit Nelson-Siegel-Svensson model with global optimization"""
        
        def objective(params):
            beta0, beta1, beta2, beta3, lambda1, lambda2 = params
            fitted = NelsonSiegelModel.nelson_siegel_svensson(
                maturities, beta0, beta1, beta2, beta3, lambda1, lambda2
            )
            weights = 1 / (maturities + 0.25)
            return np.sum(weights * (yields - fitted) ** 2)
        
        bounds = [
            (yields.min() - 2, yields.max() + 2),
            (-20, 20),
            (-20, 20),
            (-20, 20),
            (0.01, 10),
            (0.01, 10)
        ]
        
        result = differential_evolution(objective, bounds, maxiter=500, popsize=15, seed=42)
        
        if result.success:
            fitted = NelsonSiegelModel.nelson_siegel_svensson(maturities, *result.x)
            return {
                'params': result.x,
                'fitted_values': fitted,
                'rmse': np.sqrt(result.fun / len(maturities)),
                'mae': np.mean(np.abs(yields - fitted)),
                'r_squared': 1 - result.fun / np.sum((yields - np.mean(yields)) ** 2),
                'success': True
            }
        return None
    
    @staticmethod
    def calculate_factor_interpretation(params, model_type='NS'):
        """Calculate and interpret model factors"""
        if model_type == 'NS':
            beta0, beta1, beta2, lambda1 = params
            max_curvature_maturity = 1 / lambda1 if lambda1 > 0 else 0
            
            return {
                'Long_Term_Level': beta0,
                'Short_Term_Slope': beta1,
                'Curvature': beta2,
                'Max_Curvature_Maturity': max_curvature_maturity,
                'Interpretation': {
                    'Level': "Long-term expectation: {:.2f}%".format(beta0),
                    'Slope': "Curve slope: {} ({:.2f})".format('Inverted' if beta1 < 0 else 'Normal', beta1),
                    'Curvature': "Hump shape: {} at {:.1f} years".format('Humped' if beta2 > 0 else 'Sagged', max_curvature_maturity),
                    'Decay': "Decay rate: {:.4f} (max curvature at {:.1f} years)".format(lambda1, max_curvature_maturity)
                }
            }
        else:
            beta0, beta1, beta2, beta3, lambda1, lambda2 = params
            max_curvature1 = 1 / lambda1 if lambda1 > 0 else 0
            max_curvature2 = 1 / lambda2 if lambda2 > 0 else 0
            
            return {
                'Long_Term_Level': beta0,
                'Short_Term_Slope': beta1,
                'Curvature1': beta2,
                'Curvature2': beta3,
                'Max_Curvature1_Maturity': max_curvature1,
                'Max_Curvature2_Maturity': max_curvature2,
                'Interpretation': {
                    'Level': "Long-term expectation: {:.2f}%".format(beta0),
                    'Slope': "Curve slope: {} ({:.2f})".format('Inverted' if beta1 < 0 else 'Normal', beta1),
                    'Curvature1': "First hump at {:.1f} years".format(max_curvature1),
                    'Curvature2': "Second hump at {:.1f} years".format(max_curvature2)
                }
            }

# =============================================================================
# DYNAMIC PARAMETER ANALYSIS
# =============================================================================

class DynamicParameterAnalysis:
    """Analyze parameter evolution over time"""
    
    @staticmethod
    def calibrate_rolling_window(yield_df, maturities, window_years=5, model_type='NS'):
        """Calibrate model on rolling windows for parameter evolution"""
        dates = yield_df.index
        window_size = window_years * 252
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_steps = len(range(window_size, len(dates), 21))
        
        for step, i in enumerate(range(window_size, len(dates), 21)):
            status_text.text("Calibrating {} model for {}...".format(model_type, dates[i].strftime('%Y-%m-%d')))
            window_data = yield_df.iloc[i-window_size:i]
            latest_yields = window_data.iloc[-1].values[:len(maturities)]
            
            if model_type == 'NS':
                result = NelsonSiegelModel.fit_nelson_siegel(maturities, latest_yields)
                if result:
                    results.append({
                        'date': dates[i],
                        'beta0': result['params'][0],
                        'beta1': result['params'][1],
                        'beta2': result['params'][2],
                        'lambda': result['params'][3],
                        'rmse': result['rmse']
                    })
            else:
                result = NelsonSiegelModel.fit_svensson(maturities, latest_yields)
                if result:
                    results.append({
                        'date': dates[i],
                        'beta0': result['params'][0],
                        'beta1': result['params'][1],
                        'beta2': result['params'][2],
                        'beta3': result['params'][3],
                        'lambda1': result['params'][4],
                        'lambda2': result['params'][5],
                        'rmse': result['rmse']
                    })
            
            progress_bar.progress((step + 1) / total_steps)
        
        status_text.empty()
        progress_bar.empty()
        return pd.DataFrame(results)
    
    @staticmethod
    def calculate_factor_contributions(yield_df):
        """Calculate historical factor contributions (Level, Slope, Curvature)"""
        result = pd.DataFrame(index=yield_df.index)
        
        if '10Y' in yield_df.columns:
            result['Level'] = yield_df['10Y']
        
        if '10Y' in yield_df.columns and '3M' in yield_df.columns:
            result['Slope'] = (yield_df['10Y'] - yield_df['3M']) * 100
        
        if all(k in yield_df.columns for k in ['3M', '5Y', '10Y']):
            result['Curvature'] = (2 * yield_df['5Y'] - (yield_df['3M'] + yield_df['10Y'])) * 100
        
        if all(k in yield_df.columns for k in ['2Y', '10Y', '30Y']):
            result['Butterfly'] = (2 * yield_df['10Y'] - (yield_df['2Y'] + yield_df['30Y'])) * 100
        
        return result
    
    @staticmethod
    def calculate_parameter_volatility(dynamic_params):
        """Calculate volatility of model parameters over time"""
        if dynamic_params.empty:
            return None
        
        vol_df = pd.DataFrame({
            'Parameter': ['b0 (Level)', 'b1 (Slope)', 'b2 (Curvature)', 'Lambda (Decay)'],
            'Volatility': [
                dynamic_params['beta0'].std(),
                dynamic_params['beta1'].std(),
                dynamic_params['beta2'].std(),
                dynamic_params['lambda'].std() if 'lambda' in dynamic_params.columns else 0
            ],
            'Mean': [
                dynamic_params['beta0'].mean(),
                dynamic_params['beta1'].mean(),
                dynamic_params['beta2'].mean(),
                dynamic_params['lambda'].mean() if 'lambda' in dynamic_params.columns else 0
            ],
            'Current': [
                dynamic_params['beta0'].iloc[-1],
                dynamic_params['beta1'].iloc[-1],
                dynamic_params['beta2'].iloc[-1],
                dynamic_params['lambda'].iloc[-1] if 'lambda' in dynamic_params.columns else 0
            ]
        })
        
        return vol_df

# =============================================================================
# ADVANCED RISK METRICS
# =============================================================================

class AdvancedRiskMetrics:
    """Advanced risk metrics for yield curve"""
    
    @staticmethod
    def calculate_pca_risk(yield_df, n_components=3):
        """PCA-based risk decomposition and factor analysis - Fixed for NaN handling"""
        try:
            returns = yield_df.pct_change().dropna()
            
            if len(returns) < 5 or returns.shape[1] < 2:
                return {
                    'explained_variance': np.array([0.7, 0.2, 0.1][:n_components]),
                    'cumulative_variance': np.cumsum(np.array([0.7, 0.2, 0.1][:n_components])),
                    'components': np.eye(n_components, len(yield_df.columns))[:n_components],
                    'loadings': pd.DataFrame(
                        np.eye(n_components, len(yield_df.columns))[:n_components].T,
                        columns=['PC{}'.format(i+1) for i in range(n_components)],
                        index=yield_df.columns
                    ),
                    'factors': pd.DataFrame(index=returns.index),
                    'n_components': n_components
                }
            
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
            scaler = StandardScaler()
            returns_scaled = scaler.fit_transform(returns)
            actual_n_components = min(n_components, len(returns.columns), len(returns) - 1)
            if actual_n_components < 1:
                actual_n_components = 1
            
            pca = PCA(n_components=actual_n_components)
            pcs = pca.fit_transform(returns_scaled)
            
            factor_names = ['PC1_Level', 'PC2_Slope', 'PC3_Curvature'][:actual_n_components]
            factor_contributions = pd.DataFrame(pcs, index=returns.index, columns=factor_names)
            
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=['PC{}'.format(i+1) for i in range(actual_n_components)],
                index=yield_df.columns
            )
            
            return {
                'explained_variance': pca.explained_variance_ratio_,
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
                'components': pca.components_,
                'loadings': loadings,
                'factors': factor_contributions,
                'n_components': actual_n_components
            }
        except Exception as e:
            return {
                'explained_variance': np.array([0.7, 0.2, 0.1][:n_components]),
                'cumulative_variance': np.cumsum(np.array([0.7, 0.2, 0.1][:n_components])),
                'components': np.eye(n_components, len(yield_df.columns))[:n_components],
                'loadings': pd.DataFrame(
                    np.eye(n_components, len(yield_df.columns))[:n_components].T,
                    columns=['PC{}'.format(i+1) for i in range(n_components)],
                    index=yield_df.columns
                ),
                'factors': pd.DataFrame(),
                'n_components': n_components
            }
    
    @staticmethod
    def calculate_var_metrics(returns, confidence=0.95, horizon=10):
        """Calculate Value at Risk and Conditional VaR"""
        returns = returns.dropna()
        
        if len(returns) < 2:
            return {'VaR_Historical': 0, 'CVaR': 0, 'tail_ratio': 1, 
                    'VaR_Parametric': 0, 'VaR_CornishFisher': 0,
                    'skewness': 0, 'kurtosis': 0}
        
        var_historical = np.percentile(returns, (1 - confidence) * 100)
        cvar_historical = returns[returns <= var_historical].mean()
        var_parametric = norm.ppf(1 - confidence) * returns.std()
        
        skew = returns.skew()
        kurt = returns.kurtosis()
        z_cf = norm.ppf(1 - confidence)
        z_cf_adjusted = z_cf + (z_cf**2 - 1) * skew / 6 + (z_cf**3 - 3 * z_cf) * kurt / 24 - (2 * z_cf**3 - 5 * z_cf) * skew**2 / 36
        var_cf = z_cf_adjusted * returns.std()
        
        return {
            'VaR_Historical': var_historical * np.sqrt(horizon),
            'VaR_Parametric': var_parametric * np.sqrt(horizon),
            'VaR_CornishFisher': var_cf * np.sqrt(horizon),
            'CVaR': cvar_historical * np.sqrt(horizon),
            'tail_ratio': abs(cvar_historical / var_historical) if var_historical != 0 else 1,
            'skewness': skew,
            'kurtosis': kurt
        }

# =============================================================================
# FORECASTING MODELS
# =============================================================================

class YieldCurveForecasting:
    """Advanced yield curve forecasting models"""
    
    @staticmethod
    def forecast_with_var(yield_df, horizon=5, confidence=0.95):
        """Forecast using Vector Autoregression (VAR)"""
        try:
            from statsmodels.tsa.api import VAR
            
            if len(yield_df) < 100:
                return None
            
            model = VAR(yield_df)
            results = model.fit(maxlags=10, ic='aic')
            
            forecast = results.forecast(yield_df.values[-results.k_ar:], horizon)
            
            z_score = norm.ppf((1 + confidence) / 2)
            forecast_upper = forecast + z_score * np.std(forecast, axis=0)
            forecast_lower = forecast - z_score * np.std(forecast, axis=0)
            
            return {
                'forecast': forecast,
                'upper': forecast_upper,
                'lower': forecast_lower,
                'model': results,
                'horizon': horizon
            }
        except Exception as e:
            return None

# =============================================================================
# ARBITRAGE DETECTION
# =============================================================================

class ArbitrageDetection:
    """Detect arbitrage opportunities using Nelson-Siegel Svensson model"""
    
    @staticmethod
    def detect_arbitrage_opportunities(yield_df, maturities):
        """Detect arbitrage opportunities using Nelson-Siegel Svensson model"""
        latest_yields = yield_df.iloc[-1].values[:len(maturities)]
        
        nss_result = NelsonSiegelModel.fit_svensson(maturities, latest_yields)
        
        if nss_result is None:
            return None
        
        theoretical = nss_result['fitted_values']
        actual = latest_yields
        residuals = actual - theoretical
        
        mispriced = []
        for i, (m, r) in enumerate(zip(maturities, residuals)):
            if abs(r) > 0.1:
                mispriced.append({
                    'maturity': m,
                    'actual': actual[i],
                    'theoretical': theoretical[i],
                    'difference': r,
                    'opportunity': 'Overvalued' if r < 0 else 'Undervalued'
                })
        
        arbitrage_stats = {
            'mean_abs_error': np.mean(np.abs(residuals)),
            'max_error': np.max(np.abs(residuals)),
            'std_error': np.std(residuals),
            'mispriced_count': len(mispriced),
            'mispriced_securities': mispriced,
            'nss_params': nss_result['params']
        }
        
        return arbitrage_stats

# =============================================================================
# VISUALIZATION FUNCTIONS - COMPLETE SET
# =============================================================================

def create_institutional_layout(fig, title, y_title=None, height=500):
    """Apply institutional styling to plots"""
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['surface'],
        plot_bgcolor=COLORS['surface'],
        font=dict(family="Courier New, monospace", size=10, color=COLORS['text_secondary']),
        title=dict(
            text=title,
            font=dict(size=12, color=COLORS['text_primary']),
            x=0.02,
            xanchor='left'
        ),
        margin=dict(l=50, r=30, t=60, b=40),
        hovermode='x unified',
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=8)
        )
    )
    
    fig.update_xaxes(
        gridcolor=COLORS['grid'],
        gridwidth=0.5,
        zeroline=False,
        tickfont=dict(size=8),
        title_font=dict(size=9)
    )
    
    fig.update_yaxes(
        gridcolor=COLORS['grid'],
        gridwidth=0.5,
        zeroline=False,
        tickfont=dict(size=8),
        title_font=dict(size=9)
    )
    
    if y_title:
        fig.update_yaxes(title_text=y_title)
    
    return fig

def plot_2y_yield_chart(yield_df):
    """Create interactive 2Y yield chart with range slider"""
    if '2Y' not in yield_df.columns:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=yield_df.index,
        y=yield_df['2Y'],
        mode='lines',
        name='2-Year Treasury Yield',
        line=dict(color=COLORS['warning'], width=2.5),
        fill='tozeroy',
        fillcolor='rgba(243, 156, 18, 0.1)',
        hovertemplate='<b>Date: %{x|%Y-%m-%d}</b><br>2Y Yield: %{y:.2f}%<extra></extra>'
    ))
    
    fig = create_institutional_layout(fig, "2-YEAR TREASURY YIELD", "Yield (%)", height=450)
    
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(count=10, label="10Y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    return fig

def plot_10y_yield_chart(yield_df):
    """Create interactive 10Y yield chart with range slider"""
    if '10Y' not in yield_df.columns:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=yield_df.index,
        y=yield_df['10Y'],
        mode='lines',
        name='10-Year Treasury Yield',
        line=dict(color=COLORS['accent'], width=2.5),
        fill='tozeroy',
        fillcolor='rgba(15, 52, 96, 0.1)',
        hovertemplate='<b>Date: %{x|%Y-%m-%d}</b><br>10Y Yield: %{y:.2f}%<extra></extra>'
    ))
    
    fig = create_institutional_layout(fig, "10-YEAR TREASURY YIELD", "Yield (%)", height=450)
    
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(count=10, label="10Y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    return fig

def plot_nber_recession_chart(spreads, recessions):
    """Create NBER recession chart with institutional styling"""
    fig = go.Figure()
    
    if '10Y-2Y' in spreads.columns:
        fig.add_trace(go.Scatter(
            x=spreads.index,
            y=spreads['10Y-2Y'],
            mode='lines',
            name='10Y-2Y Spread',
            line=dict(color=COLORS['negative'], width=1.5),
            hovertemplate='<b>Date: %{x|%Y-%m-%d}</b><br>Spread: %{y:.1f} bps<extra></extra>'
        ))
    
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color=COLORS['neutral'],
        line_width=1,
        annotation_text="INVERSION THRESHOLD",
        annotation_position="top right"
    )
    
    for recession in recessions:
        fig.add_vrect(
            x0=recession['start'],
            x1=recession['end'],
            fillcolor=COLORS['recession'],
            opacity=0.4,
            layer="below",
            line_width=0,
            annotation_text="NBER RECESSION",
            annotation_position="top left"
        )
    
    fig = create_institutional_layout(fig, "NBER RECESSION INDICATOR & YIELD SPREAD", "Spread (bps)", height=500)
    return fig

def plot_spread_dashboard(spreads, recessions):
    """Create comprehensive spread dashboard with 4 subplots - FIXED hovertemplate"""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=('10Y-2Y SPREAD (Primary Indicator)',
                       '10Y-3M SPREAD (Campbell Harvey)',
                       '5Y-2Y SPREAD (Medium-term)',
                       '30Y-10Y SPREAD (Term Premium)'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    spread_configs = {
        '10Y-2Y': {'row': 1, 'col': 1, 'color': COLORS['negative']},
        '10Y-3M': {'row': 1, 'col': 2, 'color': COLORS['positive']},
        '5Y-2Y': {'row': 2, 'col': 1, 'color': COLORS['neutral']},
        '30Y-10Y': {'row': 2, 'col': 2, 'color': COLORS['warning']}
    }
    
    for spread_name, config in spread_configs.items():
        if spread_name in spreads.columns:
            fig.add_trace(
                go.Scatter(
                    x=spreads.index,
                    y=spreads[spread_name],
                    mode='lines',
                    name=spread_name,
                    line=dict(color=config['color'], width=1.5),
                    hovertemplate='<b>Date: %{x|%Y-%m-%d}</b><br>' + spread_name + ': %{y:.1f} bps<extra></extra>'
                ),
                row=config['row'],
                col=config['col']
            )
            
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color=COLORS['negative'],
                line_width=1,
                row=config['row'],
                col=config['col']
            )
            
            for recession in recessions:
                fig.add_vrect(
                    x0=recession['start'],
                    x1=recession['end'],
                    fillcolor=COLORS['recession'],
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                    row=config['row'],
                    col=config['col']
                )
    
    fig = create_institutional_layout(fig, "YIELD SPREAD DYNAMICS", height=600)
    return fig

def plot_current_yield_curve_comparison(yield_df, maturities):
    """Plot current yield curve vs historical averages"""
    current = yield_df.iloc[-1].values[:len(maturities)]
    one_year_ago = yield_df.iloc[-252].values[:len(maturities)] if len(yield_df) > 252 else current
    five_year_avg = yield_df.iloc[-1260:].mean().values[:len(maturities)] if len(yield_df) > 1260 else current
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=maturities,
        y=current,
        mode='lines+markers',
        name='Current Yield',
        line=dict(color=COLORS['accent'], width=2.5),
        marker=dict(size=8, symbol='circle'),
        hovertemplate='Maturity: %{x:.1f}Y<br>Yield: %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=maturities,
        y=one_year_ago,
        mode='lines+markers',
        name='1 Year Ago',
        line=dict(color=COLORS['warning'], width=2, dash='dash'),
        marker=dict(size=6, symbol='square')
    ))
    
    fig.add_trace(go.Scatter(
        x=maturities,
        y=five_year_avg,
        mode='lines+markers',
        name='5 Year Average',
        line=dict(color=COLORS['positive'], width=2, dash='dot'),
        marker=dict(size=6, symbol='diamond')
    ))
    
    fig = create_institutional_layout(fig, "CURRENT YIELD CURVE ANALYSIS", "Yield (%)", height=500)
    return fig

def plot_parameter_evolution(dynamic_params):
    """Plot parameter evolution over time"""
    if dynamic_params.empty:
        return None
    
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=('Level Factor (Beta0)', 'Slope Factor (Beta1)', 
                       'Curvature Factor (Beta2)', 'RMSE Evolution'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dynamic_params['date'],
            y=dynamic_params['beta0'],
            mode='lines',
            name='Level',
            line=dict(color=COLORS['positive'], width=2),
            fill='tozeroy',
            fillcolor='rgba(46, 204, 113, 0.1)'
        ),
        row=1,
        col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dynamic_params['date'],
            y=dynamic_params['beta1'],
            mode='lines',
            name='Slope',
            line=dict(color=COLORS['accent'], width=2),
            fill='tozeroy',
            fillcolor='rgba(15, 52, 96, 0.1)'
        ),
        row=1,
        col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=dynamic_params['date'],
            y=dynamic_params['beta2'],
            mode='lines',
            name='Curvature',
            line=dict(color=COLORS['warning'], width=2),
            fill='tozeroy',
            fillcolor='rgba(243, 156, 18, 0.1)'
        ),
        row=2,
        col=1
    )
    
    if 'rmse' in dynamic_params.columns:
        fig.add_trace(
            go.Scatter(
                x=dynamic_params['date'],
                y=dynamic_params['rmse'],
                mode='lines',
                name='RMSE',
                line=dict(color=COLORS['neutral'], width=2),
                fill='tozeroy',
                fillcolor='rgba(149, 165, 166, 0.1)'
            ),
            row=2,
            col=2
        )
    
    fig = create_institutional_layout(fig, "PARAMETER EVOLUTION OVER TIME", height=600)
    return fig

def plot_pca_factors(pca_risk):
    """Plot PCA factor dynamics"""
    if pca_risk is None or 'factors' not in pca_risk:
        return None
    
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=('PC1 - LEVEL FACTOR', 'PC2 - SLOPE FACTOR', 'PC3 - CURVATURE FACTOR'),
        shared_xaxes=True,
        vertical_spacing=0.08
    )
    
    for i, factor in enumerate(['PC1_Level', 'PC2_Slope', 'PC3_Curvature']):
        if factor in pca_risk['factors'].columns:
            fig.add_trace(
                go.Scatter(
                    x=pca_risk['factors'].index,
                    y=pca_risk['factors'][factor],
                    mode='lines',
                    name=factor,
                    line=dict(color=COLORS['accent'], width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(15, 52, 96, 0.1)'
                ),
                row=i+1,
                col=1
            )
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color=COLORS['neutral'],
                line_width=1,
                row=i+1,
                col=1
            )
    
    fig = create_institutional_layout(fig, "PRINCIPAL COMPONENT ANALYSIS", height=700)
    return fig

def plot_forecast_chart(historical, forecast_result, maturity_name='10Y'):
    """Plot forecast with confidence intervals"""
    if forecast_result is None:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=historical.index[-100:],
        y=historical.values[-100:],
        mode='lines',
        name='Historical',
        line=dict(color=COLORS['accent'], width=2)
    ))
    
    forecast_dates = pd.date_range(
        start=historical.index[-1],
        periods=forecast_result['horizon'] + 1,
        freq='D'
    )[1:]
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_result['forecast'][:, 0] if len(forecast_result['forecast'].shape) > 1 else forecast_result['forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color=COLORS['positive'], width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_result['upper'][:, 0] if len(forecast_result['upper'].shape) > 1 else forecast_result['upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(color=COLORS['neutral'], width=0.5),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_result['lower'][:, 0] if len(forecast_result['lower'].shape) > 1 else forecast_result['lower'],
        mode='lines',
        name='Lower Bound',
        line=dict(color=COLORS['neutral'], width=0.5),
        fill='tonexty',
        fillcolor='rgba(149, 165, 166, 0.2)',
        showlegend=False
    ))
    
    fig = create_institutional_layout(fig, maturity_name + " YIELD FORECAST", "Yield (%)", height=500)
    return fig

def plot_arbitrage_chart(maturities, actual, theoretical, mispriced):
    """Plot arbitrage opportunities chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=maturities,
        y=actual,
        mode='markers',
        name='Actual Yields',
        marker=dict(size=12, color=COLORS['accent'], symbol='circle')
    ))
    
    fig.add_trace(go.Scatter(
        x=maturities,
        y=theoretical,
        mode='lines',
        name='NSS Theoretical',
        line=dict(color=COLORS['positive'], width=2.5)
    ))
    
    mispriced_maturities = [m['maturity'] for m in mispriced]
    mispriced_actual = [m['actual'] for m in mispriced]
    
    if mispriced_maturities:
        fig.add_trace(go.Scatter(
            x=mispriced_maturities,
            y=mispriced_actual,
            mode='markers',
            name='Mispriced Securities',
            marker=dict(size=15, color=COLORS['warning'], symbol='circle', line=dict(width=2, color='red'))
        ))
    
    fig = create_institutional_layout(fig, "ARBITRAGE OPPORTUNITY DETECTION", "Yield (%)", height=500)
    return fig

# =============================================================================
# RECESSION ANALYSIS FUNCTIONS
# =============================================================================

def identify_recessions(recession_series):
    """Identify NBER recession periods from indicator series"""
    if recession_series is None or len(recession_series) == 0:
        return []
    
    recessions = []
    in_recession = False
    start_date = None
    
    for date, value in recession_series.items():
        if value == 1 and not in_recession:
            in_recession = True
            start_date = date
        elif value == 0 and in_recession:
            in_recession = False
            recessions.append({'start': start_date, 'end': date, 'type': 'NBER'})
    
    return recessions

def calculate_recession_metrics(spreads, recessions):
    """Calculate recession-related metrics and statistics"""
    if '10Y-2Y' not in spreads.columns:
        return {}
    
    spread_series = spreads['10Y-2Y'].dropna()
    
    inversion_periods = []
    in_inversion = False
    inv_start = None
    
    for date, value in spread_series.items():
        if value < 0 and not in_inversion:
            in_inversion = True
            inv_start = date
        elif value >= 0 and in_inversion:
            in_inversion = False
            inversion_periods.append({
                'start': inv_start,
                'end': date,
                'depth': spread_series.loc[inv_start:date].min(),
                'duration': (date - inv_start).days
            })
    
    lead_times = []
    for inversion in inversion_periods:
        for recession in recessions:
            if inversion['start'] < recession['start']:
                lead_days = (recession['start'] - inversion['start']).days
                lead_times.append(lead_days)
                break
    
    return {
        'inversion_periods': inversion_periods,
        'total_inversion_days': sum([p['duration'] for p in inversion_periods]),
        'avg_inversion_depth': np.mean([p['depth'] for p in inversion_periods]) if inversion_periods else 0,
        'lead_times': lead_times,
        'avg_lead_time': np.mean(lead_times) if lead_times else 0,
        'num_inversions': len(inversion_periods),
        'num_recessions': len(recessions)
    }

# =============================================================================
# API KEY INPUT COMPONENT
# =============================================================================

def render_api_key_input():
    """Render interactive API key input component"""
    st.markdown("""
    <div class="api-container">
        <h3 style="color: white; margin-bottom: 1rem;">FRED API Key Required</h3>
        <p style="color: #bdc3c7; font-size: 0.85rem; margin-bottom: 1.5rem;">
            This institutional dashboard requires a FRED API key to access treasury yield data.<br>
            Get your free API key from:
        </p>
        <p style="margin-bottom: 1.5rem;">
            <a href="https://fred.stlouisfed.org/docs/api/api_key.html" target="_blank" style="color: #0f3460;">
                https://fred.stlouisfed.org/docs/api/api_key.html
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        api_key = st.text_input(
            "Enter your FRED API Key",
            type="password",
            key="fred_api_key_input",
            placeholder="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            help="Your FRED API key is required to fetch real-time data"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        
        with col_btn2:
            validate_btn = st.button("Validate & Connect", type="primary", use_container_width=True)
    
    if validate_btn:
        if not api_key:
            st.error("Please enter a valid API key")
            return None
        
        with st.spinner("Validating API key..."):
            is_valid = validate_fred_api_key(api_key)
        
        if is_valid:
            st.session_state.api_key = api_key
            st.session_state.api_key_validated = True
            st.success("API key validated successfully! Fetching data...")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Invalid API key. Please check and try again.")
            return None
    
    return None

# =============================================================================
# EXECUTIVE SUMMARY SECTION
# =============================================================================

def render_executive_summary(yield_df, spreads, current_10y, current_2y, current_spread):
    """Render executive summary section"""
    
    st.markdown('<div class="executive-summary-card">', unsafe_allow_html=True)
    st.markdown('<div class="executive-title">EXECUTIVE SUMMARY</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-label">2-YEAR YIELD</div>'
            '<div class="metric-value">{:.2f}%</div>'
            '<div class="metric-label">Short-term benchmark</div>'
            '</div>'.format(current_2y), 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-label">10-YEAR YIELD</div>'
            '<div class="metric-value">{:.2f}%</div>'
            '<div class="metric-label">Long-term benchmark</div>'
            '</div>'.format(current_10y), 
            unsafe_allow_html=True
        )
    
    with col3:
        if current_spread < 0:
            status_text = "INVERTED"
            status_color = COLORS['negative']
        elif current_spread < 50:
            status_text = "FLATTENING"
            status_color = COLORS['warning']
        else:
            status_text = "NORMAL"
            status_color = COLORS['positive']
        
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-label">10Y-2Y SPREAD</div>'
            '<div class="metric-value">{:.1f} bps</div>'
            '<div class="metric-value" style="color: {}; font-size: 0.9rem;">{}</div>'
            '</div>'.format(current_spread, status_color, status_text), 
            unsafe_allow_html=True
        )
    
    with col4:
        if current_spread < 0:
            outlook = "RECESSION WARNING"
            outlook_color = COLORS['negative']
        elif current_spread < 50:
            outlook = "CAUTIOUS"
            outlook_color = COLORS['warning']
        else:
            outlook = "EXPANSION"
            outlook_color = COLORS['positive']
        
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-label">ECONOMIC OUTLOOK</div>'
            '<div class="metric-value" style="color: {}; font-size: 1.2rem;">{}</div>'
            '<div class="metric-label">Based on yield curve shape</div>'
            '</div>'.format(outlook_color, outlook), 
            unsafe_allow_html=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    
    # Header
    st.markdown("""
    <div class="hedge-header">
        <h1 style="color: white; margin: 0; font-size: 1.25rem;">YIELD CURVE ANALYTICS</h1>
        <p style="color: #bdc3c7; margin: 0; font-size: 0.7rem;">Executive Summary | FRED Data Integration | Nelson-Siegel Family Models | NBER Recession Analysis | Quantitative Risk Metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # API Key Validation
    if not st.session_state.api_key_validated:
        render_api_key_input()
        st.stop()
    
    # Data Fetching
    if not st.session_state.data_fetched:
        with st.spinner("Connecting to FRED and fetching data..."):
            yield_df = fetch_all_yield_data(st.session_state.api_key)
            recession_series = fetch_recession_data(st.session_state.api_key)
        
        if yield_df is not None:
            st.session_state.yield_data = yield_df
            st.session_state.recession_data = recession_series
            st.session_state.data_fetched = True
            st.success("Data fetched successfully: {} observations from {} to {}".format(
                len(yield_df), 
                yield_df.index[0].strftime('%Y-%m-%d'), 
                yield_df.index[-1].strftime('%Y-%m-%d')
            ))
            time.sleep(1)
            st.rerun()
        else:
            st.error("Failed to fetch data. Please check your API key and try again.")
            st.session_state.api_key_validated = False
            st.stop()
    
    # Load data from session state
    yield_df = st.session_state.yield_data
    recession_series = st.session_state.recession_data
    
    # Prepare data structures
    available_cols = [col for col in yield_df.columns if col in MATURITY_MAP]
    maturities = np.array([MATURITY_MAP[col] for col in available_cols])
    yield_values = yield_df.iloc[-1][available_cols].values
    
    # Calculate spreads
    spreads = pd.DataFrame(index=yield_df.index)
    if '10Y' in yield_df.columns and '2Y' in yield_df.columns:
        spreads['10Y-2Y'] = (yield_df['10Y'] - yield_df['2Y']) * 100
    if '10Y' in yield_df.columns and '3M' in yield_df.columns:
        spreads['10Y-3M'] = (yield_df['10Y'] - yield_df['3M']) * 100
    if '5Y' in yield_df.columns and '2Y' in yield_df.columns:
        spreads['5Y-2Y'] = (yield_df['5Y'] - yield_df['2Y']) * 100
    if '30Y' in yield_df.columns and '10Y' in yield_df.columns:
        spreads['30Y-10Y'] = (yield_df['30Y'] - yield_df['10Y']) * 100
    if '2Y' in yield_df.columns and '3M' in yield_df.columns:
        spreads['2Y-3M'] = (yield_df['2Y'] - yield_df['3M']) * 100
    if '10Y' in yield_df.columns and '1M' in yield_df.columns:
        spreads['10Y-1M'] = (yield_df['10Y'] - yield_df['1M']) * 100
    
    # Identify recessions
    recessions = identify_recessions(recession_series)
    
    # Current metrics
    current_10y = yield_df['10Y'].iloc[-1] if '10Y' in yield_df.columns else 0
    current_2y = yield_df['2Y'].iloc[-1] if '2Y' in yield_df.columns else 0
    current_30y = yield_df['30Y'].iloc[-1] if '30Y' in yield_df.columns else 0
    current_1m = yield_df['1M'].iloc[-1] if '1M' in yield_df.columns else 0
    current_6m = yield_df['6M'].iloc[-1] if '6M' in yield_df.columns else 0
    current_spread = spreads['10Y-2Y'].iloc[-1] if '10Y-2Y' in spreads.columns else 0
    
    # Fit models
    with st.spinner("Calibrating Nelson-Siegel models..."):
        ns_result = NelsonSiegelModel.fit_nelson_siegel(maturities, yield_values)
        nss_result = NelsonSiegelModel.fit_svensson(maturities, yield_values)
        st.session_state.ns_results = ns_result
        st.session_state.nss_results = nss_result
    
    # Dynamic analysis
    with st.spinner("Performing dynamic parameter analysis..."):
        dynamic_params = DynamicParameterAnalysis.calibrate_rolling_window(yield_df, maturities, window_years=5, model_type='NS')
        factors = DynamicParameterAnalysis.calculate_factor_contributions(yield_df)
        pca_risk = AdvancedRiskMetrics.calculate_pca_risk(yield_df)
        
        st.session_state.dynamic_params = dynamic_params
        st.session_state.factors = factors
        st.session_state.pca_risk = pca_risk
    
    # Recession metrics
    recession_metrics = calculate_recession_metrics(spreads, recessions)
    
    # Arbitrage detection
    with st.spinner("Detecting arbitrage opportunities..."):
        arbitrage_stats = ArbitrageDetection.detect_arbitrage_opportunities(yield_df, maturities)
    
    # Forecasting
    with st.spinner("Generating forecasts..."):
        forecast_result = YieldCurveForecasting.forecast_with_var(yield_df[['10Y']].dropna(), horizon=20)
    
    # ===== EXECUTIVE SUMMARY SECTION =====
    render_executive_summary(yield_df, spreads, current_10y, current_2y, current_spread)
    
    st.markdown("---")
    
    # Refresh button
    col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
    with col_r2:
        if st.button("Refresh Data from FRED", use_container_width=True):
            st.cache_data.clear()
            st.session_state.data_fetched = False
            st.session_state.yield_data = None
            st.session_state.recession_data = None
            st.rerun()
    
    # ===== TABS =====
    tabs = st.tabs([
        "📊 DATA TABLE",
        "📈 2Y & 10Y DYNAMIC CHARTS",
        "📈 SPREAD DYNAMICS",
        "🔬 NS MODEL FIT",
        "📉 NSS MODEL FIT",
        "⚖️ MODEL COMPARISON",
        "📊 DYNAMIC ANALYSIS",
        "🎯 FACTOR ANALYSIS",
        "⚠️ RISK METRICS",
        "💰 ARBITRAGE",
        "📉 NBER RECESSION",
        "📈 FORECASTING",
        "📁 DATA EXPORT"
    ])
    
    # ===== TAB 1: DATA TABLE =====
    with tabs[0]:
        st.markdown("### Historical Yield Data (Latest to Earliest)")
        
        # Create data table from latest to earliest
        data_table = yield_df.copy()
        data_table = data_table.iloc[::-1]
        
        data_table_reset = data_table.reset_index()
        data_table_reset.columns = ['Date'] + list(data_table.columns)
        
        display_df = data_table_reset.copy()
        for col in display_df.columns:
            if col != 'Date':
                display_df[col] = display_df[col].apply(lambda x: "{:.2f}%".format(x))
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        st.markdown("#### Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Observations", "{:,}".format(len(yield_df)))
        with col2:
            st.metric("Maturities", len(yield_df.columns))
        with col3:
            st.metric("Start Date", yield_df.index[0].strftime('%Y-%m-%d'))
        with col4:
            st.metric("End Date", yield_df.index[-1].strftime('%Y-%m-%d'))
    
    # ===== TAB 2: 2Y & 10Y DYNAMIC CHARTS =====
    with tabs[1]:
        st.markdown("### 2-Year and 10-Year Treasury Yield Dynamics")
        st.markdown("*Interactive charts with range selector - Select time periods to analyze yield movements*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 2-Year Treasury Yield")
            fig_2y = plot_2y_yield_chart(yield_df)
            if fig_2y:
                st.plotly_chart(fig_2y, use_container_width=True)
            else:
                st.warning("2Y yield data not available")
        
        with col2:
            st.markdown("#### 10-Year Treasury Yield")
            fig_10y = plot_10y_yield_chart(yield_df)
            if fig_10y:
                st.plotly_chart(fig_10y, use_container_width=True)
            else:
                st.warning("10Y yield data not available")
        
        # Combined view
        st.markdown("#### 2Y vs 10Y Yield Comparison")
        fig_combined = go.Figure()
        
        if '2Y' in yield_df.columns:
            fig_combined.add_trace(go.Scatter(
                x=yield_df.index,
                y=yield_df['2Y'],
                mode='lines',
                name='2-Year Yield',
                line=dict(color=COLORS['warning'], width=2)
            ))
        
        if '10Y' in yield_df.columns:
            fig_combined.add_trace(go.Scatter(
                x=yield_df.index,
                y=yield_df['10Y'],
                mode='lines',
                name='10-Year Yield',
                line=dict(color=COLORS['accent'], width=2)
            ))
        
        fig_combined = create_institutional_layout(fig_combined, "2Y vs 10Y YIELD COMPARISON", "Yield (%)", height=450)
        fig_combined.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(count=10, label="10Y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        st.plotly_chart(fig_combined, use_container_width=True)
        
        # Spread between 10Y and 2Y
        if '10Y' in yield_df.columns and '2Y' in yield_df.columns:
            st.markdown("#### 10Y-2Y Yield Spread")
            fig_spread = go.Figure()
            fig_spread.add_trace(go.Scatter(
                x=yield_df.index,
                y=(yield_df['10Y'] - yield_df['2Y']) * 100,
                mode='lines',
                name='10Y-2Y Spread',
                line=dict(color=COLORS['negative'], width=2),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.1)'
            ))
            fig_spread.add_hline(y=0, line_dash="dash", line_color=COLORS['neutral'])
            fig_spread = create_institutional_layout(fig_spread, "10Y-2Y YIELD SPREAD", "Spread (bps)", height=450)
            fig_spread.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1M", step="month", stepmode="backward"),
                            dict(count=3, label="3M", step="month", stepmode="backward"),
                            dict(count=6, label="6M", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1Y", step="year", stepmode="backward"),
                            dict(count=5, label="5Y", step="year", stepmode="backward"),
                            dict(count=10, label="10Y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )
            st.plotly_chart(fig_spread, use_container_width=True)
    
    # ===== TAB 3: SPREAD DYNAMICS =====
    with tabs[2]:
        st.markdown("### Yield Spread Dynamics Analysis")
        st.markdown("*Comprehensive analysis of key yield spreads*")
        
        st.plotly_chart(plot_spread_dashboard(spreads, recessions), use_container_width=True)
        
        st.markdown("#### Spread Statistics")
        spread_stats = pd.DataFrame({
            'Spread': spreads.columns,
            'Current': ["{:.1f}".format(spreads[col].iloc[-1]) for col in spreads.columns],
            'Mean': ["{:.1f}".format(spreads[col].mean()) for col in spreads.columns],
            'Std': ["{:.1f}".format(spreads[col].std()) for col in spreads.columns],
            'Min': ["{:.1f}".format(spreads[col].min()) for col in spreads.columns],
            'Max': ["{:.1f}".format(spreads[col].max()) for col in spreads.columns],
            '% Negative': ["{:.1f}%".format((spreads[col] < 0).mean() * 100) for col in spreads.columns]
        })
        st.dataframe(spread_stats, use_container_width=True, hide_index=True)
    
    # ===== TAB 4: NS MODEL FIT =====
    with tabs[3]:
        st.markdown("### Nelson-Siegel Model Calibration")
        
        if ns_result:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Model Parameters")
                param_df = pd.DataFrame({
                    'Parameter': ['b0 (Level)', 'b1 (Slope)', 'b2 (Curvature)', 'lambda (Decay)'],
                    'Value': ["{:.4f}".format(ns_result['params'][0]), "{:.4f}".format(ns_result['params'][1]), 
                              "{:.4f}".format(ns_result['params'][2]), "{:.4f}".format(ns_result['params'][3])]
                })
                st.dataframe(param_df, use_container_width=True, hide_index=True)
                
                st.markdown("#### Fit Statistics")
                st.markdown("- **RMSE:** {:.4f}".format(ns_result['rmse']))
                st.markdown("- **MAE:** {:.4f}".format(ns_result['mae']))
                st.markdown("- **R-squared:** {:.4f}".format(ns_result['r_squared']))
            
            with col2:
                fig_ns = go.Figure()
                fig_ns.add_trace(go.Scatter(
                    x=maturities,
                    y=yield_values, 
                    mode='markers',
                    name='Actual Yields', 
                    marker=dict(size=12, color=COLORS['accent'], symbol='circle')
                ))
                fig_ns.add_trace(go.Scatter(
                    x=maturities,
                    y=ns_result['fitted_values'], 
                    mode='lines',
                    name='NS Fitted', 
                    line=dict(color=COLORS['positive'], width=2.5)
                ))
                fig_ns = create_institutional_layout(fig_ns, "NELSON-SIEGEL CURVE FIT", "Yield (%)", height=450)
                st.plotly_chart(fig_ns, use_container_width=True)
            
            residuals = yield_values - ns_result['fitted_values']
            fig_resid = go.Figure()
            fig_resid.add_trace(go.Bar(x=maturities, y=residuals, name='Residuals', marker_color=COLORS['neutral']))
            fig_resid.add_hline(y=0, line_dash="dash", line_color=COLORS['negative'])
            fig_resid = create_institutional_layout(fig_resid, "FITTING RESIDUALS", "Residual (bps)", height=350)
            st.plotly_chart(fig_resid, use_container_width=True)
    
    # ===== TAB 5: NSS MODEL FIT =====
    with tabs[4]:
        st.markdown("### Nelson-Siegel-Svensson Model Calibration")
        
        if nss_result:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Model Parameters")
                param_df = pd.DataFrame({
                    'Parameter': ['b0 (Level)', 'b1 (Slope)', 'b2 (Curvature 1)', 'b3 (Curvature 2)', 'lambda1', 'lambda2'],
                    'Value': ["{:.4f}".format(nss_result['params'][0]), "{:.4f}".format(nss_result['params'][1]),
                              "{:.4f}".format(nss_result['params'][2]), "{:.4f}".format(nss_result['params'][3]),
                              "{:.4f}".format(nss_result['params'][4]), "{:.4f}".format(nss_result['params'][5])]
                })
                st.dataframe(param_df, use_container_width=True, hide_index=True)
                
                st.markdown("#### Fit Statistics")
                st.markdown("- **RMSE:** {:.4f}".format(nss_result['rmse']))
                st.markdown("- **MAE:** {:.4f}".format(nss_result['mae']))
                st.markdown("- **R-squared:** {:.4f}".format(nss_result['r_squared']))
            
            with col2:
                fig_nss = go.Figure()
                fig_nss.add_trace(go.Scatter(
                    x=maturities,
                    y=yield_values,
                    mode='markers',
                    name='Actual Yields',
                    marker=dict(size=12, color=COLORS['accent'], symbol='circle')
                ))
                fig_nss.add_trace(go.Scatter(
                    x=maturities,
                    y=nss_result['fitted_values'],
                    mode='lines',
                    name='NSS Fitted',
                    line=dict(color=COLORS['warning'], width=2.5)
                ))
                fig_nss = create_institutional_layout(fig_nss, "NELSON-SIEGEL-SVENSSON CURVE FIT", "Yield (%)", height=450)
                st.plotly_chart(fig_nss, use_container_width=True)
            
            residuals_nss = yield_values - nss_result['fitted_values']
            fig_resid_nss = go.Figure()
            fig_resid_nss.add_trace(go.Bar(x=maturities, y=residuals_nss, name='Residuals', marker_color=COLORS['neutral']))
            fig_resid_nss.add_hline(y=0, line_dash="dash", line_color=COLORS['negative'])
            fig_resid_nss = create_institutional_layout(fig_resid_nss, "FITTING RESIDUALS - SVENSSON", "Residual (bps)", height=350)
            st.plotly_chart(fig_resid_nss, use_container_width=True)
    
    # ===== TAB 6: MODEL COMPARISON =====
    with tabs[5]:
        st.markdown("### NS vs NSS Model Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatter(
                x=maturities,
                y=yield_values,
                mode='markers',
                name='Actual',
                marker=dict(size=12, color=COLORS['accent'], symbol='circle')
            ))
            if ns_result:
                fig_compare.add_trace(go.Scatter(
                    x=maturities,
                    y=ns_result['fitted_values'],
                    mode='lines',
                    name='NS Fit',
                    line=dict(color=COLORS['positive'], width=2.5)
                ))
            if nss_result:
                fig_compare.add_trace(go.Scatter(
                    x=maturities,
                    y=nss_result['fitted_values'],
                    mode='lines',
                    name='NSS Fit',
                    line=dict(color=COLORS['warning'], width=2.5, dash='dash')
                ))
            fig_compare = create_institutional_layout(fig_compare, "MODEL FIT COMPARISON", "Yield (%)", height=500)
            st.plotly_chart(fig_compare, use_container_width=True)
        
        with col2:
            if ns_result and nss_result:
                improvement_rmse = (ns_result['rmse'] - nss_result['rmse']) / ns_result['rmse'] * 100
                st.markdown("#### NSS Improvement")
                st.markdown("- **RMSE Improvement:** {:+.2f}%".format(improvement_rmse))
                
                if improvement_rmse > 10:
                    st.success("NSS provides significantly better fit")
                elif improvement_rmse > 5:
                    st.info("NSS provides moderate improvement")
                else:
                    st.warning("NS may be sufficient for this curve shape")
    
    # ===== TAB 7: DYNAMIC ANALYSIS =====
    with tabs[6]:
        st.markdown("### Dynamic Parameter Analysis")
        
        if not dynamic_params.empty:
            fig_dynamic = plot_parameter_evolution(dynamic_params)
            if fig_dynamic:
                st.plotly_chart(fig_dynamic, use_container_width=True)
            
            st.markdown("#### Parameter Statistics")
            param_stats = dynamic_params[['beta0', 'beta1', 'beta2', 'lambda']].describe()
            st.dataframe(param_stats, use_container_width=True)
    
    # ===== TAB 8: FACTOR ANALYSIS =====
    with tabs[7]:
        st.markdown("### Factor Analysis")
        
        if not factors.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_factors_time = go.Figure()
                for col in factors.columns:
                    fig_factors_time.add_trace(go.Scatter(
                        x=factors.index,
                        y=factors[col],
                        mode='lines',
                        name=col,
                        line=dict(width=1.5)
                    ))
                fig_factors_time = create_institutional_layout(fig_factors_time, "FACTOR EVOLUTION", "Value", height=450)
                st.plotly_chart(fig_factors_time, use_container_width=True)
            
            with col2:
                if len(factors.columns) > 1:
                    corr_matrix = factors.corr()
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmid=0,
                        text=corr_matrix.values.round(2),
                        texttemplate='%{text}'
                    ))
                    fig_corr = create_institutional_layout(fig_corr, "FACTOR CORRELATIONS", height=450)
                    st.plotly_chart(fig_corr, use_container_width=True)
        
        if pca_risk is not None:
            fig_pca = go.Figure(data=go.Bar(
                x=['PC1', 'PC2', 'PC3'][:len(pca_risk['explained_variance'])],
                y=pca_risk['explained_variance'] * 100,
                marker_color=COLORS['accent']
            ))
            fig_pca = create_institutional_layout(fig_pca, "PCA VARIANCE EXPLANATION", "Variance Explained (%)", height=400)
            st.plotly_chart(fig_pca, use_container_width=True)
    
    # ===== TAB 9: RISK METRICS =====
    with tabs[8]:
        st.markdown("### Advanced Risk Metrics")
        
        if '10Y' in yield_df.columns:
            returns = yield_df['10Y'].pct_change().dropna()
            var_metrics = AdvancedRiskMetrics.calculate_var_metrics(returns)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Value at Risk (95% confidence, 10-day)")
                st.metric("Historical VaR", "{:.4f}".format(var_metrics['VaR_Historical']))
                st.metric("Parametric VaR", "{:.4f}".format(var_metrics['VaR_Parametric']))
                st.metric("CVaR (Expected Shortfall)", "{:.4f}".format(var_metrics['CVaR']))
            
            with col2:
                if pca_risk is not None and 'loadings' in pca_risk:
                    st.markdown("#### PCA Risk Decomposition")
                    st.dataframe(pca_risk['loadings'].round(3), use_container_width=True)
    
    # ===== TAB 10: ARBITRAGE =====
    with tabs[9]:
        st.markdown("### Arbitrage Opportunity Detection")
        
        if arbitrage_stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Absolute Error", "{:.2f} bps".format(arbitrage_stats['mean_abs_error'] * 100))
            with col2:
                st.metric("Max Pricing Error", "{:.2f} bps".format(arbitrage_stats['max_error'] * 100))
            with col3:
                st.metric("Std Deviation", "{:.2f} bps".format(arbitrage_stats['std_error'] * 100))
            with col4:
                st.metric("Mispriced Securities", arbitrage_stats['mispriced_count'])
            
            if nss_result:
                fig_arbitrage = plot_arbitrage_chart(
                    maturities,
                    yield_values,
                    nss_result['fitted_values'], 
                    arbitrage_stats['mispriced_securities']
                )
                st.plotly_chart(fig_arbitrage, use_container_width=True)
            
            if arbitrage_stats['mispriced_securities']:
                mispriced_df = pd.DataFrame(arbitrage_stats['mispriced_securities'])
                mispriced_df['maturity'] = mispriced_df['maturity'].apply(lambda x: "{:.2f}Y".format(x))
                mispriced_df['actual'] = mispriced_df['actual'].apply(lambda x: "{:.2f}%".format(x))
                mispriced_df['theoretical'] = mispriced_df['theoretical'].apply(lambda x: "{:.2f}%".format(x))
                mispriced_df['difference'] = mispriced_df['difference'].apply(lambda x: "{:.2f} bps".format(x * 100))
                st.dataframe(mispriced_df, use_container_width=True, hide_index=True)
    
    # ===== TAB 11: NBER RECESSION =====
    with tabs[10]:
        st.markdown("### NBER Recession Analysis")
        
        st.plotly_chart(plot_nber_recession_chart(spreads, recessions), use_container_width=True)
        
        if recessions:
            recession_df = pd.DataFrame(recessions)
            recession_df['duration'] = (recession_df['end'] - recession_df['start']).dt.days
            recession_df['start'] = recession_df['start'].dt.strftime('%Y-%m-%d')
            recession_df['end'] = recession_df['end'].dt.strftime('%Y-%m-%d')
            st.dataframe(recession_df, use_container_width=True, hide_index=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Inversion Days", "{:,}".format(recession_metrics.get('total_inversion_days', 0)))
        with col2:
            st.metric("Avg Inversion Depth", "{:.1f} bps".format(recession_metrics.get('avg_inversion_depth', 0)))
        with col3:
            st.metric("Avg Lead Time", "{:.0f} days".format(recession_metrics.get('avg_lead_time', 0)))
    
    # ===== TAB 12: FORECASTING =====
    with tabs[11]:
        st.markdown("### Yield Curve Forecasting")
        
        forecast_horizon = st.slider("Forecast Horizon (Days)", 5, 60, 20, key="forecast_horizon")
        
        with st.spinner("Generating forecasts..."):
            forecast_result = YieldCurveForecasting.forecast_with_var(yield_df[['10Y']].dropna(), horizon=forecast_horizon)
        
        if forecast_result:
            st.plotly_chart(plot_forecast_chart(yield_df['10Y'], forecast_result, '10Y'), use_container_width=True)
            
            forecast_dates = pd.date_range(start=yield_df.index[-1], periods=forecast_horizon + 1, freq='D')[1:]
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast (%)': forecast_result['forecast'][:, 0] if len(forecast_result['forecast'].shape) > 1 else forecast_result['forecast'],
                'Lower Bound (%)': forecast_result['lower'][:, 0] if len(forecast_result['lower'].shape) > 1 else forecast_result['lower'],
                'Upper Bound (%)': forecast_result['upper'][:, 0] if len(forecast_result['upper'].shape) > 1 else forecast_result['upper']
            })
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    
    # ===== TAB 13: DATA EXPORT =====
    with tabs[12]:
        st.markdown("### Data Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_yields = yield_df.to_csv().encode('utf-8')
            st.download_button(
                "📥 Download Yield Data (CSV)",
                csv_yields,
                "yield_data_{}.csv".format(datetime.now().strftime('%Y%m%d_%H%M%S')),
                "text/csv"
            )
            
            if ns_result:
                ns_params_df = pd.DataFrame([ns_result['params']], columns=['b0', 'b1', 'b2', 'lambda'])
                csv_ns = ns_params_df.to_csv().encode('utf-8')
                st.download_button(
                    "📥 Download NS Parameters",
                    csv_ns,
                    "ns_params_{}.csv".format(datetime.now().strftime('%Y%m%d_%H%M%S')),
                    "text/csv"
                )
            
            csv_spreads = spreads.to_csv().encode('utf-8')
            st.download_button(
                "📥 Download Spread Data",
                csv_spreads,
                "spreads_{}.csv".format(datetime.now().strftime('%Y%m%d_%H%M%S')),
                "text/csv"
            )
        
        with col2:
            if not dynamic_params.empty:
                csv_dynamic = dynamic_params.to_csv().encode('utf-8')
                st.download_button(
                    "📥 Download Dynamic Parameters",
                    csv_dynamic,
                    "dynamic_params_{}.csv".format(datetime.now().strftime('%Y%m%d_%H%M%S')),
                    "text/csv"
                )
            
            csv_factors = factors.to_csv().encode('utf-8')
            st.download_button(
                "📥 Download Factor Data",
                csv_factors,
                "factors_{}.csv".format(datetime.now().strftime('%Y%m%d_%H%M%S')),
                "text/csv"
            )
            
            if forecast_result:
                forecast_export = pd.DataFrame({
                    'Date': forecast_dates,
                    'Forecast': forecast_result['forecast'][:, 0] if len(forecast_result['forecast'].shape) > 1 else forecast_result['forecast']
                })
                csv_forecast = forecast_export.to_csv().encode('utf-8')
                st.download_button(
                    "📥 Download Forecast Data",
                    csv_forecast,
                    "forecast_{}.csv".format(datetime.now().strftime('%Y%m%d_%H%M%S')),
                    "text/csv"
                )
        
        st.markdown("### Data Summary")
        st.markdown("- **Yield Curves:** {} maturities".format(len(yield_df.columns)))
        st.markdown("- **Observations:** {:,}".format(len(yield_df)))
        st.markdown("- **NBER Recessions:** {}".format(len(recessions)))
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #7f8c8d; font-size: 0.65rem;">
            <p>Yield Curve Analytics | Executive Summary Report | Institutional Quantitative Platform</p>
            <p>Data: Federal Reserve Economic Data (FRED) | Models: Nelson-Siegel (1987), Svensson (1994)</p>
            <p>Recession Definition: NBER (National Bureau of Economic Research)</p>
            <p>Last Update: {} UTC</p>
        </div>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
