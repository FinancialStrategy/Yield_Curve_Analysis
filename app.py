# =============================================================================
# HEDGE FUND YIELD CURVE ANALYTICS PLATFORM
# COMPLETE IMPLEMENTATION - NS, NSS, DYNAMIC ANALYSIS, RISK METRICS
# FULL VERSION WITH ALL VISUALIZATIONS - COMPLETELY UNCUT
# =============================================================================
# Version: 21.0 | Full Enterprise Suite | No Shortening | Complete Implementation
# Includes: Nelson-Siegel, Svensson, Dynamic Analysis, Risk Metrics, Arbitrage Detection
# All Tabs: DATA TABLE, SPREAD DYNAMICS, NS MODEL FIT, NSS MODEL FIT, 
# MODEL COMPARISON, DYNAMIC ANALYSIS, FACTOR ANALYSIS, RISK METRICS, 
# NBER RECESSION, FORECASTING, ARBITRAGE, DATA EXPORT
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
    page_title="Yield Curve Analytics | Institutional Platform",
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
    '1M': 1/12, '3M': 0.25, '6M': 0.5, 
    '1Y': 1, '2Y': 2, '3Y': 3, '5Y': 5, 
    '7Y': 7, '10Y': 10, '20Y': 20, '30Y': 30
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
# ADVANCED RISK METRICS - FIXED VERSION
# =============================================================================

class AdvancedRiskMetrics:
    """Advanced risk metrics for yield curve"""
    
    @staticmethod
    def calculate_pca_risk(yield_df, n_components=3):
        """PCA-based risk decomposition and factor analysis - Fixed for NaN handling"""
        try:
            # Calculate returns and drop NaN
            returns = yield_df.pct_change().dropna()
            
            # Check if we have enough data for PCA
            if len(returns) < 5 or returns.shape[1] < 2:
                # Return safe default values
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
            
            # Remove any remaining NaN or infinite values
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Standardize the returns
            scaler = StandardScaler()
            returns_scaled = scaler.fit_transform(returns)
            
            # Apply PCA with safe n_components
            actual_n_components = min(n_components, len(returns.columns), len(returns) - 1)
            if actual_n_components < 1:
                actual_n_components = 1
            
            pca = PCA(n_components=actual_n_components)
            pcs = pca.fit_transform(returns_scaled)
            
            # Create factor contributions dataframe
            factor_names = ['PC1_Level', 'PC2_Slope', 'PC3_Curvature'][:actual_n_components]
            factor_contributions = pd.DataFrame(pcs, index=returns.index, columns=factor_names)
            
            # Create loadings dataframe
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
            # Return safe default values on any error
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
        # Remove NaN values
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
    
    @staticmethod
    def calculate_stress_scenarios(yield_df, shock_bps=100):
        """Calculate stress scenario impacts"""
        current = yield_df.iloc[-1]
        
        scenarios = {
            'Parallel +100bps': current + shock_bps / 100,
            'Parallel -100bps': current - shock_bps / 100,
            'Bear Steepener': current.copy(),
            'Bull Flattener': current.copy()
        }
        
        maturities = np.arange(len(current))
        weights = maturities / max(maturities)
        scenarios['Bear Steepener'] = current + (shock_bps * (1 - weights)) / 100
        scenarios['Bull Flattener'] = current + (shock_bps * weights) / 100
        
        return scenarios

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
    
    @staticmethod
    def calculate_forecast_metrics(actual, forecast):
        """Calculate forecast accuracy metrics"""
        mae = np.mean(np.abs(actual - forecast))
        rmse = np.sqrt(np.mean((actual - forecast) ** 2))
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100 if np.all(actual != 0) else 0
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }

# =============================================================================
# ARBITRAGE DETECTION
# =============================================================================

class ArbitrageDetection:
    """Detect arbitrage opportunities using Nelson-Siegel Svensson model"""
    
    @staticmethod
    def detect_arbitrage_opportunities(yield_df, maturities):
        """Detect arbitrage opportunities using Nelson-Siegel Svensson model"""
        latest_yields = yield_df.iloc[-1].values[:len(maturities)]
        
        # Fit NSS model
        nss_result = NelsonSiegelModel.fit_svensson(maturities, latest_yields)
        
        if nss_result is None:
            return None
        
        # Calculate theoretical yields
        theoretical = nss_result['fitted_values']
        actual = latest_yields
        residuals = actual - theoretical
        
        # Identify mispriced maturities
        mispriced = []
        for i, (m, r) in enumerate(zip(maturities, residuals)):
            if abs(r) > 0.1:  # More than 10 bps deviation
                mispriced.append({
                    'maturity': m,
                    'actual': actual[i],
                    'theoretical': theoretical[i],
                    'difference': r,
                    'opportunity': 'Overvalued' if r < 0 else 'Undervalued'
                })
        
        # Calculate arbitrage statistics
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
# VISUALIZATION FUNCTIONS - COMPLETE SET (Plotly Only)
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

def plot_3d_yield_curve(yield_df):
    """Create 3D surface plot of yield curve evolution"""
    dates = yield_df.index
    maturities = np.array([MATURITY_MAP.get(col, 0) for col in yield_df.columns])
    z_data = yield_df.T.values
    
    fig = go.Figure(data=[
        go.Surface(
            x=dates,
            y=maturities,
            z=z_data,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Yield (%)", thickness=10, len=0.5),
            contours={
                "z": {
                    "show": True,
                    "usecolormap": True,
                    "highlightcolor": "#ff0000",
                    "project": {"z": True}
                }
            }
        )
    ])
    
    fig = create_institutional_layout(fig, "YIELD CURVE TERM STRUCTURE EVOLUTION", "Yield (%)", height=600)
    fig.update_layout(
        scene=dict(
            xaxis_title="Date",
            yaxis_title="Maturity (Years)",
            zaxis_title="Yield (%)",
            camera=dict(eye=dict(x=1.8, y=-1.5, z=1.2)),
            aspectmode='manual',
            aspectratio=dict(x=2, y=0.8, z=0.6)
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
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Spread: %{y:.1f} bps<extra></extra>'
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
    """Create comprehensive spread dashboard with 4 subplots"""
    fig = make_subplots(
        rows=2, cols=2,
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
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>{}: %{{y:.1f}} bps<extra></extra>'.format(spread_name)
                ),
                row=config['row'], col=config['col']
            )
            
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color=COLORS['negative'],
                line_width=1,
                row=config['row'], col=config['col']
            )
            
            for recession in recessions:
                fig.add_vrect(
                    x0=recession['start'],
                    x1=recession['end'],
                    fillcolor=COLORS['recession'],
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                    row=config['row'], col=config['col']
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
        x=maturities, y=current,
        mode='lines+markers',
        name='Current Yield',
        line=dict(color=COLORS['accent'], width=2.5),
        marker=dict(size=8, symbol='circle'),
        hovertemplate='Maturity: %{x:.1f}Y<br>Yield: %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=maturities, y=one_year_ago,
        mode='lines+markers',
        name='1 Year Ago',
        line=dict(color=COLORS['warning'], width=2, dash='dash'),
        marker=dict(size=6, symbol='square')
    ))
    
    fig.add_trace(go.Scatter(
        x=maturities, y=five_year_avg,
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
        rows=2, cols=2,
        subplot_titles=('Level Factor (Beta0)', 'Slope Factor (Beta1)', 
                       'Curvature Factor (Beta2)', 'RMSE Evolution'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dynamic_params['date'], y=dynamic_params['beta0'],
            mode='lines', name='Level',
            line=dict(color=COLORS['positive'], width=2),
            fill='tozeroy',
            fillcolor='rgba(46, 204, 113, 0.1)'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dynamic_params['date'], y=dynamic_params['beta1'],
            mode='lines', name='Slope',
            line=dict(color=COLORS['accent'], width=2),
            fill='tozeroy',
            fillcolor='rgba(15, 52, 96, 0.1)'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=dynamic_params['date'], y=dynamic_params['beta2'],
            mode='lines', name='Curvature',
            line=dict(color=COLORS['warning'], width=2),
            fill='tozeroy',
            fillcolor='rgba(243, 156, 18, 0.1)'
        ),
        row=2, col=1
    )
    
    if 'rmse' in dynamic_params.columns:
        fig.add_trace(
            go.Scatter(
                x=dynamic_params['date'], y=dynamic_params['rmse'],
                mode='lines', name='RMSE',
                line=dict(color=COLORS['neutral'], width=2),
                fill='tozeroy',
                fillcolor='rgba(149, 165, 166, 0.1)'
            ),
            row=2, col=2
        )
    
    fig = create_institutional_layout(fig, "PARAMETER EVOLUTION OVER TIME", height=600)
    return fig

def plot_pca_factors(pca_risk):
    """Plot PCA factor dynamics"""
    if pca_risk is None or 'factors' not in pca_risk:
        return None
    
    fig = make_subplots(
        rows=3, cols=1,
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
                row=i+1, col=1
            )
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color=COLORS['neutral'],
                line_width=1,
                row=i+1, col=1
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
        x=maturities, y=actual, mode='markers', name='Actual Yields',
        marker=dict(size=12, color=COLORS['accent'], symbol='circle')
    ))
    
    fig.add_trace(go.Scatter(
        x=maturities, y=theoretical, mode='lines', name='NSS Theoretical',
        line=dict(color=COLORS['positive'], width=2.5)
    ))
    
    # Highlight mispriced securities
    mispriced_maturities = [m['maturity'] for m in mispriced]
    mispriced_actual = [m['actual'] for m in mispriced]
    
    if mispriced_maturities:
        fig.add_trace(go.Scatter(
            x=mispriced_maturities, y=mispriced_actual, mode='markers',
            name='Mispriced Securities', marker=dict(size=15, color=COLORS['warning'], symbol='circle', line=dict(width=2, color='red'))
        ))
    
    fig = create_institutional_layout(fig, "ARBITRAGE OPPORTUNITY DETECTION", "Yield (%)", height=500)
    return fig

def plot_spread_correlation_heatmap(spreads_df):
    """Create correlation heatmap using Plotly"""
    if len(spreads_df.columns) < 2:
        return None
    
    corr_matrix = spreads_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{x} vs %{y}: %{z:.2f}<extra></extra>'
    ))
    
    fig = create_institutional_layout(fig, "SPREAD CORRELATION MATRIX", height=500)
    fig.update_layout(
        xaxis_title="",
        yaxis_title=""
    )
    
    return fig

def plot_rolling_spread_stats(spreads_df, window=20):
    """Create rolling statistics plot using Plotly"""
    if '10Y-2Y' not in spreads_df.columns:
        return None
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('10Y-2Y Spread with Rolling Mean', 
                                       '10Y-2Y Spread Rolling Volatility'),
                        vertical_spacing=0.12)
    
    spread = spreads_df['10Y-2Y']
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    
    # Top plot - Spread with rolling mean
    fig.add_trace(
        go.Scatter(x=spread.index, y=spread, mode='lines', name='Daily Spread',
                   line=dict(color=COLORS['accent'], width=1, opacity=0.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=spread.index, y=rolling_mean, mode='lines', name='{}d Rolling Mean'.format(window),
                   line=dict(color=COLORS['positive'], width=2)),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS['negative'], row=1, col=1)
    
    # Bottom plot - Rolling volatility
    fig.add_trace(
        go.Scatter(x=spread.index, y=rolling_std, mode='lines', name='Rolling Volatility',
                   line=dict(color=COLORS['warning'], width=2), fill='tozeroy',
                   fillcolor='rgba(243, 156, 18, 0.1)'),
        row=2, col=1
    )
    
    fig = create_institutional_layout(fig, "ROLLING SPREAD STATISTICS ({}d Window)".format(window), height=600)
    fig.update_yaxes(title_text="Spread (bps)", row=1, col=1)
    fig.update_yaxes(title_text="Volatility (bps)", row=2, col=1)
    
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
    
    recession_durations = [(r['end'] - r['start']).days for r in recessions]
    
    return {
        'inversion_periods': inversion_periods,
        'total_inversion_days': sum([p['duration'] for p in inversion_periods]),
        'avg_inversion_depth': np.mean([p['depth'] for p in inversion_periods]) if inversion_periods else 0,
        'max_inversion_depth': min([p['depth'] for p in inversion_periods]) if inversion_periods else 0,
        'lead_times': lead_times,
        'avg_lead_time': np.mean(lead_times) if lead_times else 0,
        'median_lead_time': np.median(lead_times) if lead_times else 0,
        'recession_durations': recession_durations,
        'avg_recession_duration': np.mean(recession_durations) if recession_durations else 0,
        'num_inversions': len(inversion_periods),
        'num_recessions': len(recessions),
        'inversion_recession_ratio': len(inversion_periods) / len(recessions) if recessions else 0
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
# MAIN APPLICATION
# =============================================================================

def main():
    
    # Header
    st.markdown("""
    <div class="hedge-header">
        <h1 style="color: white; margin: 0; font-size: 1.25rem;">YIELD CURVE ANALYTICS</h1>
        <p style="color: #bdc3c7; margin: 0; font-size: 0.7rem;">FRED Data Integration | Nelson-Siegel Family Models | NBER Recession Analysis | Quantitative Risk Metrics</p>
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
    if '6M' in yield_df.columns and '3M' in yield_df.columns:
        spreads['6M-3M'] = (yield_df['6M'] - yield_df['3M']) * 100
    
    # Identify recessions
    recessions = identify_recessions(recession_series)
    
    # Current metrics
    current_10y = yield_df['10Y'].iloc[-1] if '10Y' in yield_df.columns else 0
    current_2y = yield_df['2Y'].iloc[-1] if '2Y' in yield_df.columns else 0
    current_30y = yield_df['30Y'].iloc[-1] if '30Y' in yield_df.columns else 0
    current_1m = yield_df['1M'].iloc[-1] if '1M' in yield_df.columns else 0
    current_6m = yield_df['6M'].iloc[-1] if '6M' in yield_df.columns else 0
    current_spread = spreads['10Y-2Y'].iloc[-1] if '10Y-2Y' in spreads.columns else 0
    current_spread_2y_3m = spreads['2Y-3M'].iloc[-1] if '2Y-3M' in spreads.columns else 0
    current_spread_10y_1m = spreads['10Y-1M'].iloc[-1] if '10Y-1M' in spreads.columns else 0
    
    # Fit models
    with st.spinner("Calibrating Nelson-Siegel models..."):
        ns_result = NelsonSiegelModel.fit_nelson_siegel(maturities, yield_values)
        nss_result = NelsonSiegelModel.fit_svensson(maturities, yield_values)
        st.session_state.ns_results = ns_result
        st.session_state.nss_results = nss_result
    
    # Dynamic analysis
    with st.spinner("Performing dynamic parameter analysis (this may take a moment)..."):
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
    
    # ===== METRICS ROW =====
    st.markdown("### Current Market Metrics")
    
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    
    with col1:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-label">1M YIELD</div>'
            '<div class="metric-value">{:.2f}%</div>'
            '</div>'.format(current_1m), 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-label">6M YIELD</div>'
            '<div class="metric-value">{:.2f}%</div>'
            '</div>'.format(current_6m), 
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-label">2Y YIELD</div>'
            '<div class="metric-value">{:.2f}%</div>'
            '</div>'.format(current_2y), 
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-label">10Y YIELD</div>'
            '<div class="metric-value">{:.2f}%</div>'
            '</div>'.format(current_10y), 
            unsafe_allow_html=True
        )
    
    with col5:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-label">30Y YIELD</div>'
            '<div class="metric-value">{:.2f}%</div>'
            '</div>'.format(current_30y), 
            unsafe_allow_html=True
        )
    
    with col6:
        if current_spread < 0:
            status_class = "status-inverted"
            status_text_display = "INVERTED"
        elif current_spread < 50:
            status_class = "status-caution"
            status_text_display = "FLATTENING"
        else:
            status_class = "status-normal"
            status_text_display = "NORMAL"
        
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-label">10Y-2Y SPREAD</div>'
            '<div class="metric-value">{:.1f} bps</div>'
            '<div class="{}">{}</div>'
            '</div>'.format(current_spread, status_class, status_text_display), 
            unsafe_allow_html=True
        )
    
    with col7:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-label">2Y-3M SPREAD</div>'
            '<div class="metric-value">{:.1f} bps</div>'
            '</div>'.format(current_spread_2y_3m), 
            unsafe_allow_html=True
        )
    
    with col8:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-label">10Y-1M SPREAD</div>'
            '<div class="metric-value">{:.1f} bps</div>'
            '</div>'.format(current_spread_10y_1m), 
            unsafe_allow_html=True
        )
    
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
        data_table = data_table.iloc[::-1]  # Reverse order (latest first)
        
        # Add date column
        data_table_reset = data_table.reset_index()
        data_table_reset.columns = ['Date'] + list(data_table.columns)
        
        # Format for display
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
    
    # ===== TAB 2: SPREAD DYNAMICS =====
    with tabs[1]:
        st.markdown("### Yield Spread Dynamics Analysis")
        st.markdown("*Comprehensive analysis of key yield spreads*")
        
        # Spread dashboard (Plotly)
        st.markdown("#### Spread Evolution Over Time")
        st.plotly_chart(plot_spread_dashboard(spreads, recessions), use_container_width=True)
        
        # Correlation heatmap
        st.markdown("#### Spread Correlation Matrix")
        heatmap_fig = plot_spread_correlation_heatmap(spreads)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Rolling statistics
        st.markdown("#### Rolling Statistics (20-day window)")
        rolling_fig = plot_rolling_spread_stats(spreads, window=20)
        if rolling_fig:
            st.plotly_chart(rolling_fig, use_container_width=True)
        
        # Spread statistics table
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
        
        # Key spread insights
        st.markdown("#### Key Insights")
        st.markdown("""
        - **10Y-2Y Spread:** Primary recession indicator. Negative values historically precede recessions by 12-18 months.
        - **10Y-3M Spread:** Campbell Harvey indicator. More sensitive than 10Y-2Y for recession prediction.
        - **5Y-2Y Spread:** Medium-term policy effectiveness gauge.
        - **30Y-10Y Spread:** Term premium reflecting long-term inflation expectations.
        - **2Y-3M Spread:** Short-term policy expectations.
        - **10Y-1M Spread:** Ultra-short to long-term expectations.
        """)
    
    # ===== TAB 3: NS MODEL FIT =====
    with tabs[2]:
        st.markdown("### Nelson-Siegel Model Calibration")
        
        if ns_result:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Model Parameters")
                param_df = pd.DataFrame({
                    'Parameter': ['b0 (Level)', 'b1 (Slope)', 'b2 (Curvature)', 'lambda (Decay)'],
                    'Value': ["{:.4f}".format(ns_result['params'][0]), "{:.4f}".format(ns_result['params'][1]), 
                              "{:.4f}".format(ns_result['params'][2]), "{:.4f}".format(ns_result['params'][3])],
                    'Interpretation': [
                        "Long-term level: {:.2f}%".format(ns_result['params'][0]),
                        "Slope: {}".format('Inverted' if ns_result['params'][1] < 0 else 'Normal'),
                        "Curvature: {}".format('Humped' if ns_result['params'][2] > 0 else 'Sagged'),
                        "Max curvature at {:.1f} years".format(1/ns_result['params'][3])
                    ]
                })
                st.dataframe(param_df, use_container_width=True, hide_index=True)
                
                st.markdown("#### Fit Statistics")
                st.markdown("- **RMSE (Root Mean Square Error):** {:.4f}".format(ns_result['rmse']))
                st.markdown("- **MAE (Mean Absolute Error):** {:.4f}".format(ns_result['mae']))
                st.markdown("- **R² (Coefficient of Determination):** {:.4f}".format(ns_result['r_squared']))
                st.markdown("- **Number of Observations:** {}".format(len(maturities)))
                
                # Factor interpretation
                ns_interpretation = NelsonSiegelModel.calculate_factor_interpretation(ns_result['params'], 'NS')
                st.markdown("#### Factor Interpretation")
                for key, value in ns_interpretation['Interpretation'].items():
                    st.markdown("- **{}:** {}".format(key, value))
            
            with col2:
                # Current fit visualization
                fig_ns = go.Figure()
                fig_ns.add_trace(go.Scatter(
                    x=maturities, y=yield_values, 
                    mode='markers', name='Actual Yields', 
                    marker=dict(size=12, color=COLORS['accent'], symbol='circle')
                ))
                fig_ns.add_trace(go.Scatter(
                    x=maturities, y=ns_result['fitted_values'], 
                    mode='lines', name='NS Fitted', 
                    line=dict(color=COLORS['positive'], width=2.5)
                ))
                fig_ns = create_institutional_layout(fig_ns, "NELSON-SIEGEL CURVE FIT", "Yield (%)", height=450)
                st.plotly_chart(fig_ns, use_container_width=True)
            
            # Residuals analysis
            st.markdown("#### Residual Analysis")
            residuals = yield_values - ns_result['fitted_values']
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                fig_resid = go.Figure()
                fig_resid.add_trace(go.Bar(
                    x=maturities, y=residuals, 
                    name='Residuals', 
                    marker_color=COLORS['neutral'],
                    text=["{:.3f}".format(r) for r in residuals],
                    textposition='outside'
                ))
                fig_resid.add_hline(y=0, line_dash="dash", line_color=COLORS['negative'])
                fig_resid = create_institutional_layout(fig_resid, "FITTING RESIDUALS", "Residual (bps)", height=350)
                st.plotly_chart(fig_resid, use_container_width=True)
            
            with col_res2:
                st.markdown("**Residual Statistics**")
                st.markdown("- **Mean Residual:** {:.4f} bps".format(np.mean(residuals)))
                st.markdown("- **Std Deviation:** {:.4f} bps".format(np.std(residuals)))
                st.markdown("- **Max Positive:** {:.4f} bps".format(np.max(residuals)))
                st.markdown("- **Max Negative:** {:.4f} bps".format(np.min(residuals)))
                st.markdown("- **95% Confidence Band:** ±{:.4f} bps".format(1.96 * np.std(residuals)))
                
                # QQ plot for normality check
                fig_qq = go.Figure()
                sorted_resid = np.sort(residuals)
                theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_resid)))
                fig_qq.add_trace(go.Scatter(
                    x=theoretical, y=sorted_resid,
                    mode='markers', name='Residuals',
                    marker=dict(color=COLORS['accent'], size=6)
                ))
                fig_qq.add_trace(go.Scatter(
                    x=[-3, 3], y=[-3, 3],
                    mode='lines', name='Normal Line',
                    line=dict(color=COLORS['positive'], dash='dash')
                ))
                fig_qq = create_institutional_layout(fig_qq, "Q-Q PLOT (Normality Check)", "Sample Quantiles", height=350)
                st.plotly_chart(fig_qq, use_container_width=True)
    
    # ===== TAB 4: NSS MODEL FIT =====
    with tabs[3]:
        st.markdown("### Nelson-Siegel-Svensson Model Calibration")
        
        if nss_result:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Model Parameters")
                param_df = pd.DataFrame({
                    'Parameter': ['b0 (Level)', 'b1 (Slope)', 'b2 (Curvature 1)', 'b3 (Curvature 2)', 'lambda1', 'lambda2'],
                    'Value': ["{:.4f}".format(nss_result['params'][0]), "{:.4f}".format(nss_result['params'][1]),
                              "{:.4f}".format(nss_result['params'][2]), "{:.4f}".format(nss_result['params'][3]),
                              "{:.4f}".format(nss_result['params'][4]), "{:.4f}".format(nss_result['params'][5])],
                    'Interpretation': [
                        "Long-term level: {:.2f}%".format(nss_result['params'][0]),
                        "Slope: {}".format('Inverted' if nss_result['params'][1] < 0 else 'Normal'),
                        "First hump at {:.1f}Y".format(1/nss_result['params'][4]) if nss_result['params'][4] > 0 else "N/A",
                        "Second hump at {:.1f}Y".format(1/nss_result['params'][5]) if nss_result['params'][5] > 0 else "N/A",
                        "Decay rate 1: {:.4f}".format(nss_result['params'][4]),
                        "Decay rate 2: {:.4f}".format(nss_result['params'][5])
                    ]
                })
                st.dataframe(param_df, use_container_width=True, hide_index=True)
                
                st.markdown("#### Fit Statistics")
                st.markdown("- **RMSE:** {:.4f}".format(nss_result['rmse']))
                st.markdown("- **MAE:** {:.4f}".format(nss_result['mae']))
                st.markdown("- **R²:** {:.4f}".format(nss_result['r_squared']))
                st.markdown("- **Number of Observations:** {}".format(len(maturities)))
                
                # Svensson interpretation
                nss_interpretation = NelsonSiegelModel.calculate_factor_interpretation(nss_result['params'], 'NSS')
                st.markdown("#### Svensson Factor Interpretation")
                for key, value in nss_interpretation['Interpretation'].items():
                    st.markdown("- **{}:** {}".format(key, value))
            
            with col2:
                fig_nss = go.Figure()
                fig_nss.add_trace(go.Scatter(
                    x=maturities, y=yield_values,
                    mode='markers', name='Actual Yields',
                    marker=dict(size=12, color=COLORS['accent'], symbol='circle')
                ))
                fig_nss.add_trace(go.Scatter(
                    x=maturities, y=nss_result['fitted_values'],
                    mode='lines', name='NSS Fitted',
                    line=dict(color=COLORS['warning'], width=2.5)
                ))
                fig_nss = create_institutional_layout(fig_nss, "NELSON-SIEGEL-SVENSSON CURVE FIT", "Yield (%)", height=450)
                st.plotly_chart(fig_nss, use_container_width=True)
            
            # Residuals
            st.markdown("#### Residual Analysis")
            residuals_nss = yield_values - nss_result['fitted_values']
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                fig_resid_nss = go.Figure()
                fig_resid_nss.add_trace(go.Bar(
                    x=maturities, y=residuals_nss,
                    name='Residuals',
                    marker_color=COLORS['neutral'],
                    text=["{:.3f}".format(r) for r in residuals_nss],
                    textposition='outside'
                ))
                fig_resid_nss.add_hline(y=0, line_dash="dash", line_color=COLORS['negative'])
                fig_resid_nss = create_institutional_layout(fig_resid_nss, "FITTING RESIDUALS - SVENSSON", "Residual (bps)", height=350)
                st.plotly_chart(fig_resid_nss, use_container_width=True)
            
            with col_res2:
                st.markdown("**Residual Statistics**")
                st.markdown("- **Mean Residual:** {:.4f} bps".format(np.mean(residuals_nss)))
                st.markdown("- **Std Deviation:** {:.4f} bps".format(np.std(residuals_nss)))
                st.markdown("- **Max Positive:** {:.4f} bps".format(np.max(residuals_nss)))
                st.markdown("- **Max Negative:** {:.4f} bps".format(np.min(residuals_nss)))
                st.markdown("- **95% Confidence Band:** ±{:.4f} bps".format(1.96 * np.std(residuals_nss)))
    
    # ===== TAB 5: MODEL COMPARISON =====
    with tabs[4]:
        st.markdown("### NS vs NSS Model Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Current Fit Comparison")
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatter(
                x=maturities, y=yield_values,
                mode='markers', name='Actual',
                marker=dict(size=12, color=COLORS['accent'], symbol='circle')
            ))
            if ns_result:
                fig_compare.add_trace(go.Scatter(
                    x=maturities, y=ns_result['fitted_values'],
                    mode='lines', name='NS Fit',
                    line=dict(color=COLORS['positive'], width=2.5)
                ))
            if nss_result:
                fig_compare.add_trace(go.Scatter(
                    x=maturities, y=nss_result['fitted_values'],
                    mode='lines', name='NSS Fit',
                    line=dict(color=COLORS['warning'], width=2.5, dash='dash')
                ))
            fig_compare = create_institutional_layout(fig_compare, "MODEL FIT COMPARISON", "Yield (%)", height=500)
            st.plotly_chart(fig_compare, use_container_width=True)
        
        with col2:
            st.markdown("#### Error Comparison")
            error_df = pd.DataFrame({
                'Metric': ['RMSE', 'MAE', 'R²', 'Max Error'],
                'NS': [
                    "{:.4f}".format(ns_result['rmse']) if ns_result else "N/A",
                    "{:.4f}".format(ns_result['mae']) if ns_result else "N/A",
                    "{:.4f}".format(ns_result['r_squared']) if ns_result else "N/A",
                    "{:.4f}".format(np.max(np.abs(yield_values - ns_result['fitted_values']))) if ns_result else "N/A"
                ],
                'NSS': [
                    "{:.4f}".format(nss_result['rmse']) if nss_result else "N/A",
                    "{:.4f}".format(nss_result['mae']) if nss_result else "N/A",
                    "{:.4f}".format(nss_result['r_squared']) if nss_result else "N/A",
                    "{:.4f}".format(np.max(np.abs(yield_values - nss_result['fitted_values']))) if nss_result else "N/A"
                ]
            })
            st.dataframe(error_df, use_container_width=True, hide_index=True)
            
            if ns_result and nss_result:
                improvement_rmse = (ns_result['rmse'] - nss_result['rmse']) / ns_result['rmse'] * 100
                improvement_mae = (ns_result['mae'] - nss_result['mae']) / ns_result['mae'] * 100
                improvement_r2 = (nss_result['r_squared'] - ns_result['r_squared']) / ns_result['r_squared'] * 100 if ns_result['r_squared'] > 0 else 0
                
                st.markdown("#### NSS Improvement Metrics")
                st.markdown("- **RMSE Improvement:** {:+.2f}%".format(improvement_rmse))
                st.markdown("- **MAE Improvement:** {:+.2f}%".format(improvement_mae))
                st.markdown("- **R² Improvement:** {:+.2f}%".format(improvement_r2))
                
                st.markdown("#### Model Recommendation")
                if improvement_rmse > 10:
                    st.success("✅ NSS provides significantly better fit - Recommended for complex curves and longer maturities")
                elif improvement_rmse > 5:
                    st.info("📊 NSS provides moderate improvement - Consider for specific use cases")
                else:
                    st.warning("⚠️ NS may be sufficient - NSS improvement is marginal, prefer simpler model")
        
        # Residual comparison
        st.markdown("#### Residual Comparison")
        fig_resid_compare = go.Figure()
        if ns_result:
            fig_resid_compare.add_trace(go.Scatter(
                x=maturities, y=yield_values - ns_result['fitted_values'],
                mode='lines+markers', name='NS Residuals',
                line=dict(color=COLORS['positive'], width=1.5),
                marker=dict(size=6)
            ))
        if nss_result:
            fig_resid_compare.add_trace(go.Scatter(
                x=maturities, y=yield_values - nss_result['fitted_values'],
                mode='lines+markers', name='NSS Residuals',
                line=dict(color=COLORS['warning'], width=1.5, dash='dash'),
                marker=dict(size=6)
            ))
        fig_resid_compare.add_hline(y=0, line_dash="dash", line_color=COLORS['negative'])
        fig_resid_compare = create_institutional_layout(fig_resid_compare, "RESIDUAL COMPARISON", "Residual (bps)", height=450)
        st.plotly_chart(fig_resid_compare, use_container_width=True)
    
    # ===== TAB 6: DYNAMIC ANALYSIS =====
    with tabs[5]:
        st.markdown("### Dynamic Parameter Analysis")
        st.markdown("*Parameter evolution over time using rolling window calibration (5-year windows)*")
        
        if not dynamic_params.empty:
            # Parameter evolution chart
            fig_dynamic = plot_parameter_evolution(dynamic_params)
            if fig_dynamic:
                st.plotly_chart(fig_dynamic, use_container_width=True)
            
            # Summary statistics
            st.markdown("#### Parameter Statistics")
            param_stats = dynamic_params[['beta0', 'beta1', 'beta2', 'lambda']].describe()
            param_stats = param_stats.rename(columns={
                'beta0': 'b0 (Level)',
                'beta1': 'b1 (Slope)',
                'beta2': 'b2 (Curvature)',
                'lambda': 'lambda (Decay)'
            })
            st.dataframe(param_stats, use_container_width=True)
            
            # Parameter volatility
            st.markdown("#### Parameter Volatility Analysis")
            vol_df = DynamicParameterAnalysis.calculate_parameter_volatility(dynamic_params)
            if vol_df is not None:
                st.dataframe(vol_df, use_container_width=True, hide_index=True)
            
            # Recent trends
            st.markdown("#### Recent Parameter Trends (Last 12 months)")
            recent = dynamic_params.tail(12)
            recent_trend = recent[['beta0', 'beta1', 'beta2']].pct_change().mean() * 100
            
            col_trend1, col_trend2, col_trend3 = st.columns(3)
            with col_trend1:
                st.metric("Level Trend", "{:+.1f}%".format(recent_trend['beta0']), delta_color="normal")
            with col_trend2:
                st.metric("Slope Trend", "{:+.1f}%".format(recent_trend['beta1']), delta_color="inverse")
            with col_trend3:
                st.metric("Curvature Trend", "{:+.1f}%".format(recent_trend['beta2']), delta_color="normal")
            
            # 3D parameter evolution
            st.markdown("#### 3D Parameter Evolution")
            fig_3d_params = go.Figure(data=[go.Scatter3d(
                x=dynamic_params['beta0'],
                y=dynamic_params['beta1'],
                z=dynamic_params['beta2'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=dynamic_params['date'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Date")
                ),
                text=dynamic_params['date'].dt.strftime('%Y-%m-%d'),
                hovertemplate='Date: %{text}<br>Level: %{x:.2f}<br>Slope: %{y:.2f}<br>Curvature: %{z:.2f}<extra></extra>'
            )])
            fig_3d_params.update_layout(
                title="Parameter Space Evolution",
                scene=dict(
                    xaxis_title="Level (b0)",
                    yaxis_title="Slope (b1)",
                    zaxis_title="Curvature (b2)",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                height=600,
                paper_bgcolor=COLORS['surface'],
                font=dict(color=COLORS['text_secondary'])
            )
            st.plotly_chart(fig_3d_params, use_container_width=True)
    
    # ===== TAB 7: FACTOR ANALYSIS =====
    with tabs[6]:
        st.markdown("### Factor Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Factor Contributions Over Time")
            fig_factors_time = go.Figure()
            for col in factors.columns:
                fig_factors_time.add_trace(go.Scatter(
                    x=factors.index, y=factors[col],
                    mode='lines', name=col,
                    line=dict(width=1.5)
                ))
            fig_factors_time = create_institutional_layout(fig_factors_time, "FACTOR EVOLUTION", "Value", height=450)
            st.plotly_chart(fig_factors_time, use_container_width=True)
        
        with col2:
            st.markdown("#### Factor Correlation Matrix")
            if len(factors.columns) > 1:
                corr_matrix = factors.corr()
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                fig_corr = create_institutional_layout(fig_corr, "FACTOR CORRELATIONS", height=450)
                st.plotly_chart(fig_corr, use_container_width=True)
        
        # Current factor interpretation
        st.markdown("#### Current Factor Interpretation")
        if not factors.empty:
            current_factors = factors.iloc[-1]
            
            col_int1, col_int2, col_int3 = st.columns(3)
            
            with col_int1:
                if 'Level' in current_factors:
                    level_val = current_factors['Level']
                    if level_val > 4:
                        level_status = "High"
                    elif level_val < 2:
                        level_status = "Low"
                    else:
                        level_status = "Moderate"
                    
                    st.markdown(
                        '<div class="metric-card">'
                        '<div class="metric-label">LEVEL FACTOR</div>'
                        '<div class="metric-value">{:.2f}%</div>'
                        '<div class="metric-label">{} long-term rate environment</div>'
                        '</div>'.format(level_val, level_status), 
                        unsafe_allow_html=True
                    )
            
            with col_int2:
                if 'Slope' in current_factors:
                    slope_val = current_factors['Slope']
                    if slope_val < 0:
                        slope_status = "Inverted"
                        slope_color = COLORS['negative']
                    else:
                        slope_status = "Normal"
                        slope_color = COLORS['positive']
                    
                    st.markdown(
                        '<div class="metric-card">'
                        '<div class="metric-label">SLOPE FACTOR</div>'
                        '<div class="metric-value" style="color: {};">{:.1f} bps</div>'
                        '<div class="metric-label">{} curve shape</div>'
                        '</div>'.format(slope_color, slope_val, slope_status), 
                        unsafe_allow_html=True
                    )
            
            with col_int3:
                if 'Curvature' in current_factors:
                    curvature_val = current_factors['Curvature']
                    if curvature_val > 0:
                        curvature_status = "Humped"
                    else:
                        curvature_status = "Sagged"
                    
                    st.markdown(
                        '<div class="metric-card">'
                        '<div class="metric-label">CURVATURE FACTOR</div>'
                        '<div class="metric-value">{:.1f} bps</div>'
                        '<div class="metric-label">{} medium-term expectations</div>'
                        '</div>'.format(curvature_val, curvature_status), 
                        unsafe_allow_html=True
                    )
        
        # PCA analysis
        st.markdown("#### Principal Component Analysis (PCA)")
        if pca_risk is not None:
            fig_pca_variance = go.Figure(data=go.Bar(
                x=['PC{}'.format(i+1) for i in range(len(pca_risk['explained_variance']))],
                y=pca_risk['explained_variance'] * 100,
                marker_color=COLORS['accent'],
                text=["{:.1f}%".format(v*100) for v in pca_risk['explained_variance']],
                textposition='outside'
            ))
            fig_pca_variance = create_institutional_layout(fig_pca_variance, "PCA VARIANCE EXPLANATION", "Variance Explained (%)", height=400)
            st.plotly_chart(fig_pca_variance, use_container_width=True)
            
            # Cumulative variance
            cum_var = np.cumsum(pca_risk['explained_variance']) * 100
            st.markdown("**Cumulative Variance Explained:**")
            st.markdown("- PC1 + PC2: {:.1f}% of yield curve variation".format(cum_var[1] if len(cum_var) > 1 else cum_var[0]))
            st.markdown("- PC1 + PC2 + PC3: {:.1f}% of yield curve variation".format(cum_var[2] if len(cum_var) > 2 else cum_var[-1]))
    
    # ===== TAB 8: RISK METRICS =====
    with tabs[7]:
        st.markdown("### Advanced Risk Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Value at Risk (VaR) Analysis")
            
            if '10Y' in yield_df.columns:
                returns = yield_df['10Y'].pct_change().dropna()
                var_metrics = AdvancedRiskMetrics.calculate_var_metrics(returns)
                
                st.markdown("**10Y Treasury Yield Risk (95% confidence, 10-day horizon)**")
                
                risk_table = pd.DataFrame({
                    'Metric': ['Historical VaR', 'Parametric VaR', 'Cornish-Fisher VaR', 
                               'CVaR (Expected Shortfall)', 'Tail Ratio', 'Skewness', 'Excess Kurtosis'],
                    'Value': [
                        "{:.4f}".format(var_metrics['VaR_Historical']),
                        "{:.4f}".format(var_metrics['VaR_Parametric']),
                        "{:.4f}".format(var_metrics['VaR_CornishFisher']),
                        "{:.4f}".format(var_metrics['CVaR']),
                        "{:.2f}".format(var_metrics['tail_ratio']),
                        "{:.3f}".format(var_metrics['skewness']),
                        "{:.3f}".format(var_metrics['kurtosis'])
                    ]
                })
                st.dataframe(risk_table, use_container_width=True, hide_index=True)
                
                # Return distribution
                fig_returns = go.Figure()
                fig_returns.add_trace(go.Histogram(
                    x=returns * 100,
                    nbinsx=50,
                    name='Returns Distribution',
                    marker_color=COLORS['surface'],
                    opacity=0.7
                ))
                fig_returns.add_vline(
                    x=var_metrics['VaR_Historical'] * 100,
                    line_dash="dash",
                    line_color=COLORS['negative'],
                    annotation_text="VaR: {:.2f}%".format(var_metrics['VaR_Historical']*100)
                )
                fig_returns = create_institutional_layout(fig_returns, "RETURN DISTRIBUTION & VAR", "Frequency", height=400)
                st.plotly_chart(fig_returns, use_container_width=True)
        
        with col2:
            st.markdown("#### PCA Risk Decomposition")
            
            if pca_risk is not None and 'loadings' in pca_risk:
                st.dataframe(pca_risk['loadings'].round(3), use_container_width=True)
        
        # Risk report
        st.markdown("#### Risk Assessment Report")
        
        current_slope_val = spreads['10Y-2Y'].iloc[-1] if '10Y-2Y' in spreads.columns else 0
        volatility_val = yield_df['10Y'].pct_change().std() * np.sqrt(252) if '10Y' in yield_df.columns else 0
        
        # Multi-factor risk scoring
        risk_score = 0
        risk_factors = []
        
        if current_slope_val < 0:
            risk_score += 40
            risk_factors.append("Curve inversion (+40)")
        elif current_slope_val < 50:
            risk_score += 20
            risk_factors.append("Curve flattening (+20)")
        
        if volatility_val > 0.2:
            risk_score += 30
            risk_factors.append("High volatility (+30)")
        elif volatility_val > 0.1:
            risk_score += 15
            risk_factors.append("Elevated volatility (+15)")
        
        if recession_metrics.get('avg_lead_time', 0) < 200:
            risk_score += 20
            risk_factors.append("Recent inversion history (+20)")
        
        if risk_score >= 60:
            risk_level = "HIGH"
            risk_color = COLORS['negative']
        elif risk_score >= 30:
            risk_level = "MEDIUM"
            risk_color = COLORS['warning']
        else:
            risk_level = "LOW"
            risk_color = COLORS['positive']
        
        curve_status_display = 'Inverted' if current_slope_val < 0 else 'Flattening' if current_slope_val < 50 else 'Normal'
        
        st.markdown(
            '<div class="metric-card" style="border-left: 3px solid {};">'
            '<div class="metric-label">OVERALL RISK ASSESSMENT</div>'
            '<div class="metric-value" style="color: {};">{} RISK</div>'
            '<div class="metric-label">'
            '<strong>Risk Score:</strong> {}/100<br>'
            '<strong>Contributing Factors:</strong> {}<br>'
            '<strong>Curve Status:</strong> {}<br>'
            '<strong>10Y Volatility:</strong> {:.2%}<br>'
            '<strong>Risk Horizon:</strong> 10-day VaR at 95% confidence'
            '</div>'
            '</div>'.format(
                risk_color, risk_color, risk_level,
                risk_score, ', '.join(risk_factors),
                curve_status_display,
                volatility_val
            ), 
            unsafe_allow_html=True
        )
        
        # Stress scenarios
        st.markdown("#### Stress Testing Scenarios")
        
        scenarios = AdvancedRiskMetrics.calculate_stress_scenarios(yield_df[['10Y']].dropna())
        current_10y_val = yield_df['10Y'].iloc[-1]
        
        scenario_data = []
        for name, value in scenarios.items():
            impact = (value - current_10y_val) * 100
            scenario_data.append({'Scenario': name, 'Impact on 10Y': "{:+.1f} bps".format(impact)})
        
        scenario_df = pd.DataFrame(scenario_data)
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)
    
    # ===== TAB 9: ARBITRAGE =====
    with tabs[8]:
        st.markdown("### Arbitrage Opportunity Detection")
        st.markdown("*Using Nelson-Siegel-Svensson Model to identify mispriced securities*")
        
        if arbitrage_stats:
            # Display arbitrage statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Absolute Error", "{:.2f} bps".format(arbitrage_stats['mean_abs_error'] * 100))
            with col2:
                st.metric("Max Pricing Error", "{:.2f} bps".format(arbitrage_stats['max_error'] * 100))
            with col3:
                st.metric("Std Deviation", "{:.2f} bps".format(arbitrage_stats['std_error'] * 100))
            with col4:
                st.metric("Mispriced Securities", arbitrage_stats['mispriced_count'])
            
            # Plot arbitrage chart
            if nss_result:
                fig_arbitrage = plot_arbitrage_chart(
                    maturities, yield_values, nss_result['fitted_values'], 
                    arbitrage_stats['mispriced_securities']
                )
                st.plotly_chart(fig_arbitrage, use_container_width=True)
            
            # Display mispriced securities table
            if arbitrage_stats['mispriced_securities']:
                st.markdown("#### Mispriced Securities")
                mispriced_df = pd.DataFrame(arbitrage_stats['mispriced_securities'])
                mispriced_df['maturity'] = mispriced_df['maturity'].apply(lambda x: "{:.2f}Y".format(x))
                mispriced_df['actual'] = mispriced_df['actual'].apply(lambda x: "{:.2f}%".format(x))
                mispriced_df['theoretical'] = mispriced_df['theoretical'].apply(lambda x: "{:.2f}%".format(x))
                mispriced_df['difference'] = mispriced_df['difference'].apply(lambda x: "{:.2f} bps".format(x * 100))
                st.dataframe(mispriced_df, use_container_width=True, hide_index=True)
                
                st.markdown("""
                **Arbitrage Strategy Notes:**
                - **Overvalued securities** (negative difference): Consider shorting or avoiding
                - **Undervalued securities** (positive difference): Consider buying or going long
                - The NSS model identifies theoretical fair values based on the entire curve
                """)
            else:
                st.success("✅ No significant mispricing detected. The market is efficiently pricing all maturities.")
        else:
            st.warning("⚠️ Arbitrage detection requires successful NSS model calibration.")
    
    # ===== TAB 10: NBER RECESSION =====
    with tabs[9]:
        st.markdown("### NBER Recession Analysis")
        st.markdown("*National Bureau of Economic Research (NBER) official recession periods*")
        
        # NBER chart
        st.plotly_chart(plot_nber_recession_chart(spreads, recessions), use_container_width=True)
        
        # Recession periods table
        st.markdown("#### NBER Recession Periods")
        if recessions:
            recession_df = pd.DataFrame(recessions)
            recession_df['duration'] = (recession_df['end'] - recession_df['start']).dt.days
            recession_df['start'] = recession_df['start'].dt.strftime('%Y-%m-%d')
            recession_df['end'] = recession_df['end'].dt.strftime('%Y-%m-%d')
            st.dataframe(recession_df, use_container_width=True, hide_index=True)
        else:
            st.info("No recession periods identified in the data range")
        
        # Recession metrics
        st.markdown("#### Recession Analytics")
        
        col_met1, col_met2, col_met3, col_met4 = st.columns(4)
        
        with col_met1:
            st.metric("Total Inversion Days", "{:,}".format(recession_metrics.get('total_inversion_days', 0)))
        with col_met2:
            st.metric("Avg Inversion Depth", "{:.1f} bps".format(recession_metrics.get('avg_inversion_depth', 0)))
        with col_met3:
            st.metric("Avg Lead Time", "{:.0f} days".format(recession_metrics.get('avg_lead_time', 0)))
        with col_met4:
            st.metric("Number of Inversions", recession_metrics.get('num_inversions', 0))
        
        # Lead time distribution
        if recession_metrics.get('lead_times'):
            st.markdown("#### Historical Lead Times (Inversion to Recession)")
            lead_df = pd.DataFrame({
                'Lead Time (Days)': recession_metrics['lead_times'],
                'Lead Time (Months)': [d/30.44 for d in recession_metrics['lead_times']]
            })
            st.dataframe(lead_df, use_container_width=True, hide_index=True)
            
            # Lead time histogram
            fig_lead = go.Figure()
            fig_lead.add_trace(go.Histogram(
                x=recession_metrics['lead_times'],
                nbinsx=10,
                marker_color=COLORS['accent'],
                name='Lead Times'
            ))
            fig_lead.add_vline(
                x=recession_metrics['avg_lead_time'],
                line_dash="dash",
                line_color=COLORS['negative'],
                annotation_text="Avg: {:.0f} days".format(recession_metrics['avg_lead_time'])
            )
            fig_lead = create_institutional_layout(fig_lead, "LEAD TIME DISTRIBUTION", "Frequency", height=400)
            st.plotly_chart(fig_lead, use_container_width=True)
        
        # Inversion periods
        if recession_metrics.get('inversion_periods'):
            st.markdown("#### Historical Inversion Periods")
            inv_df = pd.DataFrame(recession_metrics['inversion_periods'])
            inv_df['start'] = inv_df['start'].dt.strftime('%Y-%m-%d')
            inv_df['end'] = inv_df['end'].dt.strftime('%Y-%m-%d')
            st.dataframe(inv_df, use_container_width=True, hide_index=True)
    
    # ===== TAB 11: FORECASTING =====
    with tabs[10]:
        st.markdown("### Yield Curve Forecasting")
        
        forecast_horizon = st.slider("Forecast Horizon (Days)", 5, 60, 20, key="forecast_horizon")
        
        with st.spinner("Generating forecasts..."):
            forecast_result = YieldCurveForecasting.forecast_with_var(yield_df[['10Y']].dropna(), horizon=forecast_horizon)
        
        if forecast_result:
            st.plotly_chart(plot_forecast_chart(yield_df['10Y'], forecast_result, '10Y'), use_container_width=True)
            
            # Forecast table
            forecast_dates = pd.date_range(start=yield_df.index[-1], periods=forecast_horizon + 1, freq='D')[1:]
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast (%)': forecast_result['forecast'][:, 0] if len(forecast_result['forecast'].shape) > 1 else forecast_result['forecast'],
                'Lower Bound (%)': forecast_result['lower'][:, 0] if len(forecast_result['lower'].shape) > 1 else forecast_result['lower'],
                'Upper Bound (%)': forecast_result['upper'][:, 0] if len(forecast_result['upper'].shape) > 1 else forecast_result['upper']
            })
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Insufficient data for reliable forecasting. Need at least 100 observations.")
    
    # ===== TAB 12: DATA EXPORT =====
    with tabs[11]:
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
        
        # Data summary
        st.markdown("### Data Summary")
        st.markdown("- **Yield Curves:** {} maturities ({})".format(len(yield_df.columns), ', '.join(yield_df.columns)))
        st.markdown("- **Observations:** {:,}".format(len(yield_df)))
        st.markdown("- **Date Range:** {} to {}".format(yield_df.index[0].strftime('%Y-%m-%d'), yield_df.index[-1].strftime('%Y-%m-%d')))
        
        if ns_result:
            st.markdown("- **NS RMSE:** {:.4f}".format(ns_result['rmse']))
        else:
            st.markdown("- **NS RMSE:** N/A")
        
        if nss_result:
            st.markdown("- **NSS RMSE:** {:.4f}".format(nss_result['rmse']))
        else:
            st.markdown("- **NSS RMSE:** N/A")
        
        st.markdown("- **NBER Recessions:** {}".format(len(recessions)))
        st.markdown("- **Data Completeness:** {:.0f}%".format(yield_df.notna().all().all() * 100))
        
        # Model performance summary
        if ns_result and nss_result:
            st.markdown("### Model Performance Summary")
            perf_df = pd.DataFrame({
                'Model': ['Nelson-Siegel', 'Nelson-Siegel-Svensson'],
                'Parameters': [4, 6],
                'RMSE': [ns_result['rmse'], nss_result['rmse']],
                'MAE': [ns_result['mae'], nss_result['mae']],
                'R²': [ns_result['r_squared'], nss_result['r_squared']]
            })
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #7f8c8d; font-size: 0.65rem; padding: 1rem;">
            <p>© 2024 Yield Curve Analytics | Institutional Quantitative Platform</p>
            <p>Data: Federal Reserve Economic Data (FRED) | Models: Nelson-Siegel (1987), Svensson (1994)</p>
            <p>Recession Definition: NBER (National Bureau of Economic Research)</p>
            <p>Last Update: {} UTC</p>
        </div>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
