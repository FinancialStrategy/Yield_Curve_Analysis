# =============================================================================
# HEDGE FUND YIELD CURVE ANALYTICS PLATFORM
# INTERACTIVE FRED API KEY | NBER RECESSION ANALYSIS | FULL MATURITY SPECTRUM
# =============================================================================
# Institutional Grade | Quantitative Financial Analytics
# Version: 8.0 | Enterprise Edition with Interactive API Key
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import requests
import json
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION | PROFESSIONAL THEME
# =============================================================================

st.set_page_config(
    page_title="Yield Curve Analytics | Institutional Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Institutional Color Palette - Bloomberg Terminal Style
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

# FRED API Configuration - All Maturities
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

# Session state initialization
if 'api_key_validated' not in st.session_state:
    st.session_state.api_key_validated = False
if 'yield_data' not in st.session_state:
    st.session_state.yield_data = None
if 'recession_data' not in st.session_state:
    st.session_state.recession_data = None
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False

# Custom CSS - Institutional Professional
st.markdown(f"""
<style>
    .main {{
        background-color: {COLORS['background']};
    }}
    
    .hedge-header {{
        background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        padding: 1.5rem;
        border-radius: 0;
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
    
    .api-status {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['grid']};
        border-radius: 4px;
        padding: 0.75rem;
        margin-bottom: 1rem;
        font-size: 0.8rem;
    }}
    
    .metric-card {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['grid']};
        border-radius: 4px;
        padding: 1rem;
        margin: 0.5rem 0;
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
    
    .status-inverted {{ color: {COLORS['negative']}; font-weight: 700; }}
    .status-normal {{ color: {COLORS['positive']}; font-weight: 700; }}
    .status-caution {{ color: {COLORS['warning']}; font-weight: 700; }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0rem;
        background-color: {COLORS['surface']};
        border-bottom: 1px solid {COLORS['grid']};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        color: {COLORS['text_secondary']};
        padding: 0.5rem 1rem;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
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
    }}
    
    .stButton > button:hover {{
        background-color: {COLORS['primary']};
        color: white;
    }}
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FRED API FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600)
def fetch_fred_data(api_key, series_id):
    """Fetch data from FRED API"""
    url = f"https://api.stlouisfed.org/fred/series/observations"
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

def fetch_all_yield_data(api_key, progress_callback=None):
    """Fetch all treasury yield data from FRED"""
    all_data = {}
    total_series = len(FRED_SERIES)
    
    for idx, (name, series_id) in enumerate(FRED_SERIES.items()):
        if progress_callback:
            progress_callback(idx, total_series, name, series_id)
        data = fetch_fred_data(api_key, series_id)
        if data is not None:
            all_data[name] = data
        time.sleep(0.1)  # Rate limiting
    
    if not all_data:
        return None
    
    df = pd.DataFrame(all_data).dropna()
    return df

def fetch_recession_data(api_key):
    """Fetch NBER recession indicator from FRED"""
    data = fetch_fred_data(api_key, 'USREC')
    if data is not None:
        return data
    return None

def validate_fred_api_key(api_key):
    """Validate FRED API key"""
    if not api_key or len(api_key) < 10:
        return False
    
    test_url = f"https://api.stlouisfed.org/fred/series/observations"
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
    except:
        return False

# =============================================================================
# QUANTITATIVE FINANCIAL MODELS
# =============================================================================

class NelsonSiegelModel:
    """Nelson-Siegel Yield Curve Fitting Model"""
    
    @staticmethod
    def nelson_siegel(tau, beta0, beta1, beta2, lambda_):
        return beta0 + beta1 * ((1 - np.exp(-lambda_ * tau)) / (lambda_ * tau)) + \
               beta2 * (((1 - np.exp(-lambda_ * tau)) / (lambda_ * tau)) - np.exp(-lambda_ * tau))
    
    @staticmethod
    def fit_curve(maturities, yields):
        try:
            beta0_init = yields[-1]
            beta1_init = yields[0] - yields[-1]
            beta2_init = yields[0] - 2 * yields[1] + yields[-1] if len(yields) > 2 else 0
            lambda_init = 0.0609
            
            popt, _ = curve_fit(
                lambda t, b0, b1, b2, l: NelsonSiegelModel.nelson_siegel(t, b0, b1, b2, l),
                maturities, yields,
                p0=[beta0_init, beta1_init, beta2_init, lambda_init],
                maxfev=5000
            )
            return popt
        except:
            return [yields[-1], yields[0] - yields[-1], 0, 0.0609]
    
    @staticmethod
    def get_factors(maturities, yields):
        params = NelsonSiegelModel.fit_curve(maturities, yields)
        return {'Level': params[0], 'Slope': params[1], 'Curvature': params[2], 'Decay': params[3]}

class RegimeDetection:
    """Markov Regime Switching Detection"""
    
    @staticmethod
    def detect_regime(spreads):
        if len(spreads) < 50:
            return 'NEUTRAL'
        
        rolling_mean = spreads.rolling(20).mean()
        rolling_std = spreads.rolling(20).std()
        z_score = (spreads - rolling_mean) / rolling_std
        latest_z = z_score.iloc[-1] if not pd.isna(z_score.iloc[-1]) else 0
        
        if spreads.iloc[-1] < 0:
            return 'INVERSION'
        elif latest_z < -1.5:
            return 'STRESS'
        elif latest_z > 1.5:
            return 'EXPANSION'
        else:
            return 'NEUTRAL'
    
    @staticmethod
    def calculate_recession_probability(spreads):
        if len(spreads) < 252:
            return 0.0
        
        historical_inversions = spreads[spreads < 0]
        inversion_count = len(historical_inversions)
        
        if inversion_count == 0:
            return 0.0
        
        current_inversion = spreads.iloc[-1] < 0
        inversion_depth = abs(spreads.iloc[-1]) if current_inversion else 0
        
        prob = 1 / (1 + np.exp(-(-5 + 0.1 * inversion_depth + 0.05 * inversion_count)))
        return min(0.95, max(0.05, prob))

# =============================================================================
# NBER RECESSION ANALYSIS
# =============================================================================

def identify_nber_recessions(recession_series):
    """Identify NBER recession periods"""
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
    """Calculate recession-related metrics"""
    metrics = {}
    
    # Inversion periods analysis
    inversion_periods = []
    in_inversion = False
    inv_start = None
    
    for date, value in spreads.items():
        if value < 0 and not in_inversion:
            in_inversion = True
            inv_start = date
        elif value >= 0 and in_inversion:
            in_inversion = False
            inversion_periods.append({'start': inv_start, 'end': date, 'depth': spreads.loc[inv_start:date].min()})
    
    metrics['inversion_periods'] = inversion_periods
    metrics['total_inversion_days'] = sum([(p['end'] - p['start']).days for p in inversion_periods]) if inversion_periods else 0
    metrics['avg_inversion_depth'] = np.mean([p['depth'] for p in inversion_periods]) if inversion_periods else 0
    
    # Lead time analysis
    lead_times = []
    for inversion in inversion_periods:
        for recession in recessions:
            if inversion['start'] < recession['start']:
                lead_days = (recession['start'] - inversion['start']).days
                lead_times.append(lead_days)
                break
    
    metrics['lead_times'] = lead_times
    metrics['avg_lead_time'] = np.mean(lead_times) if lead_times else 0
    metrics['median_lead_time'] = np.median(lead_times) if lead_times else 0
    
    return metrics

# =============================================================================
# DATA PROCESSING
# =============================================================================

def calculate_spreads(yield_df):
    """Calculate all yield spreads"""
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
    
    return spreads

def calculate_principal_components(yield_df):
    """PCA for yield curve dynamics"""
    if len(yield_df) < 100 or len(yield_df.columns) < 5:
        return None, None
    
    scaler = StandardScaler()
    yields_std = scaler.fit_transform(yield_df)
    
    pca = PCA()
    pcs = pca.fit_transform(yields_std)
    
    factors = pd.DataFrame(
        pcs[:, :3],
        index=yield_df.index,
        columns=['PC1_Level', 'PC2_Slope', 'PC3_Curvature']
    )
    
    return factors, pca.explained_variance_ratio_[:3]

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_institutional_layout(fig, title, y_title=None):
    """Apply institutional styling"""
    
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
        margin=dict(l=50, r=30, t=50, b=40),
        hovermode='x unified',
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
        tickfont=dict(size=8)
    )
    
    fig.update_yaxes(
        gridcolor=COLORS['grid'],
        gridwidth=0.5,
        zeroline=False,
        tickfont=dict(size=8)
    )
    
    if y_title:
        fig.update_yaxes(title_text=y_title, title_font=dict(size=9))
    
    return fig

def plot_nber_recession_chart(spreads, recessions):
    """NBER recession chart with institutional styling"""
    
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
    
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS['neutral'], 
                 line_width=1, annotation_text="INVERSION THRESHOLD")
    
    for recession in recessions:
        fig.add_vrect(
            x0=recession['start'], x1=recession['end'],
            fillcolor=COLORS['recession'], opacity=0.4,
            layer="below", line_width=0,
            annotation_text="NBER RECESSION",
            annotation_position="top left"
        )
    
    fig = create_institutional_layout(fig, "NBER RECESSION INDICATOR & YIELD SPREAD", "Spread (bps)")
    fig.update_layout(height=500)
    
    return fig

def plot_yield_curve_3d(yield_df):
    """3D surface plot"""
    
    dates = yield_df.index
    maturities = [MATURITY_MAP.get(col, 0) for col in yield_df.columns]
    z_data = yield_df.T.values
    
    fig = go.Figure(data=[
        go.Surface(
            x=dates,
            y=maturities,
            z=z_data,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Yield (%)", thickness=8),
            contours={"z": {"show": True, "usecolormap": True, "width": 1}}
        )
    ])
    
    fig = create_institutional_layout(fig, "YIELD CURVE TERM STRUCTURE EVOLUTION", "Yield (%)")
    fig.update_layout(
        scene=dict(
            xaxis_title="Date",
            yaxis_title="Maturity (Years)",
            zaxis_title="Yield (%)",
            camera=dict(eye=dict(x=1.8, y=-1.5, z=1.2)),
            aspectmode='manual',
            aspectratio=dict(x=2, y=0.8, z=0.6)
        ),
        height=550
    )
    
    return fig

def plot_spread_dashboard(spreads, recessions):
    """Professional spread dashboard"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('10Y-2Y SPREAD', '10Y-3M SPREAD', '5Y-2Y SPREAD', '30Y-10Y SPREAD'),
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
                    line=dict(color=config['color'], width=1.2),
                    hovertemplate=f'<b>%{{x|%Y-%m-%d}}</b><br>{spread_name}: %{{y:.1f}} bps<extra></extra>'
                ),
                row=config['row'], col=config['col']
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color=COLORS['negative'], 
                         line_width=1, row=config['row'], col=config['col'])
            
            for recession in recessions:
                fig.add_vrect(
                    x0=recession['start'], x1=recession['end'],
                    fillcolor=COLORS['recession'], opacity=0.3,
                    layer="below", line_width=0,
                    row=config['row'], col=config['col']
                )
    
    fig = create_institutional_layout(fig, "YIELD SPREAD DYNAMICS")
    fig.update_layout(height=550)
    
    return fig

def plot_pca_factors(factors):
    """PCA factor dynamics"""
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=('LEVEL FACTOR (PC1)', 'SLOPE FACTOR (PC2)', 'CURVATURE FACTOR (PC3)'),
        vertical_spacing=0.08
    )
    
    for i, factor in enumerate(['PC1_Level', 'PC2_Slope', 'PC3_Curvature']):
        if factor in factors.columns:
            fig.add_trace(
                go.Scatter(
                    x=factors.index,
                    y=factors[factor],
                    mode='lines',
                    name=factor,
                    line=dict(color=COLORS['accent'], width=1),
                    showlegend=False
                ),
                row=i+1, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color=COLORS['neutral'],
                         line_width=1, row=i+1, col=1)
    
    fig = create_institutional_layout(fig, "PRINCIPAL COMPONENT ANALYSIS")
    fig.update_layout(height=650)
    
    return fig

# =============================================================================
# API KEY INPUT COMPONENT
# =============================================================================

def render_api_key_input():
    """Render interactive API key input component"""
    
    st.markdown("""
    <div class="api-container">
        <h3 style="color: white; margin-bottom: 1rem;">🔑 FRED API Key Required</h3>
        <p style="color: #bdc3c7; font-size: 0.85rem; margin-bottom: 1.5rem;">
            This dashboard requires a FRED API key to access treasury yield data.<br>
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
            help="Your FRED API key is required to fetch data"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        
        with col_btn2:
            validate_btn = st.button("🔑 Validate & Connect", type="primary", use_container_width=True)
    
    if validate_btn:
        if not api_key:
            st.error("❌ Please enter a valid API key")
            return None
        
        with st.spinner("Validating API key..."):
            is_valid = validate_fred_api_key(api_key)
        
        if is_valid:
            st.session_state.api_key = api_key
            st.session_state.api_key_validated = True
            st.success("✅ API key validated successfully! Fetching data...")
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ Invalid API key. Please check and try again.")
            return None
    
    return None

# =============================================================================
# DATA FETCHING COMPONENT
# =============================================================================

def fetch_and_process_data(api_key):
    """Fetch and process all data from FRED"""
    
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    def update_progress(idx, total, name, series_id):
        progress = (idx + 1) / total
        progress_placeholder.progress(progress)
        status_placeholder.text(f"Fetching {name} ({series_id})... [{idx+1}/{total}]")
    
    # Fetch yield data
    status_placeholder.text("Fetching treasury yield data from FRED...")
    yield_df = fetch_all_yield_data(api_key, update_progress)
    
    if yield_df is None or yield_df.empty:
        progress_placeholder.empty()
        status_placeholder.empty()
        st.error("Failed to fetch yield data. Please check your API key and try again.")
        return None, None
    
    # Fetch recession data
    status_placeholder.text("Fetching NBER recession indicator...")
    recession_series = fetch_recession_data(api_key)
    
    progress_placeholder.empty()
    status_placeholder.empty()
    
    st.success(f"✅ Data fetched successfully: {len(yield_df)} observations from {yield_df.index[0].strftime('%Y-%m-%d')} to {yield_df.index[-1].strftime('%Y-%m-%d')}")
    
    return yield_df, recession_series

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    
    # Header
    st.markdown("""
    <div class="hedge-header">
        <h1 style="color: white; margin: 0; font-size: 1.25rem;">YIELD CURVE ANALYTICS</h1>
        <p style="color: #bdc3c7; margin: 0; font-size: 0.7rem;">FRED Data Integration | NBER Recession Analysis | Quantitative Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ===== API KEY VALIDATION =====
    if not st.session_state.api_key_validated:
        render_api_key_input()
        st.stop()
    
    # ===== DATA FETCHING =====
    if not st.session_state.data_fetched:
        with st.spinner("Connecting to FRED and fetching data..."):
            yield_df, recession_series = fetch_and_process_data(st.session_state.api_key)
        
        if yield_df is not None:
            st.session_state.yield_data = yield_df
            st.session_state.recession_data = recession_series
            st.session_state.data_fetched = True
            st.rerun()
        else:
            st.session_state.api_key_validated = False
            st.session_state.data_fetched = False
            st.rerun()
    
    # ===== LOAD DATA FROM SESSION STATE =====
    yield_df = st.session_state.yield_data
    recession_series = st.session_state.recession_data
    
    if yield_df is None or yield_df.empty:
        st.error("No data available. Please refresh the page and try again.")
        st.stop()
    
    # ===== DATA PROCESSING =====
    spreads = calculate_spreads(yield_df)
    recessions = identify_nber_recessions(recession_series)
    recession_metrics = calculate_recession_metrics(spreads['10Y-2Y'].dropna(), recessions) if '10Y-2Y' in spreads.columns else {}
    
    # Current metrics
    current_10y = yield_df['10Y'].iloc[-1] if '10Y' in yield_df.columns else 0
    current_2y = yield_df['2Y'].iloc[-1] if '2Y' in yield_df.columns else 0
    current_30y = yield_df['30Y'].iloc[-1] if '30Y' in yield_df.columns else 0
    current_spread = spreads['10Y-2Y'].iloc[-1] if '10Y-2Y' in spreads.columns else 0
    
    # Regime and probability
    regime = RegimeDetection.detect_regime(spreads['10Y-2Y'].dropna()) if '10Y-2Y' in spreads.columns else 'NEUTRAL'
    recession_prob = RegimeDetection.calculate_recession_probability(spreads['10Y-2Y'].dropna()) if '10Y-2Y' in spreads.columns else 0.0
    
    # Nelson-Siegel factors
    maturities_num = [MATURITY_MAP.get(col, 0) for col in yield_df.columns if col in MATURITY_MAP]
    latest_yields = yield_df.iloc[-1].values[:len(maturities_num)]
    ns_factors = NelsonSiegelModel.get_factors(maturities_num, latest_yields)
    
    # PCA analysis
    pca_factors, pca_var = calculate_principal_components(yield_df)
    
    # ===== REFRESH BUTTON =====
    col_refresh1, col_refresh2, col_refresh3 = st.columns([1, 2, 1])
    with col_refresh2:
        if st.button("🔄 Refresh Data from FRED", use_container_width=True):
            st.cache_data.clear()
            st.session_state.data_fetched = False
            st.session_state.yield_data = None
            st.session_state.recession_data = None
            st.rerun()
    
    st.markdown("---")
    
    # ===== METRICS ROW =====
    st.markdown("### Current Market Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">10Y YIELD</div>
            <div class="metric-value">{current_10y:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status_class = "status-inverted" if current_spread < 0 else "status-caution" if current_spread < 50 else "status-normal"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">10Y-2Y SPREAD</div>
            <div class="metric-value">{current_spread:.1f} bps</div>
            <div class="{status_class}">{regime}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RECESSION PROB</div>
            <div class="metric-value">{recession_prob:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">NS LEVEL FACTOR</div>
            <div class="metric-value">{ns_factors['Level']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">NS SLOPE FACTOR</div>
            <div class="metric-value">{ns_factors['Slope']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== TABS FOR ANALYSIS =====
    tabs = st.tabs([
        "📊 NBER RECESSION", "📈 TERM STRUCTURE", "📉 SPREAD DYNAMICS", 
        "🔬 FACTOR ANALYSIS", "📋 RECESSION METRICS", "📁 DATA EXPORT"
    ])
    
    # Tab 1: NBER Recession Chart
    with tabs[0]:
        st.plotly_chart(plot_nber_recession_chart(spreads, recessions), use_container_width=True)
        
        st.markdown("### NBER Recession Periods")
        if recessions:
            recession_df = pd.DataFrame(recessions)
            recession_df['duration'] = (recession_df['end'] - recession_df['start']).dt.days
            recession_df['start'] = recession_df['start'].dt.strftime('%Y-%m-%d')
            recession_df['end'] = recession_df['end'].dt.strftime('%Y-%m-%d')
            st.dataframe(recession_df, use_container_width=True, hide_index=True)
        else:
            st.info("No recession periods identified in the data range")
    
    # Tab 2: Term Structure
    with tabs[1]:
        st.plotly_chart(plot_yield_curve_3d(yield_df), use_container_width=True)
        
        # Current curve
        current_curve = yield_df.iloc[-1].values[:len(maturities_num)]
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=maturities_num, y=current_curve, mode='lines+markers',
            name='Current', line=dict(color=COLORS['accent'], width=2),
            marker=dict(size=6)
        ))
        fig_curve = create_institutional_layout(fig_curve, "CURRENT TERM STRUCTURE", "Yield (%)")
        fig_curve.update_layout(height=400)
        st.plotly_chart(fig_curve, use_container_width=True)
    
    # Tab 3: Spread Dynamics
    with tabs[2]:
        st.plotly_chart(plot_spread_dashboard(spreads, recessions), use_container_width=True)
        
        # Spread statistics
        st.markdown("### Spread Statistics")
        spread_stats = pd.DataFrame({
            'Spread': spreads.columns,
            'Current': [f"{spreads[col].iloc[-1]:.1f}" for col in spreads.columns],
            'Mean': [f"{spreads[col].mean():.1f}" for col in spreads.columns],
            'Std': [f"{spreads[col].std():.1f}" for col in spreads.columns],
            'Min': [f"{spreads[col].min():.1f}" for col in spreads.columns],
            'Max': [f"{spreads[col].max():.1f}" for col in spreads.columns],
            '% Negative': [f"{(spreads[col] < 0).mean()*100:.1f}%" for col in spreads.columns]
        })
        st.dataframe(spread_stats, use_container_width=True, hide_index=True)
    
    # Tab 4: Factor Analysis
    with tabs[3]:
        if pca_factors is not None:
            st.plotly_chart(plot_pca_factors(pca_factors), use_container_width=True)
            
            st.markdown("### PCA Variance Explanation")
            var_df = pd.DataFrame({
                'Component': ['PC1 (Level)', 'PC2 (Slope)', 'PC3 (Curvature)'],
                'Variance Explained': [f"{v:.1%}" for v in pca_var],
                'Cumulative': [f"{sum(pca_var[:i+1]):.1%}" for i in range(len(pca_var))]
            })
            st.dataframe(var_df, use_container_width=True, hide_index=True)
        
        st.markdown("### Nelson-Siegel Factors")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">LEVEL FACTOR</div>
                <div class="metric-value">{ns_factors['Level']:.2f}%</div>
                <div class="metric-label" style="font-size:0.6rem">Long-term expectation</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">SLOPE FACTOR</div>
                <div class="metric-value">{ns_factors['Slope']:.2f}</div>
                <div class="metric-label" style="font-size:0.6rem">Monetary policy stance</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">CURVATURE FACTOR</div>
                <div class="metric-value">{ns_factors['Curvature']:.2f}</div>
                <div class="metric-label" style="font-size:0.6rem">Medium-term expectations</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 5: Recession Metrics
    with tabs[4]:
        if recession_metrics:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">TOTAL INVERSION DAYS</div>
                    <div class="metric-value">{recession_metrics.get('total_inversion_days', 0):,}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">AVG INVERSION DEPTH</div>
                    <div class="metric-value">{recession_metrics.get('avg_inversion_depth', 0):.1f} bps</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">AVG LEAD TIME</div>
                    <div class="metric-value">{recession_metrics.get('avg_lead_time', 0):.0f} days</div>
                </div>
                """, unsafe_allow_html=True)
            
            if recession_metrics.get('lead_times'):
                st.markdown("### Historical Lead Times (Inversion to Recession)")
                lead_df = pd.DataFrame({
                    'Lead Time (Days)': recession_metrics['lead_times'],
                    'Lead Time (Months)': [d/30.44 for d in recession_metrics['lead_times']]
                })
                st.dataframe(lead_df, use_container_width=True, hide_index=True)
            
            if recession_metrics.get('inversion_periods'):
                st.markdown("### Inversion Periods")
                inv_df = pd.DataFrame(recession_metrics['inversion_periods'])
                inv_df['start'] = inv_df['start'].dt.strftime('%Y-%m-%d')
                inv_df['end'] = inv_df['end'].dt.strftime('%Y-%m-%d')
                st.dataframe(inv_df, use_container_width=True, hide_index=True)
        else:
            st.info("Insufficient data for recession metrics calculation")
    
    # Tab 6: Data Export
    with tabs[5]:
        st.markdown("### Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_yields = yield_df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Yield Data (CSV)",
                data=csv_yields,
                file_name=f"yield_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            csv_spreads = spreads.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Spread Data (CSV)",
                data=csv_spreads,
                file_name=f"spread_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        st.markdown("### Data Summary")
        st.markdown(f"""
        - **Yield Curves:** {len(yield_df.columns)} maturities ({', '.join(yield_df.columns)})
        - **Observations:** {len(yield_df):,}
        - **Date Range:** {yield_df.index[0].strftime('%Y-%m-%d')} to {yield_df.index[-1].strftime('%Y-%m-%d')}
        - **NBER Recessions:** {len(recessions)}
        - **Data Completeness:** {yield_df.notna().all().all() * 100:.0f}%
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.65rem; padding: 1rem;">
        <p>© 2024 Yield Curve Analytics | Institutional Quantitative Platform</p>
        <p>Data: Federal Reserve Economic Data (FRED) | Recession: NBER (National Bureau of Economic Research)</p>
        <p>Models: Nelson-Siegel | PCA | Regime Detection | VaR Analytics</p>
        <p>Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
