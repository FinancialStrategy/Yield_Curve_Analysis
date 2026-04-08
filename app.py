# =============================================================================
# HEDGE FUND YIELD CURVE ANALYTICS PLATFORM
# =============================================================================
# Institutional Grade | Quantitative Financial Analytics
# Version: 6.0 | Enterprise Edition
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
    'recession': 'rgba(52, 73, 94, 0.3)'
}

# Custom CSS - Institutional Professional
st.markdown(f"""
<style>
    /* Main container */
    .main {{
        background-color: {COLORS['background']};
    }}
    
    /* Professional header */
    .hedge-header {{
        background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        padding: 1.5rem;
        border-radius: 0;
        border-bottom: 2px solid {COLORS['accent']};
        margin-bottom: 2rem;
    }}
    
    /* Metric cards - Bloomberg style */
    .metric-card {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['grid']};
        border-radius: 4px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: none;
    }}
    
    .metric-label {{
        color: {COLORS['text_secondary']};
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    
    .metric-value {{
        color: {COLORS['text_primary']};
        font-size: 1.75rem;
        font-weight: 600;
        font-family: 'Courier New', monospace;
    }}
    
    .metric-change-positive {{
        color: {COLORS['positive']};
        font-size: 0.8rem;
    }}
    
    .metric-change-negative {{
        color: {COLORS['negative']};
        font-size: 0.8rem;
    }}
    
    /* Status indicators */
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
    
    /* Tables - Institutional style */
    .dataframe {{
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['grid']};
    }}
    
    .dataframe th {{
        background-color: {COLORS['secondary']};
        color: {COLORS['text_primary']};
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.7rem;
    }}
    
    /* Sidebar styling */
    .css-1d391kg {{
        background-color: {COLORS['primary']};
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0rem;
        background-color: {COLORS['surface']};
        border-bottom: 1px solid {COLORS['grid']};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        color: {COLORS['text_secondary']};
        padding: 0.5rem 1rem;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
    }}
    
    .stTabs [aria-selected="true"] {{
        color: {COLORS['accent']};
        border-bottom: 2px solid {COLORS['accent']};
    }}
    
    /* Hide streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# QUANTITATIVE FINANCIAL MODELS
# =============================================================================

class NelsonSiegelModel:
    """Nelson-Siegel Yield Curve Fitting Model"""
    
    @staticmethod
    def nelson_siegel(tau, beta0, beta1, beta2, lambda_):
        """Nelson-Siegel functional form"""
        return beta0 + beta1 * ((1 - np.exp(-lambda_ * tau)) / (lambda_ * tau)) + \
               beta2 * (((1 - np.exp(-lambda_ * tau)) / (lambda_ * tau)) - np.exp(-lambda_ * tau))
    
    @staticmethod
    def fit_curve(maturities, yields):
        """Fit Nelson-Siegel curve to observed yields"""
        try:
            # Initial parameter estimates
            beta0_init = yields[-1]  # Long-term level
            beta1_init = yields[0] - yields[-1]  # Short-term slope
            beta2_init = yields[0] - 2 * yields[1] + yields[-1]  # Curvature
            lambda_init = 0.0609  # Decay factor (typical for Treasury yields)
            
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
        """Extract level, slope, and curvature factors"""
        params = NelsonSiegelModel.fit_curve(maturities, yields)
        return {
            'Level': params[0],
            'Slope': params[1],
            'Curvature': params[2],
            'Decay': params[3]
        }

class RegimeDetection:
    """Markov Regime Switching Detection"""
    
    @staticmethod
    def detect_regime(spreads, threshold=-50):
        """Detect market regimes based on spread dynamics"""
        if len(spreads) < 50:
            return 'NEUTRAL'
        
        # Rolling statistics
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
    def calculate_recession_probability(spreads, lookback=252):
        """Calculate recession probability based on historical patterns"""
        if len(spreads) < lookback:
            return 0.0
        
        # Historical inversion-recession relationship
        historical_inversions = spreads[spreads < 0]
        inversion_count = len(historical_inversions)
        
        if inversion_count == 0:
            return 0.0
        
        # Probability based on inversion duration and depth
        current_inversion = spreads.iloc[-1] < 0
        inversion_depth = abs(spreads.iloc[-1]) if current_inversion else 0
        
        # Logistic regression approximation
        prob = 1 / (1 + np.exp(-(-5 + 0.1 * inversion_depth + 0.05 * inversion_count)))
        
        return min(0.95, max(0.05, prob))

class RiskMetrics:
    """Value at Risk and Risk Metrics"""
    
    @staticmethod
    def calculate_var(returns, confidence_level=0.95):
        """Calculate Value at Risk"""
        if len(returns) < 2:
            return 0.0
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def calculate_cvar(returns, confidence_level=0.95):
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) < 2:
            return 0.0
        var = RiskMetrics.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean() if len(returns[returns <= var]) > 0 else var
    
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
        """Calculate Sharpe Ratio"""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        excess_returns = returns.mean() - risk_free_rate / 252
        return excess_returns / returns.std() * np.sqrt(252)

# =============================================================================
# DATA MANAGEMENT
# =============================================================================

@st.cache_data(ttl=3600)
def fetch_institutional_data():
    """Fetch institutional-grade yield data"""
    
    tickers = {
        '1M': 'DGS1MO', '3M': 'DGS3MO', '6M': 'DGS6MO',
        '1Y': 'DGS1', '2Y': 'DGS2', '3Y': 'DGS3',
        '5Y': 'DGS5', '7Y': 'DGS7', '10Y': 'DGS10',
        '20Y': 'DGS20', '30Y': 'DGS30'
    }
    
    data = {}
    
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start="1990-01-01", progress=False)
            if not df.empty:
                data[name] = df['Close']
        except:
            continue
    
    if not data:
        st.error("Institutional data feed unavailable")
        return None
    
    df = pd.DataFrame(data).dropna()
    
    # Convert maturities to numeric for calculations
    maturity_map = {'1M': 1/12, '3M': 0.25, '6M': 0.5, '1Y': 1, '2Y': 2, 
                    '3Y': 3, '5Y': 5, '7Y': 7, '10Y': 10, '20Y': 20, '30Y': 30}
    
    return df, maturity_map

@st.cache_data(ttl=3600)
def fetch_recession_indicator():
    """Fetch NBER recession indicator"""
    try:
        df = yf.download("USREC", start="1990-01-01", progress=False)
        return df['Close'] if not df.empty else None
    except:
        return None

# =============================================================================
# QUANTITATIVE ANALYTICS
# =============================================================================

def calculate_principal_components(yield_df):
    """PCA for yield curve dynamics"""
    if len(yield_df) < 100:
        return None, None
    
    # Standardize yields
    scaler = StandardScaler()
    yields_std = scaler.fit_transform(yield_df)
    
    # Perform PCA
    pca = PCA()
    pcs = pca.fit_transform(yields_std)
    
    # Create factors DataFrame
    factors = pd.DataFrame(
        pcs[:, :3],
        index=yield_df.index,
        columns=['PC1_Level', 'PC2_Slope', 'PC3_Curvature']
    )
    
    return factors, pca.explained_variance_ratio_[:3]

def calculate_forward_rates(yield_df, maturity_map):
    """Calculate implied forward rates"""
    forwards = pd.DataFrame(index=yield_df.index)
    
    maturities = sorted([(k, v) for k, v in maturity_map.items() if k in yield_df.columns])
    
    for i in range(len(maturities) - 1):
        name1, mat1 = maturities[i]
        name2, mat2 = maturities[i + 1]
        
        if name1 in yield_df.columns and name2 in yield_df.columns:
            # Forward rate formula: (1+y2)^t2 / (1+y1)^t1 - 1
            y1 = yield_df[name1] / 100
            y2 = yield_df[name2] / 100
            
            forward = ((1 + y2) ** mat2 / (1 + y1) ** mat1) ** (1 / (mat2 - mat1)) - 1
            forwards[f'FWD_{mat1}Y_{mat2}Y'] = forward * 100
    
    return forwards

def calculate_rolling_beta(spreads, market_proxy=None):
    """Calculate rolling beta for spread sensitivity"""
    if market_proxy is None:
        market_proxy = spreads.mean(axis=1)
    
    rolling_beta = pd.Series(index=spreads.index, dtype=float)
    window = 60
    
    for i in range(window, len(spreads)):
        y = spreads.iloc[i-window:i].mean(axis=1)
        x = market_proxy.iloc[i-window:i]
        
        if len(y) == len(x) and len(y) > 10:
            covariance = np.cov(y, x)[0, 1]
            variance = np.var(x)
            rolling_beta.iloc[i] = covariance / variance if variance != 0 else 0
    
    return rolling_beta

# =============================================================================
# VISUALIZATION - INSTITUTIONAL STYLE
# =============================================================================

def create_institutional_layout(fig, title, y_title=None):
    """Apply institutional styling to plots"""
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['surface'],
        plot_bgcolor=COLORS['surface'],
        font=dict(family="Courier New, monospace", size=11, color=COLORS['text_secondary']),
        title=dict(
            text=title,
            font=dict(size=14, color=COLORS['text_primary'], family="Arial, sans-serif"),
            x=0.02,
            xanchor='left'
        ),
        margin=dict(l=60, r=30, t=60, b=40),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=9)
        )
    )
    
    fig.update_xaxes(
        gridcolor=COLORS['grid'],
        gridwidth=0.5,
        zeroline=False,
        title_font=dict(size=10),
        tickfont=dict(size=9)
    )
    
    fig.update_yaxes(
        gridcolor=COLORS['grid'],
        gridwidth=0.5,
        zeroline=False,
        title_font=dict(size=10),
        tickfont=dict(size=9)
    )
    
    if y_title:
        fig.update_yaxes(title_text=y_title)
    
    return fig

def plot_yield_curve_3d(yield_df, maturity_map):
    """3D surface plot - Institutional"""
    
    # Prepare data
    dates = yield_df.index
    maturities = [maturity_map.get(col, 0) for col in yield_df.columns]
    z_data = yield_df.T.values
    
    fig = go.Figure(data=[
        go.Surface(
            x=dates,
            y=maturities,
            z=z_data,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Yield (%)", thickness=10),
            contours={
                "z": {"show": True, "usecolormap": True, "width": 1}
            }
        )
    ])
    
    fig = create_institutional_layout(
        fig, 
        "YIELD CURVE TERM STRUCTURE EVOLUTION",
        "Yield (%)"
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title="Date",
            yaxis_title="Maturity (Years)",
            zaxis_title="Yield (%)",
            camera=dict(eye=dict(x=1.8, y=-1.5, z=1.2)),
            aspectmode='manual',
            aspectratio=dict(x=2, y=0.8, z=0.6)
        ),
        height=600
    )
    
    return fig

def plot_spread_analysis(spreads, recessions):
    """Professional spread analysis dashboard"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('10Y-2Y SPREAD', '10Y-3M SPREAD', 
                       '5Y-2Y SPREAD', '30Y-10Y SPREAD'),
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
                    hovertemplate=f'<b>%{{x|%Y-%m-%d}}</b><br>{spread_name}: %{{y:.1f}} bps<extra></extra>'
                ),
                row=config['row'], col=config['col']
            )
            
            # Zero threshold line
            fig.add_hline(y=0, line_dash="dash", line_color=COLORS['negative'], 
                         line_width=1, row=config['row'], col=config['col'])
            
            # Recession shading
            for recession in recessions:
                fig.add_vrect(
                    x0=recession['start'], x1=recession['end'],
                    fillcolor=COLORS['recession'], opacity=0.3,
                    layer="below", line_width=0,
                    row=config['row'], col=config['col']
                )
    
    fig = create_institutional_layout(fig, "YIELD SPREAD DYNAMICS")
    fig.update_layout(height=600)
    
    return fig

def plot_pca_factors(factors):
    """PCA factor dynamics visualization"""
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=('LEVEL FACTOR (PC1)', 
                                      'SLOPE FACTOR (PC2)', 
                                      'CURVATURE FACTOR (PC3)'),
                        vertical_spacing=0.08)
    
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
            
            # Zero line
            fig.add_hline(y=0, line_dash="dash", line_color=COLORS['neutral'],
                         line_width=1, row=i+1, col=1)
    
    fig = create_institutional_layout(fig, "PRINCIPAL COMPONENT ANALYSIS")
    fig.update_layout(height=700)
    
    return fig

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    
    # Header
    st.markdown("""
    <div class="hedge-header">
        <h1 style="color: white; margin: 0; font-size: 1.5rem;">YIELD CURVE ANALYTICS</h1>
        <p style="color: #bdc3c7; margin: 0; font-size: 0.75rem;">Institutional Quantitative Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data loading
    with st.spinner("Loading institutional data feeds..."):
        data_result = fetch_institutional_data()
        if data_result is None:
            st.error("Data feed unavailable")
            return
        
        yield_df, maturity_map = data_result
        recession_series = fetch_recession_indicator()
    
    # Calculate quantitative metrics
    spreads = pd.DataFrame()
    if '10Y' in yield_df.columns and '2Y' in yield_df.columns:
        spreads['10Y-2Y'] = (yield_df['10Y'] - yield_df['2Y']) * 100
    if '10Y' in yield_df.columns and '3M' in yield_df.columns:
        spreads['10Y-3M'] = (yield_df['10Y'] - yield_df['3M']) * 100
    if '5Y' in yield_df.columns and '2Y' in yield_df.columns:
        spreads['5Y-2Y'] = (yield_df['5Y'] - yield_df['2Y']) * 100
    if '30Y' in yield_df.columns and '10Y' in yield_df.columns:
        spreads['30Y-10Y'] = (yield_df['30Y'] - yield_df['10Y']) * 100
    
    # Identify recessions
    recessions = []
    if recession_series is not None:
        in_recession = False
        start_date = None
        for date, value in recession_series.items():
            if value == 1 and not in_recession:
                in_recession = True
                start_date = date
            elif value == 0 and in_recession:
                in_recession = False
                recessions.append({'start': start_date, 'end': date})
    
    # Current metrics
    current_spread = spreads['10Y-2Y'].iloc[-1] if '10Y-2Y' in spreads.columns else 0
    current_10y = yield_df['10Y'].iloc[-1] if '10Y' in yield_df.columns else 0
    current_2y = yield_df['2Y'].iloc[-1] if '2Y' in yield_df.columns else 0
    current_30y = yield_df['30Y'].iloc[-1] if '30Y' in yield_df.columns else 0
    
    # Regime detection
    regime = RegimeDetection.detect_regime(spreads['10Y-2Y'].dropna())
    recession_prob = RegimeDetection.calculate_recession_probability(spreads['10Y-2Y'].dropna())
    
    # Nelson-Siegel factors
    if len(yield_df.columns) >= 5:
        maturities_num = [maturity_map.get(col, 0) for col in yield_df.columns if col in maturity_map]
        latest_yields = yield_df.iloc[-1].values[:len(maturities_num)]
        ns_factors = NelsonSiegelModel.get_factors(maturities_num, latest_yields)
    else:
        ns_factors = {'Level': 0, 'Slope': 0, 'Curvature': 0, 'Decay': 0}
    
    # PCA analysis
    pca_factors, pca_var = calculate_principal_components(yield_df)
    
    # Forward rates
    forward_rates = calculate_forward_rates(yield_df, maturity_map)
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">10Y YIELD</div>
            <div class="metric-value">{current_10y:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">10Y-2Y SPREAD</div>
            <div class="metric-value">{current_spread:.1f} bps</div>
            <div class="metric-change-{'positive' if current_spread > 0 else 'negative'}">
                {regime}
            </div>
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
            <div class="metric-label">NS LEVEL</div>
            <div class="metric-value">{ns_factors['Level']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">NS SLOPE</div>
            <div class="metric-value">{ns_factors['Slope']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs for analysis
    tabs = st.tabs([
        "TERM STRUCTURE",
        "SPREAD DYNAMICS",
        "FACTOR ANALYSIS",
        "RISK METRICS",
        "FORWARD CURVE"
    ])
    
    with tabs[0]:
        st.plotly_chart(plot_yield_curve_3d(yield_df, maturity_map), use_container_width=True)
        
        # Current curve comparison
        if len(yield_df.columns) >= 5:
            current_curve = yield_df.iloc[-1].values[:len(maturities_num)]
            one_year_ago = yield_df.iloc[-252].values[:len(maturities_num)] if len(yield_df) > 252 else current_curve
            
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(
                x=maturities_num, y=current_curve, mode='lines+markers',
                name='Current', line=dict(color=COLORS['accent'], width=2),
                marker=dict(size=6)
            ))
            fig_curve.add_trace(go.Scatter(
                x=maturities_num, y=one_year_ago, mode='lines+markers',
                name='1Y Prior', line=dict(color=COLORS['neutral'], width=1.5, dash='dash'),
                marker=dict(size=4)
            ))
            
            fig_curve = create_institutional_layout(fig_curve, "CURRENT TERM STRUCTURE", "Yield (%)")
            fig_curve.update_layout(height=450)
            st.plotly_chart(fig_curve, use_container_width=True)
    
    with tabs[1]:
        st.plotly_chart(plot_spread_analysis(spreads, recessions), use_container_width=True)
        
        # Spread statistics table
        st.markdown("### SPREAD STATISTICS")
        spread_stats = pd.DataFrame({
            'Spread': spreads.columns,
            'Current': [f"{spreads[col].iloc[-1]:.1f} bps" for col in spreads.columns],
            'Mean': [f"{spreads[col].mean():.1f} bps" for col in spreads.columns],
            'Std': [f"{spreads[col].std():.1f} bps" for col in spreads.columns],
            'Min': [f"{spreads[col].min():.1f} bps" for col in spreads.columns],
            'Max': [f"{spreads[col].max():.1f} bps" for col in spreads.columns],
            'Skew': [f"{spreads[col].skew():.2f}" for col in spreads.columns],
            'Kurt': [f"{spreads[col].kurtosis():.2f}" for col in spreads.columns]
        })
        st.dataframe(spread_stats, use_container_width=True, hide_index=True)
    
    with tabs[2]:
        if pca_factors is not None:
            st.plotly_chart(plot_pca_factors(pca_factors), use_container_width=True)
            
            # PCA variance explanation
            st.markdown("### PCA VARIANCE EXPLANATION")
            var_df = pd.DataFrame({
                'Component': ['PC1 (Level)', 'PC2 (Slope)', 'PC3 (Curvature)'],
                'Variance Explained': [f"{v:.1%}" for v in pca_var],
                'Cumulative': [f"{sum(pca_var[:i+1]):.1%}" for i in range(len(pca_var))]
            })
            st.dataframe(var_df, use_container_width=True, hide_index=True)
        
        # Nelson-Siegel factor interpretation
        st.markdown("### NELSON-SIEGEL FACTORS")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">LEVEL FACTOR</div>
                <div class="metric-value">{ns_factors['Level']:.2f}%</div>
                <div class="metric-label" style="font-size:0.7rem">Long-term expectation</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">SLOPE FACTOR</div>
                <div class="metric-value">{ns_factors['Slope']:.2f}</div>
                <div class="metric-label" style="font-size:0.7rem">Monetary policy stance</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">CURVATURE FACTOR</div>
                <div class="metric-value">{ns_factors['Curvature']:.2f}</div>
                <div class="metric-label" style="font-size:0.7rem">Medium-term expectations</div>
            </div>
            """, unsafe_allow_html=True)
    
    with tabs[3]:
        # Risk metrics
        if len(spreads) > 0:
            returns = spreads['10Y-2Y'].pct_change().dropna()
            var_95 = RiskMetrics.calculate_var(returns)
            cvar_95 = RiskMetrics.calculate_cvar(returns)
            sharpe = RiskMetrics.calculate_sharpe_ratio(returns)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">VaR (95%)</div>
                    <div class="metric-value">{var_95:.2%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">CVaR (95%)</div>
                    <div class="metric-value">{cvar_95:.2%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">SHARPE RATIO</div>
                    <div class="metric-value">{sharpe:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">VOLATILITY (ANN)</div>
                    <div class="metric-value">{returns.std() * np.sqrt(252):.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Rolling volatility chart
            rolling_vol = returns.rolling(60).std() * np.sqrt(252)
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=rolling_vol.index, y=rolling_vol,
                mode='lines', name='Rolling Volatility',
                line=dict(color=COLORS['warning'], width=1.5),
                fill='tozeroy', fillcolor='rgba(243, 156, 18, 0.1)'
            ))
            fig_vol = create_institutional_layout(fig_vol, "ROLLING VOLATILITY (60D)", "Volatility")
            fig_vol.update_layout(height=400)
            st.plotly_chart(fig_vol, use_container_width=True)
    
    with tabs[4]:
        if not forward_rates.empty:
            # Latest forward curve
            latest_forward = forward_rates.iloc[-1].dropna()
            
            fig_fwd = go.Figure()
            fig_fwd.add_trace(go.Scatter(
                x=range(len(latest_forward)),
                y=latest_forward.values,
                mode='lines+markers',
                name='Forward Rates',
                line=dict(color=COLORS['accent'], width=2),
                marker=dict(size=6)
            ))
            
            fig_fwd = create_institutional_layout(fig_fwd, "IMPLIED FORWARD RATE CURVE", "Rate (%)")
            fig_fwd.update_layout(height=450)
            st.plotly_chart(fig_fwd, use_container_width=True)
            
            # Forward rates table
            fwd_df = pd.DataFrame({
                'Period': [f"{p.split('_')[1].replace('Y', 'Y-')}{p.split('_')[2].replace('Y', 'Y')}" 
                          for p in latest_forward.index],
                'Forward Rate': [f"{v:.2f}%" for v in latest_forward.values]
            })
            st.dataframe(fwd_df, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.7rem; padding: 1rem;">
        <p>© 2024 Yield Curve Analytics | Institutional Quantitative Platform</p>
        <p>Data: FRED | Model: Nelson-Siegel | Risk: Parametric VaR</p>
        <p>Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
