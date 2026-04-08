# ============================================================================
# PROFESSIONAL YIELD CURVE ANALYSIS - STREAMLIT CLOUD VERSION
# ============================================================================
# Author: Advanced Yield Curve Analytics
# Data Source: Federal Reserve Economic Data (FRED)
# Version: 5.0 (Python/Streamlit Cloud Edition)
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Yield Curve Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .warning-card {
        border-left-color: #d62728;
        background: linear-gradient(135deg, #fff5f5 0%, #ffe0e0 100%);
    }
    .caution-card {
        border-left-color: #ff7f0e;
        background: linear-gradient(135deg, #fff8f0 0%, #ffe8cc 100%);
    }
    .normal-card {
        border-left-color: #2ca02c;
        background: linear-gradient(135deg, #f0fff0 0%, #e0ffe0 100%);
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
        margin-top: 5px;
    }
    .term-box {
        background-color: #f0f4f8;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px;
        padding: 8px 16px;
        background-color: #f0f2f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def fetch_treasury_yields():
    """Fetch treasury yield data from FRED via yfinance"""
    
    # FRED tickers for treasury yields
    tickers = {
        '0.25': 'DGS3MO',
        '0.5': 'DGS6MO', 
        '1': 'DGS1',
        '2': 'DGS2',
        '3': 'DGS3',
        '5': 'DGS5',
        '7': 'DGS7',
        '10': 'DGS10',
        '20': 'DGS20',
        '30': 'DGS30'
    }
    
    yields_data = {}
    
    for maturity, ticker in tickers.items():
        try:
            # Fetch data from FRED via yfinance
            df = yf.download(ticker, start="1990-01-01", progress=False)
            if not df.empty:
                yields_data[maturity] = df['Close']
        except Exception as e:
            st.warning(f"Could not fetch {ticker}: {e}")
    
    if not yields_data:
        st.error("No yield data could be fetched. Please check your internet connection.")
        return None
    
    # Combine all series
    df = pd.DataFrame(yields_data)
    df = df.dropna()
    
    return df

@st.cache_data(ttl=3600)
def fetch_recession_data():
    """Fetch NBER recession indicator data"""
    try:
        # USREC from FRED (1 = recession, 0 = expansion)
        df = yf.download("USREC", start="1990-01-01", progress=False)
        if not df.empty:
            return df['Close']
    except:
        pass
    return None

# ============================================================================
# CALCULATION FUNCTIONS
# ============================================================================

def calculate_spreads(yield_df):
    """Calculate various yield spreads"""
    spreads = pd.DataFrame(index=yield_df.index)
    
    # 10Y-2Y Spread
    if '10' in yield_df.columns and '2' in yield_df.columns:
        spreads['10Y-2Y'] = yield_df['10'] - yield_df['2']
    
    # 10Y-3M Spread
    if '10' in yield_df.columns and '0.25' in yield_df.columns:
        spreads['10Y-3M'] = yield_df['10'] - yield_df['0.25']
    
    # 5Y-2Y Spread
    if '5' in yield_df.columns and '2' in yield_df.columns:
        spreads['5Y-2Y'] = yield_df['5'] - yield_df['2']
    
    # 30Y-10Y Spread
    if '30' in yield_df.columns and '10' in yield_df.columns:
        spreads['30Y-10Y'] = yield_df['30'] - yield_df['10']
    
    return spreads

def identify_recessions(recession_series):
    """Identify recession periods from NBER data"""
    if recession_series is None:
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
            recessions.append({'start': start_date, 'end': date})
    
    return recessions

def get_current_snapshot(yield_df):
    """Get current yield curve snapshot"""
    latest = yield_df.iloc[-1]
    latest_date = yield_df.index[-1]
    
    # 1 year ago
    one_year_ago = latest_date - timedelta(days=365)
    idx_1y = yield_df.index.get_indexer([one_year_ago], method='nearest')[0]
    one_year_ago_data = yield_df.iloc[idx_1y]
    
    # 5 year average
    five_years_ago = latest_date - timedelta(days=5*365)
    idx_5y = yield_df.index.get_indexer([five_years_ago], method='nearest')[0]
    five_year_avg = yield_df.iloc[idx_5y:].mean()
    
    snapshot = pd.DataFrame({
        'Maturity': yield_df.columns.astype(float),
        'Current': latest.values,
        '1 Year Ago': one_year_ago_data.values,
        '5 Year Avg': five_year_avg.values,
        'Change (1Y)': latest.values - one_year_ago_data.values
    })
    
    return snapshot, latest_date

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_3d_surface_plot(yield_df):
    """Create 3D surface plot of yield curve evolution"""
    # Prepare data for 3D surface
    dates = yield_df.index
    maturities = yield_df.columns.astype(float)
    z_data = yield_df.T.values
    
    fig = go.Figure(data=[
        go.Surface(
            x=dates,
            y=maturities,
            z=z_data,
            colorscale='Blues',
            contours={
                "z": {"show": True, "usecolormap": True, "highlightcolor": "#ff0000", "project": {"z": True}}
            },
            hovertemplate="<b>Date: %{x|%Y-%m-%d}</b><br>Maturity: %{y} years<br>Yield: %{z:.2f}%<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title="<b>U.S. Treasury Yield Curve Evolution</b><br><sup>1990 - Present | 3D Visualization</sup>",
        scene=dict(
            xaxis_title="<b>Date</b>",
            yaxis_title="<b>Maturity (Years)</b>",
            zaxis_title="<b>Yield (%)</b>",
            camera=dict(eye=dict(x=1.8, y=-1.5, z=1.2)),
            aspectmode='manual',
            aspectratio=dict(x=2, y=0.8, z=0.6)
        ),
        height=600,
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    return fig

def create_yield_curve_evolution(yield_df, recessions):
    """Create yield curve evolution chart with recession shading"""
    fig = go.Figure()
    
    # Add yield curves for selected maturities
    colors = px.colors.sequential.Blues
    maturities_to_show = ['0.25', '1', '2', '5', '10', '30']
    
    for i, maturity in enumerate(maturities_to_show):
        if maturity in yield_df.columns:
            fig.add_trace(go.Scatter(
                x=yield_df.index,
                y=yield_df[maturity],
                mode='lines',
                name=f'{maturity}Y',
                line=dict(width=1.5, color=colors[min(i+3, len(colors)-1)]),
                hovertemplate=f'<b>%{{x|%Y-%m-%d}}</b><br>{maturity}Y Yield: %{{y:.2f}}%<extra></extra>'
            ))
    
    # Add recession shading
    for recession in recessions:
        fig.add_vrect(
            x0=recession['start'], x1=recession['end'],
            fillcolor="gray", opacity=0.3,
            layer="below", line_width=0,
            annotation_text="NBER Recession",
            annotation_position="top left"
        )
    
    fig.update_layout(
        title="<b>Historical Treasury Yield Curves (1990-Present)</b>",
        xaxis_title="<b>Date</b>",
        yaxis_title="<b>Yield (%)</b>",
        hovermode='x unified',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_spread_dashboard(spreads, recessions):
    """Create spread dashboard with recession shading"""
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=('10Y-2Y Spread (Primary Indicator)',
                                      '10Y-3M Spread (Campbell Harvey)',
                                      '5Y-2Y Spread (Medium-term)',
                                      '30Y-10Y Spread (Term Premium)'),
                        vertical_spacing=0.12,
                        horizontal_spacing=0.1)
    
    spread_colors = {'10Y-2Y': '#d62728', '10Y-3M': '#1f77b4', 
                     '5Y-2Y': '#2ca02c', '30Y-10Y': '#ff7f0e'}
    
    row_col = {'10Y-2Y': (1, 1), '10Y-3M': (1, 2), '5Y-2Y': (2, 1), '30Y-10Y': (2, 2)}
    
    for spread_name in spreads.columns:
        if spread_name in row_col:
            row, col = row_col[spread_name]
            
            fig.add_trace(
                go.Scatter(
                    x=spreads.index,
                    y=spreads[spread_name],
                    mode='lines',
                    name=spread_name,
                    line=dict(color=spread_colors.get(spread_name, '#1f77b4'), width=2),
                    hovertemplate=f'<b>%{{x|%Y-%m-%d}}</b><br>{spread_name}: %{{y:.1f}} bps<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="red", 
                         opacity=0.5, row=row, col=col)
            
            # Add recession shading
            for recession in recessions:
                fig.add_vrect(
                    x0=recession['start'], x1=recession['end'],
                    fillcolor="gray", opacity=0.2,
                    layer="below", line_width=0,
                    row=row, col=col
                )
    
    fig.update_layout(
        title="<b>Yield Spread Dynamics & NBER Recession Periods</b>",
        height=600,
        showlegend=False,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Spread (bps)", row=1, col=1)
    fig.update_yaxes(title_text="Spread (bps)", row=1, col=2)
    fig.update_yaxes(title_text="Spread (bps)", row=2, col=1)
    fig.update_yaxes(title_text="Spread (bps)", row=2, col=2)
    
    return fig

def create_current_curve_comparison(snapshot):
    """Create current yield curve comparison chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=snapshot['Maturity'],
        y=snapshot['Current'],
        mode='lines+markers',
        name='Current Yield',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8, symbol='circle'),
        hovertemplate='Maturity: %{x} years<br>Current: %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=snapshot['Maturity'],
        y=snapshot['1 Year Ago'],
        mode='lines+markers',
        name='1 Year Ago',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6, symbol='square'),
        hovertemplate='Maturity: %{x} years<br>1 Year Ago: %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=snapshot['Maturity'],
        y=snapshot['5 Year Avg'],
        mode='lines+markers',
        name='5 Year Average',
        line=dict(color='#2ca02c', width=2, dash='dot'),
        marker=dict(size=6, symbol='diamond'),
        hovertemplate='Maturity: %{x} years<br>5 Year Avg: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="<b>Current Yield Curve vs Historical Averages</b>",
        xaxis_title="<b>Maturity (Years)</b>",
        yaxis_title="<b>Yield (%)</b>",
        hovermode='x unified',
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_slope_heatmap(yield_df):
    """Create heatmap of 10Y-2Y slope by month/year"""
    if '10' not in yield_df.columns or '2' not in yield_df.columns:
        return None
    
    slope = yield_df['10'] - yield_df['2']
    
    # Create year-month matrix
    slope_df = pd.DataFrame({
        'Year': slope.index.year,
        'Month': slope.index.month,
        'Slope': slope.values
    })
    
    # Aggregate by year-month
    heatmap_data = slope_df.groupby(['Year', 'Month'])['Slope'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='Year', columns='Month', values='Slope')
    heatmap_pivot = heatmap_pivot.sort_index(ascending=False)
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=month_names,
        y=heatmap_pivot.index,
        colorscale=[
            [0, '#d62728'],    # Negative (inverted) - Red
            [0.5, '#ffffff'],   # Zero - White
            [1, '#2ca02c']      # Positive - Green
        ],
        zmid=0,
        text=heatmap_pivot.values.round(1),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='Year: %{y}<br>Month: %{x}<br>Spread: %{z:.1f} bps<extra></extra>'
    ))
    
    fig.update_layout(
        title="<b>Yield Curve Slope Heatmap (10Y-2Y Spread by Month)</b><br><sup>Red = Inverted (Recession Signal), Green = Steep (Expansion)</sup>",
        xaxis_title="<b>Month</b>",
        yaxis_title="<b>Year</b>",
        height=500,
        yaxis=dict(autorange="reversed")
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Professional Yield Curve Analysis Dashboard</h1>
        <p>NBER Recession Indicators | Spread Dynamics | Executive Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔧 Controls")
        st.markdown("---")
        
        # Data refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📚 Terminology")
        st.markdown("""
        - **x10**: 10-Year Treasury Yield
        - **x30**: 30-Year Treasury Yield
        - **10Y-2Y**: Primary recession indicator
        - **10Y-3M**: Campbell Harvey indicator
        - **30Y-10Y**: Term premium
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Spread Meanings")
        st.markdown("""
        - **INVERTED (<0)**: Recession warning (12-18 months)
        - **FLATTENING (0-50 bps)**: Economic slowdown signal
        - **NORMAL (>50 bps)**: Economic expansion
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Data Source:** FRED (Federal Reserve Economic Data)
        
        **Recession Definition:** NBER (National Bureau of Economic Research)
        
        **Update Frequency:** Daily
        """)
    
    # Load data
    with st.spinner("Fetching latest yield data from FRED..."):
        yield_df = fetch_treasury_yields()
        recession_series = fetch_recession_data()
    
    if yield_df is None:
        st.error("Failed to load data. Please check your internet connection and try again.")
        return
    
    # Identify recessions
    recessions = identify_recessions(recession_series)
    
    # Calculate spreads
    spreads = calculate_spreads(yield_df)
    
    # Get current snapshot
    snapshot, latest_date = get_current_snapshot(yield_df)
    
    # Current metrics
    last_spread = spreads['10Y-2Y'].iloc[-1] if '10Y-2Y' in spreads.columns else 0
    
    if last_spread < 0:
        curve_status = "INVERTED ⚠️"
        status_class = "warning-card"
        status_message = "Recession Warning - Historically predicts recession within 12-18 months"
    elif last_spread < 50:
        curve_status = "FLATTENING 📉"
        status_class = "caution-card"
        status_message = "Economic Slowdown Signal - Monitor closely"
    else:
        curve_status = "NORMAL ✅"
        status_class = "normal-card"
        status_message = "Economic Expansion - Supportive for risk assets"
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{snapshot[snapshot['Maturity'] == 10]['Current'].values[0]:.2f}%</div>
            <div class="metric-label">Current 10Y Yield (x10)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{snapshot[snapshot['Maturity'] == 30]['Current'].values[0]:.2f}%</div>
            <div class="metric-label">Current 30Y Yield (x30)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{last_spread:.1f} bps</div>
            <div class="metric-label">10Y-2Y Spread</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card {status_class}">
            <div class="metric-value">{curve_status}</div>
            <div class="metric-label">{status_message}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Yield Curve Evolution",
        "📊 Spread Dynamics",
        "🎯 Current Analysis",
        "🗺️ Heatmap Analysis",
        "📄 Executive Summary"
    ])
    
    with tab1:
        st.markdown("### 📈 Historical Yield Curve Evolution")
        st.markdown("*Gray shaded areas represent NBER-defined recession periods*")
        
        # 3D Surface Plot
        st.plotly_chart(create_3d_surface_plot(yield_df), use_container_width=True)
        
        # 2D Evolution Chart
        st.plotly_chart(create_yield_curve_evolution(yield_df, recessions), use_container_width=True)
    
    with tab2:
        st.markdown("### 📊 Yield Spread Dynamics")
        st.markdown("*Complete analysis of all yield spreads with NBER recession shading*")
        
        # Spread Dashboard
        st.plotly_chart(create_spread_dashboard(spreads, recessions), use_container_width=True)
        
        # Spread statistics table
        st.markdown("### 📋 Spread Statistics")
        
        spread_stats = pd.DataFrame({
            'Spread': spreads.columns,
            'Current (bps)': [round(spreads[col].iloc[-1], 1) for col in spreads.columns],
            'Average (bps)': [round(spreads[col].mean(), 1) for col in spreads.columns],
            'Min (bps)': [round(spreads[col].min(), 1) for col in spreads.columns],
            'Max (bps)': [round(spreads[col].max(), 1) for col in spreads.columns],
            'Status': [
                '⚠️ INVERTED' if spreads[col].iloc[-1] < 0 else 
                '📉 FLATTENING' if spreads[col].iloc[-1] < 50 else 
                '✅ NORMAL' for col in spreads.columns
            ]
        })
        
        st.dataframe(spread_stats, use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown("### 🎯 Current Yield Curve Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Current curve comparison
            st.plotly_chart(create_current_curve_comparison(snapshot), use_container_width=True)
        
        with col2:
            # Current snapshot table
            st.markdown("#### 📊 Current Yield Curve Snapshot")
            st.dataframe(
                snapshot.round(2),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Maturity": st.column_config.NumberColumn("Maturity (Years)", format="%.2f"),
                    "Current": st.column_config.NumberColumn("Current Yield (%)", format="%.2f%%"),
                    "1 Year Ago": st.column_config.NumberColumn("1 Year Ago (%)", format="%.2f%%"),
                    "5 Year Avg": st.column_config.NumberColumn("5 Year Avg (%)", format="%.2f%%"),
                    "Change (1Y)": st.column_config.NumberColumn("Change (bps)", format="%.2f")
                }
            )
        
        # Spread dynamics explanation
        st.markdown("### 📚 Spread Dynamics Explained")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="term-box">
                <h4>🏦 What are x10, x30?</h4>
                <p><strong>x10 (10-Year Treasury Yield):</strong> ABD Hazine'nin 10 yıl vadeli tahvilinin getirisidir. Ekonomik büyüme ve enflasyon beklentilerini yansıtır.</p>
                <p><strong>x30 (30-Year Treasury Yield):</strong> ABD Hazine'nin 30 yıl vadeli tahvilinin getirisidir. En uzun vadeli devlet tahvili olup, çok uzun vadeli enflasyon ve büyüme beklentilerini yansıtır.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="term-box">
                <h4>📈 Spread Terimleri ve Anlamları:</h4>
                <ul>
                    <li><strong>10Y-2Y Spread:</strong> En önemli resesyon göstergesi. Negatif olduğunda 12-18 ay içinde resesyon sinyali verir.</li>
                    <li><strong>10Y-3M Spread:</strong> Campbell Harvey göstergesi. Daha hassas bir resesyon göstergesidir.</li>
                    <li><strong>5Y-2Y Spread:</strong> Orta vade beklentileri ve para politikasının etkinliğini ölçer.</li>
                    <li><strong>30Y-10Y Spread:</strong> Term premium. Uzun vadeli enflasyon ve büyüme beklentilerini yansıtır.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("### 🗺️ Yield Curve Slope Heatmap")
        st.markdown("*Monthly average of 10Y-2Y spread - Red indicates inversion risk, Green indicates expansion*")
        
        heatmap = create_slope_heatmap(yield_df)
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True)
    
    with tab5:
        st.markdown("### 📄 Executive Summary")
        
        # Investment recommendations
        st.markdown("#### 🎯 Investment Implications")
        
        if last_spread < 0:
            st.markdown("""
            <div class="metric-card warning-card">
                <h3>⚠️ DEFENSIVE POSITIONING RECOMMENDED</h3>
                <ul>
                    <li>Reduce duration exposure and increase cash holdings</li>
                    <li>Focus on high-quality, investment-grade fixed income</li>
                    <li>Consider floating rate instruments to mitigate duration risk</li>
                    <li>Review equity exposure and sector allocation</li>
                    <li>Prepare for potential Fed easing cycle</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif last_spread < 50:
            st.markdown("""
            <div class="metric-card caution-card">
                <h3>📊 CAUTIOUS OPTIMISM</h3>
                <ul>
                    <li>Maintain balanced duration positioning</li>
                    <li>Consider barbell strategies (short + long maturities)</li>
                    <li>Monitor economic data for growth signals</li>
                    <li>Gradually increase risk exposure on pullbacks</li>
                    <li>Prepare for curve steepener trades</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card normal-card">
                <h3>🚀 GROWTH-ORIENTED POSITIONING</h3>
                <ul>
                    <li>Extend duration to capture higher yields</li>
                    <li>Consider risk-on positioning across asset classes</li>
                    <li>Favor cyclical sectors and high-beta strategies</li>
                    <li>Monitor Fed policy for normalization signals</li>
                    <li>Implement steepener trades for economic acceleration</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Historical analysis
        st.markdown("#### ⏰ Historical Recession Lead Time Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-card normal-card">
                <div class="metric-value">12-18 Months</div>
                <div class="metric-label">Average lead time from inversion to recession</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            inversion_pct = (spreads['10Y-2Y'] < 0).sum() / len(spreads) * 100
            st.markdown(f"""
            <div class="metric-card normal-card">
                <div class="metric-value">{inversion_pct:.1f}%</div>
                <div class="metric-label">Historical time curve has been inverted</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        **Historical Patterns:**
        - **1990 Recession:** Inversion occurred 18 months prior
        - **2001 Recession:** Inversion occurred 14 months prior (dot-com bubble)
        - **2008 Financial Crisis:** Inversion occurred 22 months prior (Great Recession)
        - **2020 COVID Recession:** Brief inversion in March 2020 (pandemic-driven)
        """)
        
        # Summary statistics
        st.markdown("#### 📊 Summary Statistics")
        
        summary_stats = pd.DataFrame({
            'Metric': [
                'Total Observations',
                'Date Range',
                'Number of NBER Recessions',
                'Current 10Y-2Y Spread',
                'Historical Inversion Frequency'
            ],
            'Value': [
                f"{len(yield_df):,}",
                f"{yield_df.index[0].strftime('%Y-%m-%d')} to {yield_df.index[-1].strftime('%Y-%m-%d')}",
                f"{len(recessions)}",
                f"{last_spread:.1f} bps",
                f"{inversion_pct:.1f}%"
            ]
        })
        
        st.dataframe(summary_stats, use_container_width=True, hide_index=True)
        
        # Data download section
        st.markdown("#### 📁 Download Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_yields = yield_df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Yield Data (CSV)",
                data=csv_yields,
                file_name="historical_yields.csv",
                mime="text/csv"
            )
        
        with col2:
            csv_spreads = spreads.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Spread Data (CSV)",
                data=csv_spreads,
                file_name="historical_spreads.csv",
                mime="text/csv"
            )
        
        with col3:
            csv_snapshot = snapshot.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Current Snapshot (CSV)",
                data=csv_snapshot,
                file_name="current_snapshot.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 12px;">
        <p>Data Source: Federal Reserve Economic Data (FRED) | NBER Recession Definitions</p>
        <p>Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>This report is for informational purposes only. Past performance does not guarantee future results.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
