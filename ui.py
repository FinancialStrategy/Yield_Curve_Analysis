"""
UI Helper Module - KPI cards, API gate, CSS styling
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Optional
from config import COLORS, __version__
from data import validate_fred_api_key


def render_css() -> None:
    """Render custom CSS for the application"""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(135deg, {COLORS['bg']} 0%, {COLORS['bg2']} 100%);
        }}
        .main-title-card {{
            background: linear-gradient(90deg, {COLORS['header']} 0%, {COLORS['accent']} 100%);
            border-radius: 20px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.12);
        }}
        .main-title {{
            color: white;
            font-weight: 800;
            font-size: 1.65rem;
            margin: 0;
            letter-spacing: -0.02em;
        }}
        .main-subtitle {{
            color: rgba(255,255,255,0.86);
            font-size: 0.86rem;
            margin-top: 0.55rem;
        }}
        .metric-card {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['grid']};
            border-radius: 16px;
            padding: 1rem;
            min-height: 120px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
            transition: all 0.2s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        }}
        .metric-label {{
            color: {COLORS['text_secondary']};
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}
        .metric-value {{
            color: {COLORS['text']};
            font-size: 1.55rem;
            font-weight: 800;
            margin-top: 0.45rem;
            font-family: 'Courier New', monospace;
        }}
        .metric-sub {{
            color: {COLORS['muted']};
            font-size: 0.75rem;
            margin-top: 0.35rem;
        }}
        .metric-trend-up {{
            color: {COLORS['positive']};
            font-weight: 600;
        }}
        .metric-trend-down {{
            color: {COLORS['negative']};
            font-weight: 600;
        }}
        .note-box {{
            background: {COLORS['surface_alt']};
            border: 1px solid {COLORS['grid']};
            border-left: 4px solid {COLORS['accent']};
            border-radius: 12px;
            padding: 1rem 1.2rem;
            font-size: 0.88rem;
            margin: 1rem 0;
        }}
        .warning-box {{
            background: #fff8f0;
            border: 1px solid #f0d8b0;
            border-left: 4px solid {COLORS['warning']};
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin: 1rem 0;
        }}
        .success-box {{
            background: #f3fbf6;
            border: 1px solid #bfe3ca;
            border-left: 4px solid {COLORS['positive']};
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin: 1rem 0;
        }}
        .info-box {{
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-left: 4px solid {COLORS['info']};
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin: 1rem 0;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0;
            border-bottom: 2px solid {COLORS['grid']};
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {COLORS['text_secondary']};
            font-weight: 700;
            font-size: 0.74rem;
            text-transform: uppercase;
            padding: 8px 14px;
        }}
        .stTabs [aria-selected="true"] {{
            color: {COLORS['accent']};
            border-bottom: 3px solid {COLORS['accent']};
        }}
        .stButton > button, .stDownloadButton > button {{
            background: {COLORS['surface']};
            color: {COLORS['text']};
            border: 1px solid {COLORS['grid']};
            border-radius: 10px;
            font-weight: 700;
            transition: all 0.2s ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            border-color: {COLORS['accent']};
            color: {COLORS['accent']};
            transform: translateY(-1px);
        }}
        #MainMenu, header, footer {{
            visibility: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render the main header/title section"""
    st.markdown(
        f"""
        <div class="main-title-card">
            <div class="main-title">Dynamic Quantitative Analysis Model</div>
            <div class="main-subtitle">Institutional Fixed-Income Platform | v{__version__}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_api_gate() -> None:
    """Render API key input and validation screen"""
    st.markdown(
        f"""
        <div class="note-box" style="max-width:560px; margin:40px auto; text-align:center;">
            <b>🔑 FRED API Key Required</b><br><br>
            This institutional platform requires live U.S. Treasury data from FRED.<br>
            Get your free API key from the <b>FRED</b> website.
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        api_key = st.text_input("Enter your FRED API key", type="password", placeholder="Paste API key here")
        
        if st.button("🔐 Validate & Connect", use_container_width=True):
            if not api_key:
                st.error("Please enter a valid API key.")
                st.stop()
            
            with st.spinner("Validating API key..."):
                valid = validate_fred_api_key(api_key)
            
            if valid:
                st.session_state.api_key = api_key
                st.session_state.api_key_validated = True
                st.success("✓ API key validated successfully. Loading data...")
                st.rerun()
            else:
                st.error("Invalid API key. Please check and try again.")
    
    st.stop()


def kpi_card(label: str, value: str, sub: str, trend: Optional[float] = None, trend_label: str = "") -> None:
    """
    Render a KPI metric card
    
    Parameters
    ----------
    label : str
        Card label/title
    value : str
        Main metric value
    sub : str
        Subtitle or additional information
    trend : float, optional
        Trend percentage value
    trend_label : str
        Label for trend (e.g., "20d")
    """
    trend_html = ""
    if trend is not None and not np.isnan(trend):
        trend_class = "metric-trend-up" if trend >= 0 else "metric-trend-down"
        trend_symbol = "▲" if trend >= 0 else "▼"
        trend_html = f'<div class="{trend_class}" style="font-size:0.7rem; margin-top:0.25rem;">{trend_symbol} {abs(trend):.2f}% {trend_label}</div>'
    
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{sub}</div>{trend_html}</div>',
        unsafe_allow_html=True,
    )


def create_smart_kpi_row(yield_df: pd.DataFrame, spreads: pd.DataFrame, regime: str, regime_text: str, recession_prob: float, vix_data: pd.Series = None) -> None:
    """
    Create the smart KPI row with 6 metrics including trends
    
    Parameters
    ----------
    yield_df : pd.DataFrame
        Yield curve DataFrame
    spreads : pd.DataFrame
        Spreads DataFrame
    regime : str
        Current market regime
    regime_text : str
        Regime description
    recession_prob : float
        Recession probability
    vix_data : pd.Series, optional
        VIX time series for trend calculation
    """
    current_2y = yield_df["2Y"].iloc[-1] if "2Y" in yield_df.columns else np.nan
    current_10y = yield_df["10Y"].iloc[-1] if "10Y" in yield_df.columns else np.nan
    current_spread = spreads["10Y-2Y"].iloc[-1] if "10Y-2Y" in spreads.columns else np.nan
    
    # Calculate 20-day trends
    if "2Y" in yield_df.columns and len(yield_df) > 20:
        trend_2y = (yield_df["2Y"].iloc[-1] - yield_df["2Y"].iloc[-20]) / yield_df["2Y"].iloc[-20] * 100
        trend_10y = (yield_df["10Y"].iloc[-1] - yield_df["10Y"].iloc[-20]) / yield_df["10Y"].iloc[-20] * 100
    else:
        trend_2y = trend_10y = None
    
    # VIX trend
    current_vix = vix_data.iloc[-1] if vix_data is not None and not vix_data.empty else np.nan
    if vix_data is not None and not vix_data.empty and len(vix_data) > 20:
        trend_vix = (vix_data.iloc[-1] - vix_data.iloc[-20]) / vix_data.iloc[-20] * 100
    else:
        trend_vix = None
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        kpi_card("📊 Macro Regime", regime, regime_text)
    
    with col2:
        kpi_card("🏦 2Y Yield", f"{current_2y:.2f}%" if np.isfinite(current_2y) else "N/A", "Policy anchor", trend_2y, "20d")
    
    with col3:
        kpi_card("📈 10Y Yield", f"{current_10y:.2f}%" if np.isfinite(current_10y) else "N/A", "Benchmark", trend_10y, "20d")
    
    with col4:
        spread_display = f"{current_spread:.1f} bps" if np.isfinite(current_spread) else "N/A"
        kpi_card("🔄 10Y-2Y Spread", spread_display, "Recession signal")
    
    with col5:
        kpi_card("⚠️ Recession Prob", f"{recession_prob:.1%}", "Proxy estimate")
    
    with col6:
        kpi_card("📉 VIX", f"{current_vix:.2f}" if np.isfinite(current_vix) else "N/A", "Fear gauge", trend_vix, "20d")


def render_footer() -> None:
    """Render the application footer"""
    from datetime import datetime
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center; color:#667085; font-size:0.75rem;'>"
        f"Institutional Quantitative Platform | Algorithmic Forecasting | Advanced Monte Carlo Engine<br>"
        f"MK Istanbul Fintech LabGEN © 2026 | v{__version__} | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        f"</div>",
        unsafe_allow_html=True,
    )