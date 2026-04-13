"""
Yield Curve Institutional Platform - Main Application Entry Point
Version 37.1 - Fully Functional with Visible Technical Charts

This is the main Streamlit application that orchestrates all modules.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

# Import all modules
from config import COLORS, MATURITY_MAP, YAHOO_TICKERS, CFG, __version__
from data import fetch_yield_curve, fetch_recession_series, fetch_market_data, fetch_ohlc_data, YFINANCE_AVAILABLE
from analytics import (
    compute_spreads, classify_regime, recession_probability, identify_recessions,
    calculate_inversion_periods, calculate_lead_times, recession_hit_stats,
    factor_contributions, calculate_forward_rates, calculate_var_metrics, forecast_curve
)
from models import NelsonSiegel, model_governance, rolling_ns_parameters, MonteCarlo, Backtest
from volatility import VolatilityAnalyzer, CorrelationAnalyzer
from ml_forecast import MLForecastModel
from scenarios import generate_scenarios, get_scenario_interpretation, calculate_scenario_impact
from technical import add_technical_indicators, get_technical_signals, plot_technical_chart
from visuals import (
    chart_yield_curve, chart_yield_history, chart_spreads, chart_model_residuals,
    chart_dynamic_params, chart_factors, chart_pca, chart_rate_dynamics,
    chart_monte_carlo, chart_monte_carlo_distribution, chart_backtest, chart_scenario, chart_correlation,
    chart_technical, chart_ohlc, chart_volatility, chart_forecast
)
from ui import render_css, render_header, render_api_gate, create_smart_kpi_row, render_footer, safe_float


def safe_series_value(series: pd.Series, default: float = 0.0) -> float:
    """Safely extract the last value from a pandas Series"""
    if series is None or series.empty:
        return default
    try:
        val = series.iloc[-1]
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default


def safe_correlation(series1: pd.Series, series2: pd.Series) -> Optional[float]:
    """
    Safely calculate correlation between two pandas Series
    
    Parameters
    ----------
    series1 : pd.Series
        First time series
    series2 : pd.Series
        Second time series
    
    Returns
    -------
    float or None
        Correlation value or None if calculation fails
    """
    if series1 is None or series2 is None:
        return None
    if series1.empty or series2.empty:
        return None
    
    try:
        # Create a DataFrame with both series and drop NaN values
        df = pd.DataFrame({
            'series1': series1,
            'series2': series2
        }).dropna()
        
        # Need at least 3 observations for meaningful correlation
        if df.empty or len(df) < 3:
            return None
        
        # Calculate correlation using numpy for better control
        corr_matrix = np.corrcoef(df['series1'].values, df['series2'].values)
        corr_value = corr_matrix[0, 1]
        
        if np.isnan(corr_value):
            return None
        
        return float(corr_value)
    except Exception:
        return None


def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title="Yield Curve Institutional Platform",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Render UI components
    render_css()
    render_header()
    
    # Session state initialization
    if "api_key_validated" not in st.session_state:
        st.session_state.api_key_validated = False
    if "data_fetched" not in st.session_state:
        st.session_state.data_fetched = False
    
    # API Key validation
    if not st.session_state.api_key_validated:
        render_api_gate()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### 🎛️ Control Tower")
        
        # Date range selection
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=365*2)
        
        start = st.date_input("Start Date", start_date, max_value=end_date)
        end = st.date_input("End Date", end_date, max_value=end_date)
        
        st.markdown("---")
        
        # Analysis parameters
        with st.expander("📊 Analysis Parameters", expanded=True):
            rolling_years = st.slider("Rolling Window (years)", 2, 10, CFG.rolling_years_default)
            forecast_horizon = st.slider("Forecast Horizon (days)", 5, 60, CFG.forecast_horizon_default)
            confidence_level = st.slider("VaR Confidence Level", 0.90, 0.99, 0.95, 0.01)
        
        # Monte Carlo parameters
        with st.expander("🎲 Monte Carlo Parameters", expanded=False):
            mc_model = st.selectbox("Simulation Model", ["Geometric Brownian Motion", "Vasicek Mean-Reverting", "Ornstein-Uhlenbeck", "Jump Diffusion"])
            mc_simulations = st.slider("Number of Simulations", 1000, 20000, 8000, 1000)
            mc_horizon = st.slider("Forecast Horizon (days)", 5, 252, 20)
            mc_confidence = st.slider("Monte Carlo Confidence", 0.80, 0.99, 0.95, 0.01)
            mc_lookback = st.slider("Calibration Lookback (days)", 60, 756, 252, 21)
            mc_shock_bps = st.slider("Parallel Shock (bps)", -150, 150, 0, 5)
            mc_shock_day = st.slider("Shock Start Day", 0, 30, 1, 1)
            mc_shock_persistence = st.slider("Shock Persistence", 0.50, 1.00, 1.00, 0.05)
        
        # Machine Learning parameters
        with st.expander("🤖 Machine Learning", expanded=False):
            ml_model_type = st.selectbox("Algorithm", ["Random Forest", "Gradient Boosting"])
            ml_lags = st.slider("Autoregressive Lags", 3, 21, 5)
        
        # Backtest parameters
        with st.expander("📊 Backtest", expanded=False):
            bt_strategy = st.selectbox("Strategy Type", ["Curve Inversion", "Macro Trend", "Momentum"])
        
        # Technical Analysis
        with st.expander("📡 Technical Analysis", expanded=False):
            ohlc_period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        
        # Run button
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Fetch data on first run or when requested
    if not st.session_state.data_fetched or run_analysis:
        with st.spinner("Fetching data from FRED..."):
            # Convert dates to string format
            start_str = start.strftime("%Y-%m-%d")
            end_str = end.strftime("%Y-%m-%d")
            
            # Fetch yield curve and recession data
            yield_df = fetch_yield_curve(st.session_state.api_key, start_str, end_str)
            recession_series = fetch_recession_series(st.session_state.api_key, start_str, end_str)
            
            # Fetch VIX data
            vix_data = fetch_market_data("^VIX", start_str, end_str)
            
            if yield_df.empty:
                st.error("Failed to fetch data. Please check your API key and try again.")
                if st.button("Reset API Key"):
                    st.session_state.api_key_validated = False
                    st.rerun()
                st.stop()
            
            # Store in session state
            st.session_state.yield_df = yield_df
            st.session_state.recession_series = recession_series
            st.session_state.vix_data = vix_data
            st.session_state.data_fetched = True
            st.session_state.start_str = start_str
            st.session_state.end_str = end_str
            st.session_state.rolling_years = rolling_years
            st.session_state.forecast_horizon = forecast_horizon
            st.session_state.confidence_level = confidence_level
            st.session_state.mc_model = mc_model
            st.session_state.mc_simulations = mc_simulations
            st.session_state.mc_horizon = mc_horizon
            st.session_state.mc_confidence = mc_confidence
            st.session_state.mc_lookback = mc_lookback
            st.session_state.mc_shock_bps = mc_shock_bps
            st.session_state.mc_shock_day = mc_shock_day
            st.session_state.mc_shock_persistence = mc_shock_persistence
            st.session_state.ml_model_type = ml_model_type
            st.session_state.ml_lags = ml_lags
            st.session_state.bt_strategy = bt_strategy
            st.session_state.ohlc_period = ohlc_period
    
    # Retrieve data from session state
    yield_df = st.session_state.yield_df
    recession_series = st.session_state.recession_series
    vix_data = st.session_state.vix_data
    
    # Calculate derived data
    selected_cols = [c for c in yield_df.columns if c in MATURITY_MAP]
    maturities = np.array([MATURITY_MAP[c] for c in selected_cols])
    latest_curve = yield_df.iloc[-1][selected_cols].values
    spreads = compute_spreads(yield_df)
    recessions = identify_recessions(recession_series)
    inversions = calculate_inversion_periods(spreads)
    lead_times = calculate_lead_times(inversions, recessions)
    hit_stats = recession_hit_stats(inversions, recessions)
    regime, regime_text = classify_regime(spreads, yield_df)
    rec_prob = recession_probability(spreads)
    factors = factor_contributions(yield_df)
    forwards = calculate_forward_rates(yield_df)
    forecast_df = forecast_curve(yield_df[selected_cols], st.session_state.forecast_horizon)
    
    # Run models
    with st.spinner("Running models..."):
        ns_result = NelsonSiegel.fit(maturities, latest_curve)
        nss_result = NelsonSiegel.fit_nss(maturities, latest_curve)
        governance_df = model_governance(ns_result, nss_result)
        dynamic_params = rolling_ns_parameters(
            yield_df[selected_cols], maturities, selected_cols, st.session_state.rolling_years
        )
        pca_result = None
        scenarios = generate_scenarios(yield_df[selected_cols])
    
    # Display KPI row
    create_smart_kpi_row(yield_df, spreads, regime, regime_text, rec_prob, vix_data)
    
    # Inversion warning
    current_spread = safe_series_value(spreads["10Y-2Y"]) if "10Y-2Y" in spreads.columns else np.nan
    if not np.isnan(current_spread) and current_spread < 0:
        st.warning("⚠️ **YIELD CURVE IS INVERTED!** Historically signals recession within 6-18 months.")
    
    # Main tabs
    tabs = st.tabs([
        "📈 Yield Curve",
        "📊 Spread Analysis",
        "📐 Nelson-Siegel",
        "🎲 Monte Carlo",
        "🤖 Machine Learning",
        "🎯 Backtest",
        "⚡ Risk & Volatility",
        "🎭 Scenarios",
        "🛠 Technical Analysis",
        "💾 Export"
    ])
    
    # ========================================================================
    # TAB 1: Yield Curve
    # ========================================================================
    with tabs[0]:
        fig = chart_yield_curve(maturities, latest_curve, ns_result, nss_result, recessions)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig2 = chart_yield_history(yield_df, "2Y", COLORS["warning"], "2-Year Treasury Yield", recessions)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
        with col2:
            fig10 = chart_yield_history(yield_df, "10Y", COLORS["accent"], "10-Year Treasury Yield", recessions)
            if fig10:
                st.plotly_chart(fig10, use_container_width=True)
    
    # ========================================================================
    # TAB 2: Spread Analysis
    # ========================================================================
    with tabs[1]:
        fig = chart_spreads(spreads, recessions)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Current Spreads")
            if not spreads.empty:
                current_spreads = spreads.iloc[-1]
                for name, value in current_spreads.items():
                    if pd.notna(value):
                        st.metric(f"{name.upper()}", f"{value:.2f}%")
        
        with col2:
            st.subheader("Forward Rates")
            if not forwards.empty:
                st.line_chart(forwards)
            else:
                st.info("No forward rate data available")
    
    # ========================================================================
    # TAB 3: Nelson-Siegel Model
    # ========================================================================
    with tabs[2]:
        sub_tabs = st.tabs(["Parameters", "Governance", "Dynamic Analysis", "Factors"])
        
        with sub_tabs[0]:
            if ns_result:
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(pd.DataFrame({
                        "Parameter": ["β₀ (Level)", "β₁ (Slope)", "β₂ (Curvature)", "λ (Decay)"],
                        "Value": [f"{ns_result['params'][0]:.4f}", f"{ns_result['params'][1]:.4f}",
                                 f"{ns_result['params'][2]:.4f}", f"{ns_result['params'][3]:.4f}"]
                    }), hide_index=True, use_container_width=True)
                with col2:
                    st.metric("RMSE", f"{ns_result['rmse']*100:.2f} bps")
                    st.metric("R²", f"{ns_result['r2']:.4f}")
            
            if nss_result:
                st.markdown("### NSS Parameters")
                st.dataframe(pd.DataFrame({
                    "Parameter": ["β₀", "β₁", "β₂", "β₃", "λ₁", "λ₂"],
                    "Value": [f"{nss_result['params'][0]:.4f}", f"{nss_result['params'][1]:.4f}",
                             f"{nss_result['params'][2]:.4f}", f"{nss_result['params'][3]:.4f}",
                             f"{nss_result['params'][4]:.4f}", f"{nss_result['params'][5]:.4f}"]
                }), hide_index=True, use_container_width=True)
        
        with sub_tabs[1]:
            if not governance_df.empty:
                st.dataframe(governance_df.round(4), use_container_width=True, hide_index=True)
            fig_resid = chart_model_residuals(selected_cols, ns_result, nss_result)
            if fig_resid:
                st.plotly_chart(fig_resid, use_container_width=True)
        
        with sub_tabs[2]:
            fig_dyn = chart_dynamic_params(dynamic_params)
            if fig_dyn:
                st.plotly_chart(fig_dyn, use_container_width=True)
            else:
                st.info("Insufficient data for rolling parameter analysis")
        
        with sub_tabs[3]:
            col1, col2 = st.columns(2)
            with col1:
                fig_f = chart_factors(factors)
                if fig_f:
                    st.plotly_chart(fig_f, use_container_width=True)
            with col2:
                if pca_result:
                    fig_p = chart_pca(pca_result)
                    if fig_p:
                        st.plotly_chart(fig_p, use_container_width=True)
    
    # ========================================================================
    # TAB 4: Monte Carlo
    # ========================================================================
    with tabs[3]:
        st.subheader(f"Monte Carlo Simulation - {st.session_state.mc_model}")

        if "10Y" not in yield_df.columns or yield_df["10Y"].dropna().empty:
            st.warning("10Y yield series is required for Monte Carlo simulation.")
        else:
            y10_series = pd.to_numeric(yield_df["10Y"], errors="coerce").dropna()
            current_10y = safe_series_value(y10_series, 4.0)
            mc_preview = MonteCarlo.calibrate(y10_series, lookback=st.session_state.mc_lookback)

            top1, top2, top3, top4 = st.columns(4)
            top1.metric("Current 10Y Yield", f"{current_10y:.2f}%")
            top2.metric("Calibrated Mean", f"{mc_preview['theta']:.2f}%")
            top3.metric("Mean Reversion κ", f"{mc_preview['kappa']:.2f}")
            half_life = mc_preview.get("half_life_days", np.nan)
            top4.metric("Half-Life", f"{half_life:.1f} days" if np.isfinite(half_life) else "N/A")

            st.caption(
                f"Calibration uses the last {mc_preview['lookback_obs']} observations. "
                f"Pct-vol={mc_preview['sigma_pct']:.2%}, abs-vol={mc_preview['sigma_abs']:.2f}, "
                f"jump intensity={mc_preview['jump_lambda']:.2f} per year."
            )

            if st.button("Run Simulation", key="run_mc", use_container_width=True):
                with st.spinner(f"Running {st.session_state.mc_simulations:,} simulations..."):
                    mc_results = MonteCarlo.simulate(
                        y10_series,
                        model=st.session_state.mc_model,
                        days=st.session_state.mc_horizon,
                        sims=st.session_state.mc_simulations,
                        conf=st.session_state.mc_confidence,
                        lookback=st.session_state.mc_lookback,
                        shock_bps=st.session_state.mc_shock_bps,
                        shock_day=st.session_state.mc_shock_day,
                        shock_persistence=st.session_state.mc_shock_persistence,
                        seed=42,
                    )

                    sim_results = mc_results
                    terminal = sim_results["terminal"]
                    diag = sim_results["diagnostics"]
                    params = sim_results["params"]

                    row1 = st.columns(5)
                    row1[0].metric("Expected Terminal", f"{terminal['terminal_mean']:.2f}%")
                    row1[1].metric(f"{int(st.session_state.mc_confidence * 100)}% VaR", f"{terminal['var_level']:.2f}%")
                    row1[2].metric("CVaR / ES", f"{terminal['cvar_level']:.2f}%")
                    row1[3].metric("Prob. Yield Up", f"{terminal['prob_up']:.1%}")
                    row1[4].metric("Terminal Std", f"{terminal['terminal_std']:.2f}")

                    row2 = st.columns(5)
                    row2[0].metric("Prob. -50 bps", f"{terminal['prob_down_50bps']:.1%}")
                    row2[1].metric("Prob. +50 bps", f"{terminal['prob_up_50bps']:.1%}")
                    row2[2].metric("Terminal IQR", f"{diag['terminal_iqr']:.2f}")
                    row2[3].metric("Worst Path DD", f"{diag['worst_path_drawdown']:.2%}")
                    row2[4].metric("Shock Applied", f"{st.session_state.mc_shock_bps} bps")

                    fig = chart_monte_carlo(sim_results, params["initial"], st.session_state.mc_horizon, f"{sim_results['model_name']} Fan Chart")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    left, right = st.columns(2)
                    with left:
                        fig_dist = chart_monte_carlo_distribution(sim_results["paths"], params["initial"], "Terminal Yield Distribution")
                        if fig_dist:
                            st.plotly_chart(fig_dist, use_container_width=True)
                    with right:
                        st.markdown("### Simulation Diagnostics")
                        diag_df = pd.DataFrame({
                            "Metric": [
                                "Model", "Initial Yield", "Long-Run Mean", "Annual Drift",
                                "Annual Pct Vol", "Annual Abs Vol", "Jump Intensity",
                                "Jump Mean", "Jump Std", "Best Terminal", "Worst Terminal"
                            ],
                            "Value": [
                                sim_results["model_name"],
                                f"{params['initial']:.4f}%",
                                f"{params['theta']:.4f}%",
                                f"{params['mu']:.4f}",
                                f"{params['sigma_pct']:.4%}",
                                f"{params['sigma_abs']:.4f}",
                                f"{params['jump_lambda']:.4f}",
                                f"{params['jump_mean']:.4f}",
                                f"{params['jump_std']:.4f}",
                                f"{diag['best_terminal']:.4f}%",
                                f"{diag['worst_terminal']:.4f}%",
                            ],
                        })
                        st.dataframe(diag_df, use_container_width=True, hide_index=True)

                    terminal_df = pd.DataFrame({
                        "Percentile": ["5%", "25%", "50%", "75%", "95%"],
                        "Yield": [
                            f"{sim_results['p05'][-1]:.4f}%",
                            f"{sim_results['p25'][-1]:.4f}%",
                            f"{sim_results['median'][-1]:.4f}%",
                            f"{sim_results['p75'][-1]:.4f}%",
                            f"{sim_results['p95'][-1]:.4f}%",
                        ]
                    })
                    st.markdown("### Terminal Percentile Table")
                    st.dataframe(terminal_df, use_container_width=True, hide_index=True)


    # ========================================================================
    # TAB 5: Machine Learning
    # ========================================================================
    with tabs[4]:
        st.subheader(f"Machine Learning Forecast - {st.session_state.ml_model_type}")
        
        if st.button("Train Model", use_container_width=True):
            with st.spinner(f"Training {st.session_state.ml_model_type} model..."):
                X, y, _ = MLForecastModel.prepare_features(yield_df[selected_cols], lags=st.session_state.ml_lags)
                
                if X is not None and len(X) > 50:
                    ml_results = MLForecastModel.train_model(X, y, st.session_state.ml_model_type)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("RMSE", f"{ml_results['rmse']*100:.2f} bps")
                    col2.metric("MAE", f"{ml_results['mae']*100:.2f} bps")
                    col3.metric("R²", f"{ml_results['r2']:.3f}")
                    
                    if not ml_results['feature_importance'].empty:
                        st.subheader("Feature Importance")
                        st.dataframe(ml_results['feature_importance'], use_container_width=True)
                    
                    st.success(f"Model trained on {len(X)} samples")
                else:
                    st.warning(f"Insufficient data. Need >50 samples, have {len(X) if X is not None else 0}")
    
    # ========================================================================
    # TAB 6: Backtest
    # ========================================================================
    with tabs[5]:
        st.subheader(f"Backtest - {st.session_state.bt_strategy}")
        
        if st.button("Run Backtest", use_container_width=True):
            with st.spinner("Running backtest..."):
                result = Backtest.run(yield_df, spreads, st.session_state.bt_strategy)
                
                if result:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Return", f"{result['total_return']:.2%}")
                    col2.metric("Sharpe Ratio", f"{result['sharpe']:.2f}")
                    col3.metric("Max Drawdown", f"{result['max_drawdown']:.2%}")
                    col4.metric("Win Rate", f"{result['win_rate']:.2%}")
                    
                    fig = chart_backtest(result)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Backtest failed. Try different parameters.")
    
    # ========================================================================
    # TAB 7: Risk & Volatility
    # ========================================================================
    with tabs[6]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Volatility Analysis")
            if vix_data is not None and not vix_data.empty:
                vix_clean = vix_data.dropna()
                if not vix_clean.empty:
                    vol_regime = VolatilityAnalyzer.calculate_volatility_regime(vix_clean)
                    st.metric("Current VIX", f"{vol_regime['current_vix']:.2f}")
                    st.info(f"**Regime:** {vol_regime['regime']}")
                    st.caption(vol_regime['outlook'])
                    
                    fig_vix = chart_volatility(vix_clean, vol_regime)
                    if fig_vix:
                        st.plotly_chart(fig_vix, use_container_width=True)
                else:
                    st.info("VIX data is empty after cleaning")
            else:
                st.info("VIX data unavailable")
            
            st.subheader("Value at Risk (VaR)")
            if "10Y" in yield_df.columns:
                risk_metrics = calculate_var_metrics(yield_df["10Y"].pct_change(), st.session_state.confidence_level, 10)
                if risk_metrics:
                    st.metric("Historical VaR (10d)", f"{risk_metrics['VaR_Historical']:.4f}")
                    st.metric("Parametric VaR (10d)", f"{risk_metrics['VaR_Parametric']:.4f}")
                    st.metric("Cornish-Fisher VaR", f"{risk_metrics['VaR_CornishFisher']:.4f}")
                    st.caption(f"Skewness: {risk_metrics['Skewness']:.3f} | Kurtosis: {risk_metrics['Kurtosis']:.3f}")
        
        with col2:
            st.subheader("Correlation Analysis")
            if "10Y" in yield_df.columns and vix_data is not None and not vix_data.empty:
                vix_clean = vix_data.dropna()
                yield_clean = yield_df["10Y"].dropna()
                
                corr_value = safe_correlation(yield_clean, vix_clean)
                
                if corr_value is not None:
                    st.metric("10Y vs VIX Correlation", f"{corr_value:.3f}")
                    
                    if corr_value > 0.5:
                        st.caption("Strong positive correlation: Yields and VIX move together")
                    elif corr_value > 0.2:
                        st.caption("Weak positive correlation")
                    elif corr_value < -0.5:
                        st.caption("Strong negative correlation: Yields and VIX move opposite")
                    elif corr_value < -0.2:
                        st.caption("Weak negative correlation")
                    else:
                        st.caption("Near zero correlation: Little relationship")
                else:
                    st.info("Could not calculate correlation. Insufficient overlapping data.")
            else:
                st.info("Insufficient data for correlation analysis")
    
    # ========================================================================
    # TAB 8: Scenarios
    # ========================================================================
    with tabs[7]:
        if scenarios:
            scenario_name = st.selectbox("Select Scenario", list(scenarios.keys()))
            scenario_df = scenarios[scenario_name]
            
            fig = chart_scenario(scenario_df, scenario_name)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            current = yield_df.iloc[-1]
            impact_df = calculate_scenario_impact(current[selected_cols], scenario_df["Scenario"])
            st.dataframe(impact_df, use_container_width=True, hide_index=True)
            
            st.markdown(f'<div class="info-box">{get_scenario_interpretation(scenario_name)}</div>', unsafe_allow_html=True)
        else:
            st.info("No scenarios available")
    
    # ========================================================================
    # TAB 9: Technical Analysis (COMPLETELY REWRITTEN - VISIBLE FIGURES)
    # ========================================================================
    with tabs[8]:
        st.subheader("Technical Analysis Dashboard")
        st.markdown("""
        <div class="note-box">
        <b>📊 Technical Analysis Module</b><br><br>
        This dashboard provides professional technical analysis charts including:
        <b>Candlestick patterns</b>, <b>Moving Averages (SMA 20/50)</b>, <b>Bollinger Bands</b>,
        <b>RSI (Relative Strength Index)</b>, <b>MACD</b>, and <b>Volume analysis</b>.
        </div>
        """, unsafe_allow_html=True)
        
        if not YFINANCE_AVAILABLE:
            st.error("yfinance library is not available. Please install it with: pip install yfinance")
        else:
            # Asset selector
            tech_ticker = st.selectbox(
                "Select Asset for Technical Analysis",
                list(YAHOO_TICKERS.keys()),
                format_func=lambda x: f"{x} - {YAHOO_TICKERS[x]}",
                key="tech_ticker_main"
            )
            
            # Period selector
            tech_period = st.selectbox(
                "Select Time Period",
                ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                index=3,
                key="tech_period"
            )
            
            # Fetch fresh data for the selected ticker
            with st.spinner(f"Fetching {tech_ticker} data..."):
                tech_df = fetch_ohlc_data(tech_ticker, tech_period)
                
                if tech_df is not None and not tech_df.empty:
                    # Add technical indicators
                    tech_df = add_technical_indicators(tech_df)
                    
                    # Display current price info
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        current_price = tech_df["Close"].iloc[-1]
                        st.metric("Current Price", f"${current_price:.2f}")
                    with col2:
                        change_1d = tech_df["Close"].pct_change().iloc[-1] * 100
                        st.metric("1D Change", f"{change_1d:+.2f}%", delta=f"{change_1d:+.2f}%")
                    with col3:
                        if "SMA_20" in tech_df.columns:
                            sma20 = tech_df["SMA_20"].iloc[-1]
                            st.metric("SMA 20", f"${sma20:.2f}")
                    with col4:
                        if "SMA_50" in tech_df.columns:
                            sma50 = tech_df["SMA_50"].iloc[-1]
                            st.metric("SMA 50", f"${sma50:.2f}")
                    
                    # Generate technical signals
                    signals = get_technical_signals(tech_df)
                    
                    # Display signals in a nice format
                    st.markdown("### 📈 Current Technical Signals")
                    sig_col1, sig_col2, sig_col3 = st.columns(3)
                    with sig_col1:
                        signal_color = "🟢" if "Bullish" in signals.get("MACD", "") else "🔴" if "Bearish" in signals.get("MACD", "") else "⚪"
                        st.info(f"{signal_color} **MACD:** {signals.get('MACD', 'N/A')}")
                    with sig_col2:
                        rsi_val = signals.get("RSI", "N/A")
                        if "Oversold" in rsi_val:
                            rsi_icon = "🟢"
                        elif "Overbought" in rsi_val:
                            rsi_icon = "🔴"
                        else:
                            rsi_icon = "🟡"
                        st.info(f"{rsi_icon} **RSI:** {rsi_val}")
                    with sig_col3:
                        trend_icon = "🟢" if "Above" in signals.get("Trend", "") else "🔴"
                        st.info(f"{trend_icon} **Trend:** {signals.get('Trend', 'N/A')}")
                    
                    # Create and display the professional technical chart
                    st.markdown("### 📊 Technical Chart")
                    fig = plot_technical_chart(tech_df, tech_ticker)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Could not generate technical chart. Please try a different asset or period.")
                    
                    # Add download button for the data
                    csv_data = tech_df[["Open", "High", "Low", "Close", "Volume"]].tail(100).to_csv().encode("utf-8")
                    st.download_button(
                        "📥 Download Technical Data (CSV)",
                        csv_data,
                        f"{tech_ticker}_technical_data.csv",
                        "text/csv"
                    )
                    
                else:
                    st.error(f"Could not fetch data for {tech_ticker}. Please try another symbol or check your internet connection.")
    
    # ========================================================================
    # TAB 10: Export
    # ========================================================================
    with tabs[9]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                "📥 Download Yield Data (CSV)",
                yield_df.to_csv().encode("utf-8"),
                f"yield_data_{datetime.now():%Y%m%d}.csv",
                "text/csv"
            )
            
            st.download_button(
                "📥 Download Spreads (CSV)",
                spreads.to_csv().encode("utf-8"),
                f"spreads_{datetime.now():%Y%m%d}.csv",
                "text/csv"
            )
            
            if not forwards.empty:
                st.download_button(
                    "📥 Download Forward Rates (CSV)",
                    forwards.to_csv().encode("utf-8"),
                    f"forward_rates_{datetime.now():%Y%m%d}.csv",
                    "text/csv"
                )
        
        with col2:
            if not forecast_df.empty:
                st.download_button(
                    "📥 Download Forecast (CSV)",
                    forecast_df.to_csv().encode("utf-8"),
                    f"forecast_{datetime.now():%Y%m%d}.csv",
                    "text/csv"
                )
            
            if lead_times:
                st.download_button(
                    "📥 Download Lead Times (CSV)",
                    pd.DataFrame(lead_times).to_csv(index=False).encode("utf-8"),
                    f"lead_times_{datetime.now():%Y%m%d}.csv",
                    "text/csv"
                )
    
    # Footer
    render_footer()


if __name__ == "__main__":
    main()