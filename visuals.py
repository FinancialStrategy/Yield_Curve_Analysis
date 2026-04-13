"""
Visualization Module - All charting functions
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
from config import COLORS


def add_recession_bands(fig: go.Figure, recessions: List[dict]) -> go.Figure:
    """Add recession shading to a plotly figure"""
    for rec in recessions:
        fig.add_vrect(
            x0=rec["start"], x1=rec["end"],
            fillcolor=COLORS["recession"], opacity=0.35,
            layer="below", line_width=0
        )
    return fig


def chart_layout(fig: go.Figure, title: str, y_title: str = None, height: int = 460, x_title: str = "Date") -> go.Figure:
    """Apply consistent layout styling to a plotly figure"""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(size=12, color=COLORS["text"]),
        title=dict(text=title, x=0.01, font=dict(size=16, color=COLORS["text"])),
        margin=dict(l=60, r=30, t=80, b=50),
        hovermode="x unified",
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title=x_title, gridcolor=COLORS["grid"], showgrid=True)
    if y_title:
        fig.update_yaxes(title=y_title, gridcolor=COLORS["grid"], showgrid=True)
    return fig


def chart_yield_curve(maturities: np.ndarray, yields: np.ndarray, ns_result: dict = None, nss_result: dict = None, recessions: List[dict] = None) -> go.Figure:
    """Plot yield curve with optional model fits"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=maturities, y=yields, mode='lines+markers', name='Current Curve',
        line=dict(color=COLORS["accent"], width=2.5), marker=dict(size=10, color=COLORS["accent"])
    ))
    
    if ns_result:
        fig.add_trace(go.Scatter(
            x=maturities, y=ns_result["fitted"], mode='lines', name='NS Fit',
            line=dict(color=COLORS["positive"], width=2.0)
        ))
    
    if nss_result:
        fig.add_trace(go.Scatter(
            x=maturities, y=nss_result["fitted"], mode='lines', name='NSS Fit',
            line=dict(color=COLORS["accent3"], width=2.0, dash="dash")
        ))
    
    if recessions:
        add_recession_bands(fig, recessions)
    
    return chart_layout(fig, "U.S. Treasury Yield Curve", "Yield (%)", 460, "Maturity (Years)")


def chart_yield_history(yield_df: pd.DataFrame, tenor: str, color: str, title: str, recessions: List[dict]) -> Optional[go.Figure]:
    """Plot historical yield with recession bands"""
    if tenor not in yield_df.columns:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yield_df.index, y=yield_df[tenor], mode='lines', name=tenor,
        line=dict(color=color, width=2.2)
    ))
    add_recession_bands(fig, recessions)
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))
    return chart_layout(fig, title, "Yield (%)", 440)


def chart_spreads(spreads: pd.DataFrame, recessions: List[dict]) -> go.Figure:
    """Plot yield spreads with recession bands"""
    fig = go.Figure()
    palette = [COLORS["negative"], COLORS["accent"], COLORS["warning"], COLORS["positive"]]
    
    for i, col in enumerate(spreads.columns):
        fig.add_trace(go.Scatter(
            x=spreads.index, y=spreads[col], mode='lines', name=col,
            line=dict(color=palette[i % len(palette)], width=2)
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"])
    add_recession_bands(fig, recessions)
    return chart_layout(fig, "Treasury Yield Spreads", "Basis Points", 520)


def chart_model_residuals(selected_cols: List[str], ns_result: dict = None, nss_result: dict = None) -> Optional[go.Figure]:
    """Plot model residuals with warning flags"""
    if ns_result is None and nss_result is None:
        return None
    
    fig = go.Figure()
    
    if ns_result:
        fig.add_trace(go.Bar(
            x=selected_cols, y=ns_result["residuals"], name="NS Residuals",
            marker_color=COLORS["positive"], opacity=0.65
        ))
    
    if nss_result:
        fig.add_trace(go.Bar(
            x=selected_cols, y=nss_result["residuals"], name="NSS Residuals",
            marker_color=COLORS["warning"], opacity=0.65
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"])
    fig.add_hline(y=0.15, line_dash="dot", line_color=COLORS["negative"], line_width=1)
    fig.add_hline(y=-0.15, line_dash="dot", line_color=COLORS["negative"], line_width=1)
    
    return chart_layout(fig, "Model Residual Quality", "Residual (Yield %)", 420, "Tenor")


def chart_dynamic_params(dynamic_params: pd.DataFrame) -> Optional[go.Figure]:
    """Plot rolling Nelson-Siegel parameters"""
    if dynamic_params is None or dynamic_params.empty:
        return None
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=("β₀ (Level)", "β₁ (Slope)", "β₂ (Curvature)", "RMSE"))
    
    fig.add_trace(go.Scatter(x=dynamic_params["date"], y=dynamic_params["beta0"], mode="lines", name="β₀", line=dict(color=COLORS["positive"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=dynamic_params["date"], y=dynamic_params["beta1"], mode="lines", name="β₁", line=dict(color=COLORS["accent"])), row=1, col=2)
    fig.add_trace(go.Scatter(x=dynamic_params["date"], y=dynamic_params["beta2"], mode="lines", name="β₂", line=dict(color=COLORS["warning"])), row=2, col=1)
    fig.add_trace(go.Scatter(x=dynamic_params["date"], y=dynamic_params["rmse"], mode="lines", name="RMSE", line=dict(color=COLORS["muted"])), row=2, col=2)
    
    return chart_layout(fig, "Rolling Nelson-Siegel Parameters", height=620)


def chart_factors(factor_df: pd.DataFrame) -> Optional[go.Figure]:
    """Plot factor contributions (Level, Slope, Curvature)"""
    if factor_df is None or factor_df.empty:
        return None
    
    fig = go.Figure()
    palette = [COLORS["accent"], COLORS["warning"], COLORS["positive"], COLORS["accent2"]]
    
    for i, col in enumerate(factor_df.columns):
        fig.add_trace(go.Scatter(
            x=factor_df.index, y=factor_df[col], mode="lines", name=col,
            line=dict(color=palette[i % len(palette)], width=1.8)
        ))
    
    return chart_layout(fig, "Factor Contributions", "Value", 430)


def chart_pca(pca_risk: dict) -> Optional[go.Figure]:
    """Plot PCA explained variance"""
    if not pca_risk:
        return None
    
    fig = go.Figure()
    ev = pca_risk["explained_variance"] * 100
    fig.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(ev))], y=ev, marker_color=COLORS["accent"]))
    fig.add_trace(go.Scatter(
        x=[f"PC{i+1}" for i in range(len(ev))], y=np.cumsum(ev),
        mode='lines+markers', name='Cumulative', line=dict(color=COLORS["warning"], width=2), yaxis='y2'
    ))
    fig.update_layout(yaxis2=dict(title="Cumulative %", overlaying='y', side='right'))
    return chart_layout(fig, "PCA Explained Variance", "Percent", 430)


def chart_rate_dynamics(yield_df: pd.DataFrame, spreads: pd.DataFrame) -> Optional[go.Figure]:
    """Plot rate direction metrics"""
    if not {"2Y", "10Y"}.issubset(yield_df.columns):
        return None
    
    df = pd.DataFrame(index=yield_df.index)
    df["2Y_20D_Change"] = yield_df["2Y"].diff(20)
    df["10Y_20D_Change"] = yield_df["10Y"].diff(20)
    
    if "10Y-2Y" in spreads.columns:
        df["Slope_Momentum"] = spreads["10Y-2Y"].diff(20)
    
    df["Realized_Vol_60D"] = yield_df["10Y"].diff().rolling(60).std()
    df = df.dropna()
    
    if df.empty:
        return None
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=("2Y 20D Change", "10Y 20D Change", "Slope Momentum", "Realized Volatility"))
    
    fig.add_trace(go.Scatter(x=df.index, y=df["2Y_20D_Change"], mode="lines", name="2Y", line=dict(color=COLORS["warning"], width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["10Y_20D_Change"], mode="lines", name="10Y", line=dict(color=COLORS["accent"], width=1.8)), row=1, col=2)
    
    if "Slope_Momentum" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["Slope_Momentum"], mode="lines", name="Slope", line=dict(color=COLORS["negative"], width=1.8)), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df["Realized_Vol_60D"], mode="lines", name="Vol", line=dict(color=COLORS["positive"], width=1.8)), row=2, col=2)
    
    return chart_layout(fig, "Rate Direction Dynamics", height=640)


def chart_monte_carlo(results: dict, initial: float, days: int, title: str = "Monte Carlo Simulation") -> go.Figure:
    """Plot Monte Carlo simulation results with confidence bands"""
    fig = go.Figure()
    x_axis = np.arange(days)
    
    fig.add_trace(go.Scatter(
        x=x_axis, y=results["upper"], fill=None, mode='lines',
        line=dict(color='rgba(0,0,0,0)'), showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=x_axis, y=results["lower"], fill='tonexty', mode='lines',
        fillcolor='rgba(44, 95, 138, 0.20)', line=dict(color='rgba(0,0,0,0)'),
        name='95% Confidence Interval'
    ))
    
    fig.add_trace(go.Scatter(
        x=x_axis, y=results["mean"], mode='lines', name='Mean Path',
        line=dict(color=COLORS["accent"], width=2.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0], y=[initial], mode='markers', name='Current',
        marker=dict(size=12, color=COLORS["positive"], symbol='star')
    ))
    
    return chart_layout(fig, title, "Yield (%)", 500, "Trading Days")


def chart_backtest(results: dict) -> go.Figure:
    """Plot backtest cumulative returns"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results["cumulative"].index, y=results["cumulative"].values,
        mode='lines', name='Strategy', line=dict(color=COLORS["accent"], width=2.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=results["benchmark"].index, y=results["benchmark"].values,
        mode='lines', name='Buy & Hold', line=dict(color=COLORS["muted"], width=2, dash='dash')
    ))
    
    return chart_layout(fig, "Backtest Performance", "Cumulative Return", 500)


def chart_scenario(scenario_df: pd.DataFrame, scenario_name: str) -> go.Figure:
    """Plot scenario comparison bar chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=scenario_df.index, y=scenario_df["Current"],
        name="Current", marker_color=COLORS["accent2"]
    ))
    
    fig.add_trace(go.Bar(
        x=scenario_df.index, y=scenario_df["Scenario"],
        name=scenario_name, marker_color=COLORS["accent3"]
    ))
    
    fig.update_layout(barmode="group")
    return chart_layout(fig, f"Scenario Analysis: {scenario_name}", "Yield (%)", 460, "Tenor")


def chart_correlation(matrix: pd.DataFrame) -> go.Figure:
    """Plot correlation heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=matrix.columns,
        y=matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
    ))
    fig.update_layout(title="Correlation Matrix", height=550)
    return fig


def chart_technical(df: pd.DataFrame, ticker: str) -> Optional[go.Figure]:
    """Plot technical analysis panels (Price, RSI, MACD)"""
    if df.empty:
        return None
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f"{ticker} Price", "RSI (14)", "MACD")
    )
    
    # Price panel
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], mode="lines", name="Close",
        line=dict(color=COLORS["accent"], width=2)
    ), row=1, col=1)
    
    if "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"], mode="lines", name="BB Upper",
            line=dict(color=COLORS["muted"], width=1)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"], mode="lines", name="BB Lower",
            line=dict(color=COLORS["muted"], width=1), fill='tonexty', fillcolor=COLORS["band"]
        ), row=1, col=1)
    
    # RSI panel
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"], mode="lines", name="RSI",
            line=dict(color=COLORS["accent2"], width=1.5)
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=COLORS["negative"], row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=COLORS["positive"], row=2, col=1)
    
    # MACD panel
    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], mode="lines", name="MACD",
            line=dict(color=COLORS["positive"], width=1.5)
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_Signal"], mode="lines", name="Signal",
            line=dict(color=COLORS["negative"], width=1.5)
        ), row=3, col=1)
        
        colors = ['red' if x < 0 else 'green' for x in df["MACD_Hist"]]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_Hist"], name="Histogram", marker_color=colors
        ), row=3, col=1)
    
    return chart_layout(fig, f"Technical Analysis - {ticker}", height=760)


def chart_ohlc(df: pd.DataFrame, ticker: str) -> Optional[go.Figure]:
    """Plot candlestick chart with moving averages"""
    if df.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing=dict(line=dict(color=COLORS["positive"]), fillcolor=COLORS["positive"]),
        decreasing=dict(line=dict(color=COLORS["negative"]), fillcolor=COLORS["negative"]),
        name=ticker,
    ))
    
    if "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_20"], mode="lines", name="SMA 20",
            line=dict(color=COLORS["accent"], width=1.2)
        ))
    
    if "SMA_50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_50"], mode="lines", name="SMA 50",
            line=dict(color=COLORS["warning"], width=1.2)
        ))
    
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))
    return chart_layout(fig, f"OHLC - {ticker}", "Price", 520)


def chart_volatility(vix_data: pd.Series, vol_regime: dict) -> go.Figure:
    """Plot VIX dashboard with volatility regime"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15, row_heights=[0.6, 0.4])
    
    fig.add_trace(go.Scatter(
        x=vix_data.index, y=vix_data.values, mode='lines', name='VIX',
        line=dict(color=COLORS["warning"], width=2)
    ), row=1, col=1)
    
    fig.add_hline(y=20, line_dash="dash", line_color="orange", row=1, col=1)
    fig.add_hline(y=15, line_dash="dash", line_color="green", row=1, col=1)
    
    # Vol of vol
    from volatility import VolatilityAnalyzer
    vol_of_vol = VolatilityAnalyzer.calculate_vol_of_vol(vix_data)
    
    if len(vol_of_vol) > 0:
        fig.add_trace(go.Scatter(
            x=vol_of_vol.index, y=vol_of_vol.values, mode='lines', name='Vol of Vol',
            line=dict(color=COLORS["accent"], width=2)
        ), row=2, col=1)
    
    return chart_layout(fig, f"VIX Dashboard | Current: {vol_regime['current_vix']:.2f}", height=600)


def chart_forecast(forecast_df: pd.DataFrame) -> go.Figure:
    """Plot yield curve forecast"""
    fig = go.Figure()
    
    for col in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df[col], mode='lines',
            name=col, line=dict(width=2)
        ))
    
    return chart_layout(fig, "Yield Curve Forecast", "Yield (%)", 500)