
**Additional Parameters:**
- **b3:** Second curvature factor
- **lambda2:** Second decay factor

**Advantages:**
- Better fit for complex curve shapes
- Captures multiple humps
- Improved long-end fitting
- Handles unusual curve shapes better

**Disadvantages:**
- More parameters (6 vs 4)
- Higher risk of overfitting
- More computationally intensive
- Less stable estimates

**Model Comparison Matrix**

| Feature | NS | NSS |
|---------|----|----|
| Parameters | 4 | 6 |
| Flexibility | Moderate | High |
| Overfitting Risk | Low | Moderate |
| Computational Cost | Low | High |
| Best For | Simple curves | Complex curves |
| Stability | High | Moderate |
| Interpretation | Clear | Complex |
""")

# Factor loadings visualization
st.markdown("#### Factor Loadings Visualization")
st.markdown("*Shows how each factor contributes to yields at different maturities*")

tau_range = np.linspace(0.01, 30, 100)
lambda_val = 0.0609

factor1 = np.ones_like(tau_range)
factor2 = (1 - np.exp(-lambda_val * tau_range)) / (lambda_val * tau_range)
factor3 = factor2 - np.exp(-lambda_val * tau_range)

fig_factors = go.Figure()
fig_factors.add_trace(go.Scatter(x=tau_range, y=factor1, mode='lines', name='Level Factor (b0)', line=dict(color=COLORS['positive'], width=2)))
fig_factors.add_trace(go.Scatter(x=tau_range, y=factor2, mode='lines', name='Slope Factor (b1)', line=dict(color=COLORS['accent'], width=2)))
fig_factors.add_trace(go.Scatter(x=tau_range, y=factor3, mode='lines', name='Curvature Factor (b2)', line=dict(color=COLORS['warning'], width=2)))

fig_factors = create_institutional_layout(fig_factors, "NELSON-SIEGEL FACTOR LOADINGS", "Factor Loading", height=450)
st.plotly_chart(fig_factors, use_container_width=True)

st.markdown("""
**Interpretation Guide:**
- **Level Factor (b0):** Constant across all maturities - represents parallel shifts
- **Slope Factor (b1):** Starts at 1 and decays to 0 - influences short-term rates
- **Curvature Factor (b2):** Starts at 0, rises, then decays - affects medium-term rates
""")

# ===== TAB 2: NS MODEL FIT =====
with tabs[1]:
st.markdown("### Nelson-Siegel Model Calibration")

if ns_result:
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Model Parameters")
    param_df = pd.DataFrame({
        'Parameter': ['b0 (Level)', 'b1 (Slope)', 'b2 (Curvature)', 'lambda (Decay)'],
        'Value': [f"{ns_result['params'][0]:.4f}", f"{ns_result['params'][1]:.4f}", 
                  f"{ns_result['params'][2]:.4f}", f"{ns_result['params'][3]:.4f}"],
        'Interpretation': [
            f"Long-term level: {ns_result['params'][0]:.2f}%",
            f"Slope: {'Inverted' if ns_result['params'][1] < 0 else 'Normal'}",
            f"Curvature: {'Humped' if ns_result['params'][2] > 0 else 'Sagged'}",
            f"Max curvature at {1/ns_result['params'][3]:.1f} years"
        ]
    })
    st.dataframe(param_df, use_container_width=True, hide_index=True)
    
    st.markdown("#### Fit Statistics")
    st.markdown(f"- **RMSE (Root Mean Square Error):** {ns_result['rmse']:.4f}")
    st.markdown(f"- **MAE (Mean Absolute Error):** {ns_result['mae']:.4f}")
    st.markdown(f"- **R² (Coefficient of Determination):** {ns_result['r_squared']:.4f}")
    st.markdown(f"- **Number of Observations:** {len(maturities)}")
    
    # Factor interpretation
    ns_interpretation = NelsonSiegelModel.calculate_factor_interpretation(ns_result['params'], 'NS')
    st.markdown("#### Factor Interpretation")
    for key, value in ns_interpretation['Interpretation'].items():
        st.markdown(f"- **{key}:** {value}")

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
        text=[f"{r:.3f}" for r in residuals],
        textposition='outside'
    ))
    fig_resid.add_hline(y=0, line_dash="dash", line_color=COLORS['negative'])
    fig_resid = create_institutional_layout(fig_resid, "FITTING RESIDUALS", "Residual (bps)", height=350)
    st.plotly_chart(fig_resid, use_container_width=True)

with col_res2:
    st.markdown("**Residual Statistics**")
    st.markdown(f"- **Mean Residual:** {np.mean(residuals):.4f} bps")
    st.markdown(f"- **Std Deviation:** {np.std(residuals):.4f} bps")
    st.markdown(f"- **Max Positive:** {np.max(residuals):.4f} bps")
    st.markdown(f"- **Max Negative:** {np.min(residuals):.4f} bps")
    st.markdown(f"- **95% Confidence Band:** ±{1.96 * np.std(residuals):.4f} bps")
    
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

# ===== TAB 3: NSS MODEL FIT =====
with tabs[2]:
st.markdown("### Nelson-Siegel-Svensson Model Calibration")

if nss_result:
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Model Parameters")
    param_df = pd.DataFrame({
        'Parameter': ['b0 (Level)', 'b1 (Slope)', 'b2 (Curvature 1)', 'b3 (Curvature 2)', 'lambda1', 'lambda2'],
        'Value': [f"{nss_result['params'][0]:.4f}", f"{nss_result['params'][1]:.4f}",
                  f"{nss_result['params'][2]:.4f}", f"{nss_result['params'][3]:.4f}",
                  f"{nss_result['params'][4]:.4f}", f"{nss_result['params'][5]:.4f}"],
        'Interpretation': [
            f"Long-term level: {nss_result['params'][0]:.2f}%",
            f"Slope: {'Inverted' if nss_result['params'][1] < 0 else 'Normal'}",
            f"First hump at {1/nss_result['params'][4]:.1f}Y" if nss_result['params'][4] > 0 else "N/A",
            f"Second hump at {1/nss_result['params'][5]:.1f}Y" if nss_result['params'][5] > 0 else "N/A",
            f"Decay rate 1: {nss_result['params'][4]:.4f}",
            f"Decay rate 2: {nss_result['params'][5]:.4f}"
        ]
    })
    st.dataframe(param_df, use_container_width=True, hide_index=True)
    
    st.markdown("#### Fit Statistics")
    st.markdown(f"- **RMSE:** {nss_result['rmse']:.4f}")
    st.markdown(f"- **MAE:** {nss_result['mae']:.4f}")
    st.markdown(f"- **R²:** {nss_result['r_squared']:.4f}")
    st.markdown(f"- **Number of Observations:** {len(maturities)}")
    
    # Svensson interpretation
    nss_interpretation = NelsonSiegelModel.calculate_factor_interpretation(nss_result['params'], 'NSS')
    st.markdown("#### Svensson Factor Interpretation")
    for key, value in nss_interpretation['Interpretation'].items():
        st.markdown(f"- **{key}:** {value}")

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
        text=[f"{r:.3f}" for r in residuals_nss],
        textposition='outside'
    ))
    fig_resid_nss.add_hline(y=0, line_dash="dash", line_color=COLORS['negative'])
    fig_resid_nss = create_institutional_layout(fig_resid_nss, "FITTING RESIDUALS - SVENSSON", "Residual (bps)", height=350)
    st.plotly_chart(fig_resid_nss, use_container_width=True)

with col_res2:
    st.markdown("**Residual Statistics**")
    st.markdown(f"- **Mean Residual:** {np.mean(residuals_nss):.4f} bps")
    st.markdown(f"- **Std Deviation:** {np.std(residuals_nss):.4f} bps")
    st.markdown(f"- **Max Positive:** {np.max(residuals_nss):.4f} bps")
    st.markdown(f"- **Max Negative:** {np.min(residuals_nss):.4f} bps")
    st.markdown(f"- **95% Confidence Band:** ±{1.96 * np.std(residuals_nss):.4f} bps")

# ===== TAB 4: MODEL COMPARISON =====
with tabs[3]:
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
        f"{ns_result['rmse']:.4f}" if ns_result else "N/A",
        f"{ns_result['mae']:.4f}" if ns_result else "N/A",
        f"{ns_result['r_squared']:.4f}" if ns_result else "N/A",
        f"{np.max(np.abs(yield_values - ns_result['fitted_values'])):.4f}" if ns_result else "N/A"
    ],
    'NSS': [
        f"{nss_result['rmse']:.4f}" if nss_result else "N/A",
        f"{nss_result['mae']:.4f}" if nss_result else "N/A",
        f"{nss_result['r_squared']:.4f}" if nss_result else "N/A",
        f"{np.max(np.abs(yield_values - nss_result['fitted_values'])):.4f}" if nss_result else "N/A"
    ]
})
st.dataframe(error_df, use_container_width=True, hide_index=True)

if ns_result and nss_result:
    improvement_rmse = (ns_result['rmse'] - nss_result['rmse']) / ns_result['rmse'] * 100
    improvement_mae = (ns_result['mae'] - nss_result['mae']) / ns_result['mae'] * 100
    improvement_r2 = (nss_result['r_squared'] - ns_result['r_squared']) / ns_result['r_squared'] * 100 if ns_result['r_squared'] > 0 else 0
    
    st.markdown("#### NSS Improvement Metrics")
    st.markdown(f"- **RMSE Improvement:** {improvement_rmse:+.2f}%")
    st.markdown(f"- **MAE Improvement:** {improvement_mae:+.2f}%")
    st.markdown(f"- **R² Improvement:** {improvement_r2:+.2f}%")
    
    st.markdown("#### Model Recommendation")
    if improvement_rmse > 10:
        st.success("NSS provides significantly better fit - Recommended for complex curves and longer maturities")
    elif improvement_rmse > 5:
        st.info("NSS provides moderate improvement - Consider for specific use cases")
    else:
        st.warning("NS may be sufficient - NSS improvement is marginal, prefer simpler model")

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

# ===== TAB 5: DYNAMIC ANALYSIS =====
with tabs[4]:
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
    st.metric("Level Trend", f"{recent_trend['beta0']:+.1f}%", delta_color="normal")
with col_trend2:
    st.metric("Slope Trend", f"{recent_trend['beta1']:+.1f}%", delta_color="inverse")
with col_trend3:
    st.metric("Curvature Trend", f"{recent_trend['beta2']:+.1f}%", delta_color="normal")

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

# ===== TAB 6: FACTOR ANALYSIS =====
with tabs[5]:
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
        
        level_text = f"{level_val:.2f}%"
status_text = f"{level_status} long-term rate environment"

st.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-label">LEVEL FACTOR</div>
        <div class="metric-value">{level_text}</div>
        <div class="metric-label">{status_text}</div>
    </div>
    """, 
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
            f"""
            <div class="metric-card">
                <div class="metric-label">SLOPE FACTOR</div>
                <div class="metric-value" style="color: {slope_color};">{slope_val:.1f} bps</div>
                <div class="metric-label">{slope_status} curve shape</div>
            </div>
            """, 
            unsafe_allow_html=True
        )

with col_int3:
    if 'Curvature' in current_factors:
        curvature_val = current_factors['Curvature']
        curvature_status = "Humped" if curvature_val > 0 else "Sagged"
        
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">CURVATURE FACTOR</div>
                <div class="metric-value">{curvature_val:.1f} bps</div>
                <div class="metric-label">{curvature_status} medium-term expectations</div>
            </div>
            """, 
            unsafe_allow_html=True
        )

# PCA analysis
st.markdown("#### Principal Component Analysis (PCA)")
if pca_risk is not None:
fig_pca_variance = go.Figure(data=go.Bar(
    x=[f'PC{i+1}' for i in range(len(pca_risk['explained_variance']))],
    y=pca_risk['explained_variance'] * 100,
    marker_color=COLORS['accent'],
    text=[f"{v*100:.1f}%" for v in pca_risk['explained_variance']],
    textposition='outside'
))
fig_pca_variance = create_institutional_layout(fig_pca_variance, "PCA VARIANCE EXPLANATION", "Variance Explained (%)", height=400)
st.plotly_chart(fig_pca_variance, use_container_width=True)

# Cumulative variance
cum_var = np.cumsum(pca_risk['explained_variance']) * 100
st.markdown(f"""
**Cumulative Variance Explained:**
- PC1 + PC2: {cum_var[1]:.1f}% of yield curve variation
- PC1 + PC2 + PC3: {cum_var[2]:.1f}% of yield curve variation
""")

# ===== TAB 7: RISK METRICS =====
with tabs[6]:
st.markdown("### Advanced Risk Metrics")

col1, col2 = st.columns(2)

with col1:
st.markdown("#### Value at Risk (VaR) Analysis")

if '10Y' in yield_df.columns:
    returns = yield_df['10Y'].pct_change().dropna()
    var_metrics = AdvancedRiskMetrics.calculate_var_metrics(returns)
    
    st.markdown(f"""
    **10Y Treasury Yield Risk (95% confidence, 10-day horizon)**
    
    | Metric | Value | Interpretation |
    |--------|-------|----------------|
    | Historical VaR | {var_metrics['VaR_Historical']:.4f} | Worst expected loss |
    | Parametric VaR | {var_metrics['VaR_Parametric']:.4f} | Normal distribution assumption |
    | Cornish-Fisher VaR | {var_metrics['VaR_CornishFisher']:.4f} | Adjusted for skewness/kurtosis |
    | CVaR (Expected Shortfall) | {var_metrics['CVaR']:.4f} | Average loss beyond VaR |
    | Tail Ratio | {var_metrics['tail_ratio']:.2f} | CVaR/VaR ratio |
    | Skewness | {var_metrics['skewness']:.3f} | Distribution asymmetry |
    | Excess Kurtosis | {var_metrics['kurtosis']:.3f} | Tail thickness |
    """)
    
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
        annotation_text=f"VaR: {var_metrics['VaR_Historical']*100:.2f}%"
    )
    fig_returns = create_institutional_layout(fig_returns, "RETURN DISTRIBUTION & VAR", "Frequency", height=400)
    st.plotly_chart(fig_returns, use_container_width=True)

with col2:
st.markdown("#### PCA Risk Decomposition")

if pca_risk is not None:
    fig_pca_risk = go.Figure(data=go.Heatmap(
        z=pca_risk['loadings'].values,
        x=pca_risk['loadings'].columns,
        y=pca_risk['loadings'].index,
        colorscale='RdBu',
        zmid=0,
        text=pca_risk['loadings'].round(3).values,
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    fig_pca_risk = create_institutional_layout(fig_pca_risk, "PCA FACTOR LOADINGS", height=500)
    st.plotly_chart(fig_pca_risk, use_container_width=True)

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

st.markdown(
f"""
<div class="metric-card" style="border-left: 3px solid {risk_color};">
    <div class="metric-label">OVERALL RISK ASSESSMENT</div>
    <div class="metric-value" style="color: {risk_color};">{risk_level} RISK</div>
    <div class="metric-label">
        <strong>Risk Score:</strong> {risk_score}/100<br>
        <strong>Contributing Factors:</strong> {', '.join(risk_factors)}<br>
        <strong>Curve Status:</strong> {'Inverted' if current_slope_val < 0 else 'Flattening' if current_slope_val < 50 else 'Normal'}<br>
        <strong>10Y Volatility:</strong> {volatility_val:.2%}<br>
        <strong>Risk Horizon:</strong> 10-day VaR at 95% confidence
    </div>
</div>
""", 
unsafe_allow_html=True
)

# Stress scenarios
st.markdown("#### Stress Testing Scenarios")

scenarios = AdvancedRiskMetrics.calculate_stress_scenarios(yield_df[['10Y']].dropna())

scenario_df = pd.DataFrame({
'Scenario': list(scenarios.keys()),
'Impact on 10Y': [f"{(s - yield_df['10Y'].iloc[-1]) * 100:+.1f} bps" for s in scenarios.values()]
})
st.dataframe(scenario_df, use_container_width=True, hide_index=True)

# ===== TAB 8: NBER RECESSION =====
with tabs[7]:
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
st.metric("Total Inversion Days", f"{recession_metrics.get('total_inversion_days', 0):,}")
with col_met2:
st.metric("Avg Inversion Depth", f"{recession_metrics.get('avg_inversion_depth', 0):.1f} bps")
with col_met3:
st.metric("Avg Lead Time", f"{recession_metrics.get('avg_lead_time', 0):.0f} days")
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
    annotation_text=f"Avg: {recession_metrics['avg_lead_time']:.0f} days"
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

# ===== TAB 9: FORECASTING =====
with tabs[8]:
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

# ===== TAB 10: DATA EXPORT =====
with tabs[9]:
st.markdown("### Data Export")

col1, col2 = st.columns(2)

with col1:
csv_yields = yield_df.to_csv().encode('utf-8')
st.download_button(
    "Download Yield Data (CSV)",
    csv_yields,
    f"yield_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    "text/csv"
)

if ns_result:
    ns_params_df = pd.DataFrame([ns_result['params']], columns=['b0', 'b1', 'b2', 'lambda'])
    csv_ns = ns_params_df.to_csv().encode('utf-8')
    st.download_button(
        "Download NS Parameters",
        csv_ns,
        f"ns_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv"
    )

csv_spreads = spreads.to_csv().encode('utf-8')
st.download_button(
    "Download Spread Data",
    csv_spreads,
    f"spreads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    "text/csv"
)

with col2:
if not dynamic_params.empty:
    csv_dynamic = dynamic_params.to_csv().encode('utf-8')
    st.download_button(
        "Download Dynamic Parameters",
        csv_dynamic,
        f"dynamic_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv"
    )

csv_factors = factors.to_csv().encode('utf-8')
st.download_button(
    "Download Factor Data",
    csv_factors,
    f"factors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    "text/csv"
)

if forecast_result:
    forecast_export = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': forecast_result['forecast'][:, 0] if len(forecast_result['forecast'].shape) > 1 else forecast_result['forecast']
    })
    csv_forecast = forecast_export.to_csv().encode('utf-8')
    st.download_button(
        "Download Forecast Data",
        csv_forecast,
        f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv"
    )

# Data summary
st.markdown("### Data Summary")
st.markdown(f"- **Yield Curves:** {len(yield_df.columns)} maturities ({', '.join(yield_df.columns)})")
st.markdown(f"- **Observations:** {len(yield_df):,}")
st.markdown(f"- **Date Range:** {yield_df.index[0].strftime('%Y-%m-%d')} to {yield_df.index[-1].strftime('%Y-%m-%d')}")

if ns_result:
st.markdown(f"- **NS RMSE:** {ns_result['rmse']:.4f}")
else:
st.markdown("- **NS RMSE:** N/A")

if nss_result:
st.markdown(f"- **NSS RMSE:** {nss_result['rmse']:.4f}")
else:
st.markdown("- **NSS RMSE:** N/A")

st.markdown(f"- **NBER Recessions:** {len(recessions)}")
st.markdown(f"- **Data Completeness:** {yield_df.notna().all().all() * 100:.0f}%")

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
f"""
<div style="text-align: center; color: #7f8c8d; font-size: 0.65rem; padding: 1rem;">
<p>© 2024 Yield Curve Analytics | Institutional Quantitative Platform</p>
<p>Data: Federal Reserve Economic Data (FRED) | Models: Nelson-Siegel (1987), Svensson (1994)</p>
<p>Recession Definition: NBER (National Bureau of Economic Research)</p>
<p>Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
</div>
""", 
unsafe_allow_html=True
)

if __name__ == "__main__":
main()
