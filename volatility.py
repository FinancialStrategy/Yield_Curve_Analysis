"""
Volatility and Correlation Analysis Module
FIXED: Proper handling of pandas Series values
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, Optional


def safe_float_from_series(value: Union[pd.Series, float, None], default: float = 0.0) -> float:
    """
    Safely convert a value to float, handling pandas Series and NaN values
    
    Parameters
    ----------
    value : float, pd.Series, or None
        Input value to convert
    default : float
        Default value if conversion fails
    
    Returns
    -------
    float
        Safely converted float value
    """
    if value is None:
        return default
    if isinstance(value, pd.Series):
        if value.empty:
            return default
        val = value.iloc[-1]
        if pd.isna(val):
            return default
        return float(val)
    if isinstance(value, (int, float)):
        if np.isnan(value):
            return default
        return float(value)
    return default


class VolatilityAnalyzer:
    """
    Volatility analysis for VIX and market stress indicators
    """
    
    @staticmethod
    def calculate_volatility_regime(vix: Union[pd.Series, float, None]) -> Dict:
        """
        Classify current volatility regime based on VIX level
        
        Parameters
        ----------
        vix : pd.Series, float, or None
            VIX time series data or current VIX value
        
        Returns
        -------
        dict
            Current VIX value, regime classification, and outlook
        """
        # Safely extract current VIX value
        current_vix = safe_float_from_series(vix, 0.0)
        
        if current_vix <= 0:
            return {
                "current_vix": 0.0,
                "regime": "N/A",
                "outlook": "Data unavailable",
                "vix_percentile": "N/A"
            }
        
        # Classify regime
        if current_vix < 12:
            regime = "EXTREME COMPLACENCY"
            outlook = "High risk of volatility spike"
        elif current_vix < 15:
            regime = "LOW VOLATILITY"
            outlook = "Normal complacent market"
        elif current_vix < 20:
            regime = "NORMAL VOLATILITY"
            outlook = "Typical market conditions"
        elif current_vix < 25:
            regime = "ELEVATED VOLATILITY"
            outlook = "Increased uncertainty"
        elif current_vix < 35:
            regime = "HIGH VOLATILITY"
            outlook = "Market stress, consider hedging"
        else:
            regime = "EXTREME VOLATILITY"
            outlook = "Crisis conditions, defensive positioning"
        
        # Calculate percentile if vix is a Series with data
        percentile = "N/A"
        if isinstance(vix, pd.Series) and not vix.empty and len(vix) > 0:
            try:
                vix_clean = vix.dropna()
                if len(vix_clean) > 0:
                    percentile_val = (vix_clean < current_vix).mean()
                    percentile = f"{percentile_val * 100:.1f}%"
            except Exception:
                pass
        
        return {
            "current_vix": current_vix,
            "regime": regime,
            "outlook": outlook,
            "vix_percentile": percentile,
        }
    
    @staticmethod
    def calculate_vol_of_vol(vix: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate volatility of volatility (volatility of VIX)
        
        Parameters
        ----------
        vix : pd.Series
            VIX time series data
        window : int
            Rolling window size in days
        
        Returns
        -------
        pd.Series
            Volatility of volatility series
        """
        if vix is None or vix.empty or len(vix) < window:
            return pd.Series(dtype=float)
        
        try:
            # Ensure vix is a Series with proper values
            vix_clean = vix.dropna()
            if len(vix_clean) < window:
                return pd.Series(dtype=float)
            
            returns = vix_clean.pct_change().dropna()
            if len(returns) < window:
                return pd.Series(dtype=float)
            
            return returns.rolling(window).std() * np.sqrt(252)
        except Exception:
            return pd.Series(dtype=float)


class CorrelationAnalyzer:
    """
    Cross-asset correlation analysis
    """
    
    @staticmethod
    def calculate_correlation_matrix(assets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix for multiple assets
        
        Parameters
        ----------
        assets_df : pd.DataFrame
            DataFrame with asset prices as columns
        
        Returns
        -------
        pd.DataFrame
            Correlation matrix
        """
        if assets_df is None or assets_df.empty:
            return pd.DataFrame()
        
        try:
            returns = assets_df.pct_change().dropna()
            if returns.empty or returns.shape[1] < 2:
                return pd.DataFrame()
            return returns.corr()
        except Exception:
            return pd.DataFrame()
    
    @staticmethod
    def calculate_rolling_correlation(asset1: pd.Series, asset2: pd.Series, window: int = 60) -> pd.Series:
        """
        Calculate rolling correlation between two assets
        
        Parameters
        ----------
        asset1 : pd.Series
            First asset price series
        asset2 : pd.Series
            Second asset price series
        window : int
            Rolling window size in days
        
        Returns
        -------
        pd.Series
            Rolling correlation series
        """
        if asset1 is None or asset2 is None or asset1.empty or asset2.empty:
            return pd.Series(dtype=float)
        
        if len(asset1) < window or len(asset2) < window:
            return pd.Series(dtype=float)
        
        try:
            returns1 = asset1.pct_change().dropna()
            returns2 = asset2.pct_change().dropna()
            
            # Align indices
            common_idx = returns1.index.intersection(returns2.index)
            if len(common_idx) < window:
                return pd.Series(dtype=float)
            
            returns1 = returns1.reindex(common_idx)
            returns2 = returns2.reindex(common_idx)
            
            return returns1.rolling(window).corr(returns2)
        except Exception:
            return pd.Series(dtype=float)