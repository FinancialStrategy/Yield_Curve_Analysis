"""
Volatility and Correlation Analysis Module
"""

import pandas as pd
import numpy as np
from typing import Dict


class VolatilityAnalyzer:
    """
    Volatility analysis for VIX and market stress indicators
    """
    
    @staticmethod
    def calculate_volatility_regime(vix: pd.Series) -> Dict:
        """
        Classify current volatility regime based on VIX level
        
        Parameters
        ----------
        vix : pd.Series
            VIX time series data
        
        Returns
        -------
        dict
            Current VIX value, regime classification, and outlook
        """
        if vix.empty:
            return {"current_vix": 0, "regime": "N/A", "outlook": "Data unavailable", "vix_percentile": "N/A"}
        
        current_vix = vix.iloc[-1]
        
        if current_vix < 12:
            regime, outlook = "EXTREME COMPLACENCY", "High risk of volatility spike"
        elif current_vix < 15:
            regime, outlook = "LOW VOLATILITY", "Normal complacent market"
        elif current_vix < 20:
            regime, outlook = "NORMAL VOLATILITY", "Typical market conditions"
        elif current_vix < 25:
            regime, outlook = "ELEVATED VOLATILITY", "Increased uncertainty"
        elif current_vix < 35:
            regime, outlook = "HIGH VOLATILITY", "Market stress, consider hedging"
        else:
            regime, outlook = "EXTREME VOLATILITY", "Crisis conditions, defensive positioning"
        
        percentile = (vix < current_vix).mean()
        
        return {
            "current_vix": float(current_vix),
            "regime": regime,
            "outlook": outlook,
            "vix_percentile": f"{percentile * 100:.1f}%",
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
        if vix.empty or len(vix) < window:
            return pd.Series(dtype=float)
        
        return vix.pct_change().rolling(window).std() * np.sqrt(252)


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
        if assets_df.empty:
            return pd.DataFrame()
        
        returns = assets_df.pct_change().dropna()
        return returns.corr()
    
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
        if asset1.empty or asset2.empty or len(asset1) < window:
            return pd.Series(dtype=float)
        
        returns1 = asset1.pct_change()
        returns2 = asset2.pct_change()
        
        return returns1.rolling(window).corr(returns2)