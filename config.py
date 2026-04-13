"""
Configuration Module - Centralized constants and settings
"""

from dataclasses import dataclass
from typing import Dict

# =============================================================================
# COLOR SYSTEM
# =============================================================================

COLORS: Dict[str, str] = {
    "bg": "#eef2f7",
    "bg2": "#f7f9fc",
    "surface": "#ffffff",
    "surface_alt": "#f5f7fb",
    "header": "#1a2a3a",
    "grid": "#c8d4e0",
    "text": "#1a2a3a",
    "text_secondary": "#4a5a6a",
    "muted": "#667085",
    "accent": "#2c5f8a",
    "accent2": "#4a7c59",
    "accent3": "#c17f3a",
    "positive": "#10b981",
    "negative": "#ef4444",
    "warning": "#f59e0b",
    "info": "#3b82f6",
    "recession": "rgba(120, 130, 145, 0.18)",
    "band": "rgba(108, 142, 173, 0.10)",
}

# =============================================================================
# FRED API SERIES
# =============================================================================

FRED_SERIES: Dict[str, str] = {
    "3M": "DGS3MO",
    "2Y": "DGS2",
    "5Y": "DGS5",
    "10Y": "DGS10",
    "30Y": "DGS30",
}

RECESSION_SERIES: str = "USREC"

# =============================================================================
# MATURITY MAPPING
# =============================================================================

MATURITY_MAP: Dict[str, float] = {
    "3M": 0.25,
    "2Y": 2.0,
    "5Y": 5.0,
    "10Y": 10.0,
    "30Y": 30.0,
}

# =============================================================================
# YAHOO FINANCE TICKERS
# =============================================================================

YAHOO_TICKERS: Dict[str, str] = {
    "TLT": "20+ Year Treasury Bond ETF",
    "IEF": "7-10 Year Treasury Bond ETF",
    "SHY": "1-3 Year Treasury Bond ETF",
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq 100 ETF",
}

VOLATILITY_TICKERS: Dict[str, str] = {
    "^VIX": "CBOE Volatility Index",
}

CORRELATION_TICKERS: Dict[str, str] = {
    "^GSPC": "S&P 500",
    "GLD": "Gold",
    "UUP": "US Dollar Index",
}

# =============================================================================
# RUNTIME CONFIGURATION
# =============================================================================

@dataclass
class RuntimeConfig:
    """Runtime configuration settings"""
    history_start: str = "1990-01-01"
    timeout: int = 25
    max_retries: int = 3
    cache_ttl_sec: int = 3600
    rolling_step: int = 21
    rolling_years_default: int = 5
    forecast_horizon_default: int = 20
    mc_simulations_default: int = 5000
    var_confidence_default: float = 0.95

CFG = RuntimeConfig()

# =============================================================================
# VERSION INFORMATION
# =============================================================================

__version__ = "37.0"
__status__ = "Production - Refactored"