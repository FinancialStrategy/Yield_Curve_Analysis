"""
Analytics Module - Spreads, Regime Classification, Recession Probability
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.stats import norm
from config import MATURITY_MAP


# =============================================================================
# SPREAD CALCULATIONS
# =============================================================================

def compute_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate yield spreads from yield curve data
    
    Parameters
    ----------
    df : pd.DataFrame
        Yield curve DataFrame with maturity columns
    
    Returns
    -------
    pd.DataFrame
        Spreads in basis points
    """
    spreads = pd.DataFrame(index=df.index)
    
    if "10Y" in df and "2Y" in df:
        spreads["10Y-2Y"] = (df["10Y"] - df["2Y"]) * 100
    
    if "10Y" in df and "3M" in df:
        spreads["10Y-3M"] = (df["10Y"] - df["3M"]) * 100
    
    if "5Y" in df and "2Y" in df:
        spreads["5Y-2Y"] = (df["5Y"] - df["2Y"]) * 100
    
    if "30Y" in df and "10Y" in df:
        spreads["30Y-10Y"] = (df["30Y"] - df["10Y"]) * 100
    
    return spreads


# =============================================================================
# REGIME CLASSIFICATION
# =============================================================================

def classify_regime(spreads: pd.DataFrame, yields: pd.DataFrame) -> Tuple[str, str]:
    """
    Classify current market regime based on yield curve shape
    
    Parameters
    ----------
    spreads : pd.DataFrame
        Spreads DataFrame
    yields : pd.DataFrame
        Yield curve DataFrame
    
    Returns
    -------
    tuple
        (regime_name, regime_description)
    """
    if "10Y-2Y" not in spreads.columns or spreads.empty:
        return "Data Loading", "Waiting for data to load"
    
    spread = spreads["10Y-2Y"].iloc[-1]
    y10 = yields["10Y"].iloc[-1] if "10Y" in yields else np.nan
    
    if np.isfinite(spread) and spread < 0:
        return "Risk-off / Recession Watch", "Curve inversion signals defensive positioning"
    
    if np.isfinite(spread) and spread < 50:
        return "Neutral / Late Cycle", "Curve flattening suggests caution"
    
    if np.isfinite(y10) and y10 > 5.5:
        return "Restrictive", "Long-end rates remain restrictive"
    
    return "Risk-on / Expansion", "Positive slope supports risk-taking"


def recession_probability(spreads: pd.DataFrame) -> float:
    """
    Estimate recession probability from 10Y-2Y spread using logistic regression
    
    Parameters
    ----------
    spreads : pd.DataFrame
        Spreads DataFrame containing "10Y-2Y" column
    
    Returns
    -------
    float
        Estimated recession probability between 0 and 1
    """
    if "10Y-2Y" not in spreads.columns or spreads.empty:
        return 0.5
    
    current = spreads["10Y-2Y"].iloc[-1]
    if not np.isfinite(current):
        return 0.5
    
    # Logistic function based on historical relationship
    prob = 1 / (1 + np.exp(-(-current * 2 - 0.5)))
    return min(max(prob, 0.01), 0.99)


# =============================================================================
# RECESSION DETECTION
# =============================================================================

def identify_recessions(recession_series: pd.Series) -> List[Dict]:
    """
    Extract recession periods from NBER recession indicator
    
    Parameters
    ----------
    recession_series : pd.Series
        NBER recession indicator (1 = recession, 0 = expansion)
    
    Returns
    -------
    list
        List of dicts with 'start' and 'end' dates for each recession
    """
    if recession_series.empty:
        return []
    
    recessions = []
    in_rec = False
    start = None
    
    for date, val in recession_series.dropna().items():
        if val == 1 and not in_rec:
            in_rec = True
            start = date
        elif val == 0 and in_rec:
            recessions.append({"start": start, "end": date})
            in_rec = False
    
    return recessions


def calculate_inversion_periods(spreads: pd.DataFrame) -> List[Dict]:
    """
    Identify periods where the yield curve is inverted (10Y-2Y < 0)
    
    Parameters
    ----------
    spreads : pd.DataFrame
        Spreads DataFrame containing "10Y-2Y" column
    
    Returns
    -------
    list
        List of inversion periods with start, end, depth, and duration
    """
    if "10Y-2Y" not in spreads.columns:
        return []
    
    s = spreads["10Y-2Y"].dropna()
    periods = []
    in_inv = False
    start = None
    
    for date, value in s.items():
        if value < 0 and not in_inv:
            in_inv = True
            start = date
        elif value >= 0 and in_inv:
            periods.append({
                "start": start,
                "end": date,
                "depth": float(s.loc[start:date].min()),
                "duration_days": (date - start).days,
            })
            in_inv = False
    
    return periods


def calculate_lead_times(inversions: List[Dict], recessions: List[Dict]) -> List[Dict]:
    """
    Calculate lead times between inversion start and recession start
    
    Parameters
    ----------
    inversions : list
        List of inversion periods
    recessions : list
        List of recession periods
    
    Returns
    -------
    list
        Lead times for each inversion-recession pair
    """
    lead_times = []
    for inv in inversions:
        for rec in recessions:
            if inv["start"] < rec["start"]:
                days = (rec["start"] - inv["start"]).days
                lead_times.append({
                    "inversion_start": inv["start"],
                    "recession_start": rec["start"],
                    "lead_days": days,
                    "lead_months": days / 30.44,
                    "inversion_depth": inv["depth"],
                })
                break
    return lead_times


def recession_hit_stats(inversions: List[Dict], recessions: List[Dict]) -> Dict:
    """
    Calculate hit ratio statistics for inversion signals
    
    Parameters
    ----------
    inversions : list
        List of inversion periods
    recessions : list
        List of recession periods
    
    Returns
    -------
    dict
        Hit ratio statistics
    """
    if not inversions:
        return {"HitRatio": np.nan, "FalsePositiveRate": np.nan, "Matches": 0, "Signals": 0}
    
    leads = calculate_lead_times(inversions, recessions)
    matches = len(leads)
    signals = len(inversions)
    false_positives = signals - matches
    
    return {
        "HitRatio": matches / signals if signals else np.nan,
        "FalsePositiveRate": false_positives / signals if signals else np.nan,
        "Matches": matches,
        "Signals": signals,
    }


# =============================================================================
# FACTOR ANALYSIS
# =============================================================================

def factor_contributions(yield_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Level, Slope, and Curvature factors from yield curve
    
    Parameters
    ----------
    yield_df : pd.DataFrame
        Yield curve DataFrame
    
    Returns
    -------
    pd.DataFrame
        Factor contributions
    """
    out = pd.DataFrame(index=yield_df.index)
    
    if "10Y" in yield_df.columns:
        out["Level"] = yield_df["10Y"]
    
    if "10Y" in yield_df and "3M" in yield_df:
        out["Slope"] = (yield_df["10Y"] - yield_df["3M"]) * 100
    
    if "3M" in yield_df and "5Y" in yield_df and "10Y" in yield_df:
        out["Curvature"] = (2 * yield_df["5Y"] - (yield_df["3M"] + yield_df["10Y"])) * 100
    
    return out


def calculate_forward_rates(yield_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate implied forward rates from yield curve
    
    Parameters
    ----------
    yield_df : pd.DataFrame
        Yield curve DataFrame
    
    Returns
    -------
    pd.DataFrame
        Forward rates
    """
    if yield_df.empty:
        return pd.DataFrame()
    
    forwards = pd.DataFrame(index=yield_df.index)
    maturities = MATURITY_MAP
    
    pairs = [("3M", "2Y"), ("2Y", "5Y"), ("5Y", "10Y"), ("10Y", "30Y")]
    
    for short_term, long_term in pairs:
        if short_term in yield_df.columns and long_term in yield_df.columns:
            r1 = maturities[short_term]
            r2 = maturities[long_term]
            
            try:
                forward = (((1 + yield_df[long_term] / 100) ** r2 / 
                           (1 + yield_df[short_term] / 100) ** r1) ** 
                          (1 / (r2 - r1)) - 1) * 100
                forwards[f'{short_term}→{long_term}'] = forward
            except:
                pass
    
    return forwards


# =============================================================================
# RISK METRICS
# =============================================================================

def calculate_var_metrics(returns: pd.Series, confidence: float = 0.95, horizon: int = 10) -> Optional[Dict]:
    """
    Calculate Value at Risk metrics including Historical, Parametric, and Cornish-Fisher VaR
    
    Parameters
    ----------
    returns : pd.Series
        Daily return series
    confidence : float
        Confidence level (default 0.95)
    horizon : int
        Time horizon in days (default 10)
    
    Returns
    -------
    dict or None
        VaR metrics including Historical, Parametric, Cornish-Fisher VaR, and CVaR
    """
    returns = returns.dropna()
    if len(returns) < 20:
        return None
    
    var_hist = np.percentile(returns, (1 - confidence) * 100)
    cvar = returns[returns <= var_hist].mean()
    var_param = norm.ppf(1 - confidence) * returns.std()
    
    skewness = returns.skew()
    kurt = returns.kurtosis()
    z = norm.ppf(1 - confidence)
    z_cf = z + (z**2 - 1) * skewness / 6 + (z**3 - 3 * z) * kurt / 24 - (2 * z**3 - 5 * z) * skewness**2 / 36
    
    return {
        "VaR_Historical": float(var_hist * np.sqrt(horizon)),
        "VaR_Parametric": float(var_param * np.sqrt(horizon)),
        "VaR_CornishFisher": float(z_cf * returns.std() * np.sqrt(horizon)),
        "CVaR": float(cvar * np.sqrt(horizon)),
        "Skewness": float(skewness),
        "Kurtosis": float(kurt),
    }


def forecast_curve(yield_df: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """
    Linear forecast of yield curve using historical trend
    
    Parameters
    ----------
    yield_df : pd.DataFrame
        Yield curve DataFrame
    horizon : int
        Forecast horizon in days
    
    Returns
    -------
    pd.DataFrame
        Forecasted yields
    """
    if len(yield_df) < 120:
        return pd.DataFrame()
    
    from sklearn.linear_model import LinearRegression
    
    x = np.arange(len(yield_df)).reshape(-1, 1)
    future_x = np.arange(len(yield_df), len(yield_df) + horizon).reshape(-1, 1)
    out = {}
    
    for col in yield_df.columns:
        model = LinearRegression()
        model.fit(x, yield_df[col].values)
        out[col] = model.predict(future_x)
    
    dates = pd.bdate_range(yield_df.index[-1] + pd.Timedelta(days=1), periods=horizon)
    return pd.DataFrame(out, index=dates)