"""
Scenario Analysis Module - Deterministic shock scenarios
"""

import pandas as pd
import numpy as np
from typing import Dict
from config import MATURITY_MAP


def generate_scenarios(yield_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Generate deterministic shock scenarios for the yield curve
    
    Scenarios include:
    - Bull Steepener: Long-end falls more than short-end
    - Bear Flattener: Short-end rises more than long-end
    - Recession: Broad rally across the curve
    - Policy Easing: Short-end falls sharply
    
    Parameters
    ----------
    yield_df : pd.DataFrame
        Yield curve DataFrame with maturity columns
    
    Returns
    -------
    dict
        Dictionary of scenario DataFrames with Current and Scenario columns
    """
    if yield_df.empty:
        return {}
    
    latest = yield_df.iloc[-1].copy()
    scenarios = {}
    tenor_order = list(yield_df.columns)
    
    # Bull Steepener - long end falls more than short end
    bull = latest.copy()
    for col in tenor_order:
        m = MATURITY_MAP.get(col, 1)
        bull[col] = bull[col] - (0.08 + 0.06 * min(m / 10, 1.5))
    scenarios["Bull Steepener"] = pd.DataFrame({"Current": latest, "Scenario": bull})
    
    # Bear Flattener - short end rises more than long end
    bear = latest.copy()
    for col in tenor_order:
        m = MATURITY_MAP.get(col, 1)
        bear[col] = bear[col] + (0.14 if m <= 2 else 0.07)
    scenarios["Bear Flattener"] = pd.DataFrame({"Current": latest, "Scenario": bear})
    
    # Recession Case - broad rally, short end falls sharply
    recession = latest.copy()
    for col in tenor_order:
        m = MATURITY_MAP.get(col, 1)
        recession[col] = recession[col] - (0.22 if m <= 2 else 0.14 if m <= 10 else 0.10)
    scenarios["Recession"] = pd.DataFrame({"Current": latest, "Scenario": recession})
    
    # Policy Easing - front-end down, moderate long-end response
    easing = latest.copy()
    for col in tenor_order:
        m = MATURITY_MAP.get(col, 1)
        easing[col] = easing[col] - (0.25 if m <= 2 else 0.12 if m <= 10 else 0.06)
    scenarios["Policy Easing"] = pd.DataFrame({"Current": latest, "Scenario": easing})
    
    return scenarios


def get_scenario_interpretation(scenario_name: str) -> str:
    """
    Get interpretation text for a given scenario
    
    Parameters
    ----------
    scenario_name : str
        Name of the scenario
    
    Returns
    -------
    str
        Interpretation text
    """
    interpretations = {
        "Bull Steepener": "Long-term rates falling faster than short-term. Typically occurs during Fed easing cycles or falling inflation expectations. Positive for long-duration bonds.",
        "Bear Flattener": "Short-term rates rising faster than long-term. Often reflects hawkish central-bank policy. May precede curve inversion.",
        "Recession": "Broad rally across the curve with strongest fall at the short end. Defensive positioning recommended. Historically associated with economic downturns.",
        "Policy Easing": "Short-end yields fall sharply after expected policy cuts. Long-end reacts less. Isolates the transmission of a central-bank easing cycle."
    }
    return interpretations.get(scenario_name, "Analyze the impact of different market scenarios on the yield curve.")


def calculate_scenario_impact(current_curve: pd.Series, scenario_curve: pd.Series) -> pd.DataFrame:
    """
    Calculate the impact of a scenario on each maturity
    
    Parameters
    ----------
    current_curve : pd.Series
        Current yield curve values
    scenario_curve : pd.Series
        Scenario yield curve values
    
    Returns
    -------
    pd.DataFrame
        DataFrame with current, scenario, and change values
    """
    impact_df = pd.DataFrame({
        "Maturity": current_curve.index,
        "Current (%)": current_curve.values,
        "Scenario (%)": scenario_curve.values,
        "Change (bps)": (scenario_curve.values - current_curve.values) * 100
    })
    return impact_df