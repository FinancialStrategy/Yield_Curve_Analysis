"""
Models Module - Nelson-Siegel, Monte Carlo, Backtest Engine
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from scipy.optimize import minimize, differential_evolution


# =============================================================================
# NELSON-SIEGEL MODEL
# =============================================================================

class NelsonSiegel:
    """
    Nelson-Siegel yield curve model for parametric curve fitting
    
    The model represents the yield curve as:
    y(τ) = β₀ + β₁ * (1 - e^{-λτ})/(λτ) + β₂ * ((1 - e^{-λτ})/(λτ) - e^{-λτ})
    
    Where:
    - β₀: Long-term level factor
    - β₁: Short-term slope factor  
    - β₂: Medium-term curvature factor
    - λ: Decay parameter
    """
    
    @staticmethod
    def curve(tau, beta0, beta1, beta2, lambda1):
        """
        Calculate Nelson-Siegel curve values at given maturities
        
        Parameters
        ----------
        tau : array-like
            Maturities in years
        beta0, beta1, beta2, lambda1 : float
            Model parameters
        
        Returns
        -------
        array
            Fitted yields
        """
        tau = np.asarray(tau, dtype=float)
        x = lambda1 * np.where(tau == 0, 1e-8, tau)
        term1 = (1 - np.exp(-x)) / x
        term2 = term1 - np.exp(-x)
        return beta0 + beta1 * term1 + beta2 * term2
    
    @staticmethod
    def fit(maturities: np.ndarray, yields: np.ndarray) -> Optional[Dict]:
        """
        Fit Nelson-Siegel model to observed yields
        
        Parameters
        ----------
        maturities : np.ndarray
            Maturities in years
        yields : np.ndarray
            Observed yields in percentage
        
        Returns
        -------
        dict or None
            Fitted parameters and quality metrics
        """
        if len(maturities) == 0 or len(yields) == 0:
            return None
        
        if not np.all(np.isfinite(yields)):
            return None
        
        # Handle flat curve
        if np.std(yields) < 1e-6:
            return {
                "params": [yields[0], 0.0, 0.0, 1.0],
                "fitted": yields,
                "rmse": 0.0,
                "r2": 1.0,
                "residuals": np.zeros_like(yields),
            }
        
        def objective(params):
            fitted = NelsonSiegel.curve(maturities, *params)
            return np.sum((yields - fitted) ** 2)
        
        y_min, y_max = float(np.min(yields)), float(np.max(yields))
        bounds = [(y_min - 2, y_max + 2), (-15, 15), (-15, 15), (0.01, 5)]
        
        best, best_fun = None, np.inf
        for _ in range(8):
            try:
                x0 = [np.random.uniform(a, b) for a, b in bounds]
                res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
                if res.success and res.fun < best_fun:
                    best, best_fun = res, res.fun
            except Exception:
                continue
        
        if best is None:
            return None
        
        fitted = NelsonSiegel.curve(maturities, *best.x)
        sse = np.sum((yields - fitted) ** 2)
        sst = np.sum((yields - np.mean(yields)) ** 2)
        
        return {
            "params": best.x,
            "fitted": fitted,
            "rmse": float(np.sqrt(np.mean((yields - fitted) ** 2))),
            "r2": float(1 - sse / sst) if sst > 0 else np.nan,
            "residuals": yields - fitted,
        }
    
    @staticmethod
    def nss_curve(tau, beta0, beta1, beta2, beta3, lambda1, lambda2):
        """
        Nelson-Siegel-Svensson extension with second curvature factor
        
        Parameters
        ----------
        tau : array-like
            Maturities in years
        beta0, beta1, beta2, beta3, lambda1, lambda2 : float
            Model parameters
        
        Returns
        -------
        array
            Fitted yields
        """
        tau = np.asarray(tau, dtype=float)
        x1 = lambda1 * np.where(tau == 0, 1e-8, tau)
        x2 = lambda2 * np.where(tau == 0, 1e-8, tau)
        
        term1 = (1 - np.exp(-x1)) / x1
        term2 = term1 - np.exp(-x1)
        term3 = ((1 - np.exp(-x2)) / x2) - np.exp(-x2)
        
        return beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3
    
    @staticmethod
    def fit_nss(maturities: np.ndarray, yields: np.ndarray) -> Optional[Dict]:
        """
        Fit Nelson-Siegel-Svensson model to observed yields
        
        Parameters
        ----------
        maturities : np.ndarray
            Maturities in years
        yields : np.ndarray
            Observed yields in percentage
        
        Returns
        -------
        dict or None
            Fitted parameters and quality metrics
        """
        if len(maturities) == 0 or len(yields) == 0:
            return None
        
        if not np.all(np.isfinite(yields)):
            return None
        
        if np.std(yields) < 1e-6:
            return {
                "params": [yields[0], 0.0, 0.0, 0.0, 1.0, 1.0],
                "fitted": yields,
                "rmse": 0.0,
                "r2": 1.0,
                "residuals": np.zeros_like(yields),
            }
        
        def objective(params):
            fitted = NelsonSiegel.nss_curve(maturities, *params)
            weights = 1 / (maturities + 0.25)
            return np.sum(weights * (yields - fitted) ** 2)
        
        y_min, y_max = float(np.min(yields)), float(np.max(yields))
        bounds = [
            (y_min - 2, y_max + 2),
            (-20, 20), (-20, 20), (-20, 20),
            (0.01, 10), (0.01, 10),
        ]
        
        try:
            res = differential_evolution(objective, bounds=bounds, maxiter=220, popsize=10, polish=True, seed=42)
            if not res.success:
                return None
            fitted = NelsonSiegel.nss_curve(maturities, *res.x)
        except Exception:
            return None
        
        sse = np.sum((yields - fitted) ** 2)
        sst = np.sum((yields - np.mean(yields)) ** 2)
        
        return {
            "params": res.x,
            "fitted": fitted,
            "rmse": float(np.sqrt(np.mean((yields - fitted) ** 2))),
            "r2": float(1 - sse / sst) if sst > 0 else np.nan,
            "residuals": yields - fitted,
        }


def model_governance(ns_result: Optional[Dict], nss_result: Optional[Dict]) -> pd.DataFrame:
    """
    Evaluate model fit quality and generate governance flags
    
    Parameters
    ----------
    ns_result : dict or None
        Nelson-Siegel fit result
    nss_result : dict or None
        Nelson-Siegel-Svensson fit result
    
    Returns
    -------
    pd.DataFrame
        Governance metrics for model evaluation
    """
    rows = []
    for model_name, result in [("NS", ns_result), ("NSS", nss_result)]:
        if result is None:
            continue
        
        max_abs_residual = float(np.max(np.abs(result["residuals"])))
        rmse = result["rmse"]
        r2 = result["r2"]
        
        if rmse < 0.05 and r2 > 0.98:
            confidence = "High"
        elif rmse < 0.10 and r2 > 0.95:
            confidence = "Moderate"
        else:
            confidence = "Low"
        
        flags = []
        if max_abs_residual > 0.15:
            flags.append("Residual outlier")
        if r2 < 0.95:
            flags.append("Low fit quality")
        
        rows.append({
            "Model": model_name,
            "RMSE": rmse,
            "MAE": result.get("mae", result["rmse"]),
            "R2": r2,
            "MaxAbsResidual": max_abs_residual,
            "FitConfidence": confidence,
            "WarningFlags": ", ".join(flags) if flags else "None",
        })
    
    return pd.DataFrame(rows)


def rolling_ns_parameters(yield_df: pd.DataFrame, maturities: np.ndarray, selected_cols: List[str], years: int, step: int = 21) -> pd.DataFrame:
    """
    Calculate rolling Nelson-Siegel parameters over time
    
    Parameters
    ----------
    yield_df : pd.DataFrame
        Yield curve DataFrame
    maturities : np.ndarray
        Maturities in years
    selected_cols : list
        Selected maturity columns
    years : int
        Rolling window size in years
    step : int
        Step size in trading days
    
    Returns
    -------
    pd.DataFrame
        Rolling parameters over time
    """
    window_size = years * 252
    if len(yield_df) <= window_size + 5:
        return pd.DataFrame()
    
    rows = []
    for i in range(window_size, len(yield_df), step):
        curve = yield_df.iloc[i][selected_cols].values
        if not np.all(np.isfinite(curve)):
            continue
        
        res = NelsonSiegel.fit(maturities, curve)
        if res:
            rows.append({
                "date": yield_df.index[i],
                "beta0": res["params"][0],
                "beta1": res["params"][1],
                "beta2": res["params"][2],
                "lambda": res["params"][3],
                "rmse": res["rmse"],
            })
    
    return pd.DataFrame(rows)


# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

class MonteCarlo:
    """
    Monte Carlo simulation for yield path forecasting
    
    Supports Geometric Brownian Motion (GBM) and Vasicek mean-reverting models
    """
    
    @staticmethod
    def gbm(initial: float, mu: float, sigma: float, days: int, sims: int = 1000) -> np.ndarray:
        """
        Geometric Brownian Motion simulation
        
        Parameters
        ----------
        initial : float
            Initial yield value
        mu : float
            Drift coefficient (annualized)
        sigma : float
            Volatility coefficient (annualized)
        days : int
            Simulation horizon in trading days
        sims : int
            Number of simulation paths
        
        Returns
        -------
        np.ndarray
            Simulation paths (sims x days)
        """
        dt = 1 / 252
        paths = np.zeros((sims, days))
        paths[:, 0] = initial
        
        for i in range(1, days):
            z = np.random.standard_normal(sims)
            paths[:, i] = paths[:, i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
        return paths
    
    @staticmethod
    def vasicek(initial: float, kappa: float, theta: float, sigma: float, days: int, sims: int = 1000) -> np.ndarray:
        """
        Vasicek mean-reverting model simulation
        
        Parameters
        ----------
        initial : float
            Initial yield value
        kappa : float
            Mean-reversion speed
        theta : float
            Long-term mean level
        sigma : float
            Volatility coefficient
        days : int
            Simulation horizon in trading days
        sims : int
            Number of simulation paths
        
        Returns
        -------
        np.ndarray
            Simulation paths (sims x days)
        """
        dt = 1 / 252
        paths = np.zeros((sims, days))
        paths[:, 0] = initial
        
        for i in range(1, days):
            z = np.random.standard_normal(sims)
            dr = kappa * (theta - paths[:, i-1]) * dt + sigma * np.sqrt(dt) * z
            paths[:, i] = paths[:, i-1] + dr
        
        return paths
    
    @staticmethod
    def confidence_intervals(paths: np.ndarray, conf: float = 0.95) -> Dict:
        """
        Calculate confidence intervals for simulation paths
        
        Parameters
        ----------
        paths : np.ndarray
            Simulation paths
        conf : float
            Confidence level (default 0.95)
        
        Returns
        -------
        dict
            Mean, median, lower/upper bounds, and standard deviation
        """
        lower_p = (1 - conf) / 2 * 100
        upper_p = (1 + conf) / 2 * 100
        
        return {
            "mean": np.mean(paths, axis=0),
            "median": np.percentile(paths, 50, axis=0),
            "lower": np.percentile(paths, lower_p, axis=0),
            "upper": np.percentile(paths, upper_p, axis=0),
            "std": np.std(paths, axis=0),
        }
    
    @staticmethod
    def var(paths: np.ndarray, conf: float = 0.95) -> float:
        """
        Calculate Value at Risk from simulation paths
        
        Parameters
        ----------
        paths : np.ndarray
            Simulation paths
        conf : float
            Confidence level (default 0.95)
        
        Returns
        -------
        float
            Value at Risk at the specified confidence level
        """
        return np.percentile(paths[:, -1], (1 - conf) * 100)
    
    @staticmethod
    def cvar(paths: np.ndarray, conf: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Parameters
        ----------
        paths : np.ndarray
            Simulation paths
        conf : float
            Confidence level (default 0.95)
        
        Returns
        -------
        float
            Conditional Value at Risk
        """
        var = MonteCarlo.var(paths, conf)
        return np.mean(paths[:, -1][paths[:, -1] <= var])


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class Backtest:
    """
    Strategy backtesting engine for yield curve trading strategies
    """
    
    @staticmethod
    def run(yields: pd.DataFrame, spreads: pd.DataFrame, strategy: str = "Curve Inversion") -> Optional[Dict]:
        """
        Run backtest for a given strategy
        
        Parameters
        ----------
        yields : pd.DataFrame
            Yield curve DataFrame
        spreads : pd.DataFrame
            Spreads DataFrame
        strategy : str
            Strategy type: "Curve Inversion" or "Macro Trend"
        
        Returns
        -------
        dict or None
            Backtest results including cumulative returns, Sharpe ratio, drawdown
        """
        if "10Y" not in yields.columns:
            return None
        
        returns = yields["10Y"].pct_change().shift(-1)
        
        if strategy == "Curve Inversion":
            if "10Y-2Y" not in spreads.columns:
                return None
            signals = spreads["10Y-2Y"] < 0
        elif strategy == "Macro Trend":
            sma_50 = yields["10Y"].rolling(50).mean()
            signals = yields["10Y"] > sma_50
        elif strategy == "Momentum":
            momentum = yields["10Y"].diff(20)
            signals = momentum > 0
        else:
            return None
        
        strategy_returns = signals.shift(1) * returns
        strategy_returns = strategy_returns.fillna(0)
        
        cumulative = (1 + strategy_returns).cumprod()
        benchmark = (1 + returns.fillna(0)).cumprod()
        
        sharpe = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() > 0 else 0
        drawdown = cumulative / cumulative.cummax() - 1
        win_rate = (strategy_returns > 0).sum() / len(strategy_returns[strategy_returns != 0]) if len(strategy_returns[strategy_returns != 0]) > 0 else 0
        
        return {
            "cumulative": cumulative,
            "benchmark": benchmark,
            "sharpe": sharpe,
            "max_drawdown": float(drawdown.min()),
            "total_return": float(cumulative.iloc[-1] - 1),
            "win_rate": win_rate,
        }