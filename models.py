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
    Advanced Monte Carlo simulation engine for yield path forecasting.

    Supported models
    ----------------
    - Geometric Brownian Motion
    - Vasicek Mean-Reverting
    - Ornstein-Uhlenbeck
    - Jump Diffusion

    The engine also provides historical calibration, scenario shocks,
    distribution diagnostics, fan-chart statistics, and terminal risk metrics.
    """

    @staticmethod
    def _sanitize_series(series: pd.Series, min_obs: int = 60) -> pd.Series:
        if series is None:
            return pd.Series(dtype=float)
        s = pd.to_numeric(series, errors="coerce").dropna()
        return s if len(s) >= min_obs else pd.Series(dtype=float)

    @staticmethod
    def _fallback_params(series: pd.Series) -> Dict:
        s = MonteCarlo._sanitize_series(series, min_obs=3)
        if s.empty:
            return {
                "initial": 4.0,
                "mu": 0.0,
                "sigma_pct": 0.10,
                "sigma_abs": 0.50,
                "theta": 4.0,
                "kappa": 0.75,
                "jump_lambda": 0.10,
                "jump_mean": 0.0,
                "jump_std": 0.20,
                "last_diff": 0.0,
                "hist_mean_terminal_change": 0.0,
                "half_life_days": np.nan,
                "lookback_obs": 0,
            }

        initial = float(s.iloc[-1])
        diffs = s.diff().dropna()
        pct = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

        mu = float(pct.mean() * 252) if not pct.empty else 0.0
        sigma_pct = float(pct.std() * np.sqrt(252)) if not pct.empty else 0.10
        sigma_abs = float(diffs.std() * np.sqrt(252)) if not diffs.empty else max(abs(initial) * 0.05, 0.10)
        theta = float(s.mean())

        return {
            "initial": initial,
            "mu": mu,
            "sigma_pct": max(sigma_pct, 1e-6),
            "sigma_abs": max(sigma_abs, 1e-6),
            "theta": theta,
            "kappa": 0.75,
            "jump_lambda": 0.10,
            "jump_mean": 0.0,
            "jump_std": max(sigma_abs * 0.35, 0.05),
            "last_diff": float(diffs.iloc[-1]) if not diffs.empty else 0.0,
            "hist_mean_terminal_change": 0.0,
            "half_life_days": np.nan,
            "lookback_obs": int(len(s)),
        }

    @staticmethod
    def calibrate(yield_series: pd.Series, lookback: int = 252) -> Dict:
        """Calibrate process parameters from historical yield data."""
        s = MonteCarlo._sanitize_series(yield_series, min_obs=20)
        if s.empty:
            return MonteCarlo._fallback_params(yield_series)

        if lookback and len(s) > lookback:
            s = s.iloc[-lookback:]

        initial = float(s.iloc[-1])
        diffs = s.diff().dropna()
        pct = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

        mu = float(pct.mean() * 252) if not pct.empty else 0.0
        sigma_pct = float(pct.std() * np.sqrt(252)) if not pct.empty else 0.10
        sigma_abs = float(diffs.std() * np.sqrt(252)) if not diffs.empty else max(abs(initial) * 0.05, 0.10)
        theta = float(s.mean())

        # AR(1) calibration for mean reversion on levels
        lag = s.shift(1).dropna()
        curr = s.loc[lag.index]
        kappa = 0.75
        half_life_days = np.nan
        try:
            x = lag.values.astype(float)
            y = curr.values.astype(float)
            beta, alpha = np.polyfit(x, y, 1)
            beta = float(np.clip(beta, 1e-6, 0.9999))
            kappa = float(-np.log(beta) * 252)
            half_life_days = float(np.log(2) / kappa * 252) if kappa > 0 else np.nan
            theta = float(alpha / (1 - beta)) if abs(1 - beta) > 1e-8 else theta
        except Exception:
            pass

        centered = diffs - diffs.mean() if not diffs.empty else pd.Series(dtype=float)
        jump_threshold = 2.5 * centered.std() if not centered.empty and centered.std() > 0 else np.nan
        jumps = centered[centered.abs() > jump_threshold] if np.isfinite(jump_threshold) else pd.Series(dtype=float)
        jump_lambda = float(len(jumps) / max(len(diffs), 1) * 252) if not diffs.empty else 0.10
        jump_mean = float(jumps.mean()) if not jumps.empty else 0.0
        jump_std = float(jumps.std()) if len(jumps) > 1 else max(sigma_abs / np.sqrt(252), 0.05)

        hist_mean_terminal_change = 0.0
        horizon = min(20, max(len(s) // 10, 5))
        if len(s) > horizon:
            hist_mean_terminal_change = float((s.shift(-horizon) - s).dropna().mean())

        return {
            "initial": initial,
            "mu": mu,
            "sigma_pct": max(sigma_pct, 1e-6),
            "sigma_abs": max(sigma_abs, 1e-6),
            "theta": theta,
            "kappa": max(kappa, 1e-6),
            "jump_lambda": max(jump_lambda, 1e-6),
            "jump_mean": jump_mean,
            "jump_std": max(jump_std, 1e-6),
            "last_diff": float(diffs.iloc[-1]) if not diffs.empty else 0.0,
            "hist_mean_terminal_change": hist_mean_terminal_change,
            "half_life_days": half_life_days,
            "lookback_obs": int(len(s)),
        }

    @staticmethod
    def gbm(initial: float, mu: float, sigma: float, days: int, sims: int = 1000, seed: Optional[int] = None) -> np.ndarray:
        dt = 1 / 252
        rng = np.random.default_rng(seed)
        paths = np.zeros((sims, days + 1), dtype=float)
        paths[:, 0] = initial
        for i in range(1, days + 1):
            z = rng.standard_normal(sims)
            paths[:, i] = paths[:, i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
        return paths

    @staticmethod
    def vasicek(initial: float, kappa: float, theta: float, sigma: float, days: int, sims: int = 1000, seed: Optional[int] = None, floor: Optional[float] = None) -> np.ndarray:
        dt = 1 / 252
        rng = np.random.default_rng(seed)
        paths = np.zeros((sims, days + 1), dtype=float)
        paths[:, 0] = initial
        for i in range(1, days + 1):
            z = rng.standard_normal(sims)
            dr = kappa * (theta - paths[:, i - 1]) * dt + sigma * np.sqrt(dt) * z
            paths[:, i] = paths[:, i - 1] + dr
            if floor is not None:
                paths[:, i] = np.maximum(paths[:, i], floor)
        return paths

    @staticmethod
    def ornstein_uhlenbeck(initial: float, kappa: float, theta: float, sigma: float, days: int, sims: int = 1000, seed: Optional[int] = None) -> np.ndarray:
        return MonteCarlo.vasicek(initial, kappa, theta, sigma, days, sims, seed=seed, floor=None)

    @staticmethod
    def jump_diffusion(initial: float, mu: float, sigma: float, jump_lambda: float, jump_mean: float, jump_std: float, days: int, sims: int = 1000, seed: Optional[int] = None) -> np.ndarray:
        dt = 1 / 252
        rng = np.random.default_rng(seed)
        paths = np.zeros((sims, days + 1), dtype=float)
        paths[:, 0] = initial
        for i in range(1, days + 1):
            z = rng.standard_normal(sims)
            jump_count = rng.poisson(jump_lambda * dt, sims)
            jump_size = np.where(
                jump_count > 0,
                rng.normal(jump_mean, jump_std, sims) * jump_count,
                0.0,
            )
            diffusion = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
            paths[:, i] = np.maximum(paths[:, i - 1] * np.exp(diffusion) + jump_size, 0.0)
        return paths

    @staticmethod
    def apply_shock(paths: np.ndarray, shock_bps: float = 0.0, shock_day: int = 1, persistence: float = 1.0) -> np.ndarray:
        shocked = np.array(paths, copy=True)
        if shocked.size == 0 or shock_bps == 0:
            return shocked
        shock = shock_bps / 100.0
        start_idx = int(np.clip(shock_day, 0, shocked.shape[1] - 1))
        decay = persistence ** np.arange(shocked.shape[1] - start_idx)
        shocked[:, start_idx:] = shocked[:, start_idx:] + shock * decay
        return shocked

    @staticmethod
    def confidence_intervals(paths: np.ndarray, conf: float = 0.95) -> Dict:
        lower_p = (1 - conf) / 2 * 100
        upper_p = (1 + conf) / 2 * 100
        return {
            "mean": np.mean(paths, axis=0),
            "median": np.percentile(paths, 50, axis=0),
            "lower": np.percentile(paths, lower_p, axis=0),
            "upper": np.percentile(paths, upper_p, axis=0),
            "p05": np.percentile(paths, 5, axis=0),
            "p25": np.percentile(paths, 25, axis=0),
            "p75": np.percentile(paths, 75, axis=0),
            "p95": np.percentile(paths, 95, axis=0),
            "std": np.std(paths, axis=0),
        }

    @staticmethod
    def terminal_stats(paths: np.ndarray, initial: Optional[float] = None, conf: float = 0.95) -> Dict:
        terminal = np.asarray(paths[:, -1], dtype=float)
        if terminal.size == 0:
            return {}
        var_level = float(np.percentile(terminal, (1 - conf) * 100))
        tail = terminal[terminal <= var_level]
        cvar_level = float(np.mean(tail)) if tail.size else var_level
        initial_level = float(initial if initial is not None else paths[:, 0].mean())
        terminal_change = terminal - initial_level
        prob_up = float(np.mean(terminal > initial_level))
        prob_down_50bps = float(np.mean(terminal_change <= -0.50))
        prob_up_50bps = float(np.mean(terminal_change >= 0.50))
        return {
            "initial": initial_level,
            "terminal_mean": float(np.mean(terminal)),
            "terminal_median": float(np.median(terminal)),
            "terminal_std": float(np.std(terminal)),
            "terminal_min": float(np.min(terminal)),
            "terminal_max": float(np.max(terminal)),
            "var_level": var_level,
            "cvar_level": cvar_level,
            "prob_up": prob_up,
            "prob_down_50bps": prob_down_50bps,
            "prob_up_50bps": prob_up_50bps,
            "skew": float(pd.Series(terminal).skew()),
            "kurtosis": float(pd.Series(terminal).kurt()),
        }

    @staticmethod
    def path_diagnostics(paths: np.ndarray) -> Dict:
        if paths.size == 0:
            return {}
        cross_sectional_std = np.std(paths, axis=0)
        running_drawdown = paths / np.maximum.accumulate(paths, axis=1) - 1
        return {
            "max_cross_sectional_std": float(np.max(cross_sectional_std)),
            "terminal_iqr": float(np.percentile(paths[:, -1], 75) - np.percentile(paths[:, -1], 25)),
            "worst_path_drawdown": float(np.min(running_drawdown)),
            "best_terminal": float(np.max(paths[:, -1])),
            "worst_terminal": float(np.min(paths[:, -1])),
        }

    @staticmethod
    def summarize(paths: np.ndarray, initial: Optional[float] = None, conf: float = 0.95) -> Dict:
        out = MonteCarlo.confidence_intervals(paths, conf)
        out["terminal"] = MonteCarlo.terminal_stats(paths, initial=initial, conf=conf)
        out["diagnostics"] = MonteCarlo.path_diagnostics(paths)
        return out

    @staticmethod
    def simulate(yield_series: pd.Series, model: str = "Vasicek Mean-Reverting", days: int = 20, sims: int = 5000, conf: float = 0.95, lookback: int = 252, shock_bps: float = 0.0, shock_day: int = 1, shock_persistence: float = 1.0, seed: int = 42) -> Dict:
        params = MonteCarlo.calibrate(yield_series, lookback=lookback)
        initial = params["initial"]

        if model == "Geometric Brownian Motion":
            paths = MonteCarlo.gbm(initial, params["mu"], params["sigma_pct"], days, sims, seed=seed)
            model_short = "GBM"
        elif model == "Jump Diffusion":
            paths = MonteCarlo.jump_diffusion(
                initial, params["mu"], params["sigma_pct"], params["jump_lambda"],
                params["jump_mean"], params["jump_std"], days, sims, seed=seed
            )
            model_short = "Jump Diffusion"
        elif model == "Ornstein-Uhlenbeck":
            paths = MonteCarlo.ornstein_uhlenbeck(initial, params["kappa"], params["theta"], params["sigma_abs"], days, sims, seed=seed)
            model_short = "OU"
        else:
            paths = MonteCarlo.vasicek(initial, params["kappa"], params["theta"], params["sigma_abs"], days, sims, seed=seed)
            model_short = "Vasicek"

        shocked_paths = MonteCarlo.apply_shock(paths, shock_bps=shock_bps, shock_day=shock_day, persistence=shock_persistence)
        summary = MonteCarlo.summarize(shocked_paths, initial=initial, conf=conf)
        summary["paths"] = shocked_paths
        summary["params"] = params
        summary["model_name"] = model_short
        summary["confidence"] = conf
        summary["days"] = days
        summary["sims"] = sims
        return summary

    @staticmethod
    def var(paths: np.ndarray, conf: float = 0.95) -> float:
        return float(np.percentile(paths[:, -1], (1 - conf) * 100))

    @staticmethod
    def cvar(paths: np.ndarray, conf: float = 0.95) -> float:
        var = MonteCarlo.var(paths, conf)
        terminal = paths[:, -1]
        tail = terminal[terminal <= var]
        return float(np.mean(tail)) if tail.size else float(var)


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