"""
Machine Learning Forecast Module
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class MLForecastModel:
    """
    Machine learning models for yield curve forecasting
    
    Supports Random Forest and Gradient Boosting algorithms
    """
    
    @staticmethod
    def prepare_features(yield_df: pd.DataFrame, lags: int = 5) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
        """
        Prepare lagged features for ML model
        
        Parameters
        ----------
        yield_df : pd.DataFrame
            Yield curve DataFrame
        lags : int
            Number of lag periods to include as features
        
        Returns
        -------
        tuple
            (X_scaled, y, scaler) for training
        """
        if yield_df.empty:
            return np.array([]), np.array([]), None
        
        X, y = [], []
        for i in range(lags, len(yield_df) - 1):
            features = []
            for col in yield_df.columns:
                features.extend(yield_df[col].iloc[i-lags:i].values)
            X.append(features)
            y.append(yield_df.iloc[i + 1].values)
        
        if not X:
            return np.array([]), np.array([]), None
        
        X_arr, y_arr = np.array(X), np.array(y)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)
        
        return X_scaled, y_arr, scaler
    
    @staticmethod
    def train_model(X: np.ndarray, y: np.ndarray, model_type: str = "Random Forest", test_size: float = 0.2) -> Dict:
        """
        Train ML model on prepared features
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        model_type : str
            "Random Forest" or "Gradient Boosting"
        test_size : float
            Proportion of data to use for testing
        
        Returns
        -------
        dict
            Model performance metrics and feature importance
        """
        if len(X) == 0:
            return {}
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        if model_type == "Gradient Boosting":
            target_train = y_train[:, -1] if y_train.ndim > 1 else y_train
            target_test = y_test[:, -1] if y_test.ndim > 1 else y_test
            model = GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=3, random_state=42)
            model.fit(X_train, target_train)
            y_pred = model.predict(X_test)
            y_eval = target_test
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred_full = model.predict(X_test)
            
            if y_pred_full.ndim > 1:
                y_pred = y_pred_full[:, -1]
                y_eval = y_test[:, -1]
            else:
                y_pred = y_pred_full
                y_eval = y_test
        
        rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
        mae = mean_absolute_error(y_eval, y_pred)
        r2 = r2_score(y_eval, y_pred)
        
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "feature": [f"Lagged_Feature_{i}" for i in range(X.shape[1])],
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False).head(12)
        else:
            importance_df = pd.DataFrame()
        
        return {
            "model_used": model_type,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "feature_importance": importance_df,
        }
    
    @staticmethod
    def predict_next(model, last_values: np.ndarray, scaler: StandardScaler, lags: int) -> float:
        """
        Generate next period prediction using trained model
        
        Parameters
        ----------
        model : sklearn model
            Trained ML model
        last_values : np.ndarray
            Last 'lags' observations for each maturity
        scaler : StandardScaler
            Fitted scaler from training
        lags : int
            Number of lag periods used in training
        
        Returns
        -------
        float
            Predicted yield for next period
        """
        features = last_values.flatten().reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        
        if isinstance(prediction, np.ndarray) and len(prediction) > 0:
            return float(prediction[0])
        
        return float(prediction)