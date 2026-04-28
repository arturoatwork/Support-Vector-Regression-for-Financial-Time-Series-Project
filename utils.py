"""
Utility functions for SVR quantitative finance project.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale features using training set statistics.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        Tuple of scaled (X_train, X_test)
    """
    mean = X_train.mean()
    std = X_train.std()
    
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    return X_train_scaled, X_test_scaled


def align_data(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    target_col: str = 'Returns'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Align features with target variable.
    
    Args:
        features: Feature dataframe
        prices: Price dataframe
        target_col: Target column name
        
    Returns:
        Tuple of (X, y) aligned and dropna
    """
    # Shift target to align with features
    y = prices[target_col].shift(-1)
    
    # Align indices
    aligned = pd.concat([features, y.rename('target')], axis=1)
    aligned = aligned.dropna()
    
    X = aligned.iloc[:, :-1]
    y = aligned.iloc[:, -1]
    
    return X, y


def calculate_directional_accuracy(
    predictions: np.ndarray,
    actuals: np.ndarray
) -> float:
    """
    Calculate directional accuracy (binary classification on sign).
    
    Args:
        predictions: Model predictions
        actuals: Actual values
        
    Returns:
        Accuracy score (0-1)
    """
    pred_direction = np.sign(predictions)
    actual_direction = np.sign(actuals)
    
    accuracy = np.mean(pred_direction == actual_direction)
    return accuracy


def calculate_information_ratio(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate information ratio vs benchmark.
    
    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        periods_per_year: Trading periods per year
        
    Returns:
        Information ratio
    """
    excess_returns = returns - benchmark_returns
    annual_excess = excess_returns.mean() * periods_per_year
    annual_tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
    
    ir = annual_excess / annual_tracking_error if annual_tracking_error > 0 else 0
    return ir


def calculate_calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (return/max drawdown).
    
    Args:
        returns: Strategy returns
        periods_per_year: Trading periods per year
        
    Returns:
        Calmar ratio
    """
    cumulative = (1 + returns).cumprod()
    cummax = np.maximum.accumulate(cumulative)
    drawdown = cumulative / cummax - 1
    max_drawdown = np.abs(drawdown.min())
    
    annual_return = returns.mean() * periods_per_year
    
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0
    return calmar


def rolling_metrics(
    values: pd.Series,
    window: int = 63
) -> pd.DataFrame:
    """
    Calculate rolling metrics.
    
    Args:
        values: Time series values
        window: Rolling window size
        
    Returns:
        DataFrame with rolling mean and std
    """
    return pd.DataFrame({
        'Mean': values.rolling(window).mean(),
        'Std': values.rolling(window).std(),
        'Min': values.rolling(window).min(),
        'Max': values.rolling(window).max()
    })
