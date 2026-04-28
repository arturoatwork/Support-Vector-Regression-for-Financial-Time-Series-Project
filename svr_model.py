"""
Support Vector Regression model for financial time series prediction.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class SVRModel:
    """Wrapper for Support Vector Regression with hyperparameter optimization."""
    
    def __init__(
        self,
        kernel: str = 'rbf',
        C: float = 100,
        epsilon: float = 0.1,
        gamma: str = 'scale'
    ):
        """
        Initialize SVR model.
        
        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly')
            C: Regularization parameter
            epsilon: Epsilon in epsilon-SVR loss function
            gamma: Kernel coefficient
        """
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def build_model(self) -> SVR:
        """Build base SVR model."""
        return SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SVRModel':
        """
        Fit model to training data.
        
        Args:
            X: Features
            y: Target values
            
        Returns:
            Self
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model = self.build_model()
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"Model fitted with {len(X)} training samples")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y: Target values
            
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X)
        
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    @staticmethod
    def hyperparameter_tuning(
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[Dict] = None,
        cv_splits: int = 5
    ) -> Tuple[SVR, Dict]:
        """
        Perform hyperparameter optimization using GridSearchCV.
        
        Args:
            X: Features
            y: Target values
            param_grid: Parameter grid for search
            cv_splits: Number of cross-validation splits
            
        Returns:
            Tuple of (best_model, best_params)
        """
        if param_grid is None:
            # epsilon must be scaled relative to std(y); daily returns are ~1-2%
            # so epsilon in [0.0001, 0.005] covers 1-40% of one std(y)
            param_grid = {
                'C': [1, 10, 100, 500],
                'epsilon': [0.0001, 0.0005, 0.001, 0.005],
                'gamma': ['scale', 'auto', 0.01, 0.1]
            }
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use TimeSeriesSplit to maintain temporal order
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        grid_search = GridSearchCV(
            SVR(kernel='rbf'),
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        logger.info("Starting hyperparameter tuning...")
        grid_search.fit(X_scaled, y)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score (neg_MSE): {grid_search.best_score_:.6f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 10
    ) -> pd.Series:
        """
        Calculate permutation-based feature importance.
        
        Args:
            X: Features
            y: Target values
            n_repeats: Number of permutation repeats
            
        Returns:
            Series with feature importances
        """
        from sklearn.inspection import permutation_importance
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before importance calculation")
        
        X_scaled = self.scaler.transform(X)
        
        importance = permutation_importance(
            self.model,
            X_scaled,
            y,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )
        
        feature_imp = pd.Series(
            importance.importances_mean,
            index=X.columns
        ).sort_values(ascending=False)
        
        logger.info("Top 5 features:")
        logger.info(feature_imp.head())
        
        return feature_imp
    
    def get_model_info(self) -> Dict:
        """Get model configuration information."""
        return {
            'kernel': self.kernel,
            'C': self.C,
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'n_support_vectors': len(self.model.support_) if self.is_fitted else None,
            'is_fitted': self.is_fitted
        }
