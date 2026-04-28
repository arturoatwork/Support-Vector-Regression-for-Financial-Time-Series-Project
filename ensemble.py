"""
Ensemble methods for SVR quantitative finance project.
Implements stacking ensemble with SVR, XGBoost, LightGBM, and meta-learner.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from typing import Dict, List, Tuple, Optional, Any
import logging

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Stacking ensemble for financial time series prediction."""

    def __init__(
        self,
        base_models: Optional[List] = None,
        meta_learner: Optional[Any] = None,
        cv_folds: int = 5
    ):
        """
        Initialize ensemble model.

        Args:
            base_models: List of base model instances
            meta_learner: Meta-learner for stacking
            cv_folds: Number of cross-validation folds
        """
        if base_models is None:
            self.base_models = [
                ('svr', SVR(kernel='rbf', C=100, epsilon=0.1)),
            ]

            # Add XGBoost if available
            if XGBOOST_AVAILABLE:
                self.base_models.append(('xgboost', xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )))

            # Add LightGBM if available
            if LIGHTGBM_AVAILABLE:
                self.base_models.append(('lightgbm', lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                )))

            # Always include sklearn models
            self.base_models.extend([
                ('rf', RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )),
                ('gb', GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                ))
            ])
        else:
            self.base_models = base_models

        self.meta_learner = meta_learner or LinearRegression()
        self.cv_folds = cv_folds
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.base_predictions_train = None
        self.base_predictions_test = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnsembleModel':
        """
        Fit ensemble model using stacking.

        Args:
            X: Features
            y: Target values

        Returns:
            Self
        """
        logger.info("Fitting ensemble model...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Use TimeSeriesSplit for temporal validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)

        # Store out-of-fold predictions for meta-learner training
        meta_features = np.zeros((len(X), len(self.base_models)))

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            logger.info(f"Fitting fold {fold_idx + 1}/{self.cv_folds}")

            X_fold_train, X_fold_val = X_scaled[train_idx], X_scaled[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

            for model_idx, (name, model) in enumerate(self.base_models):
                # Fit model on training fold
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X_fold_train, y_fold_train)

                # Predict on validation fold
                val_predictions = model_clone.predict(X_fold_val)
                meta_features[val_idx, model_idx] = val_predictions

        # Train meta-learner on out-of-fold predictions
        self.meta_learner.fit(meta_features, y)

        # Train base models on full dataset
        self.fitted_base_models = []
        for name, model in self.base_models:
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_scaled, y)
            self.fitted_base_models.append((name, model_clone))

        self.is_fitted = True
        logger.info("Ensemble model fitted successfully")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using ensemble.

        Args:
            X: Features

        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)

        # Get predictions from all base models
        base_predictions = np.zeros((len(X), len(self.fitted_base_models)))

        for idx, (name, model) in enumerate(self.fitted_base_models):
            base_predictions[:, idx] = model.predict(X_scaled)

        # Meta-learner makes final prediction
        return self.meta_learner.predict(base_predictions)

    def predict_with_base_models(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions and return individual base model predictions.

        Args:
            X: Features

        Returns:
            DataFrame with ensemble and individual predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)

        predictions = {}

        # Get predictions from all base models
        for name, model in self.fitted_base_models:
            predictions[name] = model.predict(X_scaled)

        # Ensemble prediction
        base_predictions = np.column_stack([predictions[name] for name, _ in self.fitted_base_models])
        predictions['ensemble'] = self.meta_learner.predict(base_predictions)

        return pd.DataFrame(predictions, index=X.index)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate ensemble performance.

        Args:
            X: Features
            y: Target values

        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X)

        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - predictions))
        r2 = r2_score(y, predictions)

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

    def get_feature_importance(self, X: pd.DataFrame) -> pd.Series:
        """
        Get feature importance from tree-based models.

        Args:
            X: Features

        Returns:
            Average feature importance across tree models
        """
        if not self.is_fitted:
            raise ValueError("Ensemble model must be fitted before getting feature importance")

        importance_scores = []

        for name, model in self.fitted_base_models:
            if hasattr(model, 'feature_importances_'):
                importance_scores.append(model.feature_importances_)

        if not importance_scores:
            logger.warning("No tree-based models found for feature importance")
            return pd.Series(index=X.columns, dtype=float)

        # Average importance across models
        avg_importance = np.mean(importance_scores, axis=0)
        return pd.Series(avg_importance, index=X.columns).sort_values(ascending=False)

    def get_model_weights(self) -> pd.Series:
        """
        Get meta-learner coefficients (model weights).

        Returns:
            Series with model weights
        """
        if not self.is_fitted:
            raise ValueError("Ensemble model must be fitted before getting weights")

        if not hasattr(self.meta_learner, 'coef_'):
            logger.warning("Meta-learner doesn't have coefficients")
            return pd.Series(dtype=float)

        model_names = [name for name, _ in self.fitted_base_models]
        weights = self.meta_learner.coef_

        # Add intercept if available
        if hasattr(self.meta_learner, 'intercept_'):
            weights = np.append(weights, self.meta_learner.intercept_)

        return pd.Series(weights, index=model_names)

    @staticmethod
    def create_weighted_ensemble(
        models: List[Tuple[str, object]],
        weights: List[float]
    ) -> 'WeightedEnsemble':
        """
        Create a simple weighted ensemble.

        Args:
            models: List of (name, model) tuples
            weights: Weights for each model

        Returns:
            WeightedEnsemble instance
        """
        return WeightedEnsemble(models, weights)


class WeightedEnsemble:
    """Simple weighted ensemble without stacking."""

    def __init__(self, models: List[Tuple[str, object]], weights: List[float]):
        """
        Initialize weighted ensemble.

        Args:
            models: List of (name, model) tuples
            weights: Weights for each model
        """
        self.models = models
        self.weights = np.array(weights)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'WeightedEnsemble':
        """Fit all models."""
        X_scaled = self.scaler.fit_transform(X)

        for name, model in self.models:
            logger.info(f"Fitting {name}...")
            model.fit(X_scaled, y)

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make weighted predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        X_scaled = self.scaler.transform(X)

        predictions = np.zeros(len(X))
        for (name, model), weight in zip(self.models, self.weights):
            pred = model.predict(X_scaled)
            predictions += weight * pred

        return predictions


def create_financial_ensemble() -> EnsembleModel:
    """
    Create a pre-configured ensemble for financial time series.

    Returns:
        Configured EnsembleModel
    """
    base_models = [
        ('svr_rbf', SVR(kernel='rbf', C=100, epsilon=0.1)),
        ('svr_linear', SVR(kernel='linear', C=10)),
    ]

    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        base_models.append(('xgboost', xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )))

    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        base_models.append(('lightgbm', lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )))

    # Always include sklearn models
    base_models.extend([
        ('rf', RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )),
        ('gb', GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ))
    ])

    return EnsembleModel(base_models=base_models, cv_folds=5)