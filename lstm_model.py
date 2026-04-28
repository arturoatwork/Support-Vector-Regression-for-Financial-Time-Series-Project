"""
Deep learning models for SVR quantitative finance project.
Implements LSTM/GRU networks for time series prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class LSTMModel:
    """LSTM model for financial time series prediction."""

    def __init__(
        self,
        sequence_length: int = 20,
        n_features: int = 1,
        n_units: int = 50,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2
    ):
        """
        Initialize LSTM model.

        Args:
            sequence_length: Length of input sequences
            n_features: Number of input features
            n_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            epochs: Maximum training epochs
            batch_size: Training batch size
            validation_split: Validation data proportion
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras not available. Install with: pip install tensorflow")

        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_units = n_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split

        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

        self._build_model()

    def _build_model(self):
        """Build LSTM model architecture."""
        self.model = Sequential([
            LSTM(
                units=self.n_units,
                return_sequences=True,
                input_shape=(self.sequence_length, self.n_features)
            ),
            Dropout(self.dropout_rate),
            LSTM(units=self.n_units // 2, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(units=self.n_units // 4, activation='relu'),
            Dense(units=1)  # Regression output
        ])

        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )

        logger.info(f"LSTM model built with {self.model.count_params()} parameters")

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input.

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_seq, y_seq = [], []

        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = 0) -> 'LSTMModel':
        """
        Fit LSTM model to training data.

        Args:
            X: Features
            y: Target values
            verbose: Verbosity level

        Returns:
            Self
        """
        logger.info("Fitting LSTM model...")

        # Scale data
        X_scaled = self.scaler_X.fit_transform(X.values)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=verbose
        )

        # Train model
        history = self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[early_stopping],
            verbose=verbose
        )

        self.is_fitted = True
        logger.info(f"LSTM model fitted. Final loss: {history.history['loss'][-1]:.6f}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using LSTM model.

        Args:
            X: Features

        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("LSTM model must be fitted before prediction")

        X_scaled = self.scaler_X.transform(X.values)

        # Create sequences for prediction
        X_seq = []
        for i in range(len(X_scaled) - self.sequence_length + 1):
            X_seq.append(X_scaled[i:i + self.sequence_length])

        if not X_seq:
            # Handle case where we don't have enough data for sequences
            # Use the last available sequence
            X_seq = [X_scaled[-self.sequence_length:]]

        X_seq = np.array(X_seq)

        # Make predictions
        predictions_scaled = self.model.predict(X_seq, verbose=0)

        # Inverse transform predictions
        predictions = self.scaler_y.inverse_transform(predictions_scaled).flatten()

        # Pad predictions to match input length
        if len(predictions) < len(X):
            padding = np.full(len(X) - len(predictions), np.nan)
            predictions = np.concatenate([padding, predictions])

        return predictions

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate LSTM model performance.

        Args:
            X: Features
            y: Target values

        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X)

        # Remove NaN predictions for evaluation
        valid_idx = ~np.isnan(predictions)
        y_valid = y.iloc[valid_idx]
        pred_valid = predictions[valid_idx]

        if len(pred_valid) == 0:
            logger.warning("No valid predictions for evaluation")
            return {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}

        mse = mean_squared_error(y_valid, pred_valid)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_valid, pred_valid)
        r2 = r2_score(y_valid, pred_valid)

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }


class GRUModel(LSTMModel):
    """GRU model for financial time series prediction."""

    def _build_model(self):
        """Build GRU model architecture."""
        self.model = Sequential([
            GRU(
                units=self.n_units,
                return_sequences=True,
                input_shape=(self.sequence_length, self.n_features)
            ),
            Dropout(self.dropout_rate),
            GRU(units=self.n_units // 2, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(units=self.n_units // 4, activation='relu'),
            Dense(units=1)  # Regression output
        ])

        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )

        logger.info(f"GRU model built with {self.model.count_params()} parameters")


class BidirectionalLSTMModel(LSTMModel):
    """Bidirectional LSTM model for financial time series prediction."""

    def _build_model(self):
        """Build Bidirectional LSTM model architecture."""
        self.model = Sequential([
            Bidirectional(LSTM(
                units=self.n_units,
                return_sequences=True,
                input_shape=(self.sequence_length, self.n_features)
            )),
            Dropout(self.dropout_rate),
            Bidirectional(LSTM(units=self.n_units // 2, return_sequences=False)),
            Dropout(self.dropout_rate),
            Dense(units=self.n_units // 4, activation='relu'),
            Dense(units=1)  # Regression output
        ])

        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )

        logger.info(f"Bidirectional LSTM model built with {self.model.count_params()} parameters")


class DeepEnsemble:
    """Ensemble of deep learning models."""

    def __init__(self, models: Optional[List] = None):
        """
        Initialize deep ensemble.

        Args:
            models: List of deep learning model instances
        """
        if models is None:
            self.models = [
                LSTMModel(sequence_length=20, n_units=50),
                GRUModel(sequence_length=20, n_units=50),
                BidirectionalLSTMModel(sequence_length=20, n_units=50)
            ]
        else:
            self.models = models

        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'DeepEnsemble':
        """Fit all deep learning models."""
        logger.info("Fitting deep ensemble...")

        for i, model in enumerate(self.models):
            logger.info(f"Fitting model {i+1}/{len(self.models)}: {type(model).__name__}")
            model.fit(X, y, verbose=0)

        self.is_fitted = True
        logger.info("Deep ensemble fitted")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Deep ensemble must be fitted before prediction")

        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate deep ensemble."""
        predictions = self.predict(X)

        # Remove NaN predictions
        valid_idx = ~np.isnan(predictions)
        y_valid = y.iloc[valid_idx]
        pred_valid = predictions[valid_idx]

        if len(pred_valid) == 0:
            return {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}

        mse = mean_squared_error(y_valid, pred_valid)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_valid, pred_valid)
        r2 = r2_score(y_valid, pred_valid)

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }


def create_deep_models() -> Dict[str, Any]:
    """
    Create a collection of deep learning models.

    Returns:
        Dictionary of model instances
    """
    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow not available, returning empty dict")
        return {}

    return {
        'lstm': LSTMModel(sequence_length=20, n_units=64, epochs=50),
        'gru': GRUModel(sequence_length=20, n_units=64, epochs=50),
        'bidirectional_lstm': BidirectionalLSTMModel(sequence_length=20, n_units=64, epochs=50),
        'deep_ensemble': DeepEnsemble()
    }