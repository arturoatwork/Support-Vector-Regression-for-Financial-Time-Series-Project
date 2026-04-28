"""
Unit tests for SVR model functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.svr_model import SVRModel


class TestSVRModel(unittest.TestCase):
    """Tests for SVRModel class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic data
        np.random.seed(42)
        n_samples = 200
        n_features = 5

        # Generate features
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )

        # Generate target with some relationship to features
        y = X.iloc[:, 0] * 2 + X.iloc[:, 1] * -1 + np.random.randn(n_samples) * 0.1
        y = pd.Series(y, name='target')

        self.X = X
        self.y = y

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

    def test_model_initialization(self):
        """Test SVR model initialization."""
        model = SVRModel(kernel='rbf', C=100, epsilon=0.1)
        self.assertEqual(model.kernel, 'rbf')
        self.assertEqual(model.C, 100)
        self.assertEqual(model.epsilon, 0.1)
        self.assertFalse(model.is_fitted)

    def test_model_fit(self):
        """Test model fitting."""
        model = SVRModel()
        model.fit(self.X_train, self.y_train)

        self.assertTrue(model.is_fitted)
        self.assertIsNotNone(model.model)

    def test_model_predict(self):
        """Test model prediction."""
        model = SVRModel()
        model.fit(self.X_train, self.y_train)

        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))

        # Predictions should be reasonable (not all NaN or extreme values)
        self.assertFalse(np.isnan(predictions).all())
        self.assertTrue(np.isfinite(predictions).all())

    def test_model_evaluate(self):
        """Test model evaluation."""
        model = SVRModel()
        model.fit(self.X_train, self.y_train)

        metrics = model.evaluate(self.X_test, self.y_test)

        # Check that all expected metrics are present
        expected_metrics = ['MSE', 'RMSE', 'MAE', 'R2']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertFalse(np.isnan(metrics[metric]))

        # R2 should be reasonable for synthetic data
        self.assertGreater(metrics['R2'], -1.0)
        self.assertLess(metrics['R2'], 1.0)

    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning."""
        param_grid = {
            'C': [10, 100],
            'epsilon': [0.01, 0.1]
        }

        best_model, best_params = SVRModel.hyperparameter_tuning(
            self.X_train, self.y_train, param_grid, cv_splits=3
        )

        self.assertIsNotNone(best_model)
        self.assertIn('C', best_params)
        self.assertIn('epsilon', best_params)

    def test_feature_importance(self):
        """Test feature importance calculation."""
        # Use a linear kernel for more interpretable results
        model = SVRModel(kernel='linear')
        model.fit(self.X_train, self.y_train)

        importance = model.feature_importance(self.X_test, self.y_test, n_repeats=3)

        self.assertIsInstance(importance, pd.Series)
        self.assertEqual(len(importance), len(self.X_test.columns))
        self.assertEqual(importance.index.name, None)  # Should have column names

    def test_model_info(self):
        """Test model info retrieval."""
        model = SVRModel()
        model.fit(self.X_train, self.y_train)

        info = model.get_model_info()

        expected_keys = ['kernel', 'C', 'epsilon', 'gamma', 'n_support_vectors', 'is_fitted']
        for key in expected_keys:
            self.assertIn(key, info)

        self.assertTrue(info['is_fitted'])
        self.assertIsInstance(info['n_support_vectors'], int)

    def test_unfitted_model_predict(self):
        """Test that unfitted model raises error on predict."""
        model = SVRModel()

        with self.assertRaises(ValueError):
            model.predict(self.X_test)

    def test_unfitted_model_evaluate(self):
        """Test that unfitted model raises error on evaluate."""
        model = SVRModel()

        with self.assertRaises(ValueError):
            model.evaluate(self.X_test, self.y_test)

    def test_unfitted_model_importance(self):
        """Test that unfitted model raises error on feature importance."""
        model = SVRModel()

        with self.assertRaises(ValueError):
            model.feature_importance(self.X_test, self.y_test)


if __name__ == '__main__':
    unittest.main()