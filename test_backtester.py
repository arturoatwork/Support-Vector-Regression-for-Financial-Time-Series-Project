"""
Unit tests for backtester functionality.
"""

import unittest
import numpy as np
import pandas as pd
from src.backtester import WalkForwardBacktester, BacktestResults
from src.svr_model import SVRModel


class TestWalkForwardBacktester(unittest.TestCase):
    """Tests for WalkForwardBacktester class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic financial data
        np.random.seed(42)
        n_samples = 500
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

        # Generate realistic price data
        returns = np.random.randn(n_samples) * 0.02  # 2% daily vol
        prices = 100 * np.exp(np.cumsum(returns))  # Geometric Brownian motion

        # Create OHLCV data
        high = prices * (1 + np.abs(np.random.randn(n_samples)) * 0.01)
        low = prices * (1 - np.abs(np.random.randn(n_samples)) * 0.01)
        open_prices = prices * (1 + np.random.randn(n_samples) * 0.005)
        volume = np.random.randint(1000000, 10000000, n_samples)

        self.data = pd.DataFrame({
            'Open': open_prices,
            'High': high,
            'Low': low,
            'Close': prices,
            'Volume': volume,
            'Returns': returns
        }, index=dates)

        # Create features
        self.features = pd.DataFrame({
            'Return_Lag_1': returns,
            'Return_Lag_2': np.roll(returns, 1),
            'Return_Lag_3': np.roll(returns, 2),
            'Price_Lag_1': prices,
            'Volatility': pd.Series(returns).rolling(20).std(),
            'RSI': 50 + np.random.randn(n_samples) * 10,  # Mock RSI
            'MACD': np.random.randn(n_samples) * 0.01,
            'Momentum': prices - np.roll(prices, 10)
        }, index=dates).fillna(0)

        # Create model
        self.model = SVRModel(kernel='linear', C=10)  # Simple model for testing

    def test_backtester_initialization(self):
        """Test backtester initialization."""
        backtester = WalkForwardBacktester(
            model=self.model,
            data=self.data,
            features_df=self.features,
            target_col='Returns',
            train_window=100,
            test_window=20,
            step_size=20
        )

        self.assertEqual(backtester.train_window, 100)
        self.assertEqual(backtester.test_window, 20)
        self.assertEqual(backtester.step_size, 20)
        self.assertEqual(len(backtester.train_indices), 0)
        self.assertEqual(len(backtester.test_indices), 0)

    def test_backtester_run(self):
        """Test backtester run."""
        backtester = WalkForwardBacktester(
            model=self.model,
            data=self.data,
            features_df=self.features,
            target_col='Returns',
            train_window=100,
            test_window=20,
            step_size=50  # Larger step for faster testing
        )

        results = backtester.run()

        self.assertIsInstance(results, BacktestResults)
        self.assertGreater(len(results.predictions), 0)
        self.assertGreater(len(results.actuals), 0)
        self.assertEqual(len(results.predictions), len(results.actuals))

    def test_backtester_with_costs(self):
        """Test backtester with transaction costs."""
        backtester = WalkForwardBacktester(
            model=self.model,
            data=self.data,
            features_df=self.features,
            target_col='Returns',
            train_window=100,
            test_window=20,
            step_size=50
        )

        results_no_cost = backtester.run(transaction_cost=0.0)
        results_with_cost = backtester.run(transaction_cost=0.001)

        # Results should be different with costs
        self.assertNotEqual(
            results_no_cost.summary_stats['Trading']['Total_Return'],
            results_with_cost.summary_stats['Trading']['Total_Return']
        )

    def test_backtest_results_metrics(self):
        """Test backtest results metrics calculation."""
        # Create mock results
        predictions = np.random.randn(100) * 0.01
        actuals = np.random.randn(100) * 0.01
        dates = pd.date_range('2020-01-01', periods=100)
        prices = 100 * np.exp(np.cumsum(actuals))

        results = BacktestResults(
            predictions=predictions,
            actuals=actuals,
            dates=dates,
            prices=prices
        )

        # Test prediction metrics
        pred_metrics = results.calculate_metrics()
        expected_pred_keys = ['MSE', 'RMSE', 'MAE', 'R2']
        for key in expected_pred_keys:
            self.assertIn(key, pred_metrics)
            self.assertIsInstance(pred_metrics[key], (int, float))

        # Test trading metrics
        trading_metrics = results.calculate_trading_metrics()
        expected_trading_keys = [
            'Total_Return', 'Annual_Return', 'Annual_Volatility',
            'Sharpe_Ratio', 'Max_Drawdown', 'Win_Rate'
        ]
        for key in expected_trading_keys:
            self.assertIn(key, trading_metrics)
            self.assertIsInstance(trading_metrics[key], (int, float))

    def test_backtest_results_summary(self):
        """Test backtest results summary."""
        predictions = np.random.randn(100) * 0.01
        actuals = np.random.randn(100) * 0.01
        dates = pd.date_range('2020-01-01', periods=100)
        prices = 100 * np.exp(np.cumsum(actuals))

        results = BacktestResults(
            predictions=predictions,
            actuals=actuals,
            dates=dates,
            prices=prices
        )

        summary = results.summary_stats

        self.assertIn('Prediction', summary)
        self.assertIn('Trading', summary)
        self.assertIn('MSE', summary['Prediction'])
        self.assertIn('Total_Return', summary['Trading'])

    def test_backtester_edge_cases(self):
        """Test backtester edge cases."""
        # Test with very small dataset
        small_data = self.data.iloc[:50]
        small_features = self.features.iloc[:50]

        backtester = WalkForwardBacktester(
            model=self.model,
            data=small_data,
            features_df=small_features,
            target_col='Returns',
            train_window=20,
            test_window=5,
            step_size=10
        )

        results = backtester.run()

        # Should still work even with small dataset
        self.assertIsInstance(results, BacktestResults)

    def test_backtester_different_targets(self):
        """Test backtester with different target columns."""
        # Add a different target
        self.data['Log_Returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['Log_Returns'] = self.data['Log_Returns'].fillna(0)  # Fill NaN with 0

        backtester = WalkForwardBacktester(
            model=self.model,
            data=self.data,
            features_df=self.features,
            target_col='Log_Returns',
            train_window=100,
            test_window=20,
            step_size=50
        )

        results = backtester.run()

        self.assertIsInstance(results, BacktestResults)
        self.assertGreater(len(results.predictions), 0)

    def test_residuals_calculation(self):
        """Test residuals calculation in backtest results."""
        predictions = np.array([0.01, 0.02, -0.01, 0.005])
        actuals = np.array([0.015, 0.018, -0.008, 0.003])
        dates = pd.date_range('2020-01-01', periods=4)

        results = BacktestResults(
            predictions=predictions,
            actuals=actuals,
            dates=dates
        )

        expected_residuals = actuals - predictions
        np.testing.assert_array_equal(results.residuals, expected_residuals)

        # Test errors (absolute residuals)
        expected_errors = np.abs(expected_residuals)
        np.testing.assert_array_equal(results.errors, expected_errors)


class TestBacktestResults(unittest.TestCase):
    """Tests for BacktestResults class."""

    def setUp(self):
        """Set up test fixtures."""
        self.predictions = np.random.randn(100) * 0.01
        self.actuals = np.random.randn(100) * 0.01
        self.dates = pd.date_range('2020-01-01', periods=100)
        self.prices = 100 * np.exp(np.cumsum(self.actuals))

    def test_results_initialization(self):
        """Test BacktestResults initialization."""
        results = BacktestResults(
            predictions=self.predictions,
            actuals=self.actuals,
            dates=self.dates,
            prices=self.prices
        )

        np.testing.assert_array_equal(results.predictions, self.predictions)
        np.testing.assert_array_equal(results.actuals, self.actuals)
        pd.testing.assert_index_equal(results.dates, self.dates)
        np.testing.assert_array_equal(results.prices, self.prices)

        # Check residuals calculation
        expected_residuals = self.actuals - self.predictions
        np.testing.assert_array_equal(results.residuals, expected_residuals)

    def test_results_without_prices(self):
        """Test BacktestResults without price data."""
        results = BacktestResults(
            predictions=self.predictions,
            actuals=self.actuals,
            dates=self.dates,
            prices=None
        )

        self.assertIsNone(results.prices)

    def test_calculate_metrics(self):
        """Test metrics calculation."""
        results = BacktestResults(
            predictions=self.predictions,
            actuals=self.actuals,
            dates=self.dates
        )

        metrics = results.calculate_metrics()

        # Check all expected metrics are present
        expected_keys = ['MSE', 'RMSE', 'MAE', 'R2']
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (int, float, np.floating))
            self.assertFalse(np.isnan(metrics[key]))

    def test_calculate_trading_metrics(self):
        """Test trading metrics calculation."""
        results = BacktestResults(
            predictions=self.predictions,
            actuals=self.actuals,
            dates=self.dates,
            prices=self.prices
        )

        metrics = results.calculate_trading_metrics()

        # Check all expected metrics are present
        expected_keys = [
            'Total_Return', 'Annual_Return', 'Annual_Volatility',
            'Sharpe_Ratio', 'Max_Drawdown', 'Win_Rate', 'Cumulative_Return'
        ]
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (int, float, np.floating))


if __name__ == '__main__':
    unittest.main()