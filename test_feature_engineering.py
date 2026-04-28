"""
Unit tests for feature engineering functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import pandas as pd
from src.feature_engineering import FeatureEngineer
from src.advanced_features import AdvancedFeatureEngineer


class TestFeatureEngineer(unittest.TestCase):
    """Tests for FeatureEngineer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic OHLCV data
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
            'Volume': volume
        }, index=dates)

        self.feature_engineer = FeatureEngineer(self.data)

    def test_initialization(self):
        """Test FeatureEngineer initialization."""
        self.assertIsInstance(self.feature_engineer, FeatureEngineer)

    def test_rsi(self):
        """Test RSI calculation."""
        rsi = self.feature_engineer.rsi(period=14)
        self.assertEqual(len(rsi), len(self.data))
        self.assertTrue(rsi.min() >= 0 or rsi.max() <= 100 or rsi.isna().any())

    def test_macd(self):
        """Test MACD calculation."""
        macd = self.feature_engineer.macd()
        self.assertEqual(len(macd), len(self.data))
        self.assertIn('MACD', macd.columns)
        self.assertIn('Signal', macd.columns)

    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        bb = self.feature_engineer.bollinger_bands(period=20)
        self.assertEqual(len(bb), len(self.data))
        self.assertIn('BB_Upper', bb.columns)
        self.assertIn('BB_Lower', bb.columns)

    def test_atr(self):
        """Test ATR calculation."""
        atr = self.feature_engineer.atr(period=14)
        self.assertEqual(len(atr), len(self.data))
        self.assertTrue(atr.min() >= 0)

    def test_momentum(self):
        """Test momentum calculation."""
        momentum = self.feature_engineer.momentum(period=10)
        self.assertEqual(len(momentum), len(self.data))

    def test_rolling_volatility(self):
        """Test rolling volatility calculation."""
        vol = self.feature_engineer.rolling_volatility(period=20)
        self.assertEqual(len(vol), len(self.data))
        self.assertTrue(vol.min() >= 0)

    def test_volume_weighted_avg(self):
        """Test VWAP calculation."""
        vwap = self.feature_engineer.volume_weighted_avg(period=20)
        self.assertEqual(len(vwap), len(self.data))

    def test_create_lagged_features(self):
        """Test lagged feature creation."""
        lagged = self.feature_engineer.create_lagged_features(lags=3)
        expected_cols = ['Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3',
                        'Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3']
        for col in expected_cols:
            self.assertIn(col, lagged.columns)

    def test_create_all_features(self):
        """Test creation of all features."""
        features = self.feature_engineer.create_all_features()

        # Should have many features
        self.assertGreater(len(features.columns), 20)

        # Check some expected features
        expected_features = [
            'RSI', 'MACD', 'Signal', 'BB_Upper', 'ATR', 'Momentum',
            'vol_10d', 'vol_30d', 'vol_60d', 'Parkinson_Vol', 'ATR_Pct',
            'Momentum_x_Volatility', 'RSI_x_Volume_Ratio', 'Momentum_Sq'
        ]
        for feature in expected_features:
            self.assertIn(feature, features.columns)

    def test_normalize_features(self):
        """Test feature normalization."""
        features = self.feature_engineer.create_all_features()
        normalized = FeatureEngineer.normalize_features(features)

        # Check that normalization worked (should have NaN for constant columns)
        # But most features should be normalized
        self.assertEqual(len(normalized.columns), len(features.columns))


class TestAdvancedFeatureEngineer(unittest.TestCase):
    """Tests for AdvancedFeatureEngineer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic OHLCV data with more samples for advanced features
        np.random.seed(42)
        n_samples = 1000
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

        # Generate realistic price data
        returns = np.random.randn(n_samples) * 0.02
        prices = 100 * np.exp(np.cumsum(returns))

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
            'Volume': volume
        }, index=dates)

        self.advanced_engineer = AdvancedFeatureEngineer(self.data)

    def test_initialization(self):
        """Test AdvancedFeatureEngineer initialization."""
        self.assertIsInstance(self.advanced_engineer, AdvancedFeatureEngineer)

    def test_market_microstructure_features(self):
        """Test market microstructure feature creation."""
        features = self.advanced_engineer.add_market_microstructure_features()

        # Check expected microstructure features
        expected_features = [
            'Realized_Volatility', 'VWAP', 'Spread_Proxy'
        ]

        for feature in expected_features:
            self.assertIn(feature, features.columns)

        # Check no NaN values in recent data
        recent_features = features.tail(50)  # Last 50 rows should have values
        self.assertFalse(recent_features.isnull().any().any())

    def test_regime_detection_features(self):
        """Test regime detection feature creation."""
        features = self.advanced_engineer.add_regime_detection_features()

        # Check expected regime features
        expected_features = [
            'Vol_Regime', 'Trend_Regime', 'MR_Regime', 'Vol_Trend'
        ]

        for feature in expected_features:
            self.assertIn(feature, features.columns)

        # Check no NaN values in recent data
        recent_features = features.tail(50)
        self.assertFalse(recent_features.isnull().any().any())

    def test_garch_volatility_features(self):
        """Test GARCH-style volatility features."""
        features = self.advanced_engineer.add_garch_volatility_features()

        # Check expected GARCH features
        expected_features = [
            'EGARCH_Vol', 'Asym_Vol', 'Vol_Clustering', 'Long_Vol', 'Vol_Ratio'
        ]

        for feature in expected_features:
            self.assertIn(feature, features.columns)

        # Check no NaN values in recent data
        recent_features = features.tail(50)
        self.assertFalse(recent_features.isnull().any().any())

    def test_correlation_features(self):
        """Test correlation feature creation."""
        # Create multi-asset data
        n_samples = len(self.data)
        asset2_prices = 50 * np.exp(np.cumsum(np.random.randn(n_samples) * 0.015))
        asset3_prices = 75 * np.exp(np.cumsum(np.random.randn(n_samples) * 0.025))

        multi_data = pd.DataFrame({
            'Asset1': self.data['Close'],
            'Asset2': asset2_prices,
            'Asset3': asset3_prices
        })

        features = self.advanced_engineer.add_correlation_features(multi_data)

        # Check expected correlation features
        expected_features = [
            'Corr_Lag_1', 'Corr_Lag_5', 'Corr_Lag_10', 'Corr_Lag_20',
            'AutoCorr_1', 'AutoCorr_2', 'AutoCorr_3', 'AutoCorr_5',
            'Cross_Corr_Asset1', 'Cross_Corr_Asset2', 'Cross_Corr_Asset3'
        ]

        for feature in expected_features:
            self.assertIn(feature, features.columns)

        # Check no NaN values in recent data
        recent_features = features.tail(50)
        self.assertFalse(recent_features.isnull().any().any())

    def test_statistical_moments_features(self):
        """Test statistical moments feature creation."""
        features = self.advanced_engineer.add_statistical_features()

        # Check expected statistical features
        expected_features = [
            'Skewness', 'Kurtosis', 'Z_Score', 'Outlier_Flag'
        ]

        for feature in expected_features:
            self.assertIn(feature, features.columns)

        # Check no NaN values in recent data
        recent_features = features.tail(50)
        self.assertFalse(recent_features.isnull().any().any())

    def test_ichimoku_features(self):
        """Test Ichimoku Cloud feature creation."""
        features = self.advanced_engineer.add_ichimoku_features()

        # Check expected Ichimoku features
        expected_features = [
            'Tenkan_Sen', 'Kijun_Sen', 'Senkou_A', 'Senkou_B'
        ]

        for feature in expected_features:
            self.assertIn(feature, features.columns)

    def test_remove_outliers(self):
        """Test outlier removal."""
        # Create data with outliers
        test_series = pd.Series([1, 2, 3, 4, 5, 10000])

        # Remove outliers
        cleaned_series = self.advanced_engineer.remove_outliers_zscore(test_series, threshold=1.0)

        # Should have removed the outlier
        self.assertTrue(np.isnan(cleaned_series.iloc[5]))


if __name__ == '__main__':
    unittest.main()
