"""
Unit tests for data loading functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import pandas as pd
from src.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """Tests for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = DataLoader(
            ticker='AAPL',
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
    
    def test_fetch_data(self):
        """Test data fetching."""
        data = self.loader.fetch_data()
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)
        self.assertIn('Close', data.columns)
    
    def test_calculate_returns(self):
        """Test returns calculation."""
        self.loader.fetch_data()
        returns = self.loader.calculate_returns(periods=1)
        self.assertEqual(len(returns), len(self.loader.data))
        self.assertGreater(len(returns.dropna()), 0)  # Should have some valid returns
        self.assertLess(len(returns.dropna()), len(returns))  # First value should be NaN


if __name__ == '__main__':
    unittest.main()
