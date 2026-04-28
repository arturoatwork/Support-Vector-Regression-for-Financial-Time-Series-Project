"""
Data loading and preprocessing module for financial time series.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data acquisition and basic preprocessing."""
    
    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ):
        """
        Initialize DataLoader.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('1d', '1wk', '1mo')
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.data = None
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch historical price data from Yahoo Finance.
        
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching {self.ticker} data from {self.start_date} to {self.end_date}")
        
        try:
            data = yf.download(
                self.ticker,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                progress=False
            )
            
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten YahooFinance multiindex columns into first level labels for single-ticker downloads.
                data.columns = data.columns.get_level_values(0)

            logger.info(f"Downloaded {len(data)} rows")
            self.data = data
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def calculate_returns(
        self,
        data: Optional[pd.DataFrame] = None,
        periods: int = 1
    ) -> pd.Series:
        """
        Calculate daily/periodic returns.
        
        Args:
            data: DataFrame with Close prices
            periods: Number of periods for return calculation
            
        Returns:
            Series of returns
        """
        if data is None:
            data = self.data
            
        returns = data['Close'].pct_change(periods=periods)
        return returns
    
    def prepare_data(
        self,
        dropna: bool = True,
        add_returns: bool = True
    ) -> pd.DataFrame:
        """
        Prepare data for modeling.
        
        Args:
            dropna: Remove NaN values
            add_returns: Add return columns
            
        Returns:
            Cleaned DataFrame
        """
        if self.data is None:
            self.fetch_data()
        
        data = self.data.copy()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if add_returns:
            data['Returns'] = self.calculate_returns(data, periods=1)
            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        if dropna:
            data = data.dropna()
        
        logger.info(f"Data prepared: {len(data)} rows, {len(data.columns)} columns")
        return data
    
    def train_test_split(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.7,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        Preserves temporal order (no shuffling).
        
        Args:
            data: Full dataset
            train_ratio: Training set proportion
            test_ratio: Test set proportion
            val_ratio: Validation set proportion
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        n = len(data)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        train_data = data.iloc[:train_idx]
        val_data = data.iloc[train_idx:val_idx]
        test_data = data.iloc[val_idx:]
        
        logger.info(
            f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
        )
        
        return train_data, val_data, test_data
    
    def fetch_and_prepare(self) -> pd.DataFrame:
        """Convenience method to fetch and prepare data."""
        self.fetch_data()
        return self.prepare_data()
