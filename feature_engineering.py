"""
Feature engineering module for technical indicators and derived features.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Generate technical indicators and features for machine learning."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize FeatureEngineer.
        
        Args:
            data: DataFrame with OHLCV columns
        """
        self.data = data.copy()
        self.features = None
    
    def rsi(self, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD and Signal line."""
        ema_fast = self.data['Close'].ewm(span=fast).mean()
        ema_slow = self.data['Close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        })
    
    def bollinger_bands(self, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """Bollinger Bands."""
        sma = self.data['Close'].rolling(window=period).mean()
        std = self.data['Close'].rolling(window=period).std()
        
        upper_band = sma + (num_std * std)
        lower_band = sma - (num_std * std)
        
        bb_position = (self.data['Close'] - lower_band) / (upper_band - lower_band)
        
        return pd.DataFrame({
            'BB_Upper': upper_band,
            'BB_Middle': sma,
            'BB_Lower': lower_band,
            'BB_Position': bb_position
        })
    
    def atr(self, period: int = 14) -> pd.Series:
        """Average True Range for volatility."""
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def momentum(self, period: int = 10) -> pd.Series:
        """Price momentum."""
        return self.data['Close'] - self.data['Close'].shift(period)
    
    def rolling_volatility(self, period: int = 20) -> pd.Series:
        """Rolling standard deviation of returns."""
        returns = self.data['Close'].pct_change()
        return returns.rolling(window=period).std() * np.sqrt(252)

    def trend_regime(self, period: int = 20) -> pd.Series:
        """Trend regime based on the close vs the moving average."""
        sma = self.data['Close'].rolling(window=period).mean()
        return (self.data['Close'] > sma).astype(int)

    def volatility_percentile(self, vol: pd.Series, rank_window: int = 60) -> pd.Series:
        """Percentile rank of volatility within a rolling window."""
        return vol.rolling(rank_window).apply(
            lambda x: (x.iloc[-1] >= x).sum() / len(x) * 100,
            raw=False
        )

    def parkinson_volatility(self, period: int = 20) -> pd.Series:
        """Parkinson volatility estimator using the high-low range."""
        log_hl = np.log(self.data['High'] / self.data['Low'])
        parkinson = np.sqrt((1.0 / (4.0 * np.log(2.0))) * log_hl.pow(2).rolling(window=period).mean())
        return parkinson * np.sqrt(252)

    def garman_klass_volatility(self, period: int = 20) -> pd.Series:
        """Garman-Klass volatility estimator using open, high, low, close."""
        log_hl = np.log(self.data['High'] / self.data['Low'])
        log_co = np.log(self.data['Close'] / self.data['Open'])
        gk = 0.5 * log_hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)
        gk = gk.rolling(window=period).mean().clip(lower=0)
        return np.sqrt(gk) * np.sqrt(252)

    def atr_percentile(self, atr_period: int = 14, rank_window: int = 60) -> pd.Series:
        """ATR as a percentile of recent ATR values."""
        atr_series = self.atr(atr_period)
        return self.volatility_percentile(atr_series, rank_window=rank_window)

    def volume_weighted_avg(self, period: int = 20) -> pd.Series:
        """Volume-weighted average price."""
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        vwap = (typical_price * self.data['Volume']).rolling(period).sum() / \
               self.data['Volume'].rolling(period).sum()
        return vwap
    
    def create_lagged_features(self, lags: int = 5) -> pd.DataFrame:
        """Create lagged returns and prices."""
        lagged_df = pd.DataFrame(index=self.data.index)
        
        returns = self.data['Close'].pct_change()
        
        for i in range(1, lags + 1):
            lagged_df[f'Return_Lag_{i}'] = returns.shift(i)
            lagged_df[f'Price_Lag_{i}'] = self.data['Close'].shift(i)
        
        return lagged_df
    
    def create_all_features(
        self,
        lags: int = 5,
        rsi_period: int = 14,
        bb_period: int = 20,
        atr_period: int = 14,
        momentum_period: int = 10,
        vol_period: int = 20
    ) -> pd.DataFrame:
        """
        Create comprehensive feature set.
        
        Args:
            lags: Number of lags for lagged features
            rsi_period: RSI period
            bb_period: Bollinger Bands period
            atr_period: ATR period
            momentum_period: Momentum period
            vol_period: Volatility period
            
        Returns:
            DataFrame with all engineered features
        """
        features = pd.DataFrame(index=self.data.index)

        # Technical indicators
        features['RSI'] = self.rsi(rsi_period)

        macd_df = self.macd()
        features = features.join(macd_df)

        bb_df = self.bollinger_bands(bb_period)
        features = features.join(bb_df)

        features['ATR'] = self.atr(atr_period)
        features['Momentum'] = self.momentum(momentum_period)
        features['Volatility'] = self.rolling_volatility(vol_period)
        features['VWAP'] = self.volume_weighted_avg(vol_period)

        # Multi-timeframe volatility
        features['vol_10d'] = self.data['Close'].pct_change().rolling(window=10).std() * np.sqrt(252)
        features['vol_30d'] = self.data['Close'].pct_change().rolling(window=30).std() * np.sqrt(252)
        features['vol_60d'] = self.data['Close'].pct_change().rolling(window=60).std() * np.sqrt(252)

        # Volatility metrics
        features['Volatility_Pct'] = self.volatility_percentile(features['Volatility'], rank_window=60)
        features['Parkinson_Vol'] = self.parkinson_volatility(vol_period)
        features['GK_Volatility'] = self.garman_klass_volatility(vol_period)
        features['ATR_Pct'] = self.atr(atr_period) / self.data['Close']

        # Trend regime and interactions
        features['Trend_Regime'] = self.trend_regime(20)
        features['Momentum_x_Volatility'] = features['Momentum'] * features['Volatility']
        features['RSI_x_Volume_Ratio'] = features['RSI'] * (self.data['Volume'] / self.data['Volume'].rolling(20).mean())
        features['Momentum_x_Trend_Regime'] = features['Momentum'] * features['Trend_Regime']
        features['RSI_x_Volatility'] = features['RSI'] * features['Volatility']
        features['RSI_x_Volume_Change'] = features['RSI'] * self.data['Volume'].pct_change()

        # Lagged features
        lagged = self.create_lagged_features(lags)
        features = features.join(lagged)

        # Non-linear transformations
        features['Momentum_Sq'] = features['Momentum'] ** 2
        features['Volatility_Sq'] = features['Volatility'] ** 2
        features['Return_Lag_1_Cubed'] = features['Return_Lag_1'] ** 3
        features['RSI_Momentum_Volatility'] = (features['RSI'] - 50) * features['Momentum'] / (features['Volatility'] + 1e-10)
        features['Momentum_Over_Volatility'] = features['Momentum'] / (features['Volatility'] + 1e-10)
        features['RSI_over_1_plus_Volatility'] = features['RSI'] / (1 + features['Volatility'])

        # Volume features
        features['Volume_SMA'] = self.data['Volume'].rolling(20).mean()
        features['Volume_Ratio'] = self.data['Volume'] / features['Volume_SMA']

        # Returns
        features['Returns'] = self.data['Close'].pct_change()
        features['Log_Returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))

        logger.info(f"Created {len(features.columns)} features")

        return features
    
    @staticmethod
    def normalize_features(X: pd.DataFrame) -> pd.DataFrame:
        """Normalize features to zero mean and unit variance."""
        return (X - X.mean()) / X.std()
