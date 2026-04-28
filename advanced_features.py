"""
Advanced feature engineering for SVR quantitative finance project.
Includes market microstructure, regime detection, GARCH volatility, and correlation features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer:
    """Advanced feature engineering for financial time series."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize AdvancedFeatureEngineer.

        Args:
            data: DataFrame with OHLCV columns
        """
        self.data = data.copy()
        self.features = None

    def add_market_microstructure_features(self) -> pd.DataFrame:
        """
        Add market microstructure features.

        Returns:
            DataFrame with microstructure features
        """
        df = pd.DataFrame(index=self.data.index)

        # Realized volatility (daily)
        df['Realized_Volatility'] = np.log(self.data['High'] / self.data['Low']) ** 2

        # Price impact proxy (volume-weighted price change)
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        df['VWAP'] = (typical_price * self.data['Volume']).rolling(20).sum() / \
                     self.data['Volume'].rolling(20).sum()

        # Bid-ask spread proxy (high-low range)
        df['Spread_Proxy'] = (self.data['High'] - self.data['Low']) / self.data['Close']

        # Order flow imbalance proxy
        df['Order_Flow'] = (self.data['Close'] - self.data['Open']) / (self.data['High'] - self.data['Low'])

        # Volume-based features
        df['Volume_SMA'] = self.data['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = self.data['Volume'] / df['Volume_SMA']
        df['Volume_Change'] = self.data['Volume'].pct_change()

        # Price-volume correlation
        df['Price_Volume_Corr'] = self.data['Close'].rolling(20).corr(self.data['Volume'])

        return df

    def add_regime_detection_features(self) -> pd.DataFrame:
        """
        Add market regime detection features.

        Returns:
            DataFrame with regime features
        """
        df = pd.DataFrame(index=self.data.index)

        # Volatility regime (rolling volatility clusters)
        returns = self.data['Close'].pct_change()
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std()

        # Simple regime: High vol vs Low vol
        df['Vol_Regime'] = (vol_20 > vol_20.rolling(60).mean()).astype(int)

        # Trend regime (momentum-based)
        momentum_20 = self.data['Close'] - self.data['Close'].shift(20)
        momentum_60 = self.data['Close'] - self.data['Close'].shift(60)
        df['Trend_Regime'] = (momentum_20 > 0).astype(int)

        # Mean-reversion regime
        z_score = (self.data['Close'] - self.data['Close'].rolling(20).mean()) / \
                  self.data['Close'].rolling(20).std()
        df['MR_Regime'] = (abs(z_score) > 2).astype(int)

        # Volatility trend
        df['Vol_Trend'] = vol_20 - vol_60

        return df

    def add_garch_volatility_features(self) -> pd.DataFrame:
        """
        Add GARCH-style volatility features.

        Returns:
            DataFrame with GARCH-like features
        """
        df = pd.DataFrame(index=self.data.index)

        returns = self.data['Close'].pct_change().fillna(0)

        # EGARCH-like features (exponential GARCH)
        sigma2 = returns.rolling(20).var()
        df['EGARCH_Vol'] = np.sqrt(sigma2)

        # Asymmetric volatility (leverage effect)
        neg_returns = returns.where(returns < 0, 0)
        df['Asym_Vol'] = neg_returns.rolling(20).std()

        # Volatility clustering proxy
        df['Vol_Clustering'] = abs(returns) - abs(returns).rolling(20).mean()

        # Long-term volatility
        df['Long_Vol'] = returns.rolling(60).std()

        # Volatility ratio
        df['Vol_Ratio'] = df['EGARCH_Vol'] / df['Long_Vol']

        return df

    def add_correlation_features(self, external_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Add correlation-based features.

        Args:
            external_data: Additional market data for correlations

        Returns:
            DataFrame with correlation features
        """
        df = pd.DataFrame(index=self.data.index)

        returns = self.data['Close'].pct_change()

        # Rolling correlations with lagged returns
        for lag in [1, 5, 10, 20]:
            df[f'Corr_Lag_{lag}'] = returns.rolling(20).corr(returns.shift(lag))

        # Autocorrelation features
        for lag in [1, 2, 3, 5]:
            df[f'AutoCorr_{lag}'] = returns.rolling(20).apply(
                lambda x: x.autocorr(lag=lag) if len(x.dropna()) > lag else np.nan
            )

        # Cross-sectional features (if external data provided)
        if external_data is not None:
            # Assume external_data has similar structure
            for col in external_data.columns:
                if col != 'Close':
                    ext_returns = external_data[col].pct_change()
                    df[f'Cross_Corr_{col}'] = returns.rolling(20).corr(ext_returns)

        return df

    def add_statistical_features(self) -> pd.DataFrame:
        """
        Add statistical moment features.

        Returns:
            DataFrame with statistical features
        """
        df = pd.DataFrame(index=self.data.index)

        returns = self.data['Close'].pct_change()

        # Higher moments
        df['Skewness'] = returns.rolling(20).skew()
        df['Kurtosis'] = returns.rolling(20).kurt()

        # Quantile features
        df['Q25'] = returns.rolling(20).quantile(0.25)
        df['Q75'] = returns.rolling(20).quantile(0.75)
        df['IQR'] = df['Q75'] - df['Q25']

        # Outlier detection
        z_score = (returns - returns.rolling(20).mean()) / returns.rolling(20).std()
        df['Z_Score'] = z_score
        df['Outlier_Flag'] = (abs(z_score) > 3).astype(int)

        # Entropy proxy (price dispersion)
        price_range = (self.data['High'] - self.data['Low']) / self.data['Close']
        df['Price_Entropy'] = -price_range.rolling(20).apply(
            lambda x: stats.entropy(np.histogram(x.dropna(), bins=10)[0] + 1e-10)
        )

        return df

    def add_ichimoku_features(self) -> pd.DataFrame:
        """
        Add Ichimoku Cloud features.

        Returns:
            DataFrame with Ichimoku features
        """
        df = pd.DataFrame(index=self.data.index)

        # Tenkan-sen (Conversion Line)
        df['Tenkan_Sen'] = (self.data['High'].rolling(9).max() + self.data['Low'].rolling(9).min()) / 2

        # Kijun-sen (Base Line)
        df['Kijun_Sen'] = (self.data['High'].rolling(26).max() + self.data['Low'].rolling(26).min()) / 2

        # Senkou Span A (Leading Span A)
        df['Senkou_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(26)

        # Senkou Span B (Leading Span B)
        df['Senkou_B'] = ((self.data['High'].rolling(52).max() + self.data['Low'].rolling(52).min()) / 2).shift(26)

        # Chikou Span (Lagging Span)
        df['Chikou'] = self.data['Close'].shift(-26)

        # Cloud signals
        df['Cloud_Green'] = (df['Senkou_A'] > df['Senkou_B']).astype(int)
        df['Cloud_Red'] = (df['Senkou_A'] < df['Senkou_B']).astype(int)

        # TK Cross
        df['TK_Cross_Up'] = ((df['Tenkan_Sen'] > df['Kijun_Sen']) &
                            (df['Tenkan_Sen'].shift(1) <= df['Kijun_Sen'].shift(1))).astype(int)
        df['TK_Cross_Down'] = ((df['Tenkan_Sen'] < df['Kijun_Sen']) &
                              (df['Tenkan_Sen'].shift(1) >= df['Kijun_Sen'].shift(1))).astype(int)

        return df

    def create_all_advanced_features(
        self,
        external_data: Optional[pd.DataFrame] = None,
        include_ichimoku: bool = True
    ) -> pd.DataFrame:
        """
        Create comprehensive advanced feature set.

        Args:
            external_data: Additional market data for correlations
            include_ichimoku: Whether to include Ichimoku features

        Returns:
            DataFrame with all advanced features
        """
        logger.info("Creating advanced features...")

        features_df = pd.DataFrame(index=self.data.index)

        # Add all feature categories
        features_df = features_df.join(self.add_market_microstructure_features())
        features_df = features_df.join(self.add_regime_detection_features())
        features_df = features_df.join(self.add_garch_volatility_features())
        features_df = features_df.join(self.add_correlation_features(external_data))
        features_df = features_df.join(self.add_statistical_features())

        if include_ichimoku:
            features_df = features_df.join(self.add_ichimoku_features())

        # Remove NaN values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)

        logger.info(f"Created {len(features_df.columns)} advanced features")
        logger.info(f"Features: {list(features_df.columns)}")

        return features_df

    @staticmethod
    def detect_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> pd.Series:
        """
        Detect outliers using IQR method.

        Args:
            data: Input series
            multiplier: IQR multiplier

        Returns:
            Boolean series indicating outliers
        """
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        return (data < lower_bound) | (data > upper_bound)

    @staticmethod
    def remove_outliers_zscore(data: pd.Series, threshold: float = 3.0) -> pd.Series:
        """
        Remove outliers using Z-score method.

        Args:
            data: Input series
            threshold: Z-score threshold

        Returns:
            Series with outliers removed (set to NaN)
        """
        z_scores = np.abs(stats.zscore(data.dropna()))
        return data.where(z_scores < threshold, np.nan)

    @staticmethod
    def winsorize_series(data: pd.Series, limits: Tuple[float, float] = (0.05, 0.05)) -> pd.Series:
        """
        Winsorize series to limit extreme values.

        Args:
            data: Input series
            limits: Tuple of (lower_limit, upper_limit) as percentiles

        Returns:
            Winsorized series
        """
        return stats.mstats.winsorize(data, limits=limits)