"""
Walk-forward backtesting engine for model validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """Walk-forward backtesting with proper temporal validation."""
    
    def __init__(
        self,
        model,
        data: pd.DataFrame,
        features_df: pd.DataFrame,
        target_col: str = 'Returns',
        target_series: pd.Series = None,
        train_window: int = 252 * 2,  # 2 years
        test_window: int = 63,  # 3 months
        step_size: int = 63
    ):
        """
        Initialize backtester.
        
        Args:
            model: Model with fit() and predict() methods
            data: Price data
            features_df: Feature dataframe
            target_col: Target column name
            target_series: Optional aligned target series for prediction
            train_window: Training set size (trading days)
            test_window: Test set size (trading days)
            step_size: Step size for rolling window
        """
        self.model = model
        self.data = data
        self.features = features_df
        self.target_col = target_col
        self.target_series = target_series
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        
        self.predictions = []
        self.test_periods = []
        self.train_indices = []
        self.test_indices = []
    
    def run(
        self,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        confidence_threshold: float = 0.7,
        target_vol: float = 0.15,
        max_position: float = 1.0
    ) -> 'BacktestResults':
        """
        Run walk-forward backtest.
        
        Args:
            transaction_cost: Transaction cost (bps)
            slippage: Slippage (bps)
            confidence_threshold: Confidence threshold for signals
            target_vol: Target annualized volatility for sizing
            max_position: Maximum allowed position size
            
        Returns:
            BacktestResults object
        """
        logger.info("Starting walk-forward backtest...")
        
        valid_idx = self.features.dropna().index
        n_samples = len(valid_idx)
        
        all_predictions = []
        all_actuals = []
        all_dates = []
        
        # Walk-forward loop
        for i in range(0, n_samples - self.train_window - self.test_window, self.step_size):
            train_end = i + self.train_window
            test_start = train_end
            test_end = test_start + self.test_window
            
            if test_end > n_samples:
                break
            
            train_idx = valid_idx[i:train_end]
            test_idx = valid_idx[test_start:test_end]
            
            X_train = self.features.loc[train_idx].dropna()
            y_series = self.target_series if self.target_series is not None else self.data[self.target_col]
            y_train = y_series.loc[train_idx[len(train_idx)-len(X_train):]]
            
            X_test = self.features.loc[test_idx].dropna()
            y_test = y_series.loc[test_idx[len(test_idx)-len(X_test):]]
            
            # Fit on training period
            self.model.fit(X_train, y_train)
            
            # Predict on test period
            predictions = self.model.predict(X_test)
            
            all_predictions.extend(predictions)
            all_actuals.extend(y_test.values)
            all_dates.extend(y_test.index)
            
            logger.info(f"Fold {len(self.train_indices)+1}: Train {train_idx[0]} to {train_idx[-1]}, "
                       f"Test {test_idx[0]} to {test_idx[-1]}")
            
            self.train_indices.append(train_idx)
            self.test_indices.append(test_idx)
        
        results = BacktestResults(
            predictions=np.array(all_predictions),
            actuals=np.array(all_actuals),
            dates=pd.DatetimeIndex(all_dates),
            prices=self.data.loc[all_dates, 'Close'].values if 'Close' in self.data.columns else None,
            transaction_cost=transaction_cost,
            slippage=slippage,
            confidence_threshold=confidence_threshold,
            target_vol=target_vol,
            max_position=max_position
        )
        
        logger.info(f"Backtest completed with {len(self.train_indices)} folds")
        
        return results


class BacktestResults:
    """Container for backtest results and performance metrics."""
    
    def __init__(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        dates: pd.DatetimeIndex,
        prices: np.ndarray = None,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        confidence_threshold: float = 0.7,
        target_vol: float = 0.15,
        max_position: float = 1.0
    ):
        """
        Initialize results.
        
        Args:
            predictions: Model predictions
            actuals: Actual values
            dates: Date index
            prices: Close prices
            transaction_cost: Transaction cost
            slippage: Slippage
            confidence_threshold: Confidence threshold for trade filtering
            target_vol: Target annualized volatility for position sizing
            max_position: Maximum position size
        """
        self.predictions = predictions
        self.actuals = actuals
        self.dates = dates
        self.prices = prices
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.confidence_threshold = confidence_threshold
        self.target_vol = target_vol
        self.max_position = max_position

        self.residuals = actuals - predictions
        self.errors = np.abs(self.residuals)
    
    def calculate_metrics(self) -> Dict:
        """Calculate prediction metrics."""
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        mse = mean_squared_error(self.actuals, self.predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.actuals, self.predictions)
        r2 = r2_score(self.actuals, self.predictions)
        mape = np.mean(np.abs((self.actuals - self.predictions) / (self.actuals + 1e-10)))
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
    
    def calculate_trading_metrics(self) -> Dict:
        """Calculate trading performance metrics."""
        strategy_returns = self.calculate_strategy_returns()
        returns = self.actuals

        # Cumulative returns
        cum_returns = (1 + strategy_returns).cumprod() - 1
        cumulative_actual = (1 + returns).cumprod() - 1

        # Metrics
        total_return = cum_returns[-1]
        
        # Annualized return: based on number of trading days
        num_years = len(returns) / 252.0  # Convert trading days to years
        annual_return = (1 + total_return) ** (1.0 / num_years) - 1 if num_years > 0 else 0

        annual_vol = strategy_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # Maximum drawdown
        cummax = np.maximum.accumulate(1 + cum_returns)
        drawdown = (1 + cum_returns) / cummax - 1
        max_drawdown = drawdown.min()

        # Win rate
        win_rate = np.sum(strategy_returns > 0) / len(strategy_returns)

        return {
            'Total_Return': total_return,
            'Annual_Return': annual_return,
            'Annual_Volatility': annual_vol,
            'Sharpe_Ratio': sharpe,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate,
            'Cumulative_Return': cum_returns[-1]
        }

    def calculate_strategy_returns(self) -> np.ndarray:
        """Calculate strategy returns based on signals, sizing, and costs."""
        signals = np.sign(self.predictions)
        confidence = self._confidence_scores()
        signals = np.where(confidence >= self.confidence_threshold, signals, 0.0)

        position_sizes = self._position_sizing(self.actuals)
        position_sizes = np.where(signals != 0, position_sizes, 0.0)

        # Calculate strategy daily returns: signal * position_size * actual_return
        strategy_daily_returns = signals * position_sizes * self.actuals
        
        # Calculate trading costs: transaction cost on position changes
        position_changes = np.abs(np.diff(np.concatenate([[0.0], position_sizes * signals])))
        # Cost is applied as a small percentage of the position value
        trade_costs = position_changes * (self.transaction_cost + self.slippage)
        
        # Net strategy returns after costs
        return strategy_daily_returns - trade_costs

    def _confidence_scores(self) -> np.ndarray:
        """Compute percentile confidence scores for predictions."""
        abs_preds = np.abs(self.predictions)
        return pd.Series(abs_preds).rank(pct=True).values

    def _realized_volatility(self, returns: np.ndarray) -> np.ndarray:
        """Estimate realized volatility using a rolling 63-day window."""
        returns_series = pd.Series(returns)
        vol = returns_series.rolling(window=63).std() * np.sqrt(252)
        vol = vol.ffill().fillna(returns_series.std() * np.sqrt(252))
        return vol.values

    def _position_sizing(self, returns: np.ndarray) -> np.ndarray:
        """Calculate volatility-based position sizing."""
        realized_vol = self._realized_volatility(returns)
        size = np.minimum(self.target_vol / (realized_vol + 1e-10), self.max_position)
        size = np.clip(size, 0.5, self.max_position)
        regime_size = np.where(realized_vol > 0.25, 0.5, np.where(realized_vol < 0.15, 0.8, 1.0))
        return np.minimum(size, regime_size)

    @property
    def summary_stats(self) -> Dict:
        """Get comprehensive summary statistics."""
        prediction_metrics = self.calculate_metrics()
        trading_metrics = self.calculate_trading_metrics()
        
        return {
            'Prediction': prediction_metrics,
            'Trading': trading_metrics
        }
