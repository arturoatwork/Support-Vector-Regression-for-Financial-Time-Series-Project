"""
Performance metrics and risk analysis module.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class PerformanceMetrics:
    """Calculate comprehensive performance and risk metrics."""
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            
        Returns:
            Sharpe ratio
        """
        excess_returns = returns - (risk_free_rate / periods_per_year)
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_sortino_ratio(
        returns: np.ndarray,
        target_return: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino ratio (penalizes downside volatility).
        
        Args:
            returns: Array of returns
            target_return: Target return (usually 0 or risk-free rate)
            periods_per_year: Trading periods per year
            
        Returns:
            Sortino ratio
        """
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        
        annual_return = excess_returns.mean() * periods_per_year
        annual_downside_std = downside_std * np.sqrt(periods_per_year)
        
        return annual_return / annual_downside_std if annual_downside_std > 0 else 0
    
    @staticmethod
    def calculate_calmar_ratio(
        returns: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).
        
        Args:
            returns: Array of returns
            periods_per_year: Trading periods per year
            
        Returns:
            Calmar ratio
        """
        cumulative = (1 + returns).cumprod()
        cummax = np.maximum.accumulate(cumulative)
        drawdown = cumulative / cummax - 1
        max_drawdown = np.abs(drawdown.min())
        
        annual_return = returns.mean() * periods_per_year
        
        return annual_return / max_drawdown if max_drawdown > 0 else 0
    
    @staticmethod
    def calculate_information_ratio(
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate information ratio vs benchmark.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            periods_per_year: Trading periods per year
            
        Returns:
            Information ratio
        """
        excess_returns = returns - benchmark_returns
        annual_excess = excess_returns.mean() * periods_per_year
        annual_tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
        
        return annual_excess / annual_tracking_error if annual_tracking_error > 0 else 0
    
    @staticmethod
    def calculate_maximum_drawdown(returns: np.ndarray) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown and duration.
        
        Args:
            returns: Array of returns
            
        Returns:
            Tuple of (max_drawdown, start_idx, end_idx)
        """
        cumulative = (1 + returns).cumprod()
        cummax = np.maximum.accumulate(cumulative)
        drawdown = cumulative / cummax - 1
        
        max_dd_idx = np.argmin(drawdown)
        max_dd = drawdown[max_dd_idx]
        
        # Find drawdown start
        start_idx = np.where(cummax == cummax[max_dd_idx])[0]
        if len(start_idx) > 0:
            start_idx = start_idx[0]
        else:
            start_idx = 0
        
        return max_dd, start_idx, max_dd_idx
    
    @staticmethod
    def calculate_rolling_metrics(
        returns: np.ndarray,
        window: int = 63,
        periods_per_year: int = 252
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            returns: Array of returns
            window: Rolling window size
            periods_per_year: Trading periods per year
            
        Returns:
            DataFrame with rolling metrics
        """
        returns_series = pd.Series(returns)
        
        rolling_mean = returns_series.rolling(window).mean() * periods_per_year
        rolling_std = returns_series.rolling(window).std() * np.sqrt(periods_per_year)
        rolling_sharpe = rolling_mean / rolling_std
        
        return pd.DataFrame({
            'Rolling_Return': rolling_mean,
            'Rolling_Volatility': rolling_std,
            'Rolling_Sharpe': rolling_sharpe
        })
    
    @staticmethod
    def generate_performance_report(
        returns: np.ndarray,
        benchmark_returns: np.ndarray = None,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.02
    ) -> Dict:
        """
        Generate comprehensive performance report.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            periods_per_year: Trading periods per year
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary with all metrics
        """
        cumulative = (1 + returns).cumprod() - 1
        annual_return = returns.mean() * periods_per_year
        annual_vol = returns.std() * np.sqrt(periods_per_year)
        
        max_dd, _, _ = PerformanceMetrics.calculate_maximum_drawdown(returns)
        
        report = {
            'Total_Return': cumulative[-1],
            'Annual_Return': annual_return,
            'Annual_Volatility': annual_vol,
            'Sharpe_Ratio': PerformanceMetrics.calculate_sharpe_ratio(
                returns, risk_free_rate, periods_per_year
            ),
            'Sortino_Ratio': PerformanceMetrics.calculate_sortino_ratio(
                returns, 0, periods_per_year
            ),
            'Calmar_Ratio': PerformanceMetrics.calculate_calmar_ratio(
                returns, periods_per_year
            ),
            'Max_Drawdown': max_dd,
            'Win_Rate': np.sum(returns > 0) / len(returns),
            'Profit_Factor': np.sum(returns[returns > 0]) / np.abs(np.sum(returns[returns < 0])) if np.sum(returns[returns < 0]) != 0 else 0
        }
        
        if benchmark_returns is not None:
            report['Information_Ratio'] = PerformanceMetrics.calculate_information_ratio(
                returns, benchmark_returns, periods_per_year
            )
            report['Excess_Return'] = cumulative[-1] - ((1 + benchmark_returns).cumprod() - 1)[-1]
            report['Tracking_Error'] = (returns - benchmark_returns).std() * np.sqrt(periods_per_year)
        
        return report
