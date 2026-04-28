"""
Risk management utilities for the SVR quantitative finance project.
"""

import numpy as np
from typing import Sequence, Optional


class RiskManagement:
    """Calculate risk statistics for strategy returns."""

    @staticmethod
    def historical_var(returns: Sequence[float], alpha: float = 0.05) -> float:
        """Historical Value at Risk (VaR)."""
        returns = np.asarray(returns, dtype=float)
        if returns.size == 0:
            return 0.0
        return float(np.percentile(returns, 100 * alpha))

    @staticmethod
    def conditional_var(returns: Sequence[float], alpha: float = 0.05) -> float:
        """Conditional Value at Risk (CVaR), also known as expected shortfall."""
        returns = np.asarray(returns, dtype=float)
        if returns.size == 0:
            return 0.0

        var_level = RiskManagement.historical_var(returns, alpha=alpha)
        tail_losses = returns[returns <= var_level]
        return float(np.mean(tail_losses)) if tail_losses.size > 0 else var_level

    @staticmethod
    def kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
        """Calculate the Kelly fraction for position sizing."""
        if win_loss_ratio <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0

        edge = win_rate - (1.0 - win_rate) / win_loss_ratio
        return float(max(0.0, min(edge, 1.0)))

    @staticmethod
    def downside_risk(returns: Sequence[float], target: float = 0.0) -> float:
        """Calculate downside deviation relative to a target return."""
        returns = np.asarray(returns, dtype=float)
        downside = returns[returns < target] - target
        return float(np.sqrt(np.mean(downside ** 2))) if downside.size > 0 else 0.0
