"""
Statistical significance tests for forecast evaluation.

Tests implemented:
  - Diebold-Mariano (1995): equal predictive accuracy H0
  - Clark-West (2007): nested model comparison
  - Directional Accuracy (Pesaran & Timmermann, 1992)
  - Bootstrap confidence intervals (percentile method)
  - Permutation test for Sharpe ratio significance
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
    loss: str = "mse",
) -> Tuple[float, float, str]:
    """
    Diebold-Mariano test for equal predictive accuracy.

    H₀: The two forecast series have equal expected loss.
    H₁: e1 has strictly larger expected loss than e2 (one-sided).

    Parameters
    ----------
    e1, e2 : 1-D arrays of forecast errors (actuals - predictions)
    h       : forecast horizon (default 1 for 1-step ahead)
    loss    : 'mse' or 'mae'

    Returns
    -------
    dm_stat, p_value, interpretation
    """
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)

    if loss == "mse":
        d = e1 ** 2 - e2 ** 2
    elif loss == "mae":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'mse' or 'mae'")

    n = len(d)
    d_bar = np.mean(d)

    # Newey-West long-run variance (Harvey, Leybourne, Newbold, 1997)
    gamma_0 = np.var(d, ddof=1)
    gammas = []
    for lag in range(1, h + 1):
        if lag < n:
            cov = np.cov(d[lag:], d[:-lag])[0, 1]
            gammas.append((1 - lag / (h + 1)) * cov)
    long_run_var = gamma_0 + 2 * sum(gammas)
    long_run_var = max(long_run_var, 1e-15)

    dm_stat = d_bar / np.sqrt(long_run_var / n)

    # Modified t-distribution (Harvey et al. 1997)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    if p_value < 0.01:
        interp = "Model 2 significantly more accurate (p<0.01)"
    elif p_value < 0.05:
        interp = "Model 2 more accurate (p<0.05)"
    elif p_value < 0.10:
        interp = "Marginal evidence for Model 2 (p<0.10)"
    else:
        interp = "Cannot reject equal predictive accuracy (p≥0.10)"

    return float(dm_stat), float(p_value), interp


def clark_west_test(
    actual: np.ndarray,
    e_small: np.ndarray,
    e_large: np.ndarray,
) -> Tuple[float, float]:
    """
    Clark-West (2007) MSPE-adjusted test for nested models.

    Tests whether the larger (unrestricted) model adds predictive content
    beyond the smaller (restricted/benchmark) model.

    H₀: Restricted model is at least as accurate.
    H₁: Unrestricted model is more accurate.
    """
    actual = np.asarray(actual, float)
    e_small = np.asarray(e_small, float)  # restricted model errors
    e_large = np.asarray(e_large, float)  # unrestricted model errors

    # MSPE adjustment from Clark & West eq. (3)
    f_hat = e_small ** 2 - (e_large ** 2 - (actual - actual) ** 2)
    # Simplified: use the adj term directly
    adj = e_small ** 2 - e_large ** 2 + (e_small - e_large) ** 2

    f_bar = np.mean(adj)
    se = np.std(adj, ddof=1) / np.sqrt(len(adj))
    cw_stat = f_bar / se if se > 0 else 0.0

    # One-sided p-value
    p_value = 1 - stats.norm.cdf(cw_stat)

    return float(cw_stat), float(p_value)


def directional_accuracy_test(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> Dict[str, float]:
    """
    Pesaran & Timmermann (1992) directional accuracy test.

    Computes the hit rate and a z-test against the null that the model
    has no directional information (i.e., hit rate = 0.5 under H₀
    adjusted for marginal frequencies).

    Returns
    -------
    dict with keys: hit_rate, p_hat, q_hat, da_stat, p_value
    """
    actual = np.asarray(actual, float)
    predicted = np.asarray(predicted, float)

    n = len(actual)

    sign_a = np.sign(actual)
    sign_p = np.sign(predicted)
    hits = (sign_a == sign_p).astype(float)
    hit_rate = hits.mean()

    # Proportions for P&T statistic
    p_hat = (actual > 0).mean()  # fraction of positive actuals
    q_hat = (predicted > 0).mean()  # fraction of positive predictions

    # Expected hit rate under independence
    p_star = p_hat * q_hat + (1 - p_hat) * (1 - q_hat)

    var_p_star = (
        p_hat * (1 - p_hat) * q_hat ** 2 * (1 - q_hat) ** 2 * 2
        / n
    )
    var_p_star += (
        q_hat * (1 - q_hat) * p_hat ** 2 * (1 - p_hat) ** 2 * 2
        / n
    )

    da_stat = (hit_rate - p_star) / np.sqrt(max(var_p_star, 1e-15))
    p_value = 2 * (1 - stats.norm.cdf(abs(da_stat)))

    return {
        "hit_rate": hit_rate,
        "p_hat": p_hat,
        "q_hat": q_hat,
        "expected_hit_rate": p_star,
        "da_stat": float(da_stat),
        "p_value": float(p_value),
        "significant_5pct": p_value < 0.05,
    }


def bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    periods_per_year: int = 252,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap percentile confidence interval for the annualised Sharpe ratio.

    Returns
    -------
    (sharpe, lower_ci, upper_ci)
    """
    rng = np.random.default_rng(random_state)
    returns = np.asarray(returns, float)
    n = len(returns)

    sharpe_obs = returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year)

    boot_sharpes = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(returns, size=n, replace=True)
        std = sample.std(ddof=1)
        boot_sharpes[i] = sample.mean() / std * np.sqrt(periods_per_year) if std > 0 else 0.0

    lower = float(np.percentile(boot_sharpes, 100 * alpha / 2))
    upper = float(np.percentile(boot_sharpes, 100 * (1 - alpha / 2)))

    return float(sharpe_obs), lower, upper


def sharpe_permutation_test(
    strategy_returns: np.ndarray,
    n_permutations: int = 1000,
    periods_per_year: int = 252,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    One-sided permutation test for the Sharpe ratio.

    H₀: The ordering of daily returns is random (Sharpe = 0 in expectation).
    H₁: The observed Sharpe ratio is higher than random.

    Returns
    -------
    (observed_sharpe, p_value)
    """
    rng = np.random.default_rng(random_state)
    strategy_returns = np.asarray(strategy_returns, float)

    observed_sharpe = (
        strategy_returns.mean() / strategy_returns.std(ddof=1) * np.sqrt(periods_per_year)
    )

    null_sharpes = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(strategy_returns)
        std = shuffled.std(ddof=1)
        null_sharpes[i] = shuffled.mean() / std * np.sqrt(periods_per_year) if std > 0 else 0.0

    p_value = float((null_sharpes >= observed_sharpe).mean())

    return float(observed_sharpe), p_value


def print_significance_report(
    actual: np.ndarray,
    svr_preds: np.ndarray,
    ensemble_preds: Optional[np.ndarray] = None,
    strategy_returns: Optional[np.ndarray] = None,
    benchmark_label: str = "SVR",
    challenger_label: str = "Ensemble",
) -> None:
    """Print a formatted statistical significance report."""

    header = "=" * 60
    print(f"\n{header}")
    print(" STATISTICAL SIGNIFICANCE REPORT")
    print(header)

    svr_errors = actual - svr_preds
    da = directional_accuracy_test(actual, svr_preds)
    print(f"\n[1] Directional Accuracy ({benchmark_label})")
    print(f"    Hit Rate      : {da['hit_rate']:.2%}")
    print(f"    Expected (H₀) : {da['expected_hit_rate']:.2%}")
    print(f"    DA Statistic  : {da['da_stat']:.3f}")
    print(f"    p-value       : {da['p_value']:.4f}  {'✓ Significant' if da['significant_5pct'] else '✗ Not significant'}")

    if ensemble_preds is not None:
        ens_errors = actual - ensemble_preds
        dm_stat, dm_p, dm_interp = diebold_mariano_test(svr_errors, ens_errors)
        print(f"\n[2] Diebold-Mariano ({benchmark_label} vs {challenger_label})")
        print(f"    DM Statistic : {dm_stat:.3f}")
        print(f"    p-value      : {dm_p:.4f}")
        print(f"    Interpretation: {dm_interp}")

    if strategy_returns is not None:
        sharpe, lower, upper = bootstrap_sharpe_ci(strategy_returns)
        perm_sharpe, perm_p = sharpe_permutation_test(strategy_returns)
        print(f"\n[3] Sharpe Ratio Significance")
        print(f"    Observed Sharpe  : {sharpe:.3f}")
        print(f"    95% Bootstrap CI : [{lower:.3f}, {upper:.3f}]")
        print(f"    Permutation p    : {perm_p:.4f}  {'✓ Significant' if perm_p < 0.05 else '✗ Not significant'}")

    print(f"\n{header}\n")
