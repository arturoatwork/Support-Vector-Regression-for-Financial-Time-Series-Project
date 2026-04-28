"""
Visualization utilities for the SVR Quantitative Finance project.
Produces publication-quality plots for EDA, model analysis, and backtesting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

# ── Plot defaults ────────────────────────────────────────────────────────────
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
)


# ── EDA ─────────────────────────────────────────────────────────────────────

def plot_eda_dashboard(data: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Full exploratory data analysis dashboard.

    Panels:
      - Price history with high/low band
      - Daily return distribution (empirical vs normal) with VaR/CVaR
      - 21-day rolling annualised volatility
      - Normal Q-Q plot (heavy tails)
      - Return autocorrelation (serial independence check)
      - Squared-return autocorrelation (ARCH/volatility clustering)
      - Monthly returns heatmap
    """
    returns = data["Returns"].dropna()
    n = len(returns)
    ci = 1.96 / np.sqrt(n)

    fig = plt.figure(figsize=(18, 13))
    gs = gridspec.GridSpec(3, 3, fig, hspace=0.45, wspace=0.35)

    # ── Price history ─────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(data.index, data["Close"], color=PALETTE[0], lw=1.5, label="Close")
    ax0.fill_between(data.index, data["Low"], data["High"], alpha=0.12, color=PALETTE[0])
    ax0.set_title("AAPL Price History (2015–2024)", fontweight="bold", fontsize=14)
    ax0.set_ylabel("Price (USD)")
    ax0.legend(loc="upper left")

    # ── Return distribution ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    x = np.linspace(returns.min(), returns.max(), 300)
    ax1.hist(returns, bins=80, density=True, alpha=0.55, color=PALETTE[0], label="Empirical")
    ax1.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()), "r-", lw=2, label="Normal")
    var5 = np.percentile(returns, 5)
    cvar5 = returns[returns <= var5].mean()
    ax1.axvline(var5, color="orange", ls="--", lw=1.5, label=f"VaR(5%) = {var5:.2%}")
    ax1.axvline(cvar5, color="red", ls="--", lw=1.5, label=f"CVaR(5%) = {cvar5:.2%}")
    ax1.set_title("Return Distribution", fontweight="bold")
    ax1.set_xlabel("Daily Return")
    ax1.legend(fontsize=7.5)

    # ── Rolling volatility ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    vol21 = returns.rolling(21).std() * np.sqrt(252)
    ax2.plot(vol21.index, vol21, color=PALETTE[2], lw=1, label="21-day Vol")
    ax2.axhline(vol21.mean(), color="red", ls="--", lw=1, label=f"Mean {vol21.mean():.1%}")
    ax2.set_title("21-Day Rolling Volatility (Ann.)", fontweight="bold")
    ax2.set_ylabel("σ (annualised)")
    ax2.legend()

    # ── Q-Q plot ──────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    (osm, osr), (slope, intercept, _) = stats.probplot(returns, dist="norm")
    ax3.scatter(osm, osr, s=4, alpha=0.4, color=PALETTE[0])
    ax3.plot(osm, np.array(osm) * slope + intercept, "r-", lw=1.5, label="Normal reference")
    ax3.set_title("Normal Q-Q Plot (Fat Tails)", fontweight="bold")
    ax3.set_xlabel("Theoretical Quantile")
    ax3.set_ylabel("Sample Quantile")
    ax3.legend()

    # ── ACF of returns ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    lags = range(1, 21)
    acf_r = [returns.autocorr(lag=l) for l in lags]
    ax4.bar(lags, acf_r, color=PALETTE[0], alpha=0.7)
    ax4.axhline(ci, color="red", ls="--", lw=1)
    ax4.axhline(-ci, color="red", ls="--", lw=1)
    ax4.set_title("Return Autocorrelation (EMH Test)", fontweight="bold")
    ax4.set_xlabel("Lag (days)")
    ax4.set_ylabel("ACF")

    # ── ACF of squared returns (ARCH effect) ─────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    acf_r2 = [(returns ** 2).autocorr(lag=l) for l in lags]
    ax5.bar(lags, acf_r2, color=PALETTE[1], alpha=0.7)
    ax5.axhline(ci, color="red", ls="--", lw=1)
    ax5.axhline(-ci, color="red", ls="--", lw=1)
    ax5.set_title("Squared-Return ACF (ARCH Effect)", fontweight="bold")
    ax5.set_xlabel("Lag (days)")
    ax5.set_ylabel("ACF")

    # ── Monthly heatmap ───────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    monthly = data["Returns"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    df_m = pd.DataFrame({
        "Year": monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values,
    })
    pivot = df_m.pivot_table("Return", "Year", "Month", aggfunc="sum")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]
    sns.heatmap(
        pivot, ax=ax6, cmap="RdYlGn", center=0, annot=False,
        linewidths=0.4, cbar_kws={"label": "Monthly Return"},
    )
    ax6.set_title("Monthly Return Heatmap", fontweight="bold")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


# ── Feature analysis ─────────────────────────────────────────────────────────

def plot_feature_correlation(
    features: pd.DataFrame,
    top_n: int = 25,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Correlation heatmap for the top-N most-variant features."""
    top_cols = features.std().nlargest(top_n).index
    corr = features[top_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, ax=ax, cmap="coolwarm", center=0,
        annot=False, linewidths=0.3,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Feature Correlation Matrix (top {top_n} by variance)", fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def plot_feature_importance(
    importance: pd.Series,
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar chart of permutation-based feature importances."""
    top = importance.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [PALETTE[0] if v > 0 else PALETTE[3] for v in top.values]
    ax.barh(range(len(top)), top.values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Mean Permutation Importance (↓ MSE if removed)")
    ax.set_title(f"Top {top_n} Feature Importances (Permutation)", fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


# ── Kernel comparison ────────────────────────────────────────────────────────

def plot_kernel_comparison(results: Dict[str, Dict], save_path: Optional[str] = None) -> plt.Figure:
    """
    Bar chart comparison of SVR kernels across metrics.

    Parameters
    ----------
    results : dict  {kernel_name: {metric: value}}
    """
    kernels = list(results.keys())
    metrics = ["RMSE", "MAE", "R2", "Directional_Accuracy"]
    available = [m for m in metrics if m in list(results.values())[0]]

    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        vals = [results[k].get(metric, 0) for k in kernels]
        bars = ax.bar(kernels, vals, color=PALETTE[: len(kernels)], alpha=0.8)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylabel(metric)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=8,
            )

    fig.suptitle("SVR Kernel Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


# ── Backtesting ──────────────────────────────────────────────────────────────

def plot_equity_and_drawdown(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    strategy_label: str = "SVR Strategy",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Two-panel equity curve + drawdown chart.
    """
    strat = pd.Series(strategy_returns, index=dates)
    bench = pd.Series(benchmark_returns, index=dates)

    eq_strat = (1 + strat).cumprod()
    eq_bench = (1 + bench).cumprod()

    dd_strat = eq_strat / eq_strat.cummax() - 1
    dd_bench = eq_bench / eq_bench.cummax() - 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})
    fig.subplots_adjust(hspace=0.08)

    # ── Equity curve ─────────────────────────────────────────────────────
    ax1.plot(eq_strat.index, eq_strat.values, color=PALETTE[0], lw=2, label=strategy_label)
    ax1.plot(eq_bench.index, eq_bench.values, color=PALETTE[1], lw=1.5, ls="--",
             alpha=0.8, label="Buy-and-Hold")
    ax1.set_ylabel("Portfolio Value (normalised)", fontsize=11)
    ax1.set_title("Walk-Forward Backtest: Equity Curve & Drawdown", fontweight="bold", fontsize=14)
    ax1.legend(loc="upper left")
    ax1.set_xticklabels([])

    final_strat = eq_strat.iloc[-1]
    final_bench = eq_bench.iloc[-1]
    ax1.annotate(
        f"SVR: {final_strat:.2f}x",
        xy=(eq_strat.index[-1], final_strat),
        xytext=(-80, 10), textcoords="offset points",
        fontsize=9, color=PALETTE[0], fontweight="bold",
    )
    ax1.annotate(
        f"B&H: {final_bench:.2f}x",
        xy=(eq_bench.index[-1], final_bench),
        xytext=(-80, -20), textcoords="offset points",
        fontsize=9, color=PALETTE[1],
    )

    # ── Drawdown ─────────────────────────────────────────────────────────
    ax2.fill_between(dd_strat.index, dd_strat.values, 0, color=PALETTE[0], alpha=0.4,
                     label=strategy_label)
    ax2.fill_between(dd_bench.index, dd_bench.values, 0, color=PALETTE[1], alpha=0.25,
                     label="Buy-and-Hold")
    ax2.set_ylabel("Drawdown", fontsize=10)
    ax2.set_xlabel("Date")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.legend(loc="lower left", fontsize=8)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def plot_rolling_performance(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    window: int = 63,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Rolling Sharpe, Information Ratio, and hit rate."""
    strat = pd.Series(strategy_returns, index=dates)
    bench = pd.Series(benchmark_returns, index=dates)

    roll_sharpe = (strat.rolling(window).mean() / strat.rolling(window).std()) * np.sqrt(252)
    excess = strat - bench
    roll_ir = (excess.rolling(window).mean() / excess.rolling(window).std()) * np.sqrt(252)
    roll_hit = strat.rolling(window).apply(lambda x: (x > 0).mean())

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.subplots_adjust(hspace=0.1)

    for ax, series, label, color in zip(
        axes,
        [roll_sharpe, roll_ir, roll_hit],
        [f"Rolling {window}-Day Sharpe Ratio",
         f"Rolling {window}-Day Information Ratio",
         f"Rolling {window}-Day Hit Rate (Win Rate)"],
        PALETTE[:3],
    ):
        ax.plot(series.index, series.values, color=color, lw=1.5)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.fill_between(series.index, series.values, 0,
                        where=(series.values > 0), color=color, alpha=0.2)
        ax.fill_between(series.index, series.values, 0,
                        where=(series.values < 0), color="red", alpha=0.15)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontweight="bold", fontsize=11)

    axes[-1].set_xlabel("Date")
    axes[0].set_title(
        f"Rolling Performance Metrics ({window}-Day Window)",
        fontweight="bold", fontsize=13,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


# ── Risk ─────────────────────────────────────────────────────────────────────

def plot_risk_dashboard(
    strategy_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    alpha: float = 0.05,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Risk dashboard: return distribution, rolling VaR, drawdown distribution.
    """
    ret = pd.Series(strategy_returns, index=dates)
    var = np.percentile(ret, 100 * alpha)
    cvar = ret[ret <= var].mean()

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, fig, wspace=0.3)

    # ── Return distribution with VaR/CVaR ────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    x = np.linspace(ret.min(), ret.max(), 300)
    ax1.hist(ret, bins=60, density=True, alpha=0.5, color=PALETTE[0], label="Strategy")
    ax1.plot(x, stats.norm.pdf(x, ret.mean(), ret.std()), "r--", lw=1.5, label="Normal")
    ax1.axvline(var, color="orange", lw=2, label=f"VaR({alpha:.0%}) = {var:.2%}")
    ax1.axvline(cvar, color="red", lw=2, label=f"CVaR({alpha:.0%}) = {cvar:.2%}")
    ax1.fill_betweenx([0, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 10],
                      ret.min(), var, alpha=0.15, color="red")
    ax1.set_title("Return Distribution & Tail Risk", fontweight="bold")
    ax1.set_xlabel("Daily Return")
    ax1.legend(fontsize=8)

    # ── Rolling VaR ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    roll_var = ret.rolling(63).quantile(alpha)
    roll_cvar = ret.rolling(63).apply(lambda x: x[x <= np.percentile(x, 100 * alpha)].mean())
    ax2.plot(roll_var.index, roll_var.values, color="orange", lw=1.5, label=f"Rolling VaR({alpha:.0%})")
    ax2.plot(roll_cvar.index, roll_cvar.values, color="red", lw=1.5, label=f"Rolling CVaR({alpha:.0%})")
    ax2.set_title("Rolling Tail Risk (63-Day)", fontweight="bold")
    ax2.set_ylabel("Daily Return")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax2.legend()

    # ── Drawdown distribution ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    eq = (1 + ret).cumprod()
    dd = eq / eq.cummax() - 1
    ax3.hist(dd[dd < 0], bins=40, color=PALETTE[3], alpha=0.7, density=True)
    ax3.axvline(dd.min(), color="black", lw=2, ls="--",
                label=f"Max DD = {dd.min():.2%}")
    ax3.set_title("Drawdown Distribution", fontweight="bold")
    ax3.set_xlabel("Drawdown")
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax3.legend()

    plt.suptitle(f"Risk Management Dashboard  |  α = {alpha:.0%}", fontweight="bold", fontsize=13)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


# ── Regime analysis ──────────────────────────────────────────────────────────

def plot_regime_performance(
    returns: pd.Series,
    regimes: pd.Series,
    regime_labels: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Box/violin plot of strategy returns grouped by detected market regime."""
    if regime_labels is None:
        regime_labels = {r: f"Regime {r}" for r in sorted(regimes.unique())}

    df = pd.DataFrame({"Return": returns.values, "Regime": regimes.reindex(returns.index).values})
    df = df.dropna()
    df["Regime_Label"] = df["Regime"].map(regime_labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Violin plot
    order = list(regime_labels.values())
    valid_order = [l for l in order if l in df["Regime_Label"].values]
    sns.violinplot(data=df, x="Regime_Label", y="Return", order=valid_order,
                   palette=PALETTE[:len(valid_order)], ax=ax1, inner="box")
    ax1.axhline(0, color="black", lw=1, ls="--")
    ax1.set_title("Return Distribution by Market Regime", fontweight="bold")
    ax1.set_xlabel("")
    ax1.set_ylabel("Daily Return")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

    # Summary stats by regime
    summary = df.groupby("Regime_Label")["Return"].agg(
        Mean="mean", Std="std", Sharpe=lambda x: x.mean() / x.std() * np.sqrt(252),
        WinRate=lambda x: (x > 0).mean(),
    ).round(4)
    ax2.axis("off")
    tbl = ax2.table(
        cellText=summary.values,
        rowLabels=summary.index,
        colLabels=["Mean Return", "Volatility", "Ann. Sharpe", "Win Rate"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    ax2.set_title("Regime Performance Summary", fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


# ── Model comparison ─────────────────────────────────────────────────────────

def plot_model_comparison_table(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Render a styled performance comparison table."""
    df = pd.DataFrame(results).T

    fig, ax = plt.subplots(figsize=(max(10, len(df.columns) * 1.8), len(df) * 0.7 + 1.5))
    ax.axis("off")

    fmt_df = df.copy()
    for col in fmt_df.columns:
        try:
            if "Return" in col or "Drawdown" in col or "Rate" in col:
                fmt_df[col] = fmt_df[col].apply(lambda v: f"{float(v):.2%}" if pd.notna(v) else "-")
            else:
                fmt_df[col] = fmt_df[col].apply(lambda v: f"{float(v):.4f}" if pd.notna(v) else "-")
        except (ValueError, TypeError):
            pass

    tbl = ax.table(
        cellText=fmt_df.values,
        rowLabels=fmt_df.index,
        colLabels=fmt_df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 2.0)

    # Header styling
    for j in range(len(fmt_df.columns)):
        tbl[0, j].set_facecolor("#1f77b4")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Model Performance Comparison", fontweight="bold", fontsize=14, pad=20)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig
