# Support Vector Regression for Financial Time Series
### A Research-Grade Quantitative Trading Framework

**Asset**: Apple Inc. (AAPL) · 2015–2024 · ~2,500 trading days  
**Validation**: Walk-forward backtesting · No lookahead bias · Transaction costs included  
**Statistical Rigour**: Diebold-Mariano test · Pesaran-Timmermann directional accuracy · Bootstrap Sharpe CI

---

## Overview

This project implements a complete quantitative research pipeline applying **Support Vector Regression (SVR)** to financial time series prediction. It is designed to demonstrate graduate-level understanding of:

- Statistical learning theory (VC dimension, structural risk minimisation)
- Proper financial ML methodology (walk-forward validation, no data snooping)
- Kernel methods and the mathematics of ε-SVR
- Risk management (VaR, CVaR, Kelly criterion, volatility targeting)
- Ensemble learning (stacking SVR with XGBoost / LightGBM / RandomForest)
- Formal statistical testing of trading strategy significance

> **Key insight**: In efficient markets, R² ≈ 0 for next-day return prediction is *expected*, not a model failure. The relevant metric is directional accuracy and risk-adjusted excess return after transaction costs.

---

## Mathematical Foundation

The ε-SVR primal problem:

$$\min_{\mathbf{w}, b, \xi, \xi^*} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^n (\xi_i + \xi_i^*)$$

$$\text{s.t.} \quad y_i - \langle \mathbf{w}, \mathbf{x}_i \rangle - b \leq \varepsilon + \xi_i, \qquad \langle \mathbf{w}, \mathbf{x}_i \rangle + b - y_i \leq \varepsilon + \xi_i^*$$

The dual (kernel trick) enables nonlinear regression without explicit feature maps:

$$f(\mathbf{x}) = \sum_{i \in \mathcal{SV}} (\alpha_i - \alpha_i^*)\, K(\mathbf{x}_i, \mathbf{x}) + b$$

Kernels compared: **RBF** · **Linear** · **Polynomial**

---

## Project Structure

```
Support Vector Regression Project/
├── notebooks/
│   ├── 01_analysis.ipynb          # Main research notebook (full pipeline)
│   └── 02_kernel_theory.ipynb    # SVR kernel mathematics deep dive
├── src/
│   ├── feature_engineering.py     # 40+ technical & volatility features
│   ├── advanced_features.py       # Microstructure, regime, GARCH features
│   ├── svr_model.py               # SVR wrapper with hyperparameter tuning
│   ├── backtester.py              # Walk-forward engine + BacktestResults
│   ├── ensemble.py                # Stacking ensemble (SVR + trees)
│   ├── performance_metrics.py     # Sharpe, Sortino, Calmar, Info Ratio
│   ├── risk_management.py         # VaR, CVaR, Kelly criterion
│   ├── visualizer.py              # Publication-quality plotting utilities
│   ├── statistical_tests.py       # Diebold-Mariano, PT test, bootstrap CI
│   └── utils.py                   # Data alignment and scaling utilities
├── tests/
│   ├── test_data_loader.py
│   ├── test_feature_engineering.py
│   ├── test_svr_model.py
│   └── test_backtester.py
├── data/
│   └── AAPL_2015-01-01_2024-12-31.csv
├── results/                       # Generated plots and performance tables
├── config.yaml                    # Centralised configuration
├── main.py                        # Full pipeline execution script
└── requirements.txt
```

---

## Notebooks

### `01_analysis.ipynb` — Main Research Notebook

A complete, narrative-driven research paper in notebook form. Covers:

1. **ε-SVR mathematical framework** — primal/dual formulation, kernel theory, structural risk minimisation
2. **Exploratory data analysis** — return distribution, fat tails, ARCH effects, monthly return heatmap
3. **Feature engineering** — 40+ features across 6 layers with correlation analysis
4. **Kernel comparison** — RBF vs Linear vs Polynomial with out-of-sample directional accuracy
5. **Hyperparameter sensitivity** — C × ε grid search with annotated heatmap
6. **Permutation feature importance** — model-agnostic, no gradient required
7. **Walk-forward backtesting** — equity curve, drawdown, rolling Sharpe, vs buy-and-hold
8. **Risk management** — VaR/CVaR, Kelly criterion, volatility targeting
9. **Ensemble comparison** — stacking SVR + XGBoost + LightGBM + RandomForest
10. **Regime analysis** — low-vol / med-vol / high-vol performance decomposition
11. **Statistical significance** — Pesaran-Timmermann DA test, Diebold-Mariano, bootstrap Sharpe CI

### `02_kernel_theory.ipynb` — Kernel Mathematics Deep Dive

- Mercer's theorem and the Reproducing Kernel Hilbert Space
- Gram matrix structure for financial time series
- ε-insensitive vs MSE loss: why SVR naturally filters low-conviction signals
- Effect of C (regularisation) on bias-variance tradeoff with visual proof
- Effect of γ (RBF bandwidth) on model complexity
- TimeSeriesSplit vs KFold: visual demonstration of data leakage
- Empirical kernel selection guide for quantitative finance

---

## Feature Engineering

| Layer | Features | Count |
|---|---|---|
| **Technical** | RSI, MACD, Bollinger Bands, ATR, Momentum, VWAP | 12 |
| **Multi-timeframe Vol** | 10d / 20d / 30d / 60d realised σ | 4 |
| **Volatility estimators** | Parkinson, Garman-Klass, ATR%, vol percentile | 4 |
| **Microstructure** | VWAP, spread proxy, order flow, volume ratio | 5 |
| **Regime** | Trend indicator, Ichimoku Cloud, vol percentile rank | 4 |
| **Interactions** | RSI×Vol, Momentum×Vol, RSI×VolumeChange | 5 |
| **Nonlinear transforms** | Momentum², Vol², Return³, ratio features | 4 |
| **Lagged returns** | r_{t-1} through r_{t-5}, price lags | 10 |
| **Statistical moments** | Rolling skewness, kurtosis | 2 |
| **Total** | | **~50** |

---

## Backtesting Methodology

Walk-forward validation with the following parameters:

| Parameter | Value | Rationale |
|---|---|---|
| Training window | 504 days (2 years) | Sufficient history for meaningful model fit |
| Test window | 63 days (1 quarter) | Avoids overfitting to short-term patterns |
| Step size | 63 days | Non-overlapping test periods (no snooping) |
| Transaction costs | 10 bps | Conservative estimate for liquid large-cap |
| Slippage | 5 bps | Execution friction for market orders |
| Signal threshold | 70th percentile | Only trade high-conviction signals |
| Position sizing | Volatility targeting at σ = 15% p.a. | Risk-equalised across regimes |

---

## Statistical Testing

| Test | Implementation | Reference |
|---|---|---|
| Directional accuracy | Pesaran-Timmermann z-test | Pesaran & Timmermann (1992) |
| Forecast comparison | Diebold-Mariano with Newey-West variance | Diebold & Mariano (1995) |
| Nested model test | Clark-West MSPE-adjusted statistic | Clark & West (2007) |
| Sharpe confidence | Bootstrap percentile CI (2,000 resamples) | Efron & Tibshirani (1993) |
| Sharpe significance | Permutation test (1,000 permutations) | — |

---

## Performance Metrics Implemented

| Category | Metrics |
|---|---|
| **Return** | Total return, annualised return, CAGR |
| **Risk** | Annualised volatility, max drawdown, VaR(5%), CVaR(5%) |
| **Risk-adjusted** | Sharpe ratio, Sortino ratio, Calmar ratio, Information ratio |
| **Trading** | Win rate, win/loss ratio, profit factor, Kelly fraction |
| **Prediction** | RMSE, MAE, R², MAPE, directional accuracy |
| **Rolling** | 63-day rolling Sharpe, rolling IR, rolling hit rate |

---

## Ensemble Architecture

```
Training data
    │
    ├── SVR (RBF kernel, C=100)      ─── out-of-fold predictions ──┐
    ├── SVR (Linear kernel, C=10)    ─── out-of-fold predictions ──┤
    ├── XGBoost (200 trees)          ─── out-of-fold predictions ──┤ → Ridge meta-learner → prediction
    ├── LightGBM (200 trees)         ─── out-of-fold predictions ──┤
    ├── RandomForest (200 trees)     ─── out-of-fold predictions ──┤
    └── GradientBoosting (200 trees) ─── out-of-fold predictions ──┘
```

Out-of-fold predictions are generated via `TimeSeriesSplit` (5 folds) within the training window to prevent target leakage into the meta-learner.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline
python main.py

# 3. Open notebooks
jupyter notebook notebooks/01_analysis.ipynb
```

### Minimal SVR Example

```python
from src.feature_engineering import FeatureEngineer
from src.svr_model import SVRModel
from src.backtester import WalkForwardBacktester

# Build features
engineer = FeatureEngineer(data)
features = engineer.create_all_features()

# Train SVR with walk-forward validation
model = SVRModel(kernel='rbf', C=100, epsilon=0.01)
backtester = WalkForwardBacktester(
    model=model, data=data, features_df=features,
    train_window=504, test_window=63,
)
results = backtester.run(transaction_cost=0.001, target_vol=0.15)

# Evaluate
metrics = results.calculate_trading_metrics()
print(f"Sharpe: {metrics['Sharpe_Ratio']:.3f}")
print(f"Max DD: {metrics['Max_Drawdown']:.2%}")
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `scikit-learn` | SVR, scaling, TimeSeriesSplit, permutation importance |
| `numpy`, `pandas` | Numerical computing, time series handling |
| `matplotlib`, `seaborn` | Publication-quality visualisations |
| `scipy` | Statistical tests, distribution fitting |
| `xgboost`, `lightgbm` | Ensemble base models |
| `yfinance` | Market data download |

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy \
            xgboost lightgbm yfinance pyyaml
```

---

## References

1. Vapnik, V. (1995). *The Nature of Statistical Learning Theory*. Springer.
2. Smola, A.J. & Schölkopf, B. (2004). A tutorial on support vector regression. *Statistics and Computing*, 14, 199-222.
3. Pesaran, M.H. & Timmermann, A. (1992). A simple nonparametric test of predictive performance. *Journal of Business & Economic Statistics*, 10(4), 461-465.
4. Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.
5. Clark, T.E. & West, K.D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics*, 138(1), 291-311.
6. Artzner, P., Delbaen, F., Eber, J.M. & Heath, D. (1999). Coherent measures of risk. *Mathematical Finance*, 9(3), 203-228.
7. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
