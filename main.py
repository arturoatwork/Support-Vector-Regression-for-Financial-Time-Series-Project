"""
SVR pipeline for AAPL daily return prediction (2015-2024).

Runs the full research pipeline:
  - Feature engineering (83 technical / microstructure features)
  - Mutual information feature selection (top 20)
  - Walk-forward hyperparameter search scored by directional accuracy
  - Walk-forward backtest: 2-year train window, 1-quarter test, 30 folds
  - Two strategies: pure long/short SVR and long-biased overlay (30%-150%)
  - Saves predictions and metrics to results/

Usage:
    python main.py
"""

import sys, os, warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from scipy import stats
from pathlib import Path

sys.path.insert(0, '.')
from src.feature_engineering import FeatureEngineer
from src.advanced_features import AdvancedFeatureEngineer
from src.svr_model import SVRModel
from src.backtester import WalkForwardBacktester
from src.performance_metrics import PerformanceMetrics
from src.risk_management import RiskManagement

# ── helpers ───────────────────────────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, index_col=0, header=0)
    if str(raw.iloc[0, 0]) in ('AAPL', 'MSFT', 'SPY') or raw.index[0] == 'Ticker':
        raw = raw.iloc[1:]
    raw = raw.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    raw.index = pd.to_datetime(raw.index)
    raw.index.name = 'Date'
    return raw

def da(actual, predicted):
    return float((np.sign(actual) == np.sign(predicted)).mean())

def da_pval(hit_rate, n):
    z = (hit_rate - 0.5) / np.sqrt(0.25 / n)
    return float(2 * (1 - stats.norm.cdf(abs(z))))

def _da_score(y_true, y_pred):
    return float((np.sign(y_true) == np.sign(y_pred)).mean())

da_scorer = make_scorer(_da_score, greater_is_better=True)

def perf_summary(ret, label):
    ret  = np.asarray(ret, float)
    eq   = (1 + ret).cumprod()
    dd   = eq / np.maximum.accumulate(eq) - 1
    ann  = ret.mean() * 252
    vol  = ret.std()  * np.sqrt(252)
    sr   = ann / vol  if vol  > 0 else 0
    sor  = PerformanceMetrics.calculate_sortino_ratio(ret)
    cal  = PerformanceMetrics.calculate_calmar_ratio(ret)
    mdd  = dd.min()
    wr   = (ret > 0).mean()
    var5 = np.percentile(ret, 5)
    cv5  = ret[ret <= var5].mean()
    wins = ret[ret > 0]; loss = ret[ret < 0]
    wlr  = wins.mean() / (-loss.mean()) if len(loss) > 0 else 0
    kelly = RiskManagement.kelly_fraction(float(wr), float(wlr))
    return {
        'label': label, 'ann_ret': ann, 'ann_vol': vol, 'sharpe': sr,
        'sortino': sor, 'calmar': cal, 'max_dd': mdd, 'win_rate': wr,
        'win_loss': wlr, 'total': eq[-1] - 1, 'var5': var5, 'cvar5': cv5,
        'kelly_quarter': kelly / 4,
    }

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 1 · DATA")
print("=" * 62)

data = load_csv('data/AAPL_2015-01-01_2024-12-31.csv')
data['Returns']     = data['Close'].pct_change()
data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
data = data.dropna()
print(f"  {len(data):,} days  {data.index[0].date()} → {data.index[-1].date()}")
print(f"  Ann. return = {data['Returns'].mean()*252:.2%}  |  "
      f"Ann. vol = {data['Returns'].std()*np.sqrt(252):.2%}")
print(f"  Target std (1-day return) = {data['Returns'].std():.5f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 2 · FEATURE ENGINEERING  (83 raw features)")
print("=" * 62)

fe      = FeatureEngineer(data)
feat    = fe.create_all_features()
afe     = AdvancedFeatureEngineer(data)
afeat   = afe.create_all_advanced_features()

features = feat.join(afeat, rsuffix='_adv')
features = features.loc[:, ~features.columns.duplicated()]
features = features[[c for c in features.columns if not c.endswith('_adv')]]
features = features.replace([np.inf, -np.inf], np.nan)

# Next-day return target — no lookahead
target = data['Returns'].shift(-1)

common = features.index.intersection(target.index)
X_raw  = features.loc[common];  y_raw = target.loc[common]
valid  = X_raw.notna().all(axis=1) & y_raw.notna()
X_full = X_raw.loc[valid];       y_full = y_raw.loc[valid]
print(f"  Shape after alignment : {X_full.shape}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 3 · FEATURE SELECTION  (MI top 20)")
print("=" * 62)

# Fit selector only on first 70% to respect train/test discipline
sel_n    = int(len(X_full) * 0.70)
selector = SelectKBest(mutual_info_regression, k=20)
selector.fit(X_full.iloc[:sel_n], y_full.iloc[:sel_n])
sel_cols = X_full.columns[selector.get_support()]

X = X_full[sel_cols];   y = y_full
print(f"  Selected: {', '.join(sel_cols[:6])}, … ({len(sel_cols)} total)")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 4 · HYPERPARAMETER TUNING  (DA-scored, TimeSeriesSplit)")
print("=" * 62)

y_std    = y.std()
tune_n   = int(len(X) * 0.60)
X_tune   = X.iloc[:tune_n];  y_tune = y.iloc[:tune_n]
sc_tune  = StandardScaler()
X_tune_s = sc_tune.fit_transform(X_tune)

# Epsilon = 1-5% of target std; C restricted to [1,100] so C≪n/d
eps_grid = [round(y_std * f, 6) for f in [0.01, 0.03, 0.05]]
param_grid = {
    'C':       [1, 5, 10, 50],    # C < 100 avoids extreme overfitting
    'epsilon': eps_grid,
    'gamma':   ['scale'],
}

gs = GridSearchCV(SVR(kernel='rbf'), param_grid,
                  cv=TimeSeriesSplit(n_splits=5),
                  scoring=da_scorer, n_jobs=-1, verbose=0)
gs.fit(X_tune_s, y_tune)

best = gs.best_params_
print(f"  Best: C={best['C']}  epsilon={best['epsilon']:.5f}"
      f"  (eps/std = {best['epsilon']/y_std:.3f}x)  "
      f"CV-DA={gs.best_score_:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 5 · WALK-FORWARD  (train=504d, test=63d, 30 folds)")
print("=" * 62)

svr_model  = SVRModel(kernel='rbf', C=best['C'],
                       epsilon=best['epsilon'], gamma='scale')
backtester = WalkForwardBacktester(
    model=svr_model, data=data, features_df=X, target_col='Returns',
    target_series=y, train_window=252*2, test_window=63, step_size=63,
)
wf_results = backtester.run(
    transaction_cost=0.001, slippage=0.0005,
    confidence_threshold=0.55, target_vol=0.15, max_position=1.0,
)

oos_preds   = wf_results.predictions
oos_actuals = wf_results.actuals
oos_dates   = wf_results.dates
print(f"  Folds: {len(backtester.train_indices)}"
      f"  |  OOS: {oos_dates[0].date()} → {oos_dates[-1].date()}"
      f"  ({len(oos_preds):,} obs)")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 6 · STRATEGY CONSTRUCTION")
print("=" * 62)

# Confidence percentile of each prediction
conf = pd.Series(np.abs(oos_preds)).rank(pct=True).values

# ── A. Pure long/short SVR ────────────────────────────────────────────────
pure_svr_returns = wf_results.calculate_strategy_returns()

# ── B. Long-biased overlay ────────────────────────────────────────────────
# Default: 100% long AAPL; when confident → tilt to 150% (bull) or 30% (bear)
BULL_POS   = 1.5   # overweight when SVR predicts up
BEAR_POS   = 0.3   # underweight when SVR predicts down
BASE_POS   = 1.0   # neutral — always long
CONF_THOLD = 0.65  # only tilt on top-35% confidence signals

positions = np.where(
    (conf >= CONF_THOLD) & (oos_preds > 0), BULL_POS,
    np.where((conf >= CONF_THOLD) & (oos_preds < 0), BEAR_POS, BASE_POS)
)
lb_gross  = positions * oos_actuals
pos_delta = np.abs(np.diff(np.concatenate([[BASE_POS], positions])))
lb_costs  = pos_delta * (0.001 + 0.0005)   # TC + slippage on changes only
lb_returns = lb_gross - lb_costs

# ── C. Benchmark (buy and hold AAPL same OOS period) ─────────────────────
bh_returns = data['Returns'].reindex(oos_dates).fillna(0).values

print(f"  Signals used:")
print(f"    Bullish (conf ≥ {CONF_THOLD:.0%}, pred > 0): "
      f"{((conf>=CONF_THOLD)&(oos_preds>0)).sum():,} days  "
      f"→ {BULL_POS:.0%} long")
print(f"    Bearish (conf ≥ {CONF_THOLD:.0%}, pred < 0): "
      f"{((conf>=CONF_THOLD)&(oos_preds<0)).sum():,} days  "
      f"→ {BEAR_POS:.0%} long")
print(f"    Neutral (low conf): "
      f"{(conf<CONF_THOLD).sum():,} days  → {BASE_POS:.0%} long")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 7 · RESULTS")
print("=" * 62)

# Prediction quality
d_acc   = da(oos_actuals, oos_preds)
d_pval  = da_pval(d_acc, len(oos_preds))
conf_mask = conf >= 0.55
d_conf  = da(oos_actuals[conf_mask], oos_preds[conf_mask]) if conf_mask.sum() else 0
r2      = r2_score(oos_actuals, oos_preds)
pred_vs_actual_std = oos_preds.std() / oos_actuals.std()

print(f"\n── Prediction Quality ─────────────────────────────────────")
print(f"  RMSE                  : {np.sqrt(mean_squared_error(oos_actuals, oos_preds)):.6f}")
print(f"  R²                    : {r2:.4f}  (target: close to 0 is normal)")
print(f"  Pred std / Actual std : {pred_vs_actual_std:.3f}x  "
      f"({'overfit' if pred_vs_actual_std > 1.5 else 'well-calibrated'})")
print(f"  Directional Accuracy  : {d_acc:.4f}  ({d_acc*100:.2f}%)")
print(f"  DA p-value            : {d_pval:.4f}  "
      f"({'✓ significant' if d_pval < 0.10 else '✗ not significant at 10%'})")
print(f"  DA on conf. signals   : {d_conf:.4f}  ({conf_mask.sum():,} obs = {conf_mask.mean():.1%} days)")

# Strategy comparison table
s_svr = perf_summary(pure_svr_returns, 'Pure SVR L/S')
s_lb  = perf_summary(lb_returns,       'SVR Long-Biased')
s_bh  = perf_summary(bh_returns,       'Buy & Hold')

print(f"\n── Strategy Comparison  (OOS: {oos_dates[0].year}–{oos_dates[-1].year}) ──────────────")
hdr = f"  {'Metric':<24}   {'Pure SVR L/S':>14}   {'SVR Long-Biased':>16}   {'Buy & Hold':>12}"
sep = "  " + "-"*24 + "   " + "-"*14 + "   " + "-"*16 + "   " + "-"*12
print(hdr); print(sep)
rows = [
    ('Ann. Return',   'ann_ret',  '{:.2%}'),
    ('Ann. Volatility','ann_vol', '{:.2%}'),
    ('Sharpe Ratio',  'sharpe',   '{:.4f}'),
    ('Sortino Ratio', 'sortino',  '{:.4f}'),
    ('Calmar Ratio',  'calmar',   '{:.4f}'),
    ('Max Drawdown',  'max_dd',   '{:.2%}'),
    ('Total Return',  'total',    '{:.2%}'),
    ('Win Rate',      'win_rate', '{:.2%}'),
    ('Win/Loss Ratio','win_loss', '{:.3f}'),
    ('VaR  (5%,daily)','var5',   '{:.3%}'),
    ('CVaR (5%,daily)','cvar5',  '{:.3%}'),
    ('Kelly (1/4)',   'kelly_quarter', '{:.3f}'),
]
for label, key, fmt in rows:
    v1 = fmt.format(s_svr[key])
    v2 = fmt.format(s_lb[key])
    v3 = fmt.format(s_bh[key])
    print(f"  {label:<24} : {v1:>14}   {v2:>16}   {v3:>12}")

# Information ratio vs B&H
for strat, ret in [('Pure SVR L/S', pure_svr_returns), ('SVR Long-Biased', lb_returns)]:
    excess = ret - bh_returns
    ir = excess.mean() * 252 / (excess.std() * np.sqrt(252)) if excess.std() > 0 else 0
    print(f"  {'Info Ratio vs B&H':<24} : "
          f"{'':>14}   {ir:>16.4f}" if 'Long' in strat
          else f"  {'Info Ratio vs B&H':<24} : {ir:>14.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  STEP 8 · SAVE")
print("=" * 62)

Path('results').mkdir(exist_ok=True)

# Full results CSV
pd.DataFrame({
    'Date':               oos_dates,
    'Predictions':        oos_preds,
    'Actuals':            oos_actuals,
    'Residuals':          oos_actuals - oos_preds,
    'Confidence_Pct':     conf,
    'Strategy_Return':    pure_svr_returns,
    'LongBiased_Return':  lb_returns,
    'Benchmark_Return':   bh_returns,
}).to_csv('results/predictions.csv', index=False)
print("  Saved results/predictions.csv")

# Metrics summary
pd.DataFrame({
    'Pure SVR L/S':    {lab: fmt.format(s_svr[key]) for lab, key, fmt in rows},
    'SVR Long-Biased': {lab: fmt.format(s_lb[key])  for lab, key, fmt in rows},
    'Buy & Hold':      {lab: fmt.format(s_bh[key])  for lab, key, fmt in rows},
}).to_csv('results/metrics_summary.csv')
print("  Saved results/metrics_summary.csv")

print("\n" + "=" * 62)
print("  COMPLETE")
print("=" * 62)
