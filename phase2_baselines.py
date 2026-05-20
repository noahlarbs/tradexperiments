import os
import warnings
import numpy as np
import pandas as pd
import cvxpy as cp
import riskfolio as rp
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ESTIMATION_WINDOW = 252
REBALANCE_FREQ    = 21      # ~monthly

os.makedirs("data/phase2_results", exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
log_returns    = pd.read_csv("data/log_returns.csv", index_col=0, parse_dates=True)
simple_returns = np.expm1(log_returns)   # exp(x)-1, numerically stable
assets = log_returns.columns.tolist()
n      = len(assets)
dates  = log_returns.index

print(f"Loaded: {dates[0].date()} → {dates[-1].date()}, {n} assets")

# ── Strategy functions ─────────────────────────────────────────────────────────
# Each receives a (ESTIMATION_WINDOW × n_assets) DataFrame of log returns
# and returns a 1-D np.ndarray of weights summing to 1.

def equal_weight(window):
    return np.ones(window.shape[1]) / window.shape[1]


def inverse_vol(window):
    vols = window.std()
    inv  = 1.0 / vols.replace(0, np.nan)
    wts  = inv / inv.sum()
    return wts.fillna(0).values


def min_variance(window):
    n_  = window.shape[1]
    cov = window.cov().values + 1e-8 * np.eye(n_)   # small ridge for PSD
    w   = cp.Variable(n_)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))),
        [cp.sum(w) == 1, w >= 0]
    )
    prob.solve(solver=cp.CLARABEL, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return np.ones(n_) / n_
    wts = np.clip(w.value, 0, None)
    return wts / wts.sum()


def hrp(window):
    try:
        port = rp.HCPortfolio(returns=window)
        w_df = port.optimization(
            model="HRP",
            codependence="pearson",
            rm="MV",
            rf=0,
            linkage="ward",
            max_k=10,
            leaf_order=True,
        )
        wts = w_df.squeeze().reindex(window.columns).fillna(0)
        total = wts.sum()
        if total <= 0:
            raise ValueError("zero weights")
        return (wts / total).values
    except Exception as e:
        print(f"  HRP fallback — {e}")
        return np.ones(window.shape[1]) / window.shape[1]


STRATEGIES = {
    "Equal Weight": equal_weight,
    "Inverse Vol":  inverse_vol,
    "Min Variance": min_variance,
    "HRP":          hrp,
}

# ── Walk-forward engine ────────────────────────────────────────────────────────
rebal_indices = list(range(ESTIMATION_WINDOW, len(dates), REBALANCE_FREQ))
rebal_set     = set(rebal_indices)

print(f"\nRebalance schedule: {len(rebal_indices)} dates  "
      f"(first {dates[rebal_indices[0]].date()}, last {dates[rebal_indices[-1]].date()})")

all_port_rets = {}

for name, fn in STRATEGIES.items():
    print(f"\nRunning {name}...")
    current_w  = None
    ret_vals   = []
    ret_dates  = []

    for i in range(len(dates)):
        # Recompute weights at each rebalance date using only past data
        if i in rebal_set:
            window    = log_returns.iloc[i - ESTIMATION_WINDOW : i]
            current_w = fn(window)

        # Record portfolio return once we have weights
        if current_w is not None:
            port_r = (simple_returns.iloc[i].values * current_w).sum()
            ret_vals.append(port_r)
            ret_dates.append(dates[i])

    all_port_rets[name] = pd.Series(ret_vals, index=ret_dates)
    print(f"  {len(ret_vals)} return observations")

port_df = pd.DataFrame(all_port_rets)
port_df.to_csv("data/phase2_results/daily_returns.csv")
print("\nSaved data/phase2_results/daily_returns.csv")

# ── Performance metrics ────────────────────────────────────────────────────────
def compute_metrics(s, name):
    cum    = (1 + s).cumprod()
    n_yrs  = len(s) / 252
    ann_r  = cum.iloc[-1] ** (1 / n_yrs) - 1
    ann_v  = s.std() * np.sqrt(252)
    sharpe = ann_r / ann_v if ann_v > 0 else np.nan
    drawdown = (cum / cum.cummax()) - 1
    max_dd = drawdown.min()
    return {
        "Strategy":    name,
        "Ann Return":  ann_r,
        "Ann Vol":     ann_v,
        "Sharpe":      sharpe,
        "Max Drawdown": max_dd,
    }, cum

rows, cum_dict = [], {}
for name, s in all_port_rets.items():
    row, cum = compute_metrics(s, name)
    rows.append(row)
    cum_dict[name] = cum

met_df = pd.DataFrame(rows).set_index("Strategy")
met_df.to_csv("data/phase2_results/metrics.csv")

# Pretty-print
display = pd.DataFrame({
    "Ann Return":   met_df["Ann Return"].map("{:.2%}".format),
    "Ann Vol":      met_df["Ann Vol"].map("{:.2%}".format),
    "Sharpe":       met_df["Sharpe"].map("{:.2f}".format),
    "Max Drawdown": met_df["Max Drawdown"].map("{:.2%}".format),
})
print("\n─── Performance Summary " + "─" * 37)
print(display.to_string())
print("─" * 61)

# ── Cumulative return plot ─────────────────────────────────────────────────────
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
LINES  = ["-", "--", "-.", ":"]

fig, ax = plt.subplots(figsize=(13, 7))
for (name, cum), color, ls in zip(cum_dict.items(), COLORS, LINES):
    ax.plot(cum.index, cum.values, label=name, color=color, linestyle=ls, linewidth=1.7)

ax.axhline(1, color="black", linewidth=0.5, linestyle=":")
ax.set_title("Walk-Forward Backtest — Baseline Strategies", fontsize=14)
ax.set_ylabel("Growth of $1")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/phase2_results/cumulative_returns.png", dpi=150)
plt.close()
print("\nSaved data/phase2_results/cumulative_returns.png")
print("Phase 2 complete.")
