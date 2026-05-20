import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import networkx as nx
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

ESTIMATION_WINDOW = 252
REBALANCE_FREQ    = 60
FDR_THRESHOLD     = 0.05   # Benjamini-Hochberg FDR level for cointegration tests
ZSCORE_ENTRY      = 2.0
ZSCORE_EXIT       = 0.5
START             = "2018-01-01"
END               = pd.Timestamp.today().strftime("%Y-%m-%d")

os.makedirs("data/phase15_results", exist_ok=True)

# ── 1. Data ───────────────────────────────────────────────────────────────────
TICKERS_REQUESTED = [
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "ADBE", "CRM", "AMD",
    "QCOM", "TXN", "INTC", "IBM", "AMAT", "NOW", "MU", "LRCX", "ADI",
    "PANW", "KLAC", "SNPS", "CDNS", "CRWD", "FTNT", "MCHP", "NXPI",
    "MRVL", "PLTR", "WDAY", "ON"
]

print(f"\n[1/5] Downloading {len(TICKERS_REQUESTED)} Tech assets from {START} to {END}...")
raw = yf.download(TICKERS_REQUESTED, start=START, end=END, progress=False)["Close"]
raw.index = pd.to_datetime(raw.index)

# ── Survivorship transparency ─────────────────────────────────────────────────
# Any ticker without a full price history gets dropped by dropna(axis=1).
# We log exactly which ones and why so the survivorship bias is explicit.
prices_full = raw.ffill()
missing_frac = prices_full.isna().mean()
dropped = missing_frac[missing_frac > 0].sort_values(ascending=False)
kept    = missing_frac[missing_frac == 0].index.tolist()

if len(dropped) > 0:
    print(f"\nSurvivorship: {len(dropped)} ticker(s) dropped for incomplete history:")
    for tkr, frac in dropped.items():
        first_valid = raw[tkr].first_valid_index()
        print(f"  {tkr:6s}: {frac:.1%} missing  (first data: {first_valid.date() if first_valid else 'never'})")
else:
    print("Survivorship: all tickers have complete history.")

print(f"Universe after survivorship filter: {len(kept)} assets  →  {kept}")

prices     = prices_full[kept].dropna()
log_prices = np.log(prices)
log_returns  = log_prices.diff().dropna()
simple_returns = np.expm1(log_returns)

assets = log_returns.columns.tolist()
n      = len(assets)
dates  = log_returns.index
T      = len(dates)

print(f"\nLoaded {n} assets with full history, {dates[0].date()} – {dates[-1].date()}")

# ── 2. Cointegration Graph with BH Correction ─────────────────────────────────
def get_cointegration_cliques(window_log_prices):
    """
    Build a cointegration graph using Engle-Granger tests.
    Applies Benjamini-Hochberg FDR correction at FDR_THRESHOLD to control
    the expected fraction of spurious edges (without correction, ~5% of
    n*(n-1)/2 pairs would appear cointegrated by chance alone).
    Returns: (list of maximal cliques of size >= 3, n_raw_sig, n_bh_sig)
    """
    vals   = window_log_prices.values
    n_     = vals.shape[1]
    pairs  = [(i, j) for i in range(n_) for j in range(i + 1, n_)]
    n_pairs = len(pairs)

    raw_pvals = np.ones(n_pairs)
    for idx, (i, j) in enumerate(pairs):
        try:
            _, pv1, _ = coint(vals[:, i], vals[:, j], maxlag=1)
            _, pv2, _ = coint(vals[:, j], vals[:, i], maxlag=1)
            raw_pvals[idx] = min(pv1, pv2)
        except Exception:
            raw_pvals[idx] = 1.0

    n_raw_sig = int((raw_pvals < FDR_THRESHOLD).sum())

    # BH correction
    reject_bh, _, _, _ = multipletests(raw_pvals, alpha=FDR_THRESHOLD, method="fdr_bh")
    n_bh_sig = int(reject_bh.sum())

    expected_fp_raw = n_pairs * FDR_THRESHOLD
    print(f"    {n_pairs} pairs tested | raw p<{FDR_THRESHOLD}: {n_raw_sig} "
          f"(expected FP ≈ {expected_fp_raw:.0f}) | after BH correction: {n_bh_sig}")

    G = nx.Graph()
    G.add_nodes_from(range(n_))
    for (i, j), sig in zip(pairs, reject_bh):
        if sig:
            G.add_edge(i, j)

    all_cliques   = list(nx.find_cliques(G))
    valid_cliques = [c for c in all_cliques if len(c) >= 3]
    return valid_cliques, n_raw_sig, n_bh_sig

# ── 3. Walk-Forward StatArb ───────────────────────────────────────────────────
print("\n[2/5] Running Walk-Forward Graph StatArb simulation...")

port_rets      = {"Graph Clique StatArb": np.zeros(T), "S&P 500 (Benchmark)": np.zeros(T)}
active_cliques = []
position_history  = []
clique_history    = []
bh_sig_history    = []

spy_raw  = yf.download("SPY", start=START, end=END, progress=False)["Close"]
spy_rets = np.log(spy_raw / spy_raw.shift(1)).dropna()
spy_rets = spy_rets.reindex(dates).fillna(0).squeeze()
port_rets["S&P 500 (Benchmark)"] = np.expm1(spy_rets).values

for t in range(ESTIMATION_WINDOW, T - 1):
    if (t - ESTIMATION_WINDOW) % REBALANCE_FREQ == 0:
        window_lp = log_prices.iloc[t - ESTIMATION_WINDOW : t]
        print(f"  {dates[t].date()}: rebuilding graph…", end=" ")
        active_cliques, _, n_bh = get_cointegration_cliques(window_lp)
        bh_sig_history.append((dates[t], n_bh, len(active_cliques)))
        print(f"{len(active_cliques)} valid clique(s) after BH correction")

    trail_lp      = log_prices.iloc[t - 21 : t + 1].values
    signal_weights = np.zeros(n)

    for clique in active_cliques:
        k = len(clique)
        for idx in clique:
            others = [j for j in clique if j != idx]
            spread = trail_lp[:, idx] - np.mean(trail_lp[:, others], axis=1)
            mu     = spread[:-1].mean()
            std    = spread[:-1].std() + 1e-8
            z      = (spread[-1] - mu) / std

            if z > ZSCORE_ENTRY:
                signal_weights[idx] -= 1.0 / k
                for o in others:
                    signal_weights[o] += (1.0 / k) / (k - 1)
            elif z < -ZSCORE_ENTRY:
                signal_weights[idx] += 1.0 / k
                for o in others:
                    signal_weights[o] -= (1.0 / k) / (k - 1)

    total_abs = np.abs(signal_weights).sum()
    if total_abs > 1.0:
        signal_weights /= total_abs

    ret_tomorrow = simple_returns.iloc[t + 1].values
    port_rets["Graph Clique StatArb"][t + 1] = signal_weights @ ret_tomorrow
    position_history.append(np.abs(signal_weights).sum())
    clique_history.append(len(active_cliques))

# ── 4. Metrics ────────────────────────────────────────────────────────────────
print("\n[3/5] Evaluating metrics...")

returns_df = pd.DataFrame(port_rets, index=dates).iloc[ESTIMATION_WINDOW + 1:]
returns_df.to_csv("data/phase15_results/daily_returns.csv")

def metrics(s):
    s = s.dropna()
    cum   = (1 + s).cumprod()
    n_yrs = len(s) / 252
    ann_r = cum.iloc[-1] ** (1 / n_yrs) - 1
    ann_v = s.std() * np.sqrt(252)
    sh    = ann_r / ann_v if ann_v > 0 else np.nan
    dd    = ((cum / cum.cummax()) - 1).min()
    return ann_r, ann_v, sh, dd

spy_series   = returns_df["S&P 500 (Benchmark)"]
strat_series = returns_df["Graph Clique StatArb"]

corr_spy = strat_series.corr(spy_series)
vol_strat = strat_series.std()
vol_spy   = spy_series.std()
beta_spy  = corr_spy * (vol_strat / vol_spy)   # β = ρ · (σ_strat / σ_spy)

rows = []
for name in returns_df.columns:
    r, v, sh, dd = metrics(returns_df[name])
    row = {"Strategy": name, "Ann Ret": r, "Ann Vol": v, "Sharpe": sh, "Max DD": dd}
    if name == "Graph Clique StatArb":
        row["Corr to SPY"] = corr_spy
        row["Beta to SPY"] = beta_spy
    else:
        row["Corr to SPY"] = 1.0
        row["Beta to SPY"] = 1.0
    rows.append(row)

met_df = pd.DataFrame(rows).set_index("Strategy")
met_df.to_csv("data/phase15_results/metrics.csv")

disp = pd.DataFrame({
    "Ann Ret"    : met_df["Ann Ret"    ].map("{:.2%}".format),
    "Ann Vol"    : met_df["Ann Vol"    ].map("{:.2%}".format),
    "Sharpe"     : met_df["Sharpe"     ].map("{:.2f}".format),
    "Max DD"     : met_df["Max DD"     ].map("{:.2%}".format),
    "Corr/SPY"   : met_df["Corr to SPY"].map("{:.3f}".format),
    "Beta/SPY"   : met_df["Beta to SPY"].map("{:.3f}".format),
})
print("\n─── Phase 15 Metrics (Cointegration Clique StatArb — BH corrected) ────")
print(disp.to_string())
print(f"\nNote: {len(kept)} assets in universe after survivorship filter "
      f"(dropped {len(TICKERS_REQUESTED) - len(kept)}: "
      f"{[t for t in TICKERS_REQUESTED if t not in kept]})")
print(f"Beta = ρ × (σ_strat/σ_spy); Corr and Beta differ because vols differ.")

# BH summary
if bh_sig_history:
    bh_df = pd.DataFrame(bh_sig_history, columns=["Date", "BH_sig_pairs", "Valid_cliques"])
    bh_df.to_csv("data/phase15_results/cointegration_summary.csv", index=False)
    avg_cliques = bh_df["Valid_cliques"].mean()
    print(f"\nAvg cliques per rebalance (after BH): {avg_cliques:.1f}  "
          f"(max: {bh_df['Valid_cliques'].max()}, min: {bh_df['Valid_cliques'].min()})")

# ── 5. Plots ──────────────────────────────────────────────────────────────────
print("\n[4/5] Generating plots...")
STYLE = {
    "S&P 500 (Benchmark)"  : (":", "#aaaaaa"),
    "Graph Clique StatArb" : ("-", "#2a9d8f"),
}

fig, axes = plt.subplots(2, 1, figsize=(14, 11))

ax = axes[0]
for name in returns_df.columns:
    ls, col = STYLE[name]
    c = (1 + returns_df[name].dropna()).cumprod()
    ax.plot(c.index, c.values, label=name, color=col, ls=ls,
            lw=2.0 if "Clique" in name else 1.3, alpha=0.9)
ax.axhline(1, color="black", lw=0.5, ls=":")
ax.set_title("N-Dimensional Cointegration Clique StatArb vs S&P 500 (BH-corrected graph)", fontsize=12)
ax.set_ylabel("Growth of $1")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
periods = dates[ESTIMATION_WINDOW + 1:]
ax.plot(periods[:len(clique_history)],  clique_history,  color="#2a9d8f", alpha=0.7, label="# active cliques (BH)")
ax.set_title("Active Cliques per Day (after BH multiple-testing correction)", fontsize=11)
ax.set_ylabel("Count")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/phase15_results/cumulative_returns.png", dpi=150)
plt.close()

# Exposure
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(periods[:len(position_history)], position_history, color="#2a9d8f", alpha=0.8)
ax.set_title("Total Gross Market Exposure (Leverage) Over Time", fontsize=11)
ax.set_ylabel("Gross Exposure")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/phase15_results/exposure.png", dpi=150)
plt.close()

print("\n[5/5] Phase 15 complete.")
