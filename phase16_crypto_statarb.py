"""
Phase 16 — Crypto Cointegration Clique StatArb
Same BH-corrected Engle-Granger clique methodology as Phase 15,
applied to crypto spot prices via Binance US.
Benchmark: buy-and-hold BTC.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from statsmodels.stats.multitest import multipletests
import ccxt

warnings.filterwarnings("ignore")

ESTIMATION_WINDOW = 252
REBALANCE_FREQ    = 60
FDR_THRESHOLD     = 0.05
ZSCORE_ENTRY      = 2.0
START             = "2019-09-22"    # earliest common date on Binance US
END               = pd.Timestamp.today().strftime("%Y-%m-%d")

os.makedirs("data/phase16_results", exist_ok=True)

# Universe: coins with ≥ 3 years of daily data on Binance US
# Structured around 3 loosely related groups:
#   - Store of value / OG coins: BTC, ETH, LTC, BCH, ETC
#   - Smart-contract L1s:        SOL, ADA, ATOM, XLM
#   - Exchange / utility tokens: BNB, XRP, UNI
SYMBOLS_REQUESTED = [
    "BTC/USDT", "ETH/USDT", "LTC/USDT", "BCH/USDT", "ETC/USDT",
    "SOL/USDT", "ADA/USDT", "ATOM/USDT", "XLM/USDT",
    "BNB/USDT", "XRP/USDT", "UNI/USDT",
]

# ── 1. Data Download ──────────────────────────────────────────────────────────
print(f"\n[1/5] Fetching daily OHLCV from Binance US ({START} → {END})...")

exchange = ccxt.binanceus({"enableRateLimit": True})
since_ms  = int(pd.Timestamp(START).timestamp() * 1000)

def fetch_all_daily(symbol, since_ms):
    """Paginate ccxt fetch_ohlcv until we have all bars up to today."""
    all_bars = []
    cursor   = since_ms
    while True:
        bars = exchange.fetch_ohlcv(symbol, "1d", since=cursor, limit=1000)
        if not bars:
            break
        all_bars.extend(bars)
        if len(bars) < 1000:
            break
        cursor = bars[-1][0] + 86_400_000   # advance by one day in ms
        time.sleep(0.15)                     # respect rate limit
    return all_bars

raw_close = {}
for sym in SYMBOLS_REQUESTED:
    try:
        bars = fetch_all_daily(sym, since_ms)
        dates_  = pd.to_datetime([b[0] for b in bars], unit="ms", utc=True).normalize()
        closes_ = [b[4] for b in bars]
        raw_close[sym] = pd.Series(closes_, index=dates_, name=sym)
        print(f"  {sym:12s} {len(bars):4d} bars  "
              f"({dates_[0].date()} → {dates_[-1].date()})")
    except Exception as e:
        print(f"  {sym:12s} ERROR: {e}")
    time.sleep(0.2)

prices_raw = pd.DataFrame(raw_close)
prices_raw.index = prices_raw.index.tz_localize(None)
prices_raw = prices_raw.sort_index()

# ── Survivorship transparency ─────────────────────────────────────────────────
REQ_START = pd.Timestamp(START)
missing   = prices_raw.isna().mean()
dropped   = [c for c in prices_raw.columns
             if missing[c] > 0.05 or prices_raw[c].first_valid_index() > REQ_START + pd.Timedelta(days=90)]
kept      = [c for c in prices_raw.columns if c not in dropped]

if dropped:
    print(f"\nSurvivorship: dropped {len(dropped)} symbol(s) with >5% missing or "
          f"late start: {dropped}")
else:
    print("\nSurvivorship: all symbols pass.")

print(f"Final universe ({len(kept)} assets): {kept}")

prices     = prices_raw[kept].ffill(limit=3).dropna()
log_prices = np.log(prices)
log_ret    = log_prices.diff().dropna()
simple_ret = np.expm1(log_ret)

assets = log_ret.columns.tolist()
n      = len(assets)
dates  = log_ret.index
T      = len(dates)
print(f"\nBacktest window: {dates[0].date()} → {dates[-1].date()}  ({T} days, {n} assets)")

# ── 2. Cointegration Graph with BH Correction ─────────────────────────────────
def get_cointegration_cliques(window_log_prices):
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

    n_raw = int((raw_pvals < FDR_THRESHOLD).sum())
    reject_bh, _, _, _ = multipletests(raw_pvals, alpha=FDR_THRESHOLD, method="fdr_bh")
    n_bh = int(reject_bh.sum())

    exp_fp = n_pairs * FDR_THRESHOLD
    print(f"    {n_pairs} pairs | raw p<{FDR_THRESHOLD}: {n_raw} "
          f"(exp FP ≈ {exp_fp:.1f}) | BH sig: {n_bh}")

    G = nx.Graph()
    G.add_nodes_from(range(n_))
    for (i, j), sig in zip(pairs, reject_bh):
        if sig:
            G.add_edge(i, j)

    cliques = [c for c in nx.find_cliques(G) if len(c) >= 3]
    return cliques, n_raw, n_bh

# ── 3. Walk-Forward StatArb ───────────────────────────────────────────────────
print("\n[2/5] Walk-Forward Graph StatArb simulation...")

port_rets_arb = np.zeros(T)
port_rets_btc = np.zeros(T)

btc_col = assets.index("BTC/USDT") if "BTC/USDT" in assets else 0
port_rets_btc = simple_ret.iloc[:, btc_col].values

active_cliques  = []
position_hist   = []
clique_hist     = []
bh_hist         = []

for t in range(ESTIMATION_WINDOW, T - 1):
    if (t - ESTIMATION_WINDOW) % REBALANCE_FREQ == 0:
        window_lp = log_prices.iloc[t - ESTIMATION_WINDOW : t]
        print(f"  {dates[t].date()}: rebuilding…", end=" ")
        active_cliques, n_raw, n_bh = get_cointegration_cliques(window_lp)
        bh_hist.append({"date": dates[t], "n_raw": n_raw, "n_bh": n_bh,
                         "n_cliques": len(active_cliques)})
        print(f"→ {len(active_cliques)} valid clique(s)")

    trail_lp       = log_prices.iloc[t - 21 : t + 1].values
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

    ret_tomorrow             = simple_ret.iloc[t + 1].values
    port_rets_arb[t + 1]     = signal_weights @ ret_tomorrow
    position_hist.append(total_abs)
    clique_hist.append(len(active_cliques))

# ── 4. Metrics ────────────────────────────────────────────────────────────────
print("\n[3/5] Evaluating metrics...")

returns_df = pd.DataFrame({
    "Clique StatArb": port_rets_arb,
    "BTC Buy & Hold": port_rets_btc,
}, index=dates).iloc[ESTIMATION_WINDOW + 1:]

returns_df.to_csv("data/phase16_results/daily_returns.csv")

def metrics(s):
    s = s.dropna()
    cum   = (1 + s).cumprod()
    n_yrs = len(s) / 365          # crypto: 365-day year, 24/7 trading
    ann_r = cum.iloc[-1] ** (1 / n_yrs) - 1
    ann_v = s.std() * np.sqrt(365)
    sh    = ann_r / ann_v if ann_v > 0 else np.nan
    dd    = ((cum / cum.cummax()) - 1).min()
    return ann_r, ann_v, sh, dd

strat  = returns_df["Clique StatArb"]
bench  = returns_df["BTC Buy & Hold"]
corr   = strat.corr(bench)
beta   = corr * (strat.std() / bench.std())

rows = []
for nm in returns_df.columns:
    r, v, sh, dd = metrics(returns_df[nm])
    row = {"Strategy": nm, "Ann Ret": r, "Ann Vol": v, "Sharpe": sh, "Max DD": dd}
    if "StatArb" in nm:
        row["Corr/BTC"] = corr
        row["Beta/BTC"] = beta
    else:
        row["Corr/BTC"] = 1.0
        row["Beta/BTC"] = 1.0
    rows.append(row)

met_df = pd.DataFrame(rows).set_index("Strategy")
met_df.to_csv("data/phase16_results/metrics.csv")

print("\n─── Phase 16 Metrics (Crypto Clique StatArb — BH corrected) ─────────")
disp = pd.DataFrame({
    "Ann Ret" : met_df["Ann Ret"].map("{:.2%}".format),
    "Ann Vol" : met_df["Ann Vol"].map("{:.2%}".format),
    "Sharpe"  : met_df["Sharpe" ].map("{:.2f}".format),
    "Max DD"  : met_df["Max DD" ].map("{:.2%}".format),
    "Corr/BTC": met_df["Corr/BTC"].map("{:.3f}".format),
    "Beta/BTC": met_df["Beta/BTC"].map("{:.3f}".format),
})
print(disp.to_string())

if bh_hist:
    bh_df = pd.DataFrame(bh_hist)
    bh_df.to_csv("data/phase16_results/cointegration_summary.csv", index=False)
    print(f"\nCliques per rebalance — avg: {bh_df['n_cliques'].mean():.1f}  "
          f"max: {bh_df['n_cliques'].max()}  "
          f"pct with ≥1 clique: {(bh_df['n_cliques'] > 0).mean():.0%}")
    print(f"BH-significant pairs per rebalance — avg: {bh_df['n_bh'].mean():.1f}  "
          f"max: {bh_df['n_bh'].max()}")

print(f"\nUniverse: {assets}")
print(f"Survivorship note: {len(SYMBOLS_REQUESTED) - len(kept)} symbol(s) dropped "
      f"({[s for s in SYMBOLS_REQUESTED if s not in kept]})")

# ── 5. Plots ──────────────────────────────────────────────────────────────────
print("\n[4/5] Generating plots...")
fig, axes = plt.subplots(3, 1, figsize=(14, 14))

# Cumulative returns
ax = axes[0]
for nm, col, lw in [("Clique StatArb", "#2a9d8f", 2.0),
                    ("BTC Buy & Hold", "#aaaaaa",  1.3)]:
    c = (1 + returns_df[nm]).cumprod()
    ls = "-" if "Arb" in nm else ":"
    ax.plot(c.index, c.values, label=nm, color=col, lw=lw, ls=ls)
ax.axhline(1, color="black", lw=0.5, ls=":")
ax.set_title("Crypto Cointegration Clique StatArb vs BTC (BH-corrected graph)", fontsize=12)
ax.set_ylabel("Growth of $1")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Rolling 90-day Sharpe
ax = axes[1]
roll = returns_df.rolling(90)
roll_sh = (roll.mean() * 365) / (roll.std() * np.sqrt(365))
for nm, col in [("Clique StatArb", "#2a9d8f"), ("BTC Buy & Hold", "#aaaaaa")]:
    s = roll_sh[nm].dropna()
    ax.plot(s.index, s.values, label=nm, color=col, lw=1.3)
ax.axhline(0, color="black", lw=0.6, ls=":")
ax.set_title("Rolling 90-Day Sharpe", fontsize=11)
ax.set_ylabel("Annualised Sharpe")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Active cliques over time
ax = axes[2]
if clique_hist:
    ax.plot(dates[ESTIMATION_WINDOW + 1: ESTIMATION_WINDOW + 1 + len(clique_hist)],
            clique_hist, color="#2a9d8f", alpha=0.8)
ax.set_title("Active Cliques per Day (after BH correction)", fontsize=11)
ax.set_ylabel("Count")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/phase16_results/performance.png", dpi=150)
plt.close()

print("\n[5/5] Phase 16 complete.")
print(f"Results saved to data/phase16_results/")
