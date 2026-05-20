import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import riskfolio as rp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

EWMA_LAMBDA       = 0.94
ESTIMATION_WINDOW = 252
REBALANCE_FREQ    = 21
CRASH_DATE        = pd.Timestamp("2020-02-20")
PRE_CRASH_START   = pd.Timestamp("2019-11-01")   # baseline window start
PLOT_START        = pd.Timestamp("2020-01-02")    # plot from here
PLOT_END          = pd.Timestamp("2020-06-30")

CLASS_MAP = {
    "USO": "Energy",  "XLE": "Energy",  "UNG": "Energy",  "BNO": "Energy",
    "GLD": "Metals",  "SLV": "Metals",  "COPX": "Metals",
    "SPY": "Equities","EEM": "Equities","EWJ": "Equities",
    "TLT": "Bonds",   "IEF": "Bonds",   "HYG": "Bonds",
    "DBA": "Agriculture","WEAT":"Agriculture","CORN":"Agriculture",
    "FXE": "FX",      "UUP": "FX",
}

os.makedirs("data/phase6_results/diagnostics", exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
log_returns    = pd.read_csv("data/log_returns.csv", index_col=0, parse_dates=True)
simple_returns = np.expm1(log_returns)
assets = log_returns.columns.tolist()
n      = len(assets)
dates  = log_returns.index
T      = len(dates)
vals   = log_returns.values

rebal_indices = list(range(ESTIMATION_WINDOW, T, REBALANCE_FREQ))
rebal_set     = set(rebal_indices)
rebal_dates   = [dates[i] for i in rebal_indices]

ph2 = pd.read_csv("data/phase2_results/daily_returns.csv", index_col=0, parse_dates=True)
ph6 = pd.read_csv("data/phase6_results/daily_returns.csv", index_col=0, parse_dates=True)
hrp_rets  = ph2["HRP"]
ewma_rets = ph6["CentHRP-EWMA-0.94"]

print(f"Loaded: {dates[0].date()} → {dates[-1].date()}, {n} assets")

# ── Graph / EWMA helpers ───────────────────────────────────────────────────────
def cov_to_corr(S):
    d = np.sqrt(np.diag(S))
    d = np.where(d > 0, d, 1.0)
    c = S / np.outer(d, d)
    np.fill_diagonal(c, 1.0)
    return np.clip(c, -1.0, 1.0)

def build_pmfg(corr_arr, assets_):
    n_   = len(assets_)
    dist = np.sqrt(np.clip(2.0 * (1.0 - corr_arr), 0, None))
    np.fill_diagonal(dist, 0.0)
    edges = sorted((dist[i, j], assets_[i], assets_[j])
                   for i in range(n_) for j in range(i + 1, n_))
    G     = nx.Graph()
    G.add_nodes_from(assets_)
    max_e = 3 * (n_ - 2)
    for d, u, v in edges:
        if G.number_of_edges() >= max_e:
            break
        G.add_edge(u, v, weight=float(d))
        if not nx.check_planarity(G)[0]:
            G.remove_edge(u, v)
    return G

def hrp_fn(window_df):
    n_      = window_df.shape[1]
    assets_ = window_df.columns.tolist()
    try:
        port = rp.HCPortfolio(returns=window_df)
        wdf  = port.optimization(model="HRP", codependence="pearson", rm="MV",
                                 rf=0, linkage="ward", max_k=10, leaf_order=True)
        wt   = wdf.squeeze().reindex(assets_).fillna(0).values
        wt   = np.clip(wt, 0, None)
        s    = wt.sum()
        return wt / s if s > 0 else np.ones(n_) / n_
    except Exception:
        return np.ones(n_) / n_

def cent_adjust(base_w, G):
    cent = nx.degree_centrality(G)
    c    = np.array([cent.get(a, 1.0 / max(n - 1, 1)) for a in assets])
    c    = np.clip(c, 1e-6, None)
    adj  = base_w / c
    return adj / adj.sum()

# ── Pre-compute full EWMA covariance sequence (single pass) ───────────────────
print("\nRunning full EWMA sequence...")
init_days = 63
S = np.cov(vals[:init_days].T)
ewma_S_at = {}    # index → S snapshot (before update, no lookahead)

for t in range(init_days, T):
    if t in rebal_set or (dates[t] >= PRE_CRASH_START and dates[t] <= PLOT_END):
        ewma_S_at[t] = S.copy()
    r = vals[t]
    S = EWMA_LAMBDA * S + (1 - EWMA_LAMBDA) * np.outer(r, r)

print(f"  Snapshots stored: {len(ewma_S_at)}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1  Centrality speed test around the March 2020 crash
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/3] Centrality speed test (Nov 2019 – Jun 2020)...")

WATCH = ["SPY", "COPX", "HYG"]

# Daily centrality for every day in PRE_CRASH_START … PLOT_END
diag_mask    = (dates >= PRE_CRASH_START) & (dates <= PLOT_END)
diag_indices = np.where(diag_mask)[0]

ewma_cent   = {a: {} for a in WATCH}
static_cent = {a: {} for a in WATCH}

print(f"  Building PMFGs for {len(diag_indices)} days (~1 min)...")
for t in diag_indices:
    date = dates[t]

    # EWMA centrality (snapshot taken before day-t update)
    if t in ewma_S_at:
        corr_e = cov_to_corr(ewma_S_at[t])
        G_e    = build_pmfg(corr_e, assets)
        dc_e   = nx.degree_centrality(G_e)
        for a in WATCH:
            ewma_cent[a][date] = dc_e.get(a, 0.0)

    # Static 252-day rolling centrality
    start = max(0, t - ESTIMATION_WINDOW)
    win   = log_returns.iloc[start:t]
    if len(win) >= 30:
        G_s  = build_pmfg(win.corr().values, assets)
        dc_s = nx.degree_centrality(G_s)
        for a in WATCH:
            static_cent[a][date] = dc_s.get(a, 0.0)

ewma_ts   = {a: pd.Series(ewma_cent[a]).sort_index()   for a in WATCH}
static_ts = {a: pd.Series(static_cent[a]).sort_index() for a in WATCH}

# Detection lag: first post-crash day where |centrality − pre-crash mean| > 1σ
print("\n  ─── Centrality Shift Detection (>1σ from pre-crash mean) " + "─"*18)
print(f"  {'Asset':6s}  {'Method':10s}  {'Pre-μ':>6s}  {'Pre-σ':>6s}  "
      f"{'Days to detect':>14s}  Detection date")
print("  " + "─"*70)

detection = {}
for a in WATCH:
    for label, ts in [("EWMA-0.94", ewma_ts[a]), ("Static", static_ts[a])]:
        pre  = ts[(ts.index >= PRE_CRASH_START) & (ts.index < CRASH_DATE)]
        post = ts[ts.index >= CRASH_DATE]
        if len(pre) < 5 or len(post) == 0:
            continue
        mu, sigma = pre.mean(), pre.std()
        detect_date = next((d for d, v in post.items() if abs(v - mu) > sigma), None)
        lag = list(post.index).index(detect_date) if detect_date else "never"
        detection[(a, label)] = lag
        det_str = detect_date.date() if detect_date else "N/A"
        print(f"  {a:6s}  {label:10s}  {mu:6.3f}  {sigma:6.3f}  {str(lag):>14s}  {det_str}")

# Plot (Jan–Jun 2020 only)
fig, axes = plt.subplots(len(WATCH), 1, figsize=(13, 4 * len(WATCH)), sharex=True)
for ax, asset in zip(axes, WATCH):
    for label, ts, color, ls in [
        ("EWMA-0.94",  ewma_ts[asset],   "#e63946", "-"),
        ("Static-252", static_ts[asset], "#888888",  "--"),
    ]:
        seg = ts[(ts.index >= PLOT_START) & (ts.index <= PLOT_END)]
        ax.plot(seg.index, seg.values, label=label, color=color, ls=ls, lw=1.7)

    ax.axvline(CRASH_DATE, color="crimson", lw=1.4, ls=":", label="Crash (Feb 20)")
    ax.set_title(f"PMFG Degree Centrality — {asset}", fontsize=11)
    ax.set_ylabel("Centrality")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Annotate detection days in the plot margin
    ylo, yhi = ax.get_ylim()
    for label, color in [("EWMA-0.94", "#e63946"), ("Static", "#888888")]:
        lag = detection.get((asset, label))
        txt = f"{label}: {lag}d to detect" if isinstance(lag, int) else f"{label}: no shift"
        ypos = ylo + (yhi - ylo) * (0.08 if label == "EWMA-0.94" else 0.17)
        ax.text(pd.Timestamp("2020-01-05"), ypos, txt,
                fontsize=7.5, color=color, fontweight="bold")

fig.suptitle("Centrality Speed Test: EWMA-0.94 vs Static 252-Day\n(crash = Feb 20 2020)",
             fontsize=12)
plt.tight_layout()
plt.savefig("data/phase6_results/diagnostics/centrality_speed_test.png", dpi=150)
plt.close()
print("\n  Saved centrality_speed_test.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2  Weight difference over time
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/3] Rebuilding weights for all rebalance dates (~1 min)...")

hrp_w_cache  = {}
ewma_w_cache = {}

for k, i in enumerate(rebal_indices):
    date   = dates[i]
    window = log_returns.iloc[i - ESTIMATION_WINDOW : i]

    # EWMA PMFG (using pre-computed snapshot)
    corr_e = cov_to_corr(ewma_S_at[i])
    G_e    = build_pmfg(corr_e, assets)

    base_w = hrp_fn(window)
    ewma_w = cent_adjust(base_w, G_e)

    hrp_w_cache[date]  = base_w
    ewma_w_cache[date] = ewma_w

    if (k + 1) % 30 == 0 or (k + 1) == len(rebal_indices):
        print(f"  {k+1}/{len(rebal_indices)}")

# Average absolute weight difference at each rebalance date
wdiff = pd.Series(
    {date: np.mean(np.abs(ewma_w_cache[date] - hrp_w_cache[date]))
     for date in rebal_dates}
).sort_index()

CRISES = [
    ("2020-02-15", "2020-05-31", "COVID-19\n2020"),
    ("2022-01-01", "2022-12-31", "2022 Rate\nHike Cycle"),
]

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(wdiff.index, wdiff.values, color="#457b9d", lw=1.6)
ax.fill_between(wdiff.index, 0, wdiff.values, alpha=0.18, color="#457b9d")
ax.axhline(wdiff.mean(), color="#457b9d", lw=1, ls="--", alpha=0.6,
           label=f"Mean = {wdiff.mean():.3f}")

for t0s, t1s, label in CRISES:
    t0, t1 = pd.Timestamp(t0s), pd.Timestamp(t1s)
    ax.axvspan(t0, t1, alpha=0.13, color="crimson", zorder=0)
    ylo, yhi = ax.get_ylim()
    ax.text(t0 + (t1 - t0) / 2, yhi * 0.88, label,
            ha="center", va="top", fontsize=8, color="crimson", fontweight="bold")

ax.set_title("Avg Absolute Weight Difference: CentHRP-EWMA-0.94 vs HRP\n"
             "(spikes = graph structure meaningfully diverging from plain HRP)", fontsize=11)
ax.set_ylabel("Mean |Δw| per asset")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/phase6_results/diagnostics/weight_difference.png", dpi=150)
plt.close()
print("  Saved weight_difference.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3  Return attribution: which asset class bets drive the edge?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/3] Return attribution by asset class...")

# Monthly returns
def monthly_rets(s):
    for freq in ("ME", "M"):       # pandas ≥2.2 uses "ME", older uses "M"
        try:
            return s.resample(freq).apply(lambda x: (1 + x).prod() - 1)
        except Exception:
            continue

hrp_m  = monthly_rets(hrp_rets)
ewma_m = monthly_rets(ewma_rets)
common = hrp_m.index.intersection(ewma_m.index)
excess = ewma_m.loc[common] - hrp_m.loc[common]

out_months   = common[excess > 0]
under_months = common[excess < 0]

def nearest_rebal_before(month_end):
    cands = [d for d in rebal_dates if d <= month_end]
    return cands[-1] if cands else None

def avg_class_wdiff(month_list):
    acc = {cls: [] for cls in set(CLASS_MAP.values())}
    for m in month_list:
        rb = nearest_rebal_before(m)
        if rb is None or rb not in ewma_w_cache:
            continue
        diff = ewma_w_cache[rb] - hrp_w_cache[rb]
        for j, asset in enumerate(assets):
            acc[CLASS_MAP.get(asset, "Other")].append(diff[j])
    return {cls: np.mean(v) if v else 0.0 for cls, v in acc.items()}

out_diff   = avg_class_wdiff(out_months)
under_diff = avg_class_wdiff(under_months)
all_cls    = sorted(set(CLASS_MAP.values()))

# Save
attr_df = pd.DataFrame({
    "Asset Class":            all_cls,
    "Outperform Avg ΔW":     [out_diff.get(c, 0) for c in all_cls],
    "Underperform Avg ΔW":   [under_diff.get(c, 0) for c in all_cls],
})
attr_df.to_csv("data/phase6_results/diagnostics/return_attribution.csv", index=False)

print(f"\n  Outperforming months  : {len(out_months)}")
print(f"  Underperforming months: {len(under_months)}")
print(f"\n  ─── Avg Weight Diff by Asset Class (CentHRP-EWMA-0.94 minus HRP) " + "─"*8)
print(f"  {'Asset Class':14s} {'Outperform':>12s}  {'Underperform':>14s}  {'Signal?'}")
print("  " + "─"*60)
for cls in all_cls:
    ov = out_diff.get(cls, 0.0)
    uv = under_diff.get(cls, 0.0)
    # Consistent bet: same sign in both → strategy always tilts that way
    # Discriminating bet: opposite sign → tilt correlates with outcome
    if abs(ov) < 0.0005 and abs(uv) < 0.0005:
        signal = "negligible"
    elif np.sign(ov) != np.sign(uv):
        signal = "DISCRIMINATING ← key bet"
    else:
        signal = "consistent tilt"
    print(f"  {cls:14s} {ov:+.4f} ({ov*100:+.2f}pp)   {uv:+.4f} ({uv*100:+.2f}pp)   {signal}")

# Attribution bar chart
x     = np.arange(len(all_cls))
width = 0.38
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - width/2, [out_diff.get(c, 0)*100   for c in all_cls], width,
       label="Outperforming months",   color="#2a9d8f", alpha=0.87)
ax.bar(x + width/2, [under_diff.get(c, 0)*100 for c in all_cls], width,
       label="Underperforming months", color="#e63946", alpha=0.87)
ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels(all_cls, fontsize=10)
ax.set_ylabel("Avg weight diff (pp)")
ax.set_title("Return Attribution: Which Asset Class Bets Drive CentHRP-EWMA-0.94 vs HRP?\n"
             "(positive = EWMA overweights vs HRP, negative = underweights)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("data/phase6_results/diagnostics/return_attribution.png", dpi=150)
plt.close()
print("\n  Saved return_attribution.png")
print("\nPhase 6 diagnostics complete.")
