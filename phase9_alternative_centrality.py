import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import riskfolio as rp
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

ESTIMATION_WINDOW = 252
REBALANCE_FREQ    = 21
LAMBDA_EWMA       = 0.94

os.makedirs("data/phase9_results", exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
log_returns    = pd.read_csv("data/log_returns.csv", index_col=0, parse_dates=True)
simple_returns = np.expm1(log_returns)
assets = log_returns.columns.tolist()
n      = len(assets)
dates  = log_returns.index
T      = len(dates)
rebal_indices = list(range(ESTIMATION_WINDOW, T, REBALANCE_FREQ))
rebal_set     = set(rebal_indices)
rebal_dates   = [dates[i] for i in rebal_indices]

print(f"Loaded: {dates[0].date()} → {dates[-1].date()}, {n} assets")

# ── EWMA Correlation ───────────────────────────────────────────────────────────
def ewma_corr_snapshots(returns_df, lam, rebal_idx_set, all_dates, init_days=63):
    vals   = returns_df.values
    S      = np.cov(vals[:init_days].T)
    corr_at = {}

    for t in range(init_days, len(vals)):
        if t in rebal_idx_set:
            d_inv  = np.where(np.diag(S) > 0, 1.0 / np.sqrt(np.diag(S)), 0.0)
            corr   = d_inv[:, None] * S * d_inv[None, :]
            np.fill_diagonal(corr, 1.0)
            corr   = np.clip(corr, -1.0, 1.0)
            corr_at[all_dates[t]] = pd.DataFrame(corr, index=returns_df.columns, columns=returns_df.columns)
        r = vals[t]
        S = lam * S + (1.0 - lam) * np.outer(r, r)
    return corr_at

print("\n[1/5] Computing EWMA-0.94 correlation matrices...")
ewma_corr_snap = ewma_corr_snapshots(log_returns, LAMBDA_EWMA, rebal_set, dates)

# ── PMFG Construction ──────────────────────────────────────────────────────────
def corr_to_dist(corr_df):
    d = np.sqrt(np.clip(2.0 * (1.0 - corr_df.values), 0, None))
    np.fill_diagonal(d, 0.0)
    return d

def build_pmfg(corr_df):
    a    = corr_df.columns.tolist()
    n_   = len(a)
    dist = corr_to_dist(corr_df)
    edges = sorted((dist[i, j], a[i], a[j]) for i in range(n_) for j in range(i + 1, n_))
    G     = nx.Graph()
    G.add_nodes_from(a)
    max_e = 3 * (n_ - 2)
    for d, u, v in edges:
        if G.number_of_edges() >= max_e:
            break
        G.add_edge(u, v, weight=float(d))
        if not nx.check_planarity(G)[0]:
            G.remove_edge(u, v)
    return G

print("\n[2/5] Building PMFGs and computing centralities...")
pmfgs = {}
centralities = {
    "Degree": {},
    "Betweenness": {},
    "Eigenvector": {},
    "PageRank": {}
}

for date in rebal_dates:
    corr = ewma_corr_snap.get(date)
    if corr is not None:
        G = build_pmfg(corr)
        pmfgs[date] = G
        
        # Degree
        centralities["Degree"][date] = nx.degree_centrality(G)
        
        # Betweenness (weight by distance, so shortest paths are favored)
        # weight parameter is the edge attribute used for path length
        centralities["Betweenness"][date] = nx.betweenness_centrality(G, weight="weight")
        
        # Eigenvector (fallback to degree if it doesn't converge)
        try:
            # We use weight="weight" but remember eigenvector prefers large weights, 
            # and our weights are distances. So we should use 1 / weight for proximity.
            # Actually, standard eigenvector might be fine unweighted for topology.
            # Let's just use unweighted topology for eigenvector to avoid distance inversion issues.
            centralities["Eigenvector"][date] = nx.eigenvector_centrality(G, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            centralities["Eigenvector"][date] = centralities["Degree"][date]
            
        # PageRank (also unweighted topology is robust)
        centralities["PageRank"][date] = nx.pagerank(G)

# ── HRP Helpers ────────────────────────────────────────────────────────────────
def hrp_from(window_df):
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

def cent_adjust(base_w, cent_dict):
    # Scale weights by 1/centrality, renormalise
    c    = np.array([cent_dict.get(a, 1.0 / max(n - 1, 1)) for a in assets])
    # Add small epsilon to avoid divide by zero (especially for betweenness which can be 0)
    c    = np.clip(c, 1e-6, None)
    adj  = base_w / c
    return adj / adj.sum()

# ── Walk-forward Backtest ──────────────────────────────────────────────────────
print("\n[3/5] Walk-forward backtests...")

all_port_rets = {}
strat_names = list(centralities.keys())

# Let's also load HRP benchmark from phase 2 directly
ph2 = pd.read_csv("data/phase2_results/daily_returns.csv", index_col=0, parse_dates=True)
all_port_rets["HRP"] = ph2["HRP"]

for cent_name in strat_names:
    print(f"  CentHRP-EWMA-0.94-{cent_name}...")
    current_w = None
    ret_vals, ret_dates_ = [], []

    for i in range(T):
        if i in rebal_set:
            date   = dates[i]
            window = log_returns.iloc[i - ESTIMATION_WINDOW : i]
            base_w = hrp_from(window)
            
            cent_dict = centralities[cent_name].get(date)
            if cent_dict is not None:
                current_w = cent_adjust(base_w, cent_dict)
            else:
                current_w = base_w

        if current_w is not None:
            ret_vals.append((simple_returns.iloc[i].values * current_w).sum())
            ret_dates_.append(dates[i])

    strat_key = f"CentHRP-EWMA-{cent_name}"
    all_port_rets[strat_key] = pd.Series(ret_vals, index=ret_dates_)

returns_df = pd.DataFrame(all_port_rets)
returns_df.to_csv("data/phase9_results/daily_returns.csv")
print("  Saved daily_returns.csv")

# ── Metrics & Significance ─────────────────────────────────────────────────────
print("\n[4/5] Evaluating metrics and significance...")

def ann_sharpe(s):
    s = s.dropna().values
    return s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else np.nan

def psr(returns, sr_star_annual):
    s       = returns.dropna().values
    T       = len(s)
    sr_hat  = s.mean() / s.std()
    sr_star = sr_star_annual / np.sqrt(252)
    skew    = float(stats.skew(s))
    kurt    = float(stats.kurtosis(s, fisher=False))
    denom = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2
    if denom <= 0: return np.nan
    z = (sr_hat - sr_star) * np.sqrt(T - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))

def metrics(s):
    s     = s.dropna()
    cum   = (1 + s).cumprod()
    n_yrs = len(s) / 252
    ann_r = cum.iloc[-1] ** (1 / n_yrs) - 1
    ann_v = s.std() * np.sqrt(252)
    sh    = ann_r / ann_v if ann_v > 0 else np.nan
    dd    = ((cum / cum.cummax()) - 1).min()
    return ann_r, ann_v, sh, dd

hrp_sr = ann_sharpe(returns_df["HRP"])

rows = []
for name in returns_df.columns:
    r, v, sh, dd = metrics(returns_df[name])
    psr_val = psr(returns_df[name], hrp_sr) if name != "HRP" else np.nan
    rows.append({
        "Strategy": name, "Ann Ret": r, "Ann Vol": v,
        "Sharpe": sh, "Max DD": dd, "PSR vs HRP": psr_val
    })
met_df = pd.DataFrame(rows).set_index("Strategy")
met_df.to_csv("data/phase9_results/metrics.csv")

disp = pd.DataFrame({
    "Ann Ret": met_df["Ann Ret"].map("{:.2%}".format),
    "Ann Vol": met_df["Ann Vol"].map("{:.2%}".format),
    "Sharpe":  met_df["Sharpe"].map("{:.2f}".format),
    "Max DD":  met_df["Max DD"].map("{:.2%}".format),
    "PSR":     met_df["PSR vs HRP"].map(lambda x: "{:.2%}".format(x) if pd.notnull(x) else "-")
})
print("\n─── Phase 9 Metrics " + "─" * 55)
print(disp.to_string())
print("─" * 75)

# ── Plots ──────────────────────────────────────────────────────────────────────
print("\n[5/5] Generating plots...")

BENCH_STYLE = {"HRP": ("--", "#aaaaaa")}
NEW_COLORS  = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d"]
new_names   = [n for n in returns_df.columns if n != "HRP"]

# Cumulative Returns
fig, ax = plt.subplots(figsize=(14, 7))
for name, (ls, color) in BENCH_STYLE.items():
    if name in returns_df.columns:
        c = (1 + returns_df[name].dropna()).cumprod()
        ax.plot(c.index, c.values, label=name, color=color, ls=ls, lw=1.3, alpha=0.8)
for name, color in zip(new_names, NEW_COLORS):
    c = (1 + returns_df[name].dropna()).cumprod()
    ax.plot(c.index, c.values, label=name, color=color, lw=1.7)
ax.axhline(1, color="black", lw=0.5, ls=":")
ax.set_title("Cumulative Returns — Centrality Metrics (EWMA-0.94)", fontsize=13)
ax.set_ylabel("Growth of $1")
ax.legend(fontsize=8.5, ncol=2, loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/phase9_results/cumulative_returns.png", dpi=150)
plt.close()

# Rolling Sharpe
roll_sh = ((returns_df.rolling(252).mean() * np.sqrt(252)) / returns_df.rolling(252).std())
fig, ax = plt.subplots(figsize=(14, 6))
for name, (ls, color) in BENCH_STYLE.items():
    if name in roll_sh.columns:
        s = roll_sh[name].dropna()
        ax.plot(s.index, s.values, label=name, color=color, ls=ls, lw=1.3, alpha=0.8)
for name, color in zip(new_names, NEW_COLORS):
    s = roll_sh[name].dropna()
    ax.plot(s.index, s.values, label=name, color=color, lw=1.7)
ax.axhline(0, color="black", lw=0.6, ls=":")
ax.set_title("Rolling 252-Day Sharpe — Centrality Metrics", fontsize=13)
ax.set_ylabel("Annualised Sharpe Ratio")
ax.legend(fontsize=8.5, ncol=2, loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/phase9_results/rolling_sharpe.png", dpi=150)
plt.close()

print("\nPhase 9 complete.")
