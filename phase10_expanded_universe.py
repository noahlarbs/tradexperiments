import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import networkx as nx
import riskfolio as rp
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

ESTIMATION_WINDOW = 252
REBALANCE_FREQ    = 21
LAMBDA_EWMA       = 0.94
MISSING_THRESHOLD = 0.05
START = "2010-01-01"
END   = pd.Timestamp.today().strftime("%Y-%m-%d")

os.makedirs("data/phase10_results", exist_ok=True)

TICKERS = [
    # Broad Equities
    "SPY", "QQQ", "IWM", "EFA", "EEM", "VGK", "EWJ",
    # Sectors
    "XLF", "XLV", "XLK", "XLE", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE",
    # Fixed Income
    "AGG", "TLT", "IEF", "SHY", "LQD", "HYG", "EMB", "TIP", "MBB",
    # Commodities/Real Assets
    "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "COPX", "VNQ",
    # Factors
    "VIG", "MTUM", "VLUE", "QUAL", "USMV"
]

print(f"\n[1/5] Downloading {len(TICKERS)} ETFs from {START} to {END}...")
raw = yf.download(TICKERS, start=START, end=END, progress=False)["Close"]
raw.index = pd.to_datetime(raw.index)

total_days = len(raw)
missing_frac = raw.isna().mean()
dropped = missing_frac[missing_frac > MISSING_THRESHOLD].index.tolist()
kept   = missing_frac[missing_frac <= MISSING_THRESHOLD].index.tolist()

if dropped:
    print(f"WARNING: Dropped {len(dropped)} ETF(s) with >{MISSING_THRESHOLD:.0%} missing data: {dropped}")

prices = raw[kept].copy()
filled = prices.ffill(limit=5)
remaining_na = filled.isna().sum().sum()
if remaining_na:
    filled = filled.dropna()

prices = filled
log_returns = np.log(prices / prices.shift(1)).dropna()
simple_returns = np.expm1(log_returns)

assets = log_returns.columns.tolist()
n = len(assets)
dates = log_returns.index
T = len(dates)

prices.to_csv("data/phase10_results/prices.csv")
log_returns.to_csv("data/phase10_results/log_returns.csv")

print(f"Loaded: {dates[0].date()} → {dates[-1].date()}, {n} assets kept")

# ── EWMA Correlation ───────────────────────────────────────────────────────────
rebal_indices = list(range(ESTIMATION_WINDOW, T, REBALANCE_FREQ))
rebal_set     = set(rebal_indices)
rebal_dates   = [dates[i] for i in rebal_indices]

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

print("\n[2/5] Computing EWMA-0.94 correlation matrices...")
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

print("\n[3/5] Building PMFGs and computing degree centrality...")
degree_cents = {}
for date in rebal_dates:
    corr = ewma_corr_snap.get(date)
    if corr is not None:
        G = build_pmfg(corr)
        degree_cents[date] = nx.degree_centrality(G)

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
    c    = np.array([cent_dict.get(a, 1.0 / max(n - 1, 1)) for a in assets])
    c    = np.clip(c, 1e-6, None)
    adj  = base_w / c
    return adj / adj.sum()

# ── Walk-forward Backtest ──────────────────────────────────────────────────────
print("\n[4/5] Walk-forward backtests...")

all_port_rets = {
    "Equal Weight": [],
    "HRP": [],
    "CentHRP-EWMA-Degree": []
}
ret_dates_ = []

for i in range(T):
    if i in rebal_set:
        date   = dates[i]
        window = log_returns.iloc[i - ESTIMATION_WINDOW : i]
        
        eq_w = np.ones(n) / n
        hrp_w = hrp_from(window)
        
        cent_dict = degree_cents.get(date)
        if cent_dict is not None:
            cent_w = cent_adjust(hrp_w, cent_dict)
        else:
            cent_w = hrp_w
            
        current_eq_w = eq_w
        current_hrp_w = hrp_w
        current_cent_w = cent_w

    if i >= ESTIMATION_WINDOW:
        r = simple_returns.iloc[i].values
        all_port_rets["Equal Weight"].append((r * current_eq_w).sum())
        all_port_rets["HRP"].append((r * current_hrp_w).sum())
        all_port_rets["CentHRP-EWMA-Degree"].append((r * current_cent_w).sum())
        ret_dates_.append(dates[i])

returns_df = pd.DataFrame(all_port_rets, index=ret_dates_)
returns_df.to_csv("data/phase10_results/daily_returns.csv")
print("  Saved daily_returns.csv")

# ── Metrics & Significance ─────────────────────────────────────────────────────
print("\n[5/5] Evaluating metrics and significance...")

def ann_sharpe(s):
    s = s.dropna().values
    return s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else np.nan

def psr(returns, sr_star_annual):
    s       = returns.dropna().values
    T_len   = len(s)
    sr_hat  = s.mean() / s.std()
    sr_star = sr_star_annual / np.sqrt(252)
    skew    = float(stats.skew(s))
    kurt    = float(stats.kurtosis(s, fisher=False))
    denom = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2
    if denom <= 0: return np.nan
    z = (sr_hat - sr_star) * np.sqrt(T_len - 1) / np.sqrt(denom)
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
met_df.to_csv("data/phase10_results/metrics.csv")

disp = pd.DataFrame({
    "Ann Ret": met_df["Ann Ret"].map("{:.2%}".format),
    "Ann Vol": met_df["Ann Vol"].map("{:.2%}".format),
    "Sharpe":  met_df["Sharpe"].map("{:.2f}".format),
    "Max DD":  met_df["Max DD"].map("{:.2%}".format),
    "PSR":     met_df["PSR vs HRP"].map(lambda x: "{:.2%}".format(x) if pd.notnull(x) else "-")
})
print("\n─── Phase 10 Metrics (Expanded Universe) ──────────────────────────────────")
print(disp.to_string())
print("───────────────────────────────────────────────────────────────────────────")

# ── Plots ──────────────────────────────────────────────────────────────────────
BENCH_STYLE = {"HRP": ("--", "#aaaaaa"), "Equal Weight": (":", "#777777")}
NEW_COLORS  = ["#e63946"]
new_names   = ["CentHRP-EWMA-Degree"]

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
ax.set_title(f"Cumulative Returns — Expanded Universe ({n} Assets)", fontsize=13)
ax.set_ylabel("Growth of $1")
ax.legend(fontsize=8.5, ncol=2, loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/phase10_results/cumulative_returns.png", dpi=150)
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
ax.set_title(f"Rolling 252-Day Sharpe — Expanded Universe ({n} Assets)", fontsize=13)
ax.set_ylabel("Annualised Sharpe Ratio")
ax.legend(fontsize=8.5, ncol=2, loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/phase10_results/rolling_sharpe.png", dpi=150)
plt.close()

print("\nPhase 10 complete.")
