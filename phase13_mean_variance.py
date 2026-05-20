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

os.makedirs("data/phase13_results", exist_ok=True)

print("\n[1/5] Loading Sector Data (Phase 12)...")
log_returns = pd.read_csv("data/phase12_results/log_returns.csv", index_col=0, parse_dates=True)
simple_returns = np.expm1(log_returns)

# Filter for post-2010
log_returns = log_returns[log_returns.index >= '2009-01-01']
simple_returns = simple_returns[simple_returns.index >= '2009-01-01']

assets = log_returns.columns.tolist()
n = len(assets)
dates = log_returns.index
T = len(dates)

print(f"Loaded {n} assets, from {dates[0].date()} to {dates[-1].date()}")

# ── PMFG Graph and Near-PSD Function ───────────────────────────────────────────
def corr_to_dist(corr_mat):
    d = np.sqrt(np.clip(2.0 * (1.0 - corr_mat), 0, None))
    np.fill_diagonal(d, 0.0)
    return d

def build_pmfg(corr_mat):
    n_ = corr_mat.shape[0]
    dist = corr_to_dist(corr_mat)
    edges = sorted((dist[i, j], i, j) for i in range(n_) for j in range(i + 1, n_))
    G = nx.Graph()
    G.add_nodes_from(range(n_))
    max_e = 3 * (n_ - 2)
    for d, u, v in edges:
        if G.number_of_edges() >= max_e:
            break
        G.add_edge(u, v)
        if not nx.check_planarity(G)[0]:
            G.remove_edge(u, v)
    return G

def near_psd(x, epsilon=1e-8):
    """
    Ensure a symmetric matrix is Positive Semi-Definite (PSD) 
    by clipping negative eigenvalues.
    """
    eigval, eigvec = np.linalg.eigh(x)
    eigval[eigval < epsilon] = epsilon
    res = eigvec @ np.diag(eigval) @ eigvec.T
    # Enforce symmetry
    return (res + res.T) / 2.0

# ── Helper: EWMA Covariance ───────────────────────────────────────────────────
def ewma_cov(returns_df, lam):
    """Computes the full EWMA covariance matrix for the window."""
    vals = returns_df.values
    S = np.cov(vals.T)
    for t in range(1, len(vals)):
        r = vals[t]
        S = lam * S + (1.0 - lam) * np.outer(r, r)
    return S

# ── Optimization Helpers ──────────────────────────────────────────────────────
def get_weights(window_df):
    """
    Computes Standard Max Sharpe, PMFG-Filtered Max Sharpe, and HRP.
    Returns: eq_w, hrp_w, ms_w, ms_pmfg_w
    """
    n_ = window_df.shape[1]
    eq_w = np.ones(n_) / n_
    
    # Base HRP
    try:
        port_hrp = rp.HCPortfolio(returns=window_df)
        w_hrp = port_hrp.optimization(model="HRP", codependence="pearson", rm="MV", rf=0, linkage="ward", leaf_order=True)
        hrp_w = w_hrp.squeeze().reindex(assets).fillna(0).values
        hrp_w = np.clip(hrp_w, 0, None)
        hrp_w /= hrp_w.sum()
    except Exception:
        hrp_w = eq_w.copy()
        
    # EWMA Covariance for Max Sharpe
    cov_raw = ewma_cov(window_df, LAMBDA_EWMA)
    mu_raw = window_df.mean().values * 252  # Simple annualized historical mean
    
    # 1. Standard Max Sharpe
    try:
        port_ms = rp.Portfolio(returns=window_df)
        port_ms.mu = pd.DataFrame(mu_raw, index=assets, columns=['Return']).T
        port_ms.cov = pd.DataFrame(cov_raw, index=assets, columns=assets)
        w_ms_df = port_ms.optimization(model='Classic', rm='MV', obj='Sharpe', rf=0, l=0, hist=False)
        ms_w = w_ms_df.squeeze().reindex(assets).fillna(0).values
        ms_w = np.clip(ms_w, 0, None)
        s = ms_w.sum()
        ms_w = ms_w / s if s > 0 else eq_w.copy()
    except Exception:
        ms_w = eq_w.copy()
        
    # 2. PMFG-Filtered Max Sharpe
    try:
        # Get Correlation from Covariance
        d_inv = np.where(np.diag(cov_raw) > 0, 1.0 / np.sqrt(np.diag(cov_raw)), 0.0)
        corr_raw = d_inv[:, None] * cov_raw * d_inv[None, :]
        np.fill_diagonal(corr_raw, 1.0)
        corr_raw = np.clip(corr_raw, -1.0, 1.0)
        
        # PMFG Filter
        G = build_pmfg(corr_raw)
        cov_filtered = np.zeros_like(cov_raw)
        
        for i in range(n_):
            for j in range(n_):
                if i == j or G.has_edge(i, j):
                    cov_filtered[i, j] = cov_raw[i, j]
                else:
                    cov_filtered[i, j] = 0.0 # Denoise off-diagonal non-edges
        
        # Ensure Positive Semi-Definiteness
        cov_filtered_psd = near_psd(cov_filtered)
        
        port_pmfg = rp.Portfolio(returns=window_df)
        port_pmfg.mu = pd.DataFrame(mu_raw, index=assets, columns=['Return']).T
        port_pmfg.cov = pd.DataFrame(cov_filtered_psd, index=assets, columns=assets)
        
        w_pmfg_df = port_pmfg.optimization(model='Classic', rm='MV', obj='Sharpe', rf=0, l=0, hist=False)
        ms_pmfg_w = w_pmfg_df.squeeze().reindex(assets).fillna(0).values
        ms_pmfg_w = np.clip(ms_pmfg_w, 0, None)
        s = ms_pmfg_w.sum()
        ms_pmfg_w = ms_pmfg_w / s if s > 0 else eq_w.copy()
    except Exception:
        ms_pmfg_w = eq_w.copy()
        
    return eq_w, hrp_w, ms_w, ms_pmfg_w


# ── Walk-Forward Backtest ─────────────────────────────────────────────────────
print("\n[2/5] Running Walk-Forward Backtests...")

rebal_indices = list(range(ESTIMATION_WINDOW, T, REBALANCE_FREQ))
rebal_set = set(rebal_indices)

all_port_rets = {
    "Equal Weight": [],
    "HRP": [],
    "Max Sharpe (Raw)": [],
    "Max Sharpe (PMFG Filtered)": []
}
ret_dates_ = []

for i in range(T):
    if i in rebal_set:
        date = dates[i]
        window = log_returns.iloc[i - ESTIMATION_WINDOW : i]
        
        eq_w, hrp_w, ms_w, ms_pmfg_w = get_weights(window)
        
        curr_eq_w = eq_w
        curr_hrp_w = hrp_w
        curr_ms_w = ms_w
        curr_ms_pmfg_w = ms_pmfg_w

    if i >= ESTIMATION_WINDOW:
        r = simple_returns.iloc[i].values
        all_port_rets["Equal Weight"].append((r * curr_eq_w).sum())
        all_port_rets["HRP"].append((r * curr_hrp_w).sum())
        all_port_rets["Max Sharpe (Raw)"].append((r * curr_ms_w).sum())
        all_port_rets["Max Sharpe (PMFG Filtered)"].append((r * curr_ms_pmfg_w).sum())
        ret_dates_.append(dates[i])

returns_df = pd.DataFrame(all_port_rets, index=ret_dates_)

# Sub-select only 2010 onwards for accurate comparison
returns_df = returns_df[returns_df.index >= "2010-01-01"]

returns_df.to_csv("data/phase13_results/daily_returns.csv")
print("  Saved daily_returns.csv")


# ── Metrics & Significance ─────────────────────────────────────────────────────
print("\n[3/5] Evaluating metrics...")

def metrics(s):
    s     = s.dropna()
    cum   = (1 + s).cumprod()
    n_yrs = len(s) / 252
    ann_r = cum.iloc[-1] ** (1 / n_yrs) - 1
    ann_v = s.std() * np.sqrt(252)
    sh    = ann_r / ann_v if ann_v > 0 else np.nan
    dd    = ((cum / cum.cummax()) - 1).min()
    return ann_r, ann_v, sh, dd

rows = []
for name in returns_df.columns:
    r, v, sh, dd = metrics(returns_df[name])
    rows.append({
        "Strategy": name, "Ann Ret": r, "Ann Vol": v,
        "Sharpe": sh, "Max DD": dd
    })
met_df = pd.DataFrame(rows).set_index("Strategy")
met_df.to_csv("data/phase13_results/metrics.csv")

disp = pd.DataFrame({
    "Ann Ret": met_df["Ann Ret"].map("{:.2%}".format),
    "Ann Vol": met_df["Ann Vol"].map("{:.2%}".format),
    "Sharpe":  met_df["Sharpe"].map("{:.2f}".format),
    "Max DD":  met_df["Max DD"].map("{:.2%}".format),
})
print("\n─── Phase 13 Metrics (2010-2026) ──────────────────────────────────")
print(disp.to_string())
print("───────────────────────────────────────────────────────────────────")

# ── Plots ──────────────────────────────────────────────────────────────────────
print("\n[4/5] Generating Plots...")
STYLE = {
    "Equal Weight": (":", "#aaaaaa"),
    "HRP": ("--", "#888888"),
    "Max Sharpe (Raw)": ("-", "#e63946"),
    "Max Sharpe (PMFG Filtered)": ("-", "#2a9d8f")
}

# Cumulative Returns
fig, ax = plt.subplots(figsize=(14, 7))
for name in returns_df.columns:
    ls, color = STYLE[name]
    c = (1 + returns_df[name].dropna()).cumprod()
    lw = 2.0 if "Max Sharpe" in name else 1.3
    ax.plot(c.index, c.values, label=name, color=color, ls=ls, lw=lw, alpha=0.9)

ax.axhline(1, color="black", lw=0.5, ls=":")
ax.set_title("Cumulative Returns — PMFG Filtered Max Sharpe vs Benchmarks (2010-2026)", fontsize=13)
ax.set_ylabel("Growth of $1")
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/phase13_results/cumulative_returns.png", dpi=150)
plt.close()

print("\nPhase 13 complete.")
