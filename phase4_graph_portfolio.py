import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import cvxpy as cp
import riskfolio as rp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

ESTIMATION_WINDOW = 252
REBALANCE_FREQ    = 21
LAMBDAS           = [0.1, 0.5, 1.0]

os.makedirs("data/phase4_results", exist_ok=True)

CLASS_MAP = {
    "USO": "Energy",  "XLE": "Energy",  "UNG": "Energy",  "BNO": "Energy",
    "GLD": "Metals",  "SLV": "Metals",  "COPX": "Metals",
    "SPY": "Equities","EEM": "Equities","EWJ": "Equities",
    "TLT": "Bonds",   "IEF": "Bonds",   "HYG": "Bonds",
    "DBA": "Agriculture","WEAT":"Agriculture","CORN":"Agriculture",
    "FXE": "FX",      "UUP": "FX",
}
CLASS_COLORS = {
    "Energy": "tomato", "Metals": "goldenrod", "Equities": "steelblue",
    "Bonds": "mediumseagreen", "Agriculture": "saddlebrown", "FX": "mediumpurple",
    "Other": "lightgrey",
}

# ── Load ───────────────────────────────────────────────────────────────────────
log_returns    = pd.read_csv("data/log_returns.csv", index_col=0, parse_dates=True)
simple_returns = np.expm1(log_returns)
assets = log_returns.columns.tolist()
n      = len(assets)
dates  = log_returns.index

rebal_indices = list(range(ESTIMATION_WINDOW, len(dates), REBALANCE_FREQ))
rebal_set     = set(rebal_indices)
rebal_dates   = [dates[i] for i in rebal_indices]

print(f"Loaded: {dates[0].date()} → {dates[-1].date()}, {n} assets")
print(f"Rebalance dates: {len(rebal_dates)}")

# ── Graph helpers (same as phase 3) ───────────────────────────────────────────
def corr_to_dist(corr_df):
    d = np.sqrt(np.clip(2.0 * (1.0 - corr_df.values), 0, None))
    np.fill_diagonal(d, 0.0)
    return d

def build_mst(corr_df):
    a    = corr_df.columns.tolist()
    dist = corr_to_dist(corr_df)
    G    = nx.Graph()
    G.add_nodes_from(a)
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            G.add_edge(a[i], a[j], weight=float(dist[i, j]))
    return nx.minimum_spanning_tree(G, weight="weight")

def build_pmfg(corr_df):
    a    = corr_df.columns.tolist()
    n_   = len(a)
    dist = corr_to_dist(corr_df)
    edges = sorted(
        (dist[i, j], a[i], a[j]) for i in range(n_) for j in range(i + 1, n_)
    )
    G = nx.Graph()
    G.add_nodes_from(a)
    max_edges = 3 * (n_ - 2)
    for d, u, v in edges:
        if G.number_of_edges() >= max_edges:
            break
        G.add_edge(u, v, weight=float(d))
        if not nx.check_planarity(G)[0]:
            G.remove_edge(u, v)
    return G

def normalized_laplacian(G, node_list):
    """Binary normalized graph Laplacian: D^{-1/2}(D-A)D^{-1/2}."""
    A          = nx.to_numpy_array(G, nodelist=node_list, weight=None)
    d          = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    S          = np.diag(d_inv_sqrt)
    L          = np.diag(d) - A
    L_norm     = S @ L @ S
    return (L_norm + L_norm.T) / 2   # symmetrize for floating-point safety

# ── Pre-build all MST / PMFG ──────────────────────────────────────────────────
print("\n[1/3] Building MSTs...")
mst_cache = {}
for i in rebal_indices:
    w = log_returns.iloc[i - ESTIMATION_WINDOW : i]
    mst_cache[dates[i]] = build_mst(w.corr())
print(f"  Done — {len(mst_cache)} MSTs")

print("[2/3] Building PMFGs (planarity checks ~1 min)...")
pmfg_cache = {}
for k, i in enumerate(rebal_indices):
    w = log_returns.iloc[i - ESTIMATION_WINDOW : i]
    pmfg_cache[dates[i]] = build_pmfg(w.corr())
    if (k + 1) % 30 == 0 or (k + 1) == len(rebal_indices):
        print(f"  {k+1}/{len(rebal_indices)}")
print(f"  Done — {len(pmfg_cache)} PMFGs")

# ── Strategy 1: Laplacian-regularised Min-Var ─────────────────────────────────
def laplacian_minvar(window, G, lambda_val):
    """
    Minimise  w'Σw  +  λ · w'Lw   s.t.  Σw = 1, w ≥ 0

    Laplacian term penalises weight differences between connected (correlated)
    nodes, nudging the solver away from pure variance minimisation toward a
    graph-structure-aware allocation.  Combining into a single PSD matrix lets
    CLARABEL solve it in one shot.
    """
    n_   = window.shape[1]
    cov  = window.cov().values * 252          # annualise so scale matches L
    cov += 1e-8 * np.eye(n_)
    L    = normalized_laplacian(G, window.columns.tolist())
    M    = cov + lambda_val * L

    # Ensure strictly PSD (float rounding can push tiny eigenvalues negative)
    min_ev = np.linalg.eigvalsh(M).min()
    if min_ev < 0:
        M += (-min_ev + 1e-8) * np.eye(n_)
    M = (M + M.T) / 2

    w    = cp.Variable(n_)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, cp.psd_wrap(M))),
        [cp.sum(w) == 1, w >= 0],
    )
    prob.solve(solver=cp.CLARABEL, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return np.ones(n_) / n_
    wts = np.clip(w.value, 0, None)
    return wts / wts.sum()

# ── Strategy 2: Centrality-adjusted HRP ──────────────────────────────────────
def centrality_hrp(window, G):
    """
    1. Compute standard HRP weights.
    2. Scale by 1/centrality (peripheral assets get upweighted, hubs get down-
       weighted — reduces implicit systemic risk from highly-connected nodes).
    3. Renormalise to sum to 1.
    """
    n_      = window.shape[1]
    assets_ = window.columns.tolist()

    try:
        port  = rp.HCPortfolio(returns=window)
        w_df  = port.optimization(model="HRP", codependence="pearson",
                                  rm="MV", rf=0, linkage="ward",
                                  max_k=10, leaf_order=True)
        hrp_w = w_df.squeeze().reindex(assets_).fillna(0).values
        hrp_w = np.clip(hrp_w, 0, None)
        s     = hrp_w.sum()
        hrp_w = hrp_w / s if s > 0 else np.ones(n_) / n_
    except Exception:
        hrp_w = np.ones(n_) / n_

    cent = nx.degree_centrality(G)
    c    = np.array([cent.get(a, 1.0 / max(n_ - 1, 1)) for a in assets_])
    c    = np.clip(c, 1e-6, None)

    adj = hrp_w / c
    return adj / adj.sum()

# ── Walk-forward ───────────────────────────────────────────────────────────────
STRAT_DEFS = {
    **{f"Lap-MV λ={lam}": ("lap",  lam)   for lam in LAMBDAS},
    "CentHRP-MST":         ("chrp", "mst"),
    "CentHRP-PMFG":        ("chrp", "pmfg"),
}

print("\n[3/3] Walk-forward for graph strategies...")
all_port_rets  = {}
all_wgt_series = {}   # {strat: {rebal_date: weight_array}}

for strat_name, (kind, param) in STRAT_DEFS.items():
    print(f"  {strat_name}...")
    current_w = None
    ret_vals  = []
    ret_dates = []
    wgt_dict  = {}

    for i in range(len(dates)):
        if i in rebal_set:
            date   = dates[i]
            window = log_returns.iloc[i - ESTIMATION_WINDOW : i]

            if kind == "lap":
                current_w = laplacian_minvar(window, mst_cache[date], param)
            else:
                G = mst_cache[date] if param == "mst" else pmfg_cache[date]
                current_w = centrality_hrp(window, G)

            wgt_dict[date] = current_w.copy()

        if current_w is not None:
            ret_vals.append((simple_returns.iloc[i].values * current_w).sum())
            ret_dates.append(dates[i])

    all_port_rets[strat_name]  = pd.Series(ret_vals, index=ret_dates)
    all_wgt_series[strat_name] = wgt_dict

pd.DataFrame(all_port_rets).to_csv("data/phase4_results/daily_returns.csv")
print("  Saved daily_returns.csv")

# ── Metrics (phase 2 baselines + phase 4) ─────────────────────────────────────
def compute_metrics(s, name):
    cum    = (1 + s).cumprod()
    n_yrs  = len(s) / 252
    ann_r  = cum.iloc[-1] ** (1 / n_yrs) - 1
    ann_v  = s.std() * np.sqrt(252)
    sharpe = ann_r / ann_v if ann_v > 0 else np.nan
    max_dd = ((cum / cum.cummax()) - 1).min()
    return {"Strategy": name, "Ann Return": ann_r, "Ann Vol": ann_v,
            "Sharpe": sharpe, "Max Drawdown": max_dd}, cum

phase2     = pd.read_csv("data/phase2_results/daily_returns.csv",
                         index_col=0, parse_dates=True)
all_series = {**{c: phase2[c] for c in phase2.columns}, **all_port_rets}

rows, cum_dict = [], {}
for name, s in all_series.items():
    row, cum = compute_metrics(s, name)
    rows.append(row)
    cum_dict[name] = cum

met_df = pd.DataFrame(rows).set_index("Strategy")
met_df.to_csv("data/phase4_results/metrics_all.csv")

display = pd.DataFrame({
    "Ann Return":   met_df["Ann Return"].map("{:.2%}".format),
    "Ann Vol":      met_df["Ann Vol"].map("{:.2%}".format),
    "Sharpe":       met_df["Sharpe"].map("{:.2f}".format),
    "Max Drawdown": met_df["Max Drawdown"].map("{:.2%}".format),
})
print("\n─── Full Strategy Comparison " + "─" * 42)
print(display.to_string())
print("─" * 71)

# ── Plot 1: all cumulative returns ─────────────────────────────────────────────
BASELINE_STYLE = {
    "Equal Weight": ("-",  "#bbbbbb"),
    "Inverse Vol":  ("--", "#999999"),
    "Min Variance": ("-.", "#777777"),
    "HRP":          (":",  "#555555"),
}
GRAPH_COLORS = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#6a4c93"]

fig, ax = plt.subplots(figsize=(14, 7))
for name, (ls, color) in BASELINE_STYLE.items():
    if name in cum_dict:
        c = cum_dict[name]
        ax.plot(c.index, c.values, label=name, color=color,
                linestyle=ls, linewidth=1.2, alpha=0.75)
for name, color in zip(STRAT_DEFS.keys(), GRAPH_COLORS):
    c = cum_dict[name]
    ax.plot(c.index, c.values, label=name, color=color, linewidth=1.9)

ax.axhline(1, color="black", linewidth=0.5, linestyle=":")
ax.set_title("Walk-Forward Backtest — Graph Strategies vs Baselines", fontsize=13)
ax.set_ylabel("Growth of $1")
ax.legend(fontsize=8, ncol=3, loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/phase4_results/cumulative_returns_all.png", dpi=150)
plt.close()
print("\nSaved cumulative_returns_all.png")

# ── Plot 2: stacked area weight charts per graph strategy ──────────────────────
def class_weights_over_time(wgt_dict, assets_, class_map):
    """Convert {date: weight_array} to a DataFrame of class weights per date."""
    classes = list(dict.fromkeys(class_map.get(a, "Other") for a in assets_))
    rows_ = {}
    for date in sorted(wgt_dict):
        w = pd.Series(wgt_dict[date], index=assets_)
        rows_[date] = {
            cls: w[[a for a in assets_ if class_map.get(a, "Other") == cls]].sum()
            for cls in classes
        }
    return pd.DataFrame(rows_).T[classes]

for strat_name in STRAT_DEFS:
    wdf    = class_weights_over_time(all_wgt_series[strat_name], assets, CLASS_MAP)
    cols   = wdf.columns.tolist()
    colors = [CLASS_COLORS.get(c, "lightgrey") for c in cols]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.stackplot(wdf.index, wdf.T.values, labels=cols, colors=colors, alpha=0.83)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title(f"Asset Class Allocation Over Time — {strat_name}", fontsize=12)
    ax.set_ylabel("Portfolio weight")
    ax.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    safe = (strat_name.replace(" ", "_").replace("=", "")
                      .replace(".", "p").replace("λ", "L"))
    plt.savefig(f"data/phase4_results/weights_{safe}.png", dpi=150)
    plt.close()
    print(f"Saved weights_{safe}.png")

print("\nPhase 4 complete.")
