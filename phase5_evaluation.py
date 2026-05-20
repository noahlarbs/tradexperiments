import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import cvxpy as cp
import riskfolio as rp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

ESTIMATION_WINDOW = 252
REBALANCE_FREQ    = 21
LAMBDAS           = [0.1, 0.5, 1.0]
TC_BPS            = 0.001           # 10 basis points per unit of one-way turnover

os.makedirs("data/phase5_results", exist_ok=True)

# ── Load strategy returns ──────────────────────────────────────────────────────
ph2 = pd.read_csv("data/phase2_results/daily_returns.csv", index_col=0, parse_dates=True)
ph4 = pd.read_csv("data/phase4_results/daily_returns.csv", index_col=0, parse_dates=True)
all_rets    = pd.concat([ph2, ph4], axis=1)
strat_names = all_rets.columns.tolist()
n_strats    = len(strat_names)

BASELINE_NAMES = list(ph2.columns)
GRAPH_NAMES    = list(ph4.columns)

print(f"Loaded {n_strats} strategies  ({len(BASELINE_NAMES)} baselines + {len(GRAPH_NAMES)} graph)")
print(f"Period: {all_rets.index[0].date()} → {all_rets.index[-1].date()}")

# ── Load raw returns for weight reconstruction ─────────────────────────────────
log_returns   = pd.read_csv("data/log_returns.csv", index_col=0, parse_dates=True)
assets        = log_returns.columns.tolist()
n             = len(assets)
dates         = log_returns.index
rebal_indices = list(range(ESTIMATION_WINDOW, len(dates), REBALANCE_FREQ))
rebal_set     = set(rebal_indices)

# ── Graph helpers (reproduced for standalone use) ──────────────────────────────
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
    a     = corr_df.columns.tolist()
    n_    = len(a)
    dist  = corr_to_dist(corr_df)
    edges = sorted((dist[i, j], a[i], a[j]) for i in range(n_) for j in range(i+1, n_))
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

def normalized_laplacian(G, node_list):
    A  = nx.to_numpy_array(G, nodelist=node_list, weight=None)
    d  = A.sum(axis=1)
    s  = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    L  = np.diag(d) - A
    Ln = np.diag(s) @ L @ np.diag(s)
    return (Ln + Ln.T) / 2

# ── Strategy functions (condensed) ────────────────────────────────────────────
def ew(win):
    return np.ones(win.shape[1]) / win.shape[1]

def ivol(win):
    v   = win.std()
    inv = 1.0 / v.replace(0, np.nan)
    wt  = inv / inv.sum()
    return wt.fillna(0).values

def minvar(win):
    n_   = win.shape[1]
    cov  = win.cov().values + 1e-8 * np.eye(n_)
    wv   = cp.Variable(n_)
    prob = cp.Problem(cp.Minimize(cp.quad_form(wv, cp.psd_wrap(cov))),
                      [cp.sum(wv) == 1, wv >= 0])
    prob.solve(solver=cp.CLARABEL, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate") or wv.value is None:
        return np.ones(n_) / n_
    wt = np.clip(wv.value, 0, None)
    return wt / wt.sum()

def hrp_fn(win):
    n_      = win.shape[1]
    assets_ = win.columns.tolist()
    try:
        port = rp.HCPortfolio(returns=win)
        wdf  = port.optimization(model="HRP", codependence="pearson", rm="MV",
                                 rf=0, linkage="ward", max_k=10, leaf_order=True)
        wt   = wdf.squeeze().reindex(assets_).fillna(0).values
        wt   = np.clip(wt, 0, None)
        s    = wt.sum()
        return wt / s if s > 0 else np.ones(n_) / n_
    except Exception:
        return np.ones(n_) / n_

def lap_mv(win, G, lam):
    n_  = win.shape[1]
    cov = win.cov().values * 252 + 1e-8 * np.eye(n_)
    L   = normalized_laplacian(G, win.columns.tolist())
    M   = cov + lam * L
    ev  = np.linalg.eigvalsh(M).min()
    if ev < 0:
        M += (-ev + 1e-8) * np.eye(n_)
    M   = (M + M.T) / 2
    wv  = cp.Variable(n_)
    prob = cp.Problem(cp.Minimize(cp.quad_form(wv, cp.psd_wrap(M))),
                      [cp.sum(wv) == 1, wv >= 0])
    prob.solve(solver=cp.CLARABEL, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate") or wv.value is None:
        return np.ones(n_) / n_
    wt = np.clip(wv.value, 0, None)
    return wt / wt.sum()

def cent_hrp(win, G):
    n_      = win.shape[1]
    assets_ = win.columns.tolist()
    base    = hrp_fn(win)
    cent    = nx.degree_centrality(G)
    c       = np.array([cent.get(a, 1.0 / max(n_ - 1, 1)) for a in assets_])
    c       = np.clip(c, 1e-6, None)
    adj     = base / c
    return adj / adj.sum()

# Map strategy names → functions (must match CSV column names exactly)
STRAT_FNS = {
    "Equal Weight": lambda win, mst, pmfg: ew(win),
    "Inverse Vol":  lambda win, mst, pmfg: ivol(win),
    "Min Variance": lambda win, mst, pmfg: minvar(win),
    "HRP":          lambda win, mst, pmfg: hrp_fn(win),
    **{f"Lap-MV λ={lam}": (lambda win, mst, pmfg, _l=lam: lap_mv(win, mst, _l))
       for lam in LAMBDAS},
    "CentHRP-MST":  lambda win, mst, pmfg: cent_hrp(win, mst),
    "CentHRP-PMFG": lambda win, mst, pmfg: cent_hrp(win, pmfg),
}

# ── Rebuild weights in a single pass (build MST+PMFG once per rebalance date) ─
print("\nRebuilding weights for turnover analysis (~2 min)...")
wgt_history = {name: {} for name in STRAT_FNS}

for k, i in enumerate(rebal_indices):
    date   = dates[i]
    window = log_returns.iloc[i - ESTIMATION_WINDOW : i]
    corr   = window.corr()
    mst    = build_mst(corr)
    pmfg   = build_pmfg(corr)

    for name, fn in STRAT_FNS.items():
        wgt_history[name][date] = fn(window, mst, pmfg)

    if (k + 1) % 30 == 0 or (k + 1) == len(rebal_indices):
        print(f"  {k+1}/{len(rebal_indices)}")

print("  Done.")

# ── Turnover ───────────────────────────────────────────────────────────────────
def compute_turnover(wgt_dict):
    """One-way turnover at each rebalance: sum(|w_new - w_old|) / 2.
    Excludes the first rebalance (initial buy from cash)."""
    sorted_dates = sorted(wgt_dict)
    to_vals, to_dates = [], []
    for i in range(1, len(sorted_dates)):
        to_vals.append(np.sum(np.abs(wgt_dict[sorted_dates[i]] - wgt_dict[sorted_dates[i-1]])) / 2)
        to_dates.append(sorted_dates[i])
    avg = float(np.mean(to_vals)) if to_vals else 0.0
    return avg, dict(zip(to_dates, to_vals))

turnover_avg, turnover_ts = {}, {}
for name in STRAT_FNS:
    avg, ts = compute_turnover(wgt_history[name])
    turnover_avg[name] = avg
    turnover_ts[name]  = ts

# ── Transaction-cost-adjusted returns ─────────────────────────────────────────
def apply_tc(daily_rets, ts):
    adj = daily_rets.copy()
    for date, to in ts.items():
        if date in adj.index:
            adj.loc[date] -= to * TC_BPS
    return adj

adj_rets = {name: apply_tc(all_rets[name], turnover_ts[name])
            for name in STRAT_FNS if name in all_rets.columns}

# ── Metrics ────────────────────────────────────────────────────────────────────
def metrics(s):
    cum   = (1 + s).cumprod()
    n_yrs = len(s) / 252
    ann_r = cum.iloc[-1] ** (1 / n_yrs) - 1
    ann_v = s.std() * np.sqrt(252)
    sh    = ann_r / ann_v if ann_v > 0 else np.nan
    dd    = ((cum / cum.cummax()) - 1).min()
    return ann_r, ann_v, sh, dd

rows = []
for name in strat_names:
    r, v, sh, dd = metrics(all_rets[name])
    to           = turnover_avg.get(name, np.nan)
    adj_r, _, adj_sh, _ = metrics(adj_rets[name]) if name in adj_rets else (np.nan,)*4
    rows.append({"Strategy": name, "Ann Ret": r, "Ann Vol": v, "Sharpe": sh,
                 "Max DD": dd, "Turnover": to, "TC-Adj Ret": adj_r, "TC-Adj Sharpe": adj_sh})

met_df = pd.DataFrame(rows).set_index("Strategy")
met_df.to_csv("data/phase5_results/metrics_full.csv")

fmt = lambda col, f: met_df[col].map(f)
disp = pd.DataFrame({
    "Ann Ret":       fmt("Ann Ret",       "{:.2%}".format),
    "Ann Vol":       fmt("Ann Vol",       "{:.2%}".format),
    "Sharpe":        fmt("Sharpe",        "{:.2f}".format),
    "Max DD":        fmt("Max DD",        "{:.2%}".format),
    "Turnover":      fmt("Turnover",      "{:.3f}".format),
    "TC-Adj Ret":    fmt("TC-Adj Ret",    "{:.2%}".format),
    "TC-Adj Sharpe": fmt("TC-Adj Sharpe", "{:.2f}".format),
})
print("\n─── Cost-Adjusted Metrics " + "─" * 50)
print(disp.to_string())
print("─" * 76)

# ── Plot styling ───────────────────────────────────────────────────────────────
BASELINE_STYLES = {
    "Equal Weight": ("-",  "#cccccc"),
    "Inverse Vol":  ("--", "#aaaaaa"),
    "Min Variance": ("-.", "#888888"),
    "HRP":          (":",  "#666666"),
}
GRAPH_COLORS = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#6a4c93"]

def plot_baselines(ax):
    for name, (ls, color) in BASELINE_STYLES.items():
        yield name, ls, color

# ── [1] Rolling Sharpe ─────────────────────────────────────────────────────────
roll_sh = (all_rets.rolling(252).mean() * np.sqrt(252)) / all_rets.rolling(252).std()

fig, ax = plt.subplots(figsize=(14, 6))
for name, (ls, color) in BASELINE_STYLES.items():
    s = roll_sh[name].dropna()
    ax.plot(s.index, s.values, color=color, linestyle=ls, lw=1.3, label=name, alpha=0.85)
for name, color in zip(GRAPH_NAMES, GRAPH_COLORS):
    s = roll_sh[name].dropna()
    ax.plot(s.index, s.values, color=color, lw=1.7, label=name)
ax.axhline(0, color="black", lw=0.6, linestyle=":")
ax.set_title("Rolling 252-Day Sharpe Ratio — All Strategies", fontsize=13)
ax.set_ylabel("Annualised Sharpe Ratio")
ax.legend(fontsize=7.5, ncol=3, loc="upper left", framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/phase5_results/rolling_sharpe.png", dpi=150)
plt.close()
print("\nSaved rolling_sharpe.png")

# ── [2] Drawdown curves ────────────────────────────────────────────────────────
def drawdown(s):
    cum = (1 + s).cumprod()
    return (cum / cum.cummax() - 1) * 100

CRISES = [
    ("2011-07-22", "2011-12-15", "2011\nEU Debt"),
    ("2020-02-15", "2020-05-31", "COVID-19\nMar-2020"),
    ("2022-01-01", "2022-12-31", "2022 Rate\nHike Cycle"),
]

fig, ax = plt.subplots(figsize=(14, 7))
for name, (ls, color) in BASELINE_STYLES.items():
    dd = drawdown(all_rets[name])
    ax.plot(dd.index, dd.values, color=color, linestyle=ls, lw=1.2, label=name, alpha=0.8)
for name, color in zip(GRAPH_NAMES, GRAPH_COLORS):
    dd = drawdown(all_rets[name])
    ax.plot(dd.index, dd.values, color=color, lw=1.7, label=name)

# Add crisis shading after all lines are drawn so ylim is stable
for t0_str, t1_str, label in CRISES:
    t0, t1 = pd.Timestamp(t0_str), pd.Timestamp(t1_str)
    ax.axvspan(t0, t1, alpha=0.13, color="crimson", zorder=0)
    ylo, yhi = ax.get_ylim()
    ax.text(t0 + (t1 - t0) / 2, ylo * 0.85, label,
            ha="center", va="bottom", fontsize=7.5, color="crimson", fontweight="bold")

ax.set_title("Drawdown Curves — All Strategies (crisis periods shaded)", fontsize=13)
ax.set_ylabel("Drawdown (%)")
ax.legend(fontsize=7.5, ncol=3, loc="lower left", framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/phase5_results/drawdown_curves.png", dpi=150)
plt.close()
print("Saved drawdown_curves.png")

# ── [3] Calendar year returns ──────────────────────────────────────────────────
yr_rets = {}
for yr in sorted(all_rets.index.year.unique()):
    mask        = all_rets.index.year == yr
    yr_rets[yr] = (1 + all_rets.loc[mask]).prod() - 1
yr_rets = pd.DataFrame(yr_rets).T
yr_rets.index.name = "Year"
yr_rets.to_csv("data/phase5_results/calendar_year_returns.csv")

fmt_pct = lambda x: f"{x:.1%}" if pd.notna(x) else "—"
yr_display = yr_rets.apply(lambda col: col.map(fmt_pct))

print("\n─── Calendar Year Returns " + "─" * 50)
print(yr_display.to_string())
print("─" * 76)

# Styled matplotlib table
cell_text   = yr_rets.apply(lambda col: col.map(fmt_pct)).values
cell_colors = np.full(yr_rets.shape, "#f9f9f9", dtype=object)
cols_list   = list(yr_rets.columns)
for r in range(len(yr_rets)):
    row   = yr_rets.iloc[r].dropna()
    if len(row) >= 2:
        cell_colors[r, cols_list.index(row.idxmax())] = "#b7e4b7"  # green
        cell_colors[r, cols_list.index(row.idxmin())] = "#f5b7b1"  # red

fig, ax = plt.subplots(figsize=(max(14, n_strats * 1.5), len(yr_rets) * 0.5 + 1.5))
ax.axis("off")
tbl = ax.table(cellText=cell_text, rowLabels=yr_rets.index.astype(str),
               colLabels=cols_list, cellColours=cell_colors, loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.45)
ax.set_title("Calendar Year Returns  (green = best, red = worst per year)", fontsize=11, pad=8)
plt.tight_layout()
plt.savefig("data/phase5_results/calendar_year_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved calendar_year_table.png")

# ── [4] Turnover bar chart ─────────────────────────────────────────────────────
to_vals    = [turnover_avg[n] for n in strat_names]
bar_colors = ["#999999"] * len(BASELINE_NAMES) + GRAPH_COLORS[:len(GRAPH_NAMES)]

fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(range(n_strats), to_vals, color=bar_colors, alpha=0.87)
ax.set_xticks(range(n_strats))
ax.set_xticklabels(strat_names, rotation=32, ha="right", fontsize=9)
ax.set_ylabel("Avg monthly one-way turnover")
ax.set_title("Average Monthly Turnover per Strategy", fontsize=12)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("data/phase5_results/turnover.png", dpi=150)
plt.close()
print("Saved turnover.png")

# ── [5] Gross vs TC-adjusted Sharpe ───────────────────────────────────────────
gross_sh = [met_df.loc[n, "Sharpe"]        for n in strat_names]
adj_sh   = [met_df.loc[n, "TC-Adj Sharpe"] for n in strat_names]
x = np.arange(n_strats)
w = 0.38

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - w/2, gross_sh, w, label="Gross Sharpe",    color="#457b9d", alpha=0.87)
ax.bar(x + w/2, adj_sh,   w, label="TC-Adj Sharpe",   color="#e63946", alpha=0.87)
ax.set_xticks(x)
ax.set_xticklabels(strat_names, rotation=32, ha="right", fontsize=9)
ax.set_title("Gross vs Transaction-Cost-Adjusted Sharpe (10 bps / unit turnover)", fontsize=12)
ax.set_ylabel("Sharpe Ratio")
ax.axhline(0, color="black", lw=0.5)
ax.legend(fontsize=10)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("data/phase5_results/sharpe_gross_vs_adj.png", dpi=150)
plt.close()
print("Saved sharpe_gross_vs_adj.png")

# ── [6] Strategy return correlation heatmap ────────────────────────────────────
corr_mat = all_rets.corr()
corr_mat.to_csv("data/phase5_results/strategy_correlations.csv")

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr_mat, ax=ax, cmap="RdYlGn", vmin=-1, vmax=1,
            annot=True, fmt=".2f", annot_kws={"size": 8},
            linewidths=0.4, square=True)
ax.set_title("Strategy Return Correlation Matrix", fontsize=13)
plt.tight_layout()
plt.savefig("data/phase5_results/strategy_correlation_heatmap.png", dpi=150)
plt.close()
print("Saved strategy_correlation_heatmap.png")

print("\nPhase 5 complete.")
