import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import riskfolio as rp
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("WARNING: arch not found — DCC approximation will be skipped. pip install arch")

ESTIMATION_WINDOW = 252
REBALANCE_FREQ    = 21
EWMA_LAMBDAS      = [0.94, 0.97, 0.99]

os.makedirs("data/phase6_results", exist_ok=True)

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
print(f"Rebalance dates: {len(rebal_dates)}")

# ── Step 1: EWMA correlation at each rebalance date ────────────────────────────
def ewma_corr_snapshots(returns_df, lam, rebal_idx_set, all_dates, init_days=63):
    """
    Incremental EWMA covariance/correlation.
    Snapshot is taken BEFORE incorporating day-i return, so the correlation
    at rebalance index i uses data only through i-1 — no lookahead.
    """
    vals   = returns_df.values
    S      = np.cov(vals[:init_days].T)   # initialise with short sample cov
    corr_at = {}
    cov_at  = {}

    for t in range(init_days, len(vals)):
        # Snapshot before update → uses data up to t-1
        if t in rebal_idx_set:
            d_inv  = np.where(np.diag(S) > 0, 1.0 / np.sqrt(np.diag(S)), 0.0)
            corr   = d_inv[:, None] * S * d_inv[None, :]
            np.fill_diagonal(corr, 1.0)
            corr   = np.clip(corr, -1.0, 1.0)
            corr_at[all_dates[t]] = pd.DataFrame(corr, index=returns_df.columns,
                                                  columns=returns_df.columns)
            cov_at[all_dates[t]]  = S.copy()
        r = vals[t]
        S = lam * S + (1.0 - lam) * np.outer(r, r)

    return corr_at, cov_at

print("\n[1/5] Computing EWMA correlation matrices...")
ewma_corr_all = {}
ewma_cov_all  = {}
for lam in EWMA_LAMBDAS:
    corr_snap, cov_snap = ewma_corr_snapshots(log_returns, lam, rebal_set, dates)
    ewma_corr_all[lam] = corr_snap
    ewma_cov_all[lam]  = cov_snap
    print(f"  λ={lam}: {len(corr_snap)} matrices")

# ── Step 2: DCC approximation via per-asset GARCH + EWMA on residuals ─────────
print("\n[2/5] Fitting GARCH(1,1) per asset...")
garch_std_resid = log_returns.copy().astype(float)
garch_success   = 0
dcc_available   = False

if ARCH_AVAILABLE:
    for asset in assets:
        r_pct = log_returns[asset].dropna() * 100   # scale to % for stability
        try:
            res = arch_model(r_pct, vol="Garch", p=1, q=1,
                             dist="normal", rescale=False).fit(
                disp="off", options={"maxiter": 500, "ftol": 1e-6}
            )
            cond_vol = res.conditional_volatility.reindex(dates) / 100
            # Fallback to rolling std for any missing dates
            roll_std = log_returns[asset].rolling(21, min_periods=5).std()
            cond_vol = cond_vol.fillna(roll_std).replace(0, np.nan).fillna(roll_std)
            garch_std_resid[asset] = log_returns[asset] / cond_vol
            garch_success += 1
        except Exception as e:
            print(f"  {asset}: GARCH failed ({e}) — using raw returns")

    garch_std_resid = garch_std_resid.ffill().bfill()
    print(f"  GARCH converged: {garch_success}/{n} assets")

    if garch_success >= int(n * 0.8):
        # Note: GARCH parameters use full-sample data — mild lookahead in
        # volatility filter; the correlation EWMA itself has no lookahead.
        dcc_corr_snap, dcc_cov_snap = ewma_corr_snapshots(
            garch_std_resid, 0.94, rebal_set, dates
        )
        dcc_available = True
        print(f"  DCC-approx (GARCH resids + EWMA λ=0.94): {len(dcc_corr_snap)} matrices")
    else:
        print("  <80% GARCH fits — DCC approximation skipped")
else:
    print("  arch not installed — DCC skipped")

# ── Graph helpers ──────────────────────────────────────────────────────────────
def corr_to_dist(corr_df):
    d = np.sqrt(np.clip(2.0 * (1.0 - corr_df.values), 0, None))
    np.fill_diagonal(d, 0.0)
    return d

def build_pmfg(corr_df):
    a    = corr_df.columns.tolist()
    n_   = len(a)
    dist = corr_to_dist(corr_df)
    edges = sorted((dist[i, j], a[i], a[j])
                   for i in range(n_) for j in range(i + 1, n_))
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

# ── Step 3: Build all PMFGs in a single pass ───────────────────────────────────
print("\n[3/5] Building PMFGs for all methods (single pass, ~3–5 min)...")

method_keys = [f"EWMA-{lam}" for lam in EWMA_LAMBDAS] + ["Static-252"]
if dcc_available:
    method_keys.append("DCC-approx")

dynamic_pmfgs = {k: {} for k in method_keys}
degree_cent   = {k: {} for k in method_keys}   # {method: {date: {asset: centrality}}}

for k, i in enumerate(rebal_indices):
    date   = dates[i]
    window = log_returns.iloc[i - ESTIMATION_WINDOW : i]

    # EWMA PMFGs
    for lam in EWMA_LAMBDAS:
        key  = f"EWMA-{lam}"
        corr = ewma_corr_all[lam].get(date)
        if corr is not None:
            G = build_pmfg(corr)
            dynamic_pmfgs[key][date] = G
            degree_cent[key][date]   = nx.degree_centrality(G)

    # DCC PMFG
    if dcc_available:
        corr = dcc_corr_snap.get(date)
        if corr is not None:
            G = build_pmfg(corr)
            dynamic_pmfgs["DCC-approx"][date] = G
            degree_cent["DCC-approx"][date]   = nx.degree_centrality(G)

    # Static rolling-window PMFG (for centrality comparison)
    G_static = build_pmfg(window.corr())
    dynamic_pmfgs["Static-252"][date] = G_static
    degree_cent["Static-252"][date]   = nx.degree_centrality(G_static)

    if (k + 1) % 30 == 0 or (k + 1) == len(rebal_indices):
        print(f"  {k+1}/{len(rebal_indices)}")

print("  Done.")

# ── HRP helpers ────────────────────────────────────────────────────────────────
def hrp_from(window_df):
    """Standard HRP on a returns DataFrame (raw or residuals)."""
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
    """Scale weights by 1/centrality, renormalise."""
    cent = nx.degree_centrality(G)
    c    = np.array([cent.get(a, 1.0 / max(n - 1, 1)) for a in assets])
    c    = np.clip(c, 1e-6, None)
    adj  = base_w / c
    return adj / adj.sum()

# ── Step 4: Walk-forward for all strategy variants ─────────────────────────────
print("\n[4/5] Walk-forward backtests...")

# Strategy config: (name, pmfg_key, use_garch_residuals_for_hrp)
strat_configs = [
    ("CentHRP-EWMA-0.94",  "EWMA-0.94",    False),
    ("CentHRP-EWMA-0.97",  "EWMA-0.97",    False),
    ("CentHRP-EWMA-0.99",  "EWMA-0.99",    False),
    ("DynHRP-EWMA-0.94",   "EWMA-0.94",    True),   # HRP on GARCH residuals
]
if dcc_available:
    strat_configs += [
        ("CentHRP-DCC",    "DCC-approx",   False),
        ("DynHRP-DCC",     "DCC-approx",   True),
    ]

all_port_rets = {}

for strat_name, pmfg_key, use_resid in strat_configs:
    print(f"  {strat_name}...")
    pmfg_cache = dynamic_pmfgs[pmfg_key]
    current_w  = None
    ret_vals, ret_dates_ = [], []

    for i in range(T):
        if i in rebal_set:
            date   = dates[i]
            window = log_returns.iloc[i - ESTIMATION_WINDOW : i]
            G      = pmfg_cache.get(date) or dynamic_pmfgs["Static-252"].get(date)

            if use_resid:
                w_resid = garch_std_resid.iloc[i - ESTIMATION_WINDOW : i][assets]
                w_resid = w_resid.dropna()
                base_w  = hrp_from(w_resid) if len(w_resid) >= 30 else hrp_from(window)
            else:
                base_w = hrp_from(window)

            current_w = cent_adjust(base_w, G)

        if current_w is not None:
            ret_vals.append((simple_returns.iloc[i].values * current_w).sum())
            ret_dates_.append(dates[i])

    all_port_rets[strat_name] = pd.Series(ret_vals, index=ret_dates_)

pd.DataFrame(all_port_rets).to_csv("data/phase6_results/daily_returns.csv")
print("  Saved daily_returns.csv")

# ── Load benchmarks from earlier phases ────────────────────────────────────────
ph2 = pd.read_csv("data/phase2_results/daily_returns.csv", index_col=0, parse_dates=True)
ph4 = pd.read_csv("data/phase4_results/daily_returns.csv", index_col=0, parse_dates=True)

benchmarks = pd.DataFrame({
    "HRP":                  ph2["HRP"],
    "CentHRP-PMFG-Static":  ph4["CentHRP-PMFG"],
})
all_compare = pd.concat([benchmarks, pd.DataFrame(all_port_rets)], axis=1)

# ── Metrics ────────────────────────────────────────────────────────────────────
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
for name in all_compare.columns:
    r, v, sh, dd = metrics(all_compare[name])
    rows.append({"Strategy": name, "Ann Ret": r, "Ann Vol": v,
                 "Sharpe": sh, "Max DD": dd})
met_df = pd.DataFrame(rows).set_index("Strategy")
met_df.to_csv("data/phase6_results/metrics.csv")

disp = pd.DataFrame({
    "Ann Ret": met_df["Ann Ret"].map("{:.2%}".format),
    "Ann Vol": met_df["Ann Vol"].map("{:.2%}".format),
    "Sharpe":  met_df["Sharpe"].map("{:.2f}".format),
    "Max DD":  met_df["Max DD"].map("{:.2%}".format),
})
print("\n─── Phase 6 Metrics " + "─" * 55)
print(disp.to_string())
print("─" * 75)

ewma_cent_names = [n for n in all_compare.columns
                   if n.startswith("CentHRP-EWMA")]
if ewma_cent_names:
    best  = met_df.loc[ewma_cent_names, "Sharpe"].idxmax()
    worst = met_df.loc[ewma_cent_names, "Sharpe"].idxmin()
    print(f"\nBest EWMA lambda  : {best} → Sharpe {met_df.loc[best,  'Sharpe']:.2f}")
    print(f"Worst EWMA lambda : {worst} → Sharpe {met_df.loc[worst, 'Sharpe']:.2f}")
    print(
        f"\nInterpretation: {'faster' if '0.94' in best else 'slower'}-decay EWMA "
        f"({'λ=0.94 reacts quickly to regime shifts' if '0.94' in best else 'λ=0.99 is more stable, less whipsaw'})"
    )

# ── Step 5: Centrality dynamics comparison ─────────────────────────────────────
print("\n[5/5] Plotting centrality dynamics and results...")

WATCH_ASSETS = ["SPY", "COPX"]
CRISES = [
    ("2020-02-15", "2020-05-31", "COVID-19\nMar-2020"),
    ("2022-01-01", "2022-12-31", "2022 Rate\nHike Cycle"),
]

cent_methods = {"Static-252": ("#888888", "--"),
                "EWMA-0.94":  ("#e63946", "-")}
if dcc_available:
    cent_methods["DCC-approx"] = ("#2a9d8f", "-.")

# Save centrality time series
for method in cent_methods:
    df = pd.DataFrame(degree_cent[method]).T
    df.index.name = "date"
    df.to_csv(f"data/phase6_results/centrality_{method.replace('-','_')}.csv")

fig, axes = plt.subplots(len(WATCH_ASSETS), 1,
                         figsize=(13, 4.5 * len(WATCH_ASSETS)), sharex=True)
if len(WATCH_ASSETS) == 1:
    axes = [axes]

for ax, asset in zip(axes, WATCH_ASSETS):
    for method, (color, ls) in cent_methods.items():
        ts = pd.Series({date: degree_cent[method][date].get(asset, np.nan)
                        for date in rebal_dates
                        if date in degree_cent[method]})
        ax.plot(ts.index, ts.values, label=method, color=color, ls=ls, lw=1.6)

    # Crisis shading after lines so ylim is settled
    for t0s, t1s, label in CRISES:
        t0, t1 = pd.Timestamp(t0s), pd.Timestamp(t1s)
        ax.axvspan(t0, t1, alpha=0.13, color="crimson", zorder=0)
        ylo, yhi = ax.get_ylim()
        ax.text(t0 + (t1 - t0) / 2, ylo + (yhi - ylo) * 0.04, label,
                ha="center", va="bottom", fontsize=7.5,
                color="crimson", fontweight="bold")

    ax.set_title(f"PMFG Degree Centrality — {asset}", fontsize=11)
    ax.set_ylabel("Centrality")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle("Centrality Dynamics: Static vs EWMA vs DCC", fontsize=13)
plt.tight_layout()
plt.savefig("data/phase6_results/centrality_dynamics.png", dpi=150)
plt.close()
print("  Saved centrality_dynamics.png")

# ── Plot: cumulative returns ────────────────────────────────────────────────────
BENCH_STYLE = {"HRP": ("--", "#aaaaaa"), "CentHRP-PMFG-Static": ("-.", "#777777")}
NEW_COLORS  = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#6a4c93", "#a8dadc"]
new_names   = [n for n in all_compare.columns if n not in BENCH_STYLE]

fig, ax = plt.subplots(figsize=(14, 7))
for name, (ls, color) in BENCH_STYLE.items():
    if name in all_compare.columns:
        c = (1 + all_compare[name].dropna()).cumprod()
        ax.plot(c.index, c.values, label=name, color=color, ls=ls, lw=1.3, alpha=0.8)
for name, color in zip(new_names, NEW_COLORS):
    c = (1 + all_compare[name].dropna()).cumprod()
    ax.plot(c.index, c.values, label=name, color=color, lw=1.7)
ax.axhline(1, color="black", lw=0.5, ls=":")
ax.set_title("Cumulative Returns — Dynamic Correlation Strategies vs Benchmarks",
             fontsize=13)
ax.set_ylabel("Growth of $1")
ax.legend(fontsize=8.5, ncol=2, loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/phase6_results/cumulative_returns.png", dpi=150)
plt.close()
print("  Saved cumulative_returns.png")

# ── Plot: rolling 252-day Sharpe ───────────────────────────────────────────────
roll_sh = ((all_compare.rolling(252).mean() * np.sqrt(252))
           / all_compare.rolling(252).std())

fig, ax = plt.subplots(figsize=(14, 6))
for name, (ls, color) in BENCH_STYLE.items():
    if name in roll_sh.columns:
        s = roll_sh[name].dropna()
        ax.plot(s.index, s.values, label=name, color=color, ls=ls, lw=1.3, alpha=0.8)
for name, color in zip(new_names, NEW_COLORS):
    s = roll_sh[name].dropna()
    ax.plot(s.index, s.values, label=name, color=color, lw=1.7)
ax.axhline(0, color="black", lw=0.6, ls=":")
ax.set_title("Rolling 252-Day Sharpe — Dynamic Correlation Strategies", fontsize=13)
ax.set_ylabel("Annualised Sharpe Ratio")
ax.legend(fontsize=8.5, ncol=2, loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/phase6_results/rolling_sharpe.png", dpi=150)
plt.close()
print("  Saved rolling_sharpe.png")

print("\nPhase 6 complete.")
