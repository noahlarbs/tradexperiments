import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import networkx as nx
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ESTIMATION_WINDOW = 252
MAX_LAG           = 5       # average net-flow over lags 1..5
FLOW_PCTILE       = 80      # sparsify: keep top 20% of positive net-flow edges
START             = "2010-01-01"
END               = pd.Timestamp.today().strftime("%Y-%m-%d")
TRAIN_END         = "2018-12-31"   # everything after this is genuine OOS

os.makedirs("data/phase14_results", exist_ok=True)

TICKERS = ["USO", "UNG", "GLD", "SLV", "COPX", "DBA", "UUP", "FXA", "EEM"]

print(f"\n[1/5] Downloading {len(TICKERS)} assets from {START} to {END}...")
raw = yf.download(TICKERS, start=START, end=END, progress=False)["Close"]
raw.index = pd.to_datetime(raw.index)

prices = raw.ffill(limit=5).dropna()
log_returns    = np.log(prices / prices.shift(1)).dropna()
simple_returns = np.expm1(log_returns)

assets = log_returns.columns.tolist()
n      = len(assets)
dates  = log_returns.index
T      = len(dates)

train_cutoff   = pd.Timestamp(TRAIN_END)
test_start_idx = next(i for i, d in enumerate(dates) if d > train_cutoff)
print(f"Loaded {n} assets, {dates[0].date()} – {dates[-1].date()}")
print(f"Train: {dates[0].date()} – {TRAIN_END}  |  OOS test: {dates[test_start_idx].date()} – {dates[-1].date()}")

# ── Lag-k Net Information Flow ─────────────────────────────────────────────────
def net_flow_matrix(window_vals, max_lag=MAX_LAG):
    """
    Average directed net-information-flow over lags k = 1 .. max_lag.
    N[i,j] > 0  means i historically leads j.
    Uses lag-1 cross-correlation at each lag independently and averages.
    """
    W, N_ = window_vals.shape
    N_total = np.zeros((N_, N_))

    for k in range(1, max_lag + 1):
        R_lag = window_vals[: W - k, :]   # leader slice  [W-k, N]
        R_con = window_vals[k:,       :]   # follower slice [W-k, N]

        mu_l, std_l = R_lag.mean(0), R_lag.std(0) + 1e-8
        mu_c, std_c = R_con.mean(0), R_con.std(0) + 1e-8

        Z_lag = (R_lag - mu_l) / std_l
        Z_con = (R_con - mu_c) / std_c

        Wk    = Z_lag.shape[0]
        M_k   = (Z_lag.T @ Z_con) / Wk   # M[i,j] = corr(R_i[t-k], R_j[t])
        N_total += M_k - M_k.T            # net flow i→j minus j→i

    return N_total / max_lag

vals = log_returns.values

# ── Strategy Simulation ────────────────────────────────────────────────────────
print(f"\n[2/5] Simulating strategies (lag-1..{MAX_LAG} net flow)...")

port_rets = {
    "Equal Weight"              : np.zeros(T),
    "XS Momentum (1-day)"       : np.zeros(T),
    "Graph Spillover"           : np.zeros(T),
}

port_rets["Equal Weight"] = simple_returns.mean(axis=1).values

for t in range(ESTIMATION_WINDOW, T - 1):
    window      = vals[t - ESTIMATION_WINDOW : t + 1]
    ret_today   = vals[t]                           # known at end of day t
    ret_tmrw    = simple_returns.iloc[t + 1].values # realized next day

    # ── Cross-Sectional Momentum (1-day) ──────────────────────────────────────
    # Long top half by today's return, short bottom half — equal-notional legs
    ranks = np.argsort(ret_today)
    half  = max(n // 2, 1)
    w_xs  = np.zeros(n)
    w_xs[ranks[:half]]  = -1.0 / half
    w_xs[ranks[-half:]] =  1.0 / half
    port_rets["XS Momentum (1-day)"][t + 1] = w_xs @ ret_tmrw

    # ── Graph Spillover Momentum ───────────────────────────────────────────────
    N_mat     = net_flow_matrix(window)
    pos_vals  = N_mat[N_mat > 0]
    threshold = np.percentile(pos_vals, FLOW_PCTILE) if len(pos_vals) else 0.0
    A         = np.where(N_mat >= threshold, N_mat, 0.0)   # sparse directed adj

    # Signal for j = weighted sum of leaders' returns today
    signal = A.T @ ret_today
    w_graph = np.maximum(signal, 0.0)
    s = w_graph.sum()
    w_graph = w_graph / s if s > 1e-12 else np.ones(n) / n

    port_rets["Graph Spillover"][t + 1] = w_graph @ ret_tmrw

returns_df = pd.DataFrame(port_rets, index=dates).iloc[ESTIMATION_WINDOW + 1:]
returns_df.to_csv("data/phase14_results/daily_returns.csv")

# ── Metrics (full / train / OOS) ──────────────────────────────────────────────
def metrics(s):
    s = s[s != 0].dropna()
    if len(s) < 60:
        return dict(zip(["Ann Ret","Ann Vol","Sharpe","Max DD"],
                        [np.nan]*4))
    cum   = (1 + s).cumprod()
    n_yrs = len(s) / 252
    ann_r = cum.iloc[-1] ** (1 / n_yrs) - 1
    ann_v = s.std() * np.sqrt(252)
    sh    = ann_r / ann_v if ann_v > 0 else np.nan
    dd    = ((cum / cum.cummax()) - 1).min()
    return {"Ann Ret": ann_r, "Ann Vol": ann_v, "Sharpe": sh, "Max DD": dd}

def fmt(df):
    return pd.DataFrame({
        "Ann Ret": df["Ann Ret"].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "n/a"),
        "Ann Vol": df["Ann Vol"].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "n/a"),
        "Sharpe" : df["Sharpe" ].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "n/a"),
        "Max DD" : df["Max DD" ].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "n/a"),
    })

print("\n[3/5] Computing metrics...")
for label, mask in [
    ("Full 2011–present",     None),
    ("Train 2011–2018",       returns_df.index <= train_cutoff),
    ("OOS  2019–present",     returns_df.index >  train_cutoff),
]:
    sub = returns_df if mask is None else returns_df.loc[mask]
    rows = [{"Strategy": nm, **metrics(sub[nm])} for nm in sub.columns]
    met  = pd.DataFrame(rows).set_index("Strategy")
    if "OOS" in label:
        met.to_csv("data/phase14_results/metrics_oos.csv")
    elif "Train" in label:
        met.to_csv("data/phase14_results/metrics_train.csv")
    else:
        met.to_csv("data/phase14_results/metrics.csv")
    print(f"\n─── Phase 14 {label} ─────────────────────────────────────────────")
    print(fmt(met).to_string())

# ── Directed Graph Snapshot ────────────────────────────────────────────────────
print("\n[4/5] Saving directed-graph snapshot (most recent window)...")
N_last  = net_flow_matrix(vals[-ESTIMATION_WINDOW:])
thresh  = np.percentile(N_last[N_last > 0], FLOW_PCTILE) if (N_last > 0).any() else 0
A_last  = np.where(N_last >= thresh, N_last, 0.0)

G = nx.DiGraph()
G.add_nodes_from(assets)
for i in range(n):
    for j in range(n):
        if A_last[i, j] > 0:
            G.add_edge(assets[i], assets[j], weight=float(A_last[i, j]))

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
edge_weights = [G[u][v]["weight"] * 10 for u, v in G.edges()]
nx.draw_networkx_nodes(G, pos, node_size=800, node_color="lightblue", edgecolors="black")
nx.draw_networkx_edges(G, pos, edge_color="#e63946", arrowsize=20,
                       width=edge_weights, connectionstyle="arc3,rad=0.1")
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
plt.title(f"Directed Net-Flow Graph (avg lag 1–{MAX_LAG}, snapshot {END})\nThick arrows = high predictive flow",
          fontsize=13)
plt.axis("off")
plt.tight_layout()
plt.savefig("data/phase14_results/directed_graph.png", dpi=150)
plt.close()

# ── Performance Plot ───────────────────────────────────────────────────────────
print("[5/5] Generating cumulative return plot...")
STYLE = {
    "Equal Weight"        : (":",  "#aaaaaa"),
    "XS Momentum (1-day)" : ("--", "#888888"),
    "Graph Spillover"     : ("-",  "#e63946"),
}

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

for ax, period, label in [
    (axes[0], returns_df.index <= train_cutoff, f"Train 2011–{TRAIN_END[:4]}"),
    (axes[1], returns_df.index >  train_cutoff, f"OOS   {dates[test_start_idx].year}–present"),
]:
    sub = returns_df.loc[period]
    for nm in sub.columns:
        s = sub[nm]
        s = s[s != 0].reindex(sub.index).fillna(0)
        c = (1 + s).cumprod()
        ls, col = STYLE.get(nm, ("-", "steelblue"))
        lw = 2.0 if "Graph" in nm else 1.3
        ax.plot(c.index, c.values, label=nm, color=col, ls=ls, lw=lw, alpha=0.9)
    ax.axhline(1, color="black", lw=0.5, ls=":")
    ax.set_title(f"Cumulative Returns — {label}", fontsize=12)
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/phase14_results/cumulative_returns.png", dpi=150)
plt.close()

print("\nPhase 14 complete.")
