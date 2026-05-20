import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity as nx_modularity

warnings.filterwarnings("ignore")

os.makedirs("data/phase3_results", exist_ok=True)

ESTIMATION_WINDOW = 252
REBALANCE_FREQ    = 21

# ── Asset class colours ────────────────────────────────────────────────────────
CLASS_MAP = {
    "USO": ("Energy",      "tomato"),
    "XLE": ("Energy",      "tomato"),
    "UNG": ("Energy",      "tomato"),
    "BNO": ("Energy",      "tomato"),
    "GLD": ("Metals",      "goldenrod"),
    "SLV": ("Metals",      "goldenrod"),
    "COPX":("Metals",      "goldenrod"),
    "SPY": ("Equities",    "steelblue"),
    "EEM": ("Equities",    "steelblue"),
    "EWJ": ("Equities",    "steelblue"),
    "TLT": ("Bonds",       "mediumseagreen"),
    "IEF": ("Bonds",       "mediumseagreen"),
    "HYG": ("Bonds",       "mediumseagreen"),
    "DBA": ("Agriculture", "saddlebrown"),
    "WEAT":("Agriculture", "saddlebrown"),
    "CORN":("Agriculture", "saddlebrown"),
    "FXE": ("FX",          "mediumpurple"),
    "UUP": ("FX",          "mediumpurple"),
}
LEGEND_CLASSES = {
    "Energy": "tomato", "Metals": "goldenrod", "Equities": "steelblue",
    "Bonds": "mediumseagreen", "Agriculture": "saddlebrown", "FX": "mediumpurple",
}

# ── Load ───────────────────────────────────────────────────────────────────────
log_returns = pd.read_csv("data/log_returns.csv", index_col=0, parse_dates=True)
assets = log_returns.columns.tolist()
n      = len(assets)
dates  = log_returns.index
print(f"Loaded: {dates[0].date()} → {dates[-1].date()}, {n} assets")

rebal_indices = list(range(ESTIMATION_WINDOW, len(dates), REBALANCE_FREQ))
rebal_dates   = [dates[i] for i in rebal_indices]
print(f"Rebalance dates: {len(rebal_dates)}")

# ── Step 1: Rolling correlation matrices ───────────────────────────────────────
print("\n[1/5] Computing rolling correlations...")
corr_matrices = {}
for i in rebal_indices:
    window = log_returns.iloc[i - ESTIMATION_WINDOW : i]
    corr_matrices[dates[i]] = window.corr()
print(f"  Done — {len(corr_matrices)} matrices")

# ── Shared helpers ─────────────────────────────────────────────────────────────
def corr_to_dist(corr_df):
    """Mantegna (1999) correlation distance: sqrt(2*(1-rho)), range [0,2]."""
    d = np.sqrt(np.clip(2.0 * (1.0 - corr_df.values), 0, None))
    np.fill_diagonal(d, 0.0)
    return d

def complete_graph(corr_df):
    """Fully connected graph with distance edge weights."""
    assets_ = corr_df.columns.tolist()
    dist    = corr_to_dist(corr_df)
    G = nx.Graph()
    G.add_nodes_from(assets_)
    for i in range(len(assets_)):
        for j in range(i + 1, len(assets_)):
            G.add_edge(assets_[i], assets_[j], weight=float(dist[i, j]))
    return G

# ── Step 2: MST ────────────────────────────────────────────────────────────────
print("\n[2/5] Building MSTs...")
mst_graphs = {date: nx.minimum_spanning_tree(complete_graph(corr), weight="weight")
              for date, corr in corr_matrices.items()}
print(f"  Done — {len(mst_graphs)} MSTs  (edges per graph: {n - 1})")

# ── Step 3: PMFG ──────────────────────────────────────────────────────────────
def build_pmfg(corr_df):
    """
    Planar Maximally Filtered Graph (Tumminello et al. 2005).
    Greedily add shortest-distance edges while maintaining planarity.
    Target: 3*(n-2) edges.
    """
    assets_ = corr_df.columns.tolist()
    n_      = len(assets_)
    dist    = corr_to_dist(corr_df)
    max_edges = 3 * (n_ - 2)

    # All candidate edges sorted by distance ascending
    edge_candidates = sorted(
        [(dist[i, j], assets_[i], assets_[j])
         for i in range(n_) for j in range(i + 1, n_)]
    )

    G = nx.Graph()
    G.add_nodes_from(assets_)
    for d, u, v in edge_candidates:
        if G.number_of_edges() >= max_edges:
            break
        G.add_edge(u, v, weight=float(d))
        is_planar, _ = nx.check_planarity(G)
        if not is_planar:
            G.remove_edge(u, v)
    return G

print("\n[3/5] Building PMFGs (planarity checks — takes ~1 min)...")
pmfg_graphs = {}
for k, (date, corr) in enumerate(corr_matrices.items()):
    pmfg_graphs[date] = build_pmfg(corr)
    if (k + 1) % 30 == 0 or (k + 1) == len(corr_matrices):
        target = 3 * (n - 2)
        actual = pmfg_graphs[date].number_of_edges()
        print(f"  {k+1:3d}/{len(corr_matrices)}  last graph: {actual}/{target} edges")
print(f"  Done — {len(pmfg_graphs)} PMFGs")

# ── Step 4: Visualisation ──────────────────────────────────────────────────────
print("\n[4/5] Drawing network plots...")

def find_nearest_rebal(target_str):
    t = pd.Timestamp(target_str)
    return min(rebal_dates, key=lambda d: abs(d - t))

VIZ_TARGETS = {
    "calm_2017":   find_nearest_rebal("2017-07-01"),
    "crisis_2020": find_nearest_rebal("2020-03-15"),
    "recent_2024": find_nearest_rebal("2024-01-15"),
}
VIZ_LABELS = {
    "calm_2017":   "Calm (mid-2017)",
    "crisis_2020": "Crisis (Mar-2020)",
    "recent_2024": "Recent (2024)",
}

legend_patches = [mpatches.Patch(color=c, label=cls) for cls, c in LEGEND_CLASSES.items()]

def draw_graph(G, ax, title, seed=0):
    node_colors = [CLASS_MAP.get(nd, ("?", "lightgrey"))[1] for nd in G.nodes()]
    pos = nx.spring_layout(G, seed=seed)

    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    if edge_weights:
        w_min, w_max = min(edge_weights), max(edge_weights)
        span = w_max - w_min + 1e-9
        # thicker line = shorter distance = stronger correlation
        widths = [2.5 * (1 - (w - w_min) / span) + 0.4 for w in edge_weights]
    else:
        widths = [1.0]

    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, alpha=0.45, edge_color="grey")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=480, alpha=0.92)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_weight="bold")
    ax.set_title(title, fontsize=10, pad=6)
    ax.axis("off")

for key, date in VIZ_TARGETS.items():
    label = VIZ_LABELS[key]

    # MST
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_graph(mst_graphs[date], ax, f"MST — {label}  ({date.date()})")
    fig.legend(handles=legend_patches, loc="lower center", ncol=6, fontsize=8, framealpha=0.8)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(f"data/phase3_results/{key}_mst.png", dpi=150, bbox_inches="tight")
    plt.close()

    # PMFG
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_graph(pmfg_graphs[date], ax, f"PMFG — {label}  ({date.date()})")
    fig.legend(handles=legend_patches, loc="lower center", ncol=6, fontsize=8, framealpha=0.8)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(f"data/phase3_results/{key}_pmfg.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved {key}_mst.png + {key}_pmfg.png")

# ── Step 5: Network statistics over time ───────────────────────────────────────
print("\n[5/5] Computing network statistics over time...")

def safe_aspl(G):
    """Average shortest path length; handles disconnected graphs."""
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G, weight="weight")
    # fallback: average over connected components
    lengths = []
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        if sub.number_of_nodes() > 1:
            lengths.append(nx.average_shortest_path_length(sub, weight="weight"))
    return float(np.mean(lengths)) if lengths else np.nan

def safe_modularity(G):
    try:
        comms = greedy_modularity_communities(G)
        return nx_modularity(G, comms)
    except Exception:
        return np.nan

rows = []
for date in rebal_dates:
    mst  = mst_graphs[date]
    pmfg = pmfg_graphs[date]

    mst_dc  = nx.degree_centrality(mst)
    pmfg_dc = nx.degree_centrality(pmfg)

    row = {
        "date":            date,
        "mst_aspl":        safe_aspl(mst),
        "mst_modularity":  safe_modularity(mst),
        "pmfg_aspl":       safe_aspl(pmfg),
        "pmfg_density":    nx.density(pmfg),
        "pmfg_modularity": safe_modularity(pmfg),
    }
    for a in assets:
        row[f"mst_deg_{a}"]  = mst_dc.get(a, 0.0)
        row[f"pmfg_deg_{a}"] = pmfg_dc.get(a, 0.0)
    rows.append(row)

stats = pd.DataFrame(rows).set_index("date")
stats.to_csv("data/phase3_results/network_stats.csv")
print("  Saved network_stats.csv")

# ── Stats plot 1: structural metrics ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

axes[0, 0].plot(stats.index, stats["mst_aspl"], color="steelblue", lw=1.3)
axes[0, 0].set_title("MST — Avg Shortest Path Length")
axes[0, 0].set_ylabel("Weighted distance")
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(stats.index, stats["pmfg_aspl"], color="darkorange", lw=1.3)
axes[0, 1].set_title("PMFG — Avg Shortest Path Length")
axes[0, 1].set_ylabel("Weighted distance")
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(stats.index, stats["mst_modularity"],  color="steelblue",  lw=1.3, label="MST")
axes[1, 0].plot(stats.index, stats["pmfg_modularity"], color="darkorange", lw=1.3, label="PMFG")
axes[1, 0].set_title("Modularity")
axes[1, 0].set_ylabel("Q")
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(stats.index, stats["pmfg_density"], color="mediumseagreen", lw=1.3)
axes[1, 1].set_title("PMFG — Graph Density")
axes[1, 1].set_ylabel("Density")
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle("Correlation Network — Structural Statistics Over Time", fontsize=13)
plt.tight_layout()
plt.savefig("data/phase3_results/stats_structural.png", dpi=150)
plt.close()
print("  Saved stats_structural.png")

# ── Stats plot 2: top-5 MST centrality over time ──────────────────────────────
mst_deg_cols = [c for c in stats.columns if c.startswith("mst_deg_")]
avg_cent = stats[mst_deg_cols].mean().sort_values(ascending=False)
top5_assets = [c.replace("mst_deg_", "") for c in avg_cent.index[:5]]

fig, ax = plt.subplots(figsize=(13, 5))
for asset in top5_assets:
    col   = f"mst_deg_{asset}"
    color = CLASS_MAP.get(asset, ("?", "grey"))[1]
    ax.plot(stats.index, stats[col], label=asset, color=color, lw=1.4)

# Mark crisis window
ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"),
           alpha=0.12, color="red", label="2020 crisis")
ax.set_title("MST Degree Centrality — Top 5 Assets Over Time", fontsize=12)
ax.set_ylabel("Degree centrality")
ax.legend(fontsize=9, ncol=3)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/phase3_results/stats_centrality.png", dpi=150)
plt.close()
print("  Saved stats_centrality.png")

# ── Summary printout ───────────────────────────────────────────────────────────
print("\n─── Most Central Assets — MST (avg degree centrality) ──────")
for a, v in zip(top5_assets, avg_cent.values[:5]):
    cls = CLASS_MAP.get(a, ("?",))[0]
    print(f"  {a:6s} ({cls:11s}): {v:.3f}")

pmfg_deg_cols = [c for c in stats.columns if c.startswith("pmfg_deg_")]
avg_pmfg = stats[pmfg_deg_cols].mean().sort_values(ascending=False)
top5_pmfg = [c.replace("pmfg_deg_", "") for c in avg_pmfg.index[:5]]
print("\n─── Most Central Assets — PMFG (avg degree centrality) ─────")
for a, v in zip(top5_pmfg, avg_pmfg.values[:5]):
    cls = CLASS_MAP.get(a, ("?",))[0]
    print(f"  {a:6s} ({cls:11s}): {v:.3f}")

# Crisis vs calm centrality shift (MST)
calm_mask   = (stats.index >= "2016-01-01") & (stats.index <= "2019-12-31")
crisis_mask = (stats.index >= "2020-01-01") & (stats.index <= "2020-12-31")

calm_cent   = stats.loc[calm_mask,   mst_deg_cols].mean()
crisis_cent = stats.loc[crisis_mask, mst_deg_cols].mean()
delta = (crisis_cent - calm_cent).sort_values(ascending=False)
delta.index = [c.replace("mst_deg_", "") for c in delta.index]

print("\n─── MST Centrality Shift: Crisis-2020 vs Calm-2016-2019 ────")
print("  Gaining centrality (became more central hubs):")
for a, v in delta.head(5).items():
    print(f"  {a:6s}: {v:+.3f}")
print("  Losing centrality (retreated to periphery):")
for a, v in delta.tail(5).items():
    print(f"  {a:6s}: {v:+.3f}")

print("\nPhase 3 complete.")
