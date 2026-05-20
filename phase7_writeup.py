import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# ── Strategy palette (consistent across all 4 panels) ─────────────────────────
STRATS = {
    "Equal Weight":        ("#aaaaaa", "-",  1.2, "Equal Weight"),
    "HRP":                 ("#666666", "--", 1.4, "HRP"),
    "CentHRP-PMFG-Static": ("#457b9d", "-",  1.8, "CentHRP-Static"),
    "CentHRP-EWMA-0.94":   ("#e63946", "-",  2.2, "CentHRP-EWMA-0.94"),
}
COLORS = [v[0] for v in STRATS.values()]

SHORT_LABELS = ["Equal\nWeight", "HRP", "CentHRP\nStatic", "CentHRP\nEWMA-0.94"]

CRASH  = pd.Timestamp("2020-02-20")
CRISES = [
    (pd.Timestamp("2020-02-15"), pd.Timestamp("2020-06-01"), "COVID-19\n(2020)"),
    (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"), "Rate Hike\nCycle (2022)"),
]

# ── Load returns ───────────────────────────────────────────────────────────────
ph2 = pd.read_csv("data/phase2_results/daily_returns.csv", index_col=0, parse_dates=True)
ph4 = pd.read_csv("data/phase4_results/daily_returns.csv", index_col=0, parse_dates=True)
ph6 = pd.read_csv("data/phase6_results/daily_returns.csv", index_col=0, parse_dates=True)

rets = pd.DataFrame({
    "Equal Weight":        ph2["Equal Weight"],
    "HRP":                 ph2["HRP"],
    "CentHRP-PMFG-Static": ph4["CentHRP-PMFG"],
    "CentHRP-EWMA-0.94":   ph6["CentHRP-EWMA-0.94"],
})

# ── Load SPY centrality from phase 6 CSVs ─────────────────────────────────────
cent_static = pd.read_csv("data/phase6_results/centrality_Static_252.csv",
                          index_col=0, parse_dates=True)["SPY"]
cent_ewma   = pd.read_csv("data/phase6_results/centrality_EWMA_0.94.csv",
                          index_col=0, parse_dates=True)["SPY"]

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

strat_list  = list(rets.columns)
met         = {name: metrics(rets[name]) for name in strat_list}
sharpe_vals = [met[n][2] for n in strat_list]
dd_vals     = [met[n][3] for n in strat_list]

# ── Build figure ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30)
ax_tl = fig.add_subplot(gs[0, 0])
ax_tr = fig.add_subplot(gs[0, 1])
ax_bl = fig.add_subplot(gs[1, 0])
ax_br = fig.add_subplot(gs[1, 1])

# ──────────────────────────────────────────────────────────────────────────────
# Panel A — Cumulative returns
# ──────────────────────────────────────────────────────────────────────────────
for name, (color, ls, lw, label) in STRATS.items():
    cum = (1 + rets[name].dropna()).cumprod()
    ax_tl.plot(cum.index, cum.values, label=label,
               color=color, ls=ls, lw=lw)

ax_tl.axhline(1, color="black", lw=0.5, ls=":")
ax_tl.set_title("A  Cumulative Returns", fontsize=11, fontweight="bold", loc="left")
ax_tl.set_ylabel("Growth of $1")
ax_tl.legend(fontsize=9, loc="upper left", framealpha=0.88)
ax_tl.grid(True, alpha=0.22)
ax_tl.tick_params(axis="x", rotation=12)

# ──────────────────────────────────────────────────────────────────────────────
# Panel B — Drawdown curves
# ──────────────────────────────────────────────────────────────────────────────
for name, (color, ls, lw, label) in STRATS.items():
    s   = rets[name].dropna()
    cum = (1 + s).cumprod()
    dd  = (cum / cum.cummax() - 1) * 100
    ax_tr.plot(dd.index, dd.values, label=label,
               color=color, ls=ls, lw=lw)

for t0, t1, _ in CRISES:
    ax_tr.axvspan(t0, t1, alpha=0.11, color="crimson", zorder=0)

ylo, yhi = ax_tr.get_ylim()
for t0, t1, label in CRISES:
    ax_tr.text(t0 + (t1 - t0) / 2, ylo * 0.82, label,
               ha="center", va="bottom", fontsize=7, color="crimson", fontweight="bold")

ax_tr.set_title("B  Drawdown Curves", fontsize=11, fontweight="bold", loc="left")
ax_tr.set_ylabel("Drawdown (%)")
ax_tr.legend(fontsize=9, loc="lower left", framealpha=0.88)
ax_tr.grid(True, alpha=0.22)
ax_tr.tick_params(axis="x", rotation=12)

# ──────────────────────────────────────────────────────────────────────────────
# Panel C — SPY centrality mechanism plot
# ──────────────────────────────────────────────────────────────────────────────
ax_bl.plot(cent_static.index, cent_static.values,
           color="#666666", ls="--", lw=1.6, label="Static 252-day")
ax_bl.plot(cent_ewma.index, cent_ewma.values,
           color="#e63946", ls="-",  lw=2.0, label="EWMA λ=0.94")

ax_bl.axvline(CRASH, color="crimson", lw=1.3, ls=":", zorder=5)

ylo_bl, yhi_bl = ax_bl.get_ylim()
ax_bl.text(CRASH + pd.Timedelta(days=100),
           ylo_bl + (yhi_bl - ylo_bl) * 0.88,
           "COVID\ncrash", ha="left", va="top", fontsize=7.5,
           color="crimson", fontweight="bold")

ax_bl.set_title("C  SPY Degree Centrality (PMFG)\n"
                "      EWMA rewires the graph faster under stress",
                fontsize=11, fontweight="bold", loc="left")
ax_bl.set_ylabel("Degree centrality")
ax_bl.legend(fontsize=9, loc="upper right", framealpha=0.88)
ax_bl.grid(True, alpha=0.22)
ax_bl.tick_params(axis="x", rotation=12)

# ──────────────────────────────────────────────────────────────────────────────
# Panel D — Sharpe + |Max Drawdown| grouped bars
# ──────────────────────────────────────────────────────────────────────────────
x = np.arange(4)
w = 0.35

bars_sh = ax_br.bar(x - w / 2, sharpe_vals, w,
                    color=COLORS, alpha=0.92, zorder=3, label="Sharpe Ratio")
ax_br.set_ylabel("Sharpe Ratio", fontsize=9)
ax_br.set_ylim(0, max(sharpe_vals) * 1.30)
ax_br.grid(True, axis="y", alpha=0.22, zorder=0)

ax_br2 = ax_br.twinx()
bars_dd = ax_br2.bar(x + w / 2, np.abs(dd_vals) * 100, w,
                     color=COLORS, alpha=0.42, hatch="///", zorder=3,
                     label="|Max Drawdown| %")
ax_br2.set_ylabel("|Max Drawdown| %", fontsize=9, color="#555555")
ax_br2.set_ylim(0, max(np.abs(dd_vals)) * 100 * 1.35)
ax_br2.tick_params(axis="y", colors="#555555")

for bar, v in zip(bars_sh, sharpe_vals):
    ax_br.text(bar.get_x() + bar.get_width() / 2, v + 0.013,
               f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

for bar, v in zip(bars_dd, np.abs(dd_vals) * 100):
    ax_br2.text(bar.get_x() + bar.get_width() / 2, v + 0.4,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5, color="#444444")

ax_br.set_xticks(x)
ax_br.set_xticklabels(SHORT_LABELS, fontsize=9)
ax_br.set_title("D  Strategy Metrics", fontsize=11, fontweight="bold", loc="left")

h_sh = mpatches.Patch(color="#555555", alpha=0.92, label="Sharpe Ratio  (left axis)")
h_dd = mpatches.Patch(color="#555555", alpha=0.42, hatch="///",
                      label="|Max Drawdown| %  (right axis)")
ax_br.legend(handles=[h_sh, h_dd], fontsize=8, loc="upper right", framealpha=0.88)

# ── Super-title and save ───────────────────────────────────────────────────────
fig.suptitle(
    "Graph-Aware Portfolio Construction: PMFG Centrality + Dynamic Correlation",
    fontsize=14, fontweight="bold"
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("data/final_summary.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved data/final_summary.png")

# ── Print metrics for RESULTS.md cross-check ──────────────────────────────────
print("\nMetrics (4 strategies):")
for name in strat_list:
    r, v, sh, dd = met[name]
    print(f"  {name:26s}  ret={r:.2%}  vol={v:.2%}  sharpe={sh:.2f}  dd={dd:.2%}")
