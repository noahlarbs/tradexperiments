# Graph Portfolio Construction: Honest Research Log

A systematic exploration of graph theory applied to multi-asset portfolio construction.
Walk-forward backtests only — no lookahead, monthly rebalance, 252-day estimation windows.

---

## Quick Summary

The strongest finding in this project is also the simplest: **adding an EWMA graph
rewiring mechanism on top of HRP gives the best risk-adjusted profile of any strategy
tested** — Sharpe 0.89 vs HRP's 0.79 on a 17-asset universe (2011–2026). Everything
built afterward either confirmed that finding from different angles, failed to beat it,
or exposed its own bugs.

---

## Phases 1–8: Core Result

17-ETF multi-asset universe (energy, metals, equities, bonds, agriculture, FX).
Walk-forward 2011–2026.

| Strategy | Ann Ret | Ann Vol | Sharpe | Max DD |
|---|---|---|---|---|
| Equal Weight | 3.76% | 11.29% | 0.33 | −35.73% |
| HRP | 2.68% | 3.41% | 0.79 | −9.21% |
| CentHRP-PMFG (static 252d window) | 2.38% | 3.19% | 0.75 | −6.49% |
| **CentHRP-EWMA-0.94** | **2.77%** | **3.13%** | **0.89** | **−6.89%** |

**Mechanism (phase 6 diagnostics):** EWMA detects the COPX centrality shift within
0 days of the Feb 2020 crash vs 12 days for the static window. Faster rewiring →
faster weight reduction on newly-central (systemic) nodes → smaller drawdown.

**Statistical caution (phase 8):** Bootstrap p-value for the Degree-EWMA edge over HRP
is 0.16 (not significant). DSR passes multiple-testing correction at 95%, but the effect
is not decisive. The Sharpe lift is real but fragile.

---

## Phase 9: Alternative Centralities

Same 17-ETF universe. All centralities tested on EWMA-0.94 base.

| Centrality | Sharpe | PSR vs HRP | Notes |
|---|---|---|---|
| Degree | 0.89 | 0.64 | Best; not statistically significant |
| PageRank | 0.86 | 0.61 | Similar story |
| Eigenvector | 0.79 | 0.50 | Essentially HRP |
| Betweenness | 0.72 | 0.41 | Worse — betweenness uses distance weights that interact poorly with HRP |

Degree wins, but no centrality is clearly better than the others at statistical significance.

---

## Phase 10: Expanded Universe (34 → 25 ETFs)

| Strategy | Ann Ret | Sharpe | Max DD | Note |
|---|---|---|---|---|
| Equal Weight | 7.41% | 0.65 | −26.1% | — |
| HRP | 1.79% | 0.76 | −10.0% | RP overweights bonds |
| CentHRP-Degree | 1.73% | 0.79 | −9.9% | Marginal improvement |

HRP's absolute return (1.79%) versus Equal Weight (7.41%) exposes the fixed-income drag
problem: risk-parity overweights low-vol bonds, which structurally reduces nominal returns.
The graph adjustment is marginal (SR 0.76 → 0.79). PSR = 0.54 — not significant.

---

## Phase 12: Methodological Finding — HRP and PMFG Are Redundant in Homogeneous Universes

Restricted universe to 9 US Equity Sector ETFs (XLF, XLV, XLK, XLE, XLI, XLP, XLY, XLU, XLB).

| Strategy | Ann Ret | Sharpe | Max DD |
|---|---|---|---|
| Equal Weight | 12.91% | 0.78 | −36.9% |
| HRP | 12.90% | 0.86 | −32.2% |
| CentHRP-EWMA-Degree | 12.83% | **0.87** | −32.2% |

**Key finding:** CentHRP is virtually identical to HRP (0.87 vs 0.86). HRP's hierarchical
Ward linkage already identifies and isolates the densely-correlated clusters that PMFG
degree-centrality is trying to penalize. Inside a homogeneous single-asset-class universe,
the graph adds no information that the correlation structure itself doesn't already contain.
The centrality overlay is redundant.

---

## Phase 13: PMFG-Filtered Mean-Variance

Tested whether zeroing non-PMFG-edge off-diagonal covariance entries stabilizes
Max-Sharpe optimization (sector ETF universe, 2010–2026).

| Strategy | Ann Ret | Sharpe | Max DD |
|---|---|---|---|
| HRP | 12.85% | 0.85 | −32.5% |
| Max Sharpe (raw EWMA cov) | 11.64% | 0.66 | −26.0% |
| Max Sharpe (PMFG-filtered cov) | 11.56% | 0.68 | −31.6% |

**Finding:** PMFG filtering modestly reduces estimation noise (vol 17.5% → 17.0%) but
the improvement is negligible. Both Max-Sharpe variants underperform HRP. The binding
constraint is noisy expected-return estimation (μ), not the covariance matrix —
a well-known result. Denoising Σ alone doesn't fix Markowitz.

---

## Phase 14: Directed Lead-Lag Networks (Rebuilt — lag 1–5 averaging, explicit OOS)

9 macro/commodity ETFs (USO, UNG, GLD, SLV, COPX, DBA, UUP, FXA, EEM).
Signal: directed net information flow averaged over lags 1–5, sparsified at 80th percentile.
Trade: long assets receiving strong net inflow weighted by today's source returns.

| Period | Strategy | Ann Ret | Sharpe | Max DD |
|---|---|---|---|---|
| Full 2011–2026 | Equal Weight | 1.61% | 0.12 | −57.0% |
| Full 2011–2026 | XS Momentum (1d) | −3.91% | −0.19 | −56.8% |
| Full 2011–2026 | **Graph Spillover** | **7.02%** | **0.34** | **−42.9%** |
| Train 2011–2018 | Equal Weight | −6.97% | −0.55 | −54.5% |
| Train 2011–2018 | Graph Spillover | 0.78% | **0.04** | −42.9% |
| OOS 2019–2026 | Equal Weight | 11.42% | 0.75 | −28.0% |
| OOS 2019–2026 | Graph Spillover | 13.95% | 0.63 | −41.5% |

**Honest assessment:** The full-period Sharpe (0.34) is misleading. The train-period Sharpe
is 0.04 — essentially flat. The apparent performance is concentrated in 2019–2026 which was
a commodity bull market (COVID supply shock, energy transition). In the true OOS period the
strategy *underperforms equal weight* (0.63 vs 0.75) while taking more drawdown (−41.5% vs
−28.0%). The directed graph concept is interesting but this implementation does not robustly
extract alpha on this universe.

---

## Phase 15: Cointegration Clique StatArb (Fixed — survivorship, BH correction, beta)

28 US tech stocks (PLTR and CRWD dropped — IPO'd after 2018, missing >15% of history).
Engle-Granger cointegration with **Benjamini-Hochberg FDR correction** at 5%.
60-day graph rebuild. Z-score entry ±2.0.

| Strategy | Ann Ret | Sharpe | Max DD | Corr/SPY | Beta/SPY |
|---|---|---|---|---|---|
| Graph Clique StatArb (BH) | 1.21% | 0.44 | −4.69% | 0.037 | 0.005 |
| S&P 500 | 17.47% | 0.90 | −33.72% | 1.00 | 1.00 |

**What the multiple-testing correction reveals:**
- Without correction: raw p<0.05 finds 10–85 "cointegrated" pairs per rebalance (expected
  false positives at nominal α=5% are ~19 per rebalance on 378 pairs tested).
- After BH correction at FDR=5%: valid cliques in only 1 of 31 rebalance periods
  (Feb 2021). Average cliques per rebalance: 0.1.
- The strategy is essentially never active. The near-zero Sharpe (0.44) with near-zero
  volatility (2.75%) and tiny drawdown (−4.69%) is the statistical signature of a strategy
  that barely trades.

**Conclusion:** The original Phase 15 "market-neutral alpha" with Sharpe 0.23 and
Beta 0.06 was driven by unadjusted p-values creating spurious cointegration edges in every
rebalance. After proper multiple-testing correction, no reliable cointegration structure
exists in this 28-stock tech universe at 252-day rolling windows.

Note: "Beta to SPY" now correctly computed as ρ × (σ_strat / σ_spy). The original code
reported correlation as beta — they differ when strategies have different volatility to the
benchmark.

---

## Cross-Phase Conclusions

1. **EWMA graph rewiring on HRP** is the most defensible contribution (phases 1–8). The
   mechanism is diagnosed, the effect is real, but statistical significance is marginal.

2. **Graph topology is redundant inside homogeneous asset classes** (phase 12). HRP already
   captures cluster structure via hierarchical linkage. Apply graph methods across uncorrelated
   asset classes, not within them.

3. **PMFG doesn't fix Markowitz** (phase 13). Denoising Σ is not the binding constraint.
   Expected-return estimation is, and graph theory offers nothing there.

4. **Directed lead-lag graphs are conceptually sound but implementation-sensitive** (phase 14).
   Train-period performance (SR 0.04) reveals the OOS gains were driven by regime, not
   strategy alpha. Requires a more careful universe selection and signal construction.

5. **Cointegration statarb requires multiple-testing correction** (phase 15). Without BH or
   Bonferroni, every rolling-window backtest on 30+ stocks will find dozens of spurious
   cointegrated pairs and generate fake cliques. After correction, the tech universe shows
   almost no persistent cointegration.

---

## Known Limitations

- ETF proxies have tracking error vs futures; commodity ETFs have contango drag absent from
  equity benchmarks.
- All universes small (9–28 assets); centrality effects likely stronger at larger n.
- Phase 14 has no fitted parameters to overfit, but performance is regime-dependent.
- Phase 15 with BH correction leaves almost no tradable signal; a more fertile universe
  (crypto, microcaps, commodities) may have more genuine cointegration structure.
- DCC-GARCH in phase 6 has mild lookahead (vol parameters fit on full dataset); does not
  affect headline CentHRP-EWMA results.
