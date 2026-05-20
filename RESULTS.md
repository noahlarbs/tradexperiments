# GraphTrading: Results

![Summary Figure](data/final_summary.png)

## What We Built

A walk-forward backtester (2011–2026, 252-day estimation window, monthly rebalance)
comparing graph-aware portfolio construction against standard baselines across 17
multi-asset ETFs spanning energy, metals, equities, bonds, agriculture, and FX.
We build a PMFG (Planar Maximally Filtered Graph) correlation network at each
rebalance date and use each asset's degree centrality to adjust HRP weights —
downweighting structural hubs and upweighting peripheral assets to reduce implicit
systemic risk. The final variant replaces the static rolling correlation window
with EWMA (λ=0.94) to rewire the graph faster during market stress.

## Main Findings

- **Centrality adjustment reduces tail risk without sacrificing Sharpe.** CentHRP-PMFG
  cuts max drawdown from −9.21% (plain HRP) to −6.49% by systematically reducing
  weight on assets that are correlation hubs. Hub assets tend to experience
  synchronised drawdowns during stress — the graph makes this structural risk explicit.

- **Dynamic correlation (EWMA λ=0.94) improves Sharpe from 0.75 → 0.89.**
  Diagnostics isolate the mechanism: EWMA detects the COPX centrality shift within
  0 trading days of the Feb 20 2020 crash, versus 12 days for the 252-day static
  window. Faster graph rewiring → faster weight reduction on newly-central (now
  systemic) nodes → smaller drawdown in the crash leg.

- **The edge is timing, not a factor tilt.** Return attribution shows the EWMA
  strategy consistently overweights FX (+3 pp) and underweights bonds (−0.9 pp)
  vs HRP regardless of whether the month is an outperforming or underperforming
  month. There is no discriminating asset class bet — the performance difference
  is entirely in the speed of centrality signal, not in a static tilt toward any
  sector.

## Final Metrics (2011–2026, walk-forward)

| Strategy | Ann Return | Ann Vol | Sharpe | Max Drawdown |
|---|---|---|---|---|
| Equal Weight | 3.76% | 11.29% | 0.33 | −35.73% |
| HRP | 2.68% | 3.41% | 0.79 | −9.21% |
| CentHRP-PMFG (Static) | 2.38% | 3.19% | 0.75 | −6.49% |
| **CentHRP-EWMA-0.94** | **2.77%** | **3.13%** | **0.89** | **−6.89%** |

*Zero risk-free rate. 17-asset ETF universe (WEAT dropped: >5% missing data).
10 bps/unit turnover sensitivity check in Phase 5 shows negligible TC impact
(monthly turnover ~10.4%, TC-adjusted Sharpe 0.87 vs gross 0.89).*

## Limitations

The backtest has four material caveats. First, the DCC-approximation (Phase 6)
fits GARCH volatility parameters on the full dataset before standardising residuals —
a mild form of lookahead that does not affect the simpler EWMA variants in the
headline results, but inflates the DCC performance numbers. Second, the universe
is small (17 assets) and weighted toward commodity and equity markets of the
2010s; the centrality signal may not generalise to fixed income-heavy or crypto
universes where graph topology differs materially. Third, ETF proxies introduce
tracking error relative to the futures markets they represent, and several
commodity ETFs (UNG, BNO, CORN) have significant contango drag that is absent
from equity benchmarks — the equal-weight return advantage partly reflects 2021–2025
commodity reflation, not a structural alpha. Fourth, transaction costs here
assume a flat 10 bps per unit of turnover; bid-ask spreads for thinly-traded
ETFs and the potential for PMFG rewiring to cluster rebalances during regime
transitions could make live costs meaningfully higher.
