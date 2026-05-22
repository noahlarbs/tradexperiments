"""
Phase 18 — Combined ETF + Futures Portfolio
Blends CentHRP-EWMA-0.94 (17 ETFs, Phase 1-8) with TSMOM+ENB+VoV (11 futures, Phase 17).

The two strategies are structurally uncorrelated:
  CentHRP:       long-only ETF cross-asset, HRP weights, 200d trend filter, vol cap
  TSMOM+ENB+VoV: long-short futures momentum, daily positions, dual-regime vol scaling

Key insight: the same ENB metric used to improve TSMOM doubles as a portfolio
ALLOCATION signal — when futures markets are highly correlated (low ENB, few real bets),
shift capital from TSMOM toward CentHRP; when independent (high ENB), shift toward TSMOM.

Usage: python3 phase18_combined_portfolio.py

RESULTS (train 2011-2017, OOS 2018-2024):

  Individual sleeves:
    CentHRP-EWMA (17 ETFs)     OOS  +4.2%  6.6%  Sharpe 0.64  DD -19.4%
    TSMOM+ENB+VoV (11 futures) OOS +13.6% 17.3%  Sharpe 0.79  DD -28.8%
    OOS correlation between sleeves: 0.27  ← genuinely uncorrelated

  Combinations:
    50/50 raw                  OOS  +8.9% 10.0%  Sharpe 0.88  DD -13.5%
    50/50 vol-equal (10% each) OOS  +8.6%  8.5%  Sharpe 1.01  DD -12.9%  ← clean 1.0+
    Risk-parity (81%/19%)      OOS  +6.0%  7.0%  Sharpe 0.86  DD -16.0%
    OOS-optimal (95%/5%)       OOS  +9.8% 10.3%  Sharpe 0.95  DD -20.9%
    ENB-adaptive (vol-eq)      OOS  +9.3%  8.9%  Sharpe 1.04  DD -11.2%  ← BEST

  ENB-adaptive wins: w_futures = 0.3 + 0.4 × ENB_percentile → [0.3, 0.7]
  When futures markets are correlated (low ENB), tilt to ETFs. When independent
  (high ENB), TSMOM has more genuine bets — tilt to futures.

HONEST CAVEAT:
  Train-period combined Sharpe is weak (0.08-0.16 for vol-equal variants) because
  the futures sleeve barely works in 2011-2017. The combination is driven almost
  entirely by OOS performance of both strategies improving together. The 0.27
  correlation is what makes the OOS Sharpe 1.0+ possible — not any overlay cleverness.
  Both strategies should be paper-traded independently before allocating real capital.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import networkx as nx
import riskfolio as rp
from scipy.optimize import minimize

from phase17_futures_tsmom_graph import (
    load_returns, ewma_corr, build_pmfg, effective_num_bets,
    precompute_graphs, compute_vov_scale, compute_strategy,
)

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS_ETF = [
    "DBC", "GLD", "SLV", "USO", "UNG", "CORN", "COPX",
    "SPY", "EEM", "EFA", "TLT", "IEF", "AGG", "LQD",
    "UUP", "FXA", "VNQ",
]

START      = "2010-01-01"
END        = "2024-12-31"
TRAIN_END  = "2017-12-31"
OOS_START  = "2018-01-01"

LAMBDA_EWMA   = 0.94
EST_WIN       = 252
REBAL_FREQ    = 5      # recompute ETF weights every 5 trading days
VOL_TARGET    = 0.15
VOL_LOOKBACK  = 20
MA_WINDOW     = 200
TCOST_ETF     = 0.0001  # ETF tcost lower than futures
TCOST_FUT     = 0.0002


# ── CentHRP backtest ──────────────────────────────────────────────────────────
def load_etf_returns():
    print("Downloading ETF data…")
    raw = yf.download(TICKERS_ETF, start=START, end=END,
                      progress=False, auto_adjust=True)["Close"]
    log_ret = np.log(raw / raw.shift(1))
    missing = log_ret.isna().mean()
    drop = missing[missing > 0.05].index.tolist()
    if drop:
        print(f"  Dropping ETFs with >5% missing: {drop}")
    active = [t for t in TICKERS_ETF if t not in drop]
    log_ret = log_ret[active].ffill(limit=5).dropna()
    print(f"  {len(log_ret)} trading days, {len(active)} ETFs: {active}")
    return log_ret


def compute_hrp_weights(log_ret_window):
    """Compute CentHRP-EWMA-0.94 weights for a given window."""
    tickers = log_ret_window.columns.tolist()
    n       = len(tickers)

    # EWMA correlation → PMFG → degree centrality
    corr = ewma_corr(log_ret_window)
    G    = build_pmfg(corr)
    cent = nx.degree_centrality(G)

    # HRP
    try:
        port  = rp.HCPortfolio(returns=log_ret_window)
        wdf   = port.optimization(model="HRP", codependence="pearson", rm="MV",
                                  rf=0, linkage="ward", max_k=10, leaf_order=True)
        hrp_w = wdf.squeeze().reindex(tickers).fillna(0).values
        hrp_w = np.clip(hrp_w, 0, None)
        s     = hrp_w.sum()
        hrp_w = hrp_w / s if s > 0 else np.ones(n) / n
    except Exception:
        hrp_w = np.ones(n) / n

    # Centrality adjustment: downweight hubs
    c   = np.array([cent.get(t, 1.0 / max(n - 1, 1)) for t in tickers])
    c   = np.clip(c, 1e-6, None)
    adj = hrp_w / c
    adj /= adj.sum()
    return pd.Series(adj, index=tickers)


def run_centurp_backtest(log_ret_etf):
    """
    Rolling CentHRP-EWMA backtest.
    Weights recomputed every REBAL_FREQ days using past EST_WIN days.
    Applies 200d MA trend filter and 15% vol targeting.
    Returns daily portfolio return Series.
    """
    tickers = log_ret_etf.columns.tolist()
    T       = len(log_ret_etf)
    prices  = np.exp(log_ret_etf.cumsum())  # reconstructed price index

    weights_df = pd.DataFrame(0.0, index=log_ret_etf.index, columns=tickers)
    last_w     = pd.Series(1.0 / len(tickers), index=tickers)

    print("Computing rolling CentHRP weights…")
    for t in range(EST_WIN, T):
        if t % REBAL_FREQ == 0:
            window = log_ret_etf.iloc[t - EST_WIN : t]
            last_w = compute_hrp_weights(window)

            # Trend filter: zero out assets below 200d MA
            if t >= MA_WINDOW:
                ma200 = prices.iloc[t - MA_WINDOW : t].mean()
                below = [tk for tk in last_w.index if prices.iloc[t][tk] < ma200[tk]]
                if below:
                    last_w[below] = 0.0
                    s = last_w.sum()
                    if s > 0:
                        last_w /= s
                    else:
                        last_w = pd.Series(1.0 / len(tickers), index=tickers)

            if t % 500 == 0:
                print(f"  {log_ret_etf.index[t].date()}  ({t}/{T})")

        weights_df.iloc[t] = last_w

    # Vol targeting: scale down when realized portfolio vol > 15%
    port_ret_raw = (weights_df.shift(1) * log_ret_etf).sum(axis=1, min_count=1)
    book_vol     = (port_ret_raw.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK)
                    .std().shift(1) * np.sqrt(252)).clip(lower=0.01, upper=10.0)
    vol_scale    = (VOL_TARGET / book_vol).clip(upper=1.0)   # cap at 1 (no leverage for ETFs)
    positions    = weights_df.multiply(vol_scale, axis=0)

    # Transaction costs
    tcost    = (positions.diff().abs() * TCOST_ETF).sum(axis=1, min_count=1).fillna(0)
    port_ret = (positions.shift(1) * log_ret_etf).sum(axis=1, min_count=1) - tcost
    return port_ret.dropna()


# ── TSMOM+ENB+VoV backtest (reuse Phase 17) ──────────────────────────────────
def run_tsmom_backtest():
    log_ret_fut = load_returns()
    cent_df, incoming_df, gfevd_store, enb_series = precompute_graphs(log_ret_fut)
    enb_pct        = enb_series.expanding(min_periods=63).rank(pct=True)
    regime_scale_s = 0.5 + enb_pct
    vov_scale_s    = compute_vov_scale(log_ret_fut)
    port_ret = compute_strategy(
        log_ret_fut, False, False, True,
        cent_df, incoming_df, None,
        regime_scale_s=regime_scale_s,
        vov_scale_s=vov_scale_s,
    )
    return port_ret, enb_pct, log_ret_fut


# ── Combination schemes ───────────────────────────────────────────────────────
def vol_scale_to_target(ret_series, target_vol=0.10, lookback=63):
    """Scale a return series so its realized vol equals target_vol."""
    roll_vol = ret_series.rolling(lookback, min_periods=lookback).std().shift(1) * np.sqrt(252)
    roll_vol = roll_vol.clip(lower=0.01)
    scale    = (target_vol / roll_vol).clip(upper=3.0)
    return ret_series * scale


def combine(etf_ret, fut_ret, w_etf, w_fut, vol_eq_target=None):
    """
    Combine two return series with given weights.
    If vol_eq_target is set, each series is first scaled to that vol level.
    """
    aligned = pd.DataFrame({"etf": etf_ret, "fut": fut_ret}).dropna()
    if vol_eq_target is not None:
        aligned["etf"] = vol_scale_to_target(aligned["etf"], vol_eq_target)
        aligned["fut"] = vol_scale_to_target(aligned["fut"], vol_eq_target)
    return (w_etf * aligned["etf"] + w_fut * aligned["fut"]).dropna()


def optimal_weights_train(etf_ret, fut_ret, train_end=TRAIN_END):
    """
    Find the allocation (w_etf, 1-w_etf) that maximizes Sharpe on the train period.
    Both series scaled to 10% vol first, then combined.
    """
    aligned = pd.DataFrame({"etf": etf_ret, "fut": fut_ret}).dropna()
    train   = aligned[aligned.index <= train_end]

    etf_v = vol_scale_to_target(train["etf"], 0.10)
    fut_v = vol_scale_to_target(train["fut"], 0.10)

    def neg_sharpe(w):
        w_e = float(np.clip(w[0], 0.01, 0.99))
        comb = w_e * etf_v + (1 - w_e) * fut_v
        ann_ret = comb.mean() * 252
        ann_vol = comb.std() * np.sqrt(252)
        return -(ann_ret / ann_vol) if ann_vol > 0 else 0.0

    res = minimize(neg_sharpe, [0.5], bounds=[(0.05, 0.95)], method="L-BFGS-B")
    w_opt = float(res.x[0])
    print(f"  Optimal train allocation: {w_opt:.1%} ETF / {1-w_opt:.1%} Futures")
    return w_opt


def enb_adaptive_combine(etf_ret, fut_ret, enb_pct, fut_index):
    """
    ENB-adaptive allocation: when ENB is high (independent futures markets),
    tilt toward futures (TSMOM has more genuine independent bets).
    When ENB is low, tilt toward ETFs (correlated futures → less value from TSMOM).
    Weights: w_fut = 0.3 + 0.4 × enb_pct  → [0.3, 0.7]
             w_etf = 1 - w_fut              → [0.3, 0.7]
    Both scaled to 10% vol before combining.
    """
    aligned = pd.DataFrame({"etf": etf_ret, "fut": fut_ret}).dropna()
    etf_v = vol_scale_to_target(aligned["etf"], 0.10)
    fut_v = vol_scale_to_target(aligned["fut"], 0.10)

    enb_on_dates = enb_pct.reindex(aligned.index).ffill().fillna(0.5)
    w_fut_s = (0.3 + 0.4 * enb_on_dates).clip(0.3, 0.7)
    w_etf_s = 1.0 - w_fut_s

    return (w_etf_s * etf_v + w_fut_s * fut_v).dropna()


# ── Performance reporting ─────────────────────────────────────────────────────
def perf_stats(ret_series):
    r = ret_series.dropna()
    if len(r) < 30:
        return {"Ann Ret": "N/A", "Ann Vol": "N/A", "Sharpe": "N/A", "Max DD": "N/A", "Calmar": "N/A"}
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cum     = (1 + r).cumprod()
    max_dd  = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar  = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "Ann Ret": f"{ann_ret:+.1%}",
        "Ann Vol": f"{ann_vol:.1%}",
        "Sharpe":  f"{sharpe:.2f}",
        "Max DD":  f"{max_dd:.1%}",
        "Calmar":  f"{calmar:.2f}" if not np.isnan(calmar) else "N/A",
    }


def print_combined_results(all_returns):
    train_mask = lambda r: r[r.index <= TRAIN_END]
    oos_mask   = lambda r: r[r.index >= OOS_START]

    cols = ["Ann Ret", "Ann Vol", "Sharpe", "Max DD", "Calmar"]
    w = 40
    print("\n" + "=" * 85)
    print("Combined ETF + Futures Portfolio — Phase 18")
    print("=" * 85)
    print(f"{'Strategy':<{w}} {'Period':<15}", end="")
    for c in cols:
        print(f"  {c:>9}", end="")
    print()
    print("-" * 85)

    for name, ret in all_returns.items():
        for lbl, subset in [("Full 2011-2024", ret),
                            ("Train 2011-2017", train_mask(ret)),
                            ("OOS  2018-2024", oos_mask(ret))]:
            m = perf_stats(subset)
            print(f"{name:<{w}} {lbl:<15}", end="")
            for c in cols:
                print(f"  {m[c]:>9}", end="")
            print()
        print()

    print("=" * 85)
    corr_note = "\nCorrelation matrix (OOS daily returns):"
    print(corr_note)
    oos_df = pd.DataFrame({k: v[v.index >= OOS_START] for k, v in all_returns.items()}).dropna()
    print(oos_df.corr().round(2).to_string())


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── CentHRP sleeve ────────────────────────────────────────────────────────
    log_ret_etf = load_etf_returns()
    etf_ret     = run_centurp_backtest(log_ret_etf)

    # ── TSMOM+ENB+VoV sleeve ─────────────────────────────────────────────────
    print("\nRunning TSMOM+ENB+VoV futures backtest…")
    fut_ret, enb_pct, log_ret_fut = run_tsmom_backtest()

    # Align to common date range (both strategies active)
    common_start = max(etf_ret.index.min(), fut_ret.index.min(), pd.Timestamp(START))
    etf_ret = etf_ret[etf_ret.index >= common_start]
    fut_ret = fut_ret[fut_ret.index >= common_start]

    # ── Print individual sleeve stats ─────────────────────────────────────────
    print("\n── Individual sleeve performance ───────────────────────────────────")
    individual = {
        "CentHRP-EWMA (17 ETFs)":       etf_ret,
        "TSMOM+ENB+VoV (11 futures)":   fut_ret,
    }
    train_mask = lambda r: r[r.index <= TRAIN_END]
    oos_mask   = lambda r: r[r.index >= OOS_START]
    for name, ret in individual.items():
        for lbl, subset in [("Full", ret), ("Train", train_mask(ret)), ("OOS", oos_mask(ret))]:
            m = perf_stats(subset)
            print(f"  {name:35s} {lbl:6s}  "
                  f"Ret {m['Ann Ret']:>6s}  Vol {m['Ann Vol']:>5s}  "
                  f"Sharpe {m['Sharpe']:>5s}  DD {m['Max DD']:>7s}")
        print()

    # ── Find optimal train allocation ─────────────────────────────────────────
    print("Finding optimal allocation on train period…")
    w_opt = optimal_weights_train(etf_ret, fut_ret)

    # ── Build all combination variants ───────────────────────────────────────
    # Risk contribution per sleeve at natural vol:
    etf_vol = etf_ret[etf_ret.index <= TRAIN_END].std() * np.sqrt(252)
    fut_vol = fut_ret[fut_ret.index <= TRAIN_END].std() * np.sqrt(252)
    w_rp_etf = (1 / etf_vol) / (1 / etf_vol + 1 / fut_vol)  # risk-parity weight
    print(f"  Risk-parity weights: {w_rp_etf:.1%} ETF / {1-w_rp_etf:.1%} Futures  "
          f"(ETF vol {etf_vol:.1%}, Futures vol {fut_vol:.1%})")

    combined = {
        "CentHRP-EWMA (17 ETFs)":          etf_ret,
        "TSMOM+ENB+VoV (11 futures)":      fut_ret,
        "50/50 raw":                        combine(etf_ret, fut_ret, 0.5, 0.5),
        "50/50 vol-equal (10% each)":       combine(etf_ret, fut_ret, 0.5, 0.5, 0.10),
        f"Risk-parity ({w_rp_etf:.0%}/{1-w_rp_etf:.0%})":
                                            combine(etf_ret, fut_ret, w_rp_etf, 1-w_rp_etf),
        f"OOS-optimal ({w_opt:.0%}/{1-w_opt:.0%} vol-eq)":
                                            combine(etf_ret, fut_ret, w_opt, 1-w_opt, 0.10),
        "ENB-adaptive (vol-eq)":            enb_adaptive_combine(etf_ret, fut_ret, enb_pct, fut_ret.index),
    }

    print_combined_results(combined)


if __name__ == "__main__":
    main()
