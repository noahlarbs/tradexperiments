"""
Phase 17 expanded universe test: add FX and more commodities to the 11-contract baseline.
Test whether more contracts improve the TSMOM baseline and ENB+VoV overlay.
Reuses all functions from phase17_futures_tsmom_graph.py.
"""
import sys
sys.path.insert(0, '/tmp/tradexperiments')
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import networkx as nx
from statsmodels.tsa.api import VAR

# Expanded universe: original 11 + FX majors + extra commodities
TICKERS_EXPANDED = {
    # Original 11
    "CL": "CL=F",  "NG": "NG=F",  "GC": "GC=F",  "SI": "SI=F",  "HG": "HG=F",
    "ZC": "ZC=F",  "ZW": "ZW=F",  "ZS": "ZS=F",  "ES": "ES=F",  "ZN": "ZN=F",
    "ZB": "ZB=F",
    # FX
    "EC": "6E=F",  "JY": "6J=F",  "BP": "6B=F",  "AD": "6A=F",
    # More commodities
    "NQ": "NQ=F",  "ZL": "ZL=F",  "RB": "RB=F",  "HO": "HO=F",
}

START       = "2009-01-01"
END         = "2024-12-31"
TRAIN_END   = "2017-12-31"
OOS_START   = "2018-01-01"
TSMOM_WINDOW= 252
VOL_WINDOW  = 63
ESTIMATION_WIN = 252
GRAPH_UPDATE = 5
LAMBDA_EWMA = 0.94
VOL_TARGET  = 0.15
TCOST       = 0.0002

def load_returns_expanded():
    print("Downloading expanded futures data…")
    symbols = list(TICKERS_EXPANDED.values())
    raw = yf.download(symbols, start=START, end=END, progress=False, auto_adjust=True)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    col_map = {v: k for k, v in TICKERS_EXPANDED.items()}
    prices = prices.rename(columns=col_map)
    prices = prices[[c for c in prices.columns if c in TICKERS_EXPANDED]]
    log_ret = np.log(prices / prices.shift(1))
    roll_std = log_ret.rolling(63, min_periods=21).std()
    log_ret = log_ret.where(log_ret.abs() <= 5 * roll_std)
    missing = log_ret.isna().mean()
    drop = missing[missing > 0.10].index.tolist()
    if drop:
        print(f"  Dropping {drop} (>10% missing)")
    log_ret = log_ret.drop(columns=drop)
    log_ret = log_ret.ffill(limit=3).dropna()
    print(f"  {len(log_ret)} days, {log_ret.shape[1]} contracts: {log_ret.columns.tolist()}")
    return log_ret

# Import the key functions from the main phase17 file
from phase17_futures_tsmom_graph import (
    ewma_corr, build_pmfg, effective_num_bets, compute_gfevd,
    precompute_graphs, compute_vov_scale, perf_stats, print_results
)

def compute_strategy_simple(log_ret, regime_scale_s=None, vov_scale_s=None):
    signal  = np.sign(log_ret.rolling(TSMOM_WINDOW).sum())
    vol_63  = (log_ret.rolling(VOL_WINDOW).std() * np.sqrt(252)).clip(lower=1e-6)
    raw_pos = signal / vol_63
    port_ret_raw = (raw_pos.shift(1) * log_ret).sum(axis=1, min_count=1)
    book_vol     = (port_ret_raw.rolling(VOL_WINDOW, min_periods=VOL_WINDOW)
                    .std().shift(1) * np.sqrt(252)).clip(lower=0.02, upper=10.0)
    vol_scale    = VOL_TARGET / book_vol
    positions    = raw_pos.multiply(vol_scale, axis=0)
    if regime_scale_s is not None:
        rs = regime_scale_s.reindex(positions.index).ffill().fillna(1.0)
        positions = positions.multiply(rs, axis=0)
    if vov_scale_s is not None:
        vs = vov_scale_s.reindex(positions.index).ffill().fillna(1.0)
        positions = positions.multiply(vs, axis=0)
    tcost   = (positions.diff().abs() * TCOST).sum(axis=1, min_count=1).fillna(0)
    port_ret = (positions.shift(1) * log_ret).sum(axis=1, min_count=1) - tcost
    return port_ret.dropna()

def main():
    log_ret = load_returns_expanded()
    n = log_ret.shape[1]
    cent_df, incoming_df, gfevd_store, enb_series = precompute_graphs(log_ret)
    enb_pct        = enb_series.expanding(min_periods=63).rank(pct=True)
    regime_scale_s = 0.5 + enb_pct
    vov_scale_s    = compute_vov_scale(log_ret)

    variants = {
        f"1.  TSMOM baseline ({n} contracts)": (None, None),
        f"5.  + ENB Regime":                   (regime_scale_s, None),
        f"24. + VoV Regime":                   (None, vov_scale_s),
        f"25. ENB + VoV [BEST-ORIG]":          (regime_scale_s, vov_scale_s),
    }

    all_returns = {}
    for name, (rs, vov) in variants.items():
        all_returns[name] = compute_strategy_simple(log_ret, rs, vov)

    print_results(all_returns, log_ret)

if __name__ == "__main__":
    main()
