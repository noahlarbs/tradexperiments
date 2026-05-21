"""
Phase 17 — TSMOM + Graph Risk Overlay on Multi-Sector Futures
Walk-forward OOS ablation (train 2010-2017, OOS 2018-2024) comparing:
  1. TSMOM baseline (Moskowitz, Ooi & Pedersen 2012)
  2. TSMOM + PMFG degree-centrality position scaling
  3. TSMOM + Diebold-Yilmaz generalized spillover overlay
  4. TSMOM + centrality + spillover (full model)

Prior work:
  TSMOM:           Moskowitz, Ooi & Pedersen (2012) JFE
  PMFG centrality: Tumminello et al. (2005) + Phases 1-8 of this project
  D-Y spillover:   Diebold & Yilmaz (2012), GFEVD: Pesaran & Shin (1998)

Novel contribution:
  D-Y spillover used as a position-sizing modifier (dual role: risk de-risking
  via incoming-spillover scaling + lead amplification via same-direction sender
  weighting). Stacked with PMFG structural centrality on a TSMOM futures book.
  OOS ablation isolates each layer's marginal contribution honestly.

Usage: python3 phase17_futures_tsmom_graph.py

Results (11 contracts: CL NG GC SI HG ZC ZW ZS ES ZN ZB — DX=F delisted):

  Strategy               Period     Ann Ret  Ann Vol  Sharpe  Max DD
  TSMOM (baseline)       Full        +6.0%    15.5%    0.39   -42.9%
  TSMOM (baseline)       Train       +2.2%    15.5%    0.14   -37.1%
  TSMOM (baseline)       OOS         +10.2%   15.6%    0.66   -25.5%
  TSMOM + Centrality     OOS          +9.6%   15.5%    0.62   -25.5%
  TSMOM + Spillover      OOS         +10.1%   15.6%    0.65   -25.3%
  TSMOM + Cent+Spill     OOS          +9.5%   15.4%    0.62   -25.5%

Honest findings:
  - TSMOM OOS Sharpe 0.66 confirms the strategy works on this futures universe.
    The low train Sharpe (0.14) reflects the known 2010-2017 "CTA winter"
    (commodity price whipsaw, QE distortions).
  - Graph overlays are neutral-to-negative OOS. Neither PMFG centrality nor
    D-Y spillover improve on the plain TSMOM baseline.
  - Centrality specifically hurts (-0.04 OOS Sharpe). This is the opposite of
    Phases 1-8, where centrality helped on the multi-asset ETF universe.
    Hypothesis: in a universe where all assets share the same signal type
    (price momentum), centrality penalizes the assets that are most correlated
    — but those same assets are often the ones with the clearest trend signals.
    The effect that worked in a heterogeneous ETF universe (penalizing
    contagion hubs across asset classes) misfires inside a more homogeneous
    commodity/rates universe where hub = strong trending market.
  - The TSMOM-only strategy is the keeper. Run it with vol targeting.
    Graph overlays add complexity with no OOS benefit here.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import networkx as nx
from statsmodels.tsa.api import VAR

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS = {
    "CL": "CL=F",   # Crude oil
    "NG": "NG=F",   # Natural gas
    "GC": "GC=F",   # Gold
    "SI": "SI=F",   # Silver
    "HG": "HG=F",   # Copper
    "ZC": "ZC=F",   # Corn
    "ZW": "ZW=F",   # Wheat
    "ZS": "ZS=F",   # Soybeans
    "ES": "ES=F",   # S&P 500 E-mini
    "ZN": "ZN=F",   # 10-year treasury
    "ZB": "ZB=F",   # 30-year treasury
    "DX": "DX=F",   # US dollar index
}

START           = "2009-01-01"
END             = "2024-12-31"
TRAIN_END       = "2017-12-31"
OOS_START       = "2018-01-01"

TSMOM_WINDOW    = 252    # lookback for momentum signal
VOL_WINDOW      = 63     # lookback for realized vol
ESTIMATION_WIN  = 252    # rolling window for graph estimation
GRAPH_UPDATE    = 5      # recompute graphs every N trading days
LAMBDA_EWMA     = 0.94
VAR_LAG         = 1
FEVD_HORIZON    = 10     # H-step ahead FEVD

VOL_TARGET      = 0.15   # annualized portfolio vol target
ALPHA_LEAD      = 0.30   # lead amplification strength (theoretically motivated)
BETA_SPILLOVER  = 0.50   # incoming-spillover de-risking strength
MIN_SEND        = 0.10   # min sender share to count as a lead signal
TCOST           = 0.0002 # 0.02% one-way per trade


# ── Data ──────────────────────────────────────────────────────────────────────
def load_returns():
    print("Downloading futures data…")
    symbols = list(TICKERS.values())
    raw = yf.download(symbols, start=START, end=END, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw

    col_map = {v: k for k, v in TICKERS.items()}
    prices = prices.rename(columns=col_map)
    # keep only tickers we know about
    prices = prices[[c for c in prices.columns if c in TICKERS]]

    log_ret = np.log(prices / prices.shift(1))

    # Roll-day outlier filter: cap at ±5 rolling σ
    roll_std = log_ret.rolling(63, min_periods=21).std()
    log_ret = log_ret.where(log_ret.abs() <= 5 * roll_std)

    missing = log_ret.isna().mean()
    drop = missing[missing > 0.10].index.tolist()
    if drop:
        print(f"  Dropping {drop} (>10% missing)")
    log_ret = log_ret.drop(columns=drop)

    log_ret = log_ret.ffill(limit=3).dropna()
    print(f"  {len(log_ret)} trading days, {log_ret.shape[1]} contracts: "
          f"{log_ret.columns.tolist()}")
    return log_ret


# ── Graph utilities ───────────────────────────────────────────────────────────
def ewma_corr(returns_df, lam=LAMBDA_EWMA, init_days=63):
    vals = returns_df.values
    S = np.cov(vals[:init_days].T) + 1e-8 * np.eye(vals.shape[1])
    for r in vals[init_days:]:
        S = lam * S + (1 - lam) * np.outer(r, r)
    d = np.diag(S)
    d_inv = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    C = d_inv[:, None] * S * d_inv[None, :]
    np.fill_diagonal(C, 1.0)
    return pd.DataFrame(np.clip(C, -1, 1),
                        index=returns_df.columns, columns=returns_df.columns)


def build_pmfg(corr_df):
    a = corr_df.columns.tolist()
    n = len(a)
    dist = np.sqrt(np.clip(2 * (1 - corr_df.values), 0, None))
    np.fill_diagonal(dist, 0)
    edges = sorted((dist[i, j], a[i], a[j])
                   for i in range(n) for j in range(i + 1, n))
    G = nx.Graph()
    G.add_nodes_from(a)
    for d, u, v in edges:
        if G.number_of_edges() >= 3 * (n - 2):
            break
        G.add_edge(u, v, weight=float(d))
        if not nx.check_planarity(G)[0]:
            G.remove_edge(u, v)
    return G


def compute_gfevd(window_df, p=VAR_LAG, H=FEVD_HORIZON):
    """
    Generalized FEVD (Pesaran-Shin 1998) via VAR(p).
    result[i, j] = normalized share of i's H-step forecast error variance due to j.
    """
    tickers = window_df.columns.tolist()
    n = len(tickers)
    uniform = pd.DataFrame(1.0 / n, index=tickers, columns=tickers)
    try:
        res = VAR(window_df.values).fit(p, trend="n")
        A     = res.coefs[0]       # n×n companion matrix for lag 1
        Sigma = res.sigma_u        # n×n residual covariance

        # MA coefficients: Ψ_0 = I, Ψ_h = Ψ_{h-1} A  (VAR(1))
        Psi = [np.eye(n)]
        for _ in range(1, H):
            Psi.append(Psi[-1] @ A)

        sigma_diag = np.diag(Sigma)
        theta = np.zeros((n, n))

        for i in range(n):
            ei = np.zeros(n); ei[i] = 1.0
            denom = sum(float(ei @ Psi[h] @ Sigma @ Psi[h].T @ ei) for h in range(H))
            if denom < 1e-12:
                theta[i, :] = 1.0 / n
                continue
            for j in range(n):
                ej = np.zeros(n); ej[j] = 1.0
                numer = (1.0 / sigma_diag[j]) * sum(
                    float(ei @ Psi[h] @ Sigma @ ej) ** 2 for h in range(H)
                )
                theta[i, j] = numer / denom

        row_sums = theta.sum(axis=1, keepdims=True)
        theta /= np.where(row_sums < 1e-12, 1.0, row_sums)
        return pd.DataFrame(theta, index=tickers, columns=tickers)
    except Exception:
        return uniform


# ── Rolling graph precomputation ──────────────────────────────────────────────
def precompute_graphs(log_ret):
    """
    Precompute PMFG centrality and GFEVD spillover on a rolling weekly basis.
    Returns:
      cent_df:        (T × n) DataFrame of degree centrality per day
      incoming_df:    (T × n) DataFrame of total incoming spillover fraction
      gfevd_store:    dict {t_idx: n×n np.array} (weekly entries, daily lookups)
    """
    tickers = log_ret.columns.tolist()
    n       = len(tickers)
    T       = len(log_ret)

    cent_df     = pd.DataFrame(1.0 / n, index=log_ret.index, columns=tickers)
    incoming_df = pd.DataFrame(0.0,     index=log_ret.index, columns=tickers)
    gfevd_store = {}

    last_cent     = np.full(n, 1.0 / n)
    last_gfevd    = np.ones((n, n)) / n
    last_incoming = np.zeros(n)

    print("Precomputing rolling graphs (weekly)…")
    for t in range(ESTIMATION_WIN, T):
        if t % GRAPH_UPDATE == 0:
            window = log_ret.iloc[t - ESTIMATION_WIN : t]

            # PMFG centrality
            corr = ewma_corr(window)
            G    = build_pmfg(corr)
            cent = nx.degree_centrality(G)
            last_cent = np.array([cent.get(tk, 1.0 / max(n - 1, 1)) for tk in tickers])

            # GFEVD
            gfevd_df  = compute_gfevd(window)
            last_gfevd    = gfevd_df.values.copy()
            last_incoming = 1.0 - np.diag(last_gfevd)  # fraction from external shocks

            if t % 500 == 0:
                print(f"  {log_ret.index[t].date()}  ({t}/{T})")

        cent_df.iloc[t]     = last_cent
        incoming_df.iloc[t] = last_incoming
        gfevd_store[t]      = last_gfevd

    return cent_df.ffill(), incoming_df.ffill(), gfevd_store


def compute_lead_amplifier(tsmom_signals, gfevd_store, log_ret):
    """
    Lead signal: if sender j explains > MIN_SEND of i's variance AND has the
    same TSMOM direction as i, amplify i's position by ALPHA_LEAD × spillover share.
    Returns (T × n) DataFrame of daily amplifiers (≥ 1.0).
    """
    tickers = log_ret.columns.tolist()
    n       = len(tickers)
    T       = len(log_ret)
    amp_arr = np.ones((T, n))

    for t in range(ESTIMATION_WIN, T):
        gfevd_mat = gfevd_store.get(t, np.ones((n, n)) / n)
        sig = tsmom_signals.iloc[t].values

        if np.any(np.isnan(sig)):
            continue

        same_dir     = (np.sign(sig[:, None]) * np.sign(sig[None, :])) > 0
        above_thresh = gfevd_mat > MIN_SEND
        off_diag     = ~np.eye(n, dtype=bool)

        boost = (gfevd_mat * same_dir * above_thresh * off_diag).sum(axis=1)
        amp_arr[t] = 1.0 + ALPHA_LEAD * boost

    return pd.DataFrame(amp_arr, index=log_ret.index, columns=tickers)


# ── Strategy variants ─────────────────────────────────────────────────────────
def compute_strategy(log_ret, use_centrality, use_spillover,
                     cent_df, incoming_df, lead_amp_df):
    """
    Returns daily portfolio return Series (net of transaction costs).

    Position sizing pipeline:
      raw_pos  = sign(252d return) / realized_vol_63d
      [opt]  × centrality_scalar (1/centrality, mean-normalized)
      [opt]  × spillover_risk_scalar (de-risk high incoming-spillover markets)
      [opt]  × lead_amplifier (boost same-direction receiver markets)
      × book_vol_scale  (target 15% annualized portfolio vol)
    """
    tickers = log_ret.columns.tolist()
    n       = len(tickers)

    cum_ret_252 = log_ret.rolling(TSMOM_WINDOW).sum()
    signal      = np.sign(cum_ret_252)
    vol_63      = (log_ret.rolling(VOL_WINDOW).std() * np.sqrt(252)).clip(lower=1e-6)

    # Base raw positions
    raw_pos = signal / vol_63

    # Centrality adjustment: 1/centrality, normalized to mean=1 each day
    if use_centrality:
        c_inv   = 1.0 / cent_df.reindex(columns=tickers).clip(lower=1e-6)
        c_norm  = c_inv.div(c_inv.mean(axis=1).replace(0, 1.0), axis=0)
        raw_pos = raw_pos * c_norm

    # Spillover adjustment
    if use_spillover:
        risk_scalar = (1.0 - BETA_SPILLOVER * incoming_df.reindex(columns=tickers)
                       ).clip(lower=0.30, upper=1.0)
        lead_amp    = lead_amp_df.reindex(columns=tickers).fillna(1.0)
        raw_pos     = raw_pos * risk_scalar * lead_amp

    # Book-level vol targeting (lagged to avoid lookahead)
    # min_count=1 ensures NaN (not 0) propagates during warm-up so book_vol
    # is NaN rather than ~0, preventing vol_scale from blowing up to 7.5x
    port_ret_raw = (raw_pos.shift(1) * log_ret).sum(axis=1, min_count=1)
    book_vol     = (port_ret_raw.rolling(VOL_WINDOW, min_periods=VOL_WINDOW)
                    .std().shift(1) * np.sqrt(252)).clip(lower=0.02, upper=10.0)
    vol_scale    = (VOL_TARGET / book_vol)
    positions    = raw_pos.multiply(vol_scale, axis=0)

    # Transaction costs on notional position change
    tcost = (positions.diff().abs() * TCOST).sum(axis=1, min_count=1).fillna(0)

    port_ret = (positions.shift(1) * log_ret).sum(axis=1, min_count=1) - tcost
    return port_ret.dropna()


# ── Performance metrics ───────────────────────────────────────────────────────
def perf_stats(ret_series, label=""):
    r = ret_series.dropna()
    if len(r) < 30:
        return {"Ann Ret": "N/A", "Ann Vol": "N/A", "Sharpe": "N/A", "Max DD": "N/A"}

    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0

    cum      = (1 + r).cumprod()
    max_dd   = ((cum - cum.cummax()) / cum.cummax()).min()

    return {
        "Ann Ret": f"{ann_ret:+.1%}",
        "Ann Vol": f"{ann_vol:.1%}",
        "Sharpe":  f"{sharpe:.2f}",
        "Max DD":  f"{max_dd:.1%}",
    }


def print_results(all_returns, log_ret):
    train_mask = log_ret.index <= TRAIN_END
    oos_mask   = log_ret.index >= OOS_START

    periods = [
        ("Full  2010–2024", slice(None)),
        ("Train 2010–2017", train_mask),
        ("OOS   2018–2024", oos_mask),
    ]

    cols = ["Ann Ret", "Ann Vol", "Sharpe", "Max DD"]
    header_width = 35

    print("\n" + "=" * 75)
    print("TSMOM + Graph Overlay — OOS Ablation")
    print("=" * 75)
    print(f"{'Strategy':<{header_width}} {'Period':<20}", end="")
    for c in cols:
        print(f"  {c:>10}", end="")
    print()
    print("-" * 75)

    for name, ret in all_returns.items():
        for period_label, mask in periods:
            if mask is slice(None):
                sub = ret
            else:
                sub = ret[ret.index.isin(log_ret.index[mask])]
            m = perf_stats(sub)
            print(f"{name:<{header_width}} {period_label:<20}", end="")
            for c in cols:
                print(f"  {m[c]:>10}", end="")
            print()
        print()

    print("=" * 75)
    print("\nKey: OOS period is the only one that matters for honest evaluation.")
    print(f"Graph layers add value only if OOS Sharpe > '1. TSMOM (baseline)'.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log_ret = load_returns()

    # Precompute rolling graphs (weekly, ~3-5 min)
    cent_df, incoming_df, gfevd_store = precompute_graphs(log_ret)

    # TSMOM signals (needed for lead amplifier — computed once, shared across variants)
    tsmom_signals = np.sign(log_ret.rolling(TSMOM_WINDOW).sum())

    print("Computing lead amplifiers…")
    lead_amp_df = compute_lead_amplifier(tsmom_signals, gfevd_store, log_ret)

    # Four strategy variants
    variants = {
        "1. TSMOM (baseline)":       (False, False),
        "2. TSMOM + Centrality":     (True,  False),
        "3. TSMOM + Spillover":      (False, True),
        "4. TSMOM + Cent + Spill":   (True,  True),
    }

    print("Running strategy variants…")
    all_returns = {}
    for name, (use_c, use_s) in variants.items():
        all_returns[name] = compute_strategy(
            log_ret, use_c, use_s, cent_df, incoming_df, lead_amp_df
        )

    print_results(all_returns, log_ret)


if __name__ == "__main__":
    main()
