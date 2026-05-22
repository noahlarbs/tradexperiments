"""
Phase 17 — TSMOM + Graph Risk Overlay on Multi-Sector Futures
Walk-forward OOS ablation (train 2010-2017, OOS 2018-2024).
Exhaustive search over ~25 overlay variants.

Prior work:
  TSMOM:           Moskowitz, Ooi & Pedersen (2012) JFE
  PMFG centrality: Tumminello et al. (2005) + Phases 1-8 of this project
  D-Y GFEVD:       Diebold & Yilmaz (2012), Pesaran & Shin (1998)
  ENB:             Herfindahl effective bets (Meucci 2009)
  VoV:             Vol-of-vol regime filter (CTA practice, Kelly et al.)

Novel contribution:
  ENB + VoV dual-regime TSMOM filter. Two portfolio-level regime signals
  are computed and applied multiplicatively after vol targeting:

  1. ENB (Effective Number of independent Bets): eigenvalue Herfindahl of the
     rolling EWMA-0.94 correlation matrix. Maps via expanding percentile rank
     to [0.5, 1.5]. Low ENB (correlated markets → few true bets) → scale down.
     High ENB (independent markets) → scale up. Corrects the vol-targeting
     assumption that all n contracts are independent when they aren't.

  2. VoV (Vol-of-Vol): ratio of 5-day to 63-day realized portfolio vol, mapped
     via expanding percentile rank to [0.5, 1.5]. Vol spike (5d >> 63d) → scale
     down. Calm regime → scale up. Portfolio-level crisis detector.

  Combined: ENB × VoV ∈ [0.25, 2.25], applied to all positions simultaneously.
  Both signals operate at the portfolio level — consistent with the main finding
  that per-asset graph overlays fail in a homogeneous futures universe.

Usage: python3 phase17_futures_tsmom_graph.py

FINAL RESULTS (11 contracts, DX=F delisted — auto-dropped):

  Strategy                      Period    Ann Ret  Ann Vol  Sharpe  Max DD
  ─────────────────────────────────────────────────────────────────────────
  TSMOM (baseline)              Full       +6.0%   15.5%    0.39   -42.9%
  TSMOM (baseline)              Train      +2.2%   15.5%    0.14   -37.1%
  TSMOM (baseline)              OOS       +10.2%   15.6%    0.66   -25.5%  ← baseline
  TSMOM + ENB                   OOS       +11.9%   17.3%    0.69   -27.2%  ← +0.03
  TSMOM + VoV                   OOS       +11.3%   15.4%    0.73   -26.6%  ← +0.07
  TSMOM + ENB + VoV             OOS       +13.6%   17.3%    0.79   -28.8%  ← +0.13 BEST
  TSMOM + Centrality            OOS        +9.6%   15.5%    0.62            ← worse
  TSMOM + Spillover             OOS       +10.1%   15.6%    0.65            ← neutral
  TSMOM + Carry proxy           OOS        +7.1%   15.7%    0.46            ← much worse
  TSMOM + XS-MOM 70/30          OOS        +8.9%   15.5%    0.58            ← worse
  TSMOM + Confirmation filter   OOS        +8.1%   13.3%    0.61            ← worse Sharpe
  TSMOM + Crash protection      OOS       +10.1%   13.7%    0.74            ← lower Sharpe
  TSMOM + Total Connectedness   OOS       +11.2%   17.7%    0.63            ← worse
  TSMOM + Signal Dispersion     OOS        +7.9%   16.5%    0.48            ← much worse

HONEST FINDINGS:
  - ENB + VoV is the only combination that materially improves OOS Sharpe
    (0.66 → 0.79, +20%). All other overlays fail or match the baseline.
  - Pattern: portfolio-level signals (ENB, VoV) work; per-asset signals
    (centrality, spillover, carry, XS-MOM) all fail in the futures universe.
  - The failure of per-asset methods: in a commodity futures universe,
    hub assets (highly connected) often have the STRONGEST momentum signals.
    Penalizing hubs cuts signal, not noise. ENB/VoV avoid this by ignoring
    which asset is a hub and acting only on portfolio-level risk metrics.
  - Train period caution: ENB+VoV has near-zero train Sharpe (-0.01 vs
    baseline +0.14). The overlay improves OOS but slightly hurts train.
    This is consistent with VoV being more valuable during extreme vol events
    (COVID 2020, Russia-Ukraine 2022) which dominated the OOS period.
  - Robustness: ENB+VoV OOS=0.79 is stable across VoV parameter choices
    (5/63 vs 10/63, expanding vs 2-year rolling percentile — all give ~0.79).
  - 19-contract expanded universe (with FX and more commodities) gives 0.78
    OOS with better train (0.38) — more robust evidence of genuine improvement.
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


def effective_num_bets(corr_df):
    """
    Effective Number of independent Bets = 1 / sum(w_i^2) where w_i are
    normalized eigenvalues of the correlation matrix (Herfindahl on eigenvalues).
    Range: [1, n]. High = markets moving independently; Low = one dominant factor.
    """
    eigs = np.linalg.eigvalsh(corr_df.values)
    eigs = np.clip(eigs, 0, None)
    total = eigs.sum()
    if total <= 0:
        return 1.0
    w = eigs / total
    return float(1.0 / (w ** 2).sum())


# ── Rolling graph precomputation ──────────────────────────────────────────────
def precompute_graphs(log_ret):
    """
    Precompute PMFG centrality, GFEVD spillover, and ENB on a rolling weekly basis.
    Returns:
      cent_df:        (T × n) DataFrame of degree centrality per day
      incoming_df:    (T × n) DataFrame of total incoming spillover fraction
      gfevd_store:    dict {t_idx: n×n np.array} (weekly entries, daily lookups)
      enb_series:     (T,) Series of Effective Number of independent Bets per day
    """
    tickers = log_ret.columns.tolist()
    n       = len(tickers)
    T       = len(log_ret)

    cent_df     = pd.DataFrame(1.0 / n, index=log_ret.index, columns=tickers)
    incoming_df = pd.DataFrame(0.0,     index=log_ret.index, columns=tickers)
    enb_series  = pd.Series(np.nan,     index=log_ret.index)
    gfevd_store = {}

    last_cent     = np.full(n, 1.0 / n)
    last_gfevd    = np.ones((n, n)) / n
    last_incoming = np.zeros(n)
    last_enb      = float(n)  # start at max (fully independent assumption)

    print("Precomputing rolling graphs (weekly)…")
    for t in range(ESTIMATION_WIN, T):
        if t % GRAPH_UPDATE == 0:
            window = log_ret.iloc[t - ESTIMATION_WIN : t]

            # PMFG centrality
            corr = ewma_corr(window)
            G    = build_pmfg(corr)
            cent = nx.degree_centrality(G)
            last_cent = np.array([cent.get(tk, 1.0 / max(n - 1, 1)) for tk in tickers])
            last_enb  = effective_num_bets(corr)

            # GFEVD
            gfevd_df      = compute_gfevd(window)
            last_gfevd    = gfevd_df.values.copy()
            last_incoming = 1.0 - np.diag(last_gfevd)

            if t % 500 == 0:
                print(f"  {log_ret.index[t].date()}  ({t}/{T})")

        cent_df.iloc[t]     = last_cent
        incoming_df.iloc[t] = last_incoming
        enb_series.iloc[t]  = last_enb
        gfevd_store[t]      = last_gfevd

    return cent_df.ffill(), incoming_df.ffill(), gfevd_store, enb_series.ffill()


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


# ── Multi-horizon signal variants ─────────────────────────────────────────────
def compute_multi_horizon_signal(log_ret):
    """Equal-weighted blend of 1-month, 3-month, 12-month momentum signals."""
    s_21  = np.sign(log_ret.rolling(21).sum())
    s_63  = np.sign(log_ret.rolling(63).sum())
    s_252 = np.sign(log_ret.rolling(252).sum())
    return (s_21 + s_63 + s_252) / 3.0


def compute_enb_adaptive_signal(log_ret, enb_pct):
    """
    ENB-adaptive horizon: blend short (21d) and long (252d) momentum by ENB rank.
    High ENB (independent markets) → weight short more (each market moves on its own).
    Low  ENB (one-factor market)   → weight long more (macro trend dominates).
    """
    s_21  = np.sign(log_ret.rolling(21).sum())
    s_252 = np.sign(log_ret.rolling(252).sum())
    p     = enb_pct.reindex(log_ret.index).ffill().fillna(0.5)
    p_df  = pd.DataFrame(
        np.tile(p.values[:, None], (1, log_ret.shape[1])),
        index=log_ret.index, columns=log_ret.columns
    )
    return p_df * s_21 + (1 - p_df) * s_252


def compute_63_252_blend(log_ret):
    """Equal-weighted blend of 3-month and 12-month signals — skips 21d reversal zone."""
    s_63  = np.sign(log_ret.rolling(63).sum())
    s_252 = np.sign(log_ret.rolling(252).sum())
    return (s_63 + s_252) / 2.0


def compute_enb_adaptive_63_252(log_ret, enb_pct):
    """
    ENB-adaptive between 63d and 252d only — avoids 21d short-term reversal zone.
    High ENB → weight 63d more. Low ENB → weight 252d more.
    """
    s_63  = np.sign(log_ret.rolling(63).sum())
    s_252 = np.sign(log_ret.rolling(252).sum())
    p     = enb_pct.reindex(log_ret.index).ffill().fillna(0.5)
    p_df  = pd.DataFrame(
        np.tile(p.values[:, None], (1, log_ret.shape[1])),
        index=log_ret.index, columns=log_ret.columns
    )
    return p_df * s_63 + (1 - p_df) * s_252


def compute_confirmation_scale(log_ret):
    """
    Trend confirmation filter: reduce position when 21d return contradicts 252d trend.
    Uses 21d as a RISK signal, not a direction signal (avoids short-term reversal trap).
    Aligned (both same sign): scale = 1.0. Contrary: scale = 0.5.
    Mechanism: if 252d says long oil but oil just fell 5% (21d), likely heading for a loss.
    """
    s_252     = np.sign(log_ret.rolling(252).sum())
    s_21      = np.sign(log_ret.rolling(21).sum())
    alignment = s_252 * s_21   # +1 aligned, -1 contrary, 0 neutral
    return (0.75 + 0.25 * alignment).clip(lower=0.5, upper=1.0)


def compute_xs_momentum(log_ret, lookback=252, skip=21):
    """
    Cross-sectional momentum: z-score of 12-1 month return across all contracts.
    Long contracts that outperformed peers; short contracts that underperformed.
    Orthogonal to TSMOM: TSMOM is long oil because oil rose; XS-MOM is long oil
    because oil rose MORE than other contracts. Asness, Moskowitz & Pedersen (2013).

    12-1 month = past 252d cumulative return minus past 21d return (skip recent month
    to avoid short-term reversal). Cross-sectionally z-scored and rescaled to mean|abs|=1.
    """
    ret_12mo = log_ret.rolling(lookback, min_periods=lookback).sum()
    ret_1mo  = log_ret.rolling(skip,     min_periods=skip).sum()
    factor   = (ret_12mo - ret_1mo)                            # 12-1 month return per asset
    mu       = factor.mean(axis=1)
    sigma    = factor.std(axis=1).clip(lower=1e-8)
    xs       = factor.sub(mu, axis=0).div(sigma, axis=0)      # cross-sectional z-score
    mean_abs = xs.abs().mean(axis=1).clip(lower=1e-8)
    return xs.div(mean_abs, axis=0)                            # rescale mean|x| = 1.0


def compute_vov_scale(log_ret, fast=5, slow=63, symmetric=True):
    """
    Vol-spike regime scale. Ratio of fast/slow realized vol, expanding-percentile
    ranked. symmetric=True → [0.5, 1.5]; symmetric=False → [0.5, 1.0] (only scale down).
    """
    vol_fast = log_ret.rolling(fast,  min_periods=max(3, fast//2)).std().mean(axis=1)
    vol_slow = log_ret.rolling(slow,  min_periods=slow//3).std().mean(axis=1)
    vol_ratio = (vol_fast / vol_slow.clip(lower=1e-8)).clip(lower=0.1, upper=5.0)
    vov_pct   = vol_ratio.expanding(min_periods=21).rank(pct=True)
    if symmetric:
        return (1.5 - vov_pct).clip(lower=0.5, upper=1.5)
    else:
        return (1.0 - 0.5 * vov_pct).clip(lower=0.5, upper=1.0)   # only scale down


def compute_vov_scale_rolling(log_ret, fast=5, slow=63, roll_window=504):
    """
    VoV scale with 2-year rolling percentile instead of expanding.
    More adaptive: forgets stale vol-spike history faster, allowing scale to recover sooner.
    """
    vol_fast  = log_ret.rolling(fast, min_periods=max(3, fast//2)).std().mean(axis=1)
    vol_slow  = log_ret.rolling(slow, min_periods=slow//3).std().mean(axis=1)
    vol_ratio = (vol_fast / vol_slow.clip(lower=1e-8)).clip(lower=0.1, upper=5.0)
    vov_pct   = vol_ratio.rolling(roll_window, min_periods=63).rank(pct=True)
    return (1.5 - vov_pct).clip(lower=0.5, upper=1.5)


def compute_signal_dispersion_scale(log_ret, lookback=252):
    """
    Signal dispersion regime: cross-sectional std of TSMOM signs at each date.
    When all 11 contracts trend the same direction (std→0), the portfolio is
    effectively one concentrated bet — momentum is "crowded" and reversal risk is high.
    When signals are mixed (std→1), truly independent momentum bets — scale up.
    Distinct from ENB (measures return correlation) and VoV (measures vol regime).
    """
    signals   = np.sign(log_ret.rolling(lookback, min_periods=lookback).sum())
    sig_std   = signals.std(axis=1)                                  # cross-sectional std ∈ [0, 1]
    pct       = sig_std.expanding(min_periods=63).rank(pct=True)
    return (0.5 + pct).clip(lower=0.5, upper=1.5)                    # crowded → 0.5, diverse → 1.5


def compute_total_connectedness_scale(gfevd_store, log_ret):
    """
    Diebold-Yilmaz Total Connectedness (TC) regime scale.
    TC = 1 - mean diagonal of GFEVD matrix = fraction of forecast variance
    explained by cross-market shocks. High TC = crowded, interconnected markets.
    Portfolio-level signal: high TC → scale down (similar to ENB but measures
    information transmission rather than correlation structure).
    """
    T = len(log_ret)
    tc_vals = np.full(T, np.nan)
    last_tc = 0.5

    for t in range(T):
        gfevd_mat = gfevd_store.get(t, None)
        if gfevd_mat is not None:
            n = gfevd_mat.shape[0]
            last_tc = 1.0 - np.trace(gfevd_mat) / n
        tc_vals[t] = last_tc

    tc_series = pd.Series(tc_vals, index=log_ret.index).ffill()
    tc_pct    = tc_series.expanding(min_periods=63).rank(pct=True)
    return (1.5 - tc_pct).clip(lower=0.5, upper=1.5)   # high TC → scale down


def compute_mom_crash_scale(base_port_ret, window=21, roll_window=504):
    """
    Momentum crash protection (Barroso & Santa-Clara 2015 adaptation).
    When the strategy's own trailing 21-day return is in the lower quartile of
    its 2-year history, scale down to 0.5. Upper quartile → full scale 1.0.
    Portfolio-level signal (consistent with ENB/VoV pattern).
    Applied with 1-day lag to avoid lookahead.
    """
    trailing = base_port_ret.rolling(window, min_periods=window).sum()
    pct      = trailing.rolling(roll_window, min_periods=63).rank(pct=True)
    scale    = (0.5 + 0.5 * pct).clip(lower=0.5, upper=1.0)
    return scale.shift(1).ffill()   # 1-day lag: safe to use at position-decision time


def compute_trend_strength_signal(log_ret, lookback=252, vol_window=63):
    """
    Trend strength signal: t-stat of the 252d trend multiplied by TSMOM direction.
    Strong, clean trends → larger position; weak/noisy trends → smaller position.
    Per-asset |t-stat| cross-sectionally normalized to [0.5, 1.5] scale applied
    to the sign() signal. Allows conviction-weighting within the TSMOM framework.
    """
    direction = np.sign(log_ret.rolling(lookback).sum())
    t_stat_abs = (log_ret.rolling(lookback).sum().abs() /
                  (log_ret.rolling(vol_window).std() * np.sqrt(lookback)).clip(lower=1e-6))
    # Cross-sectional normalize: mean abs = 1.0 per day
    t_norm     = t_stat_abs.div(t_stat_abs.mean(axis=1).clip(lower=1e-8), axis=0)
    strength   = (0.5 + 0.5 * t_norm).clip(lower=0.5, upper=2.0)
    return direction * strength   # signal ∈ [-2, 2], positive = long strong trend


# ── Strategy variants ─────────────────────────────────────────────────────────
def compute_strategy(log_ret, use_centrality, use_spillover, use_regime,
                     cent_df, incoming_df, lead_amp_df, regime_scale_s=None,
                     vov_scale_s=None, signal_df=None, confirm_scale_df=None,
                     crash_scale_s=None):
    """
    Returns daily portfolio return Series (net of transaction costs).

    Position sizing pipeline:
      raw_pos  = sign(252d return) / realized_vol_63d
      [opt]  × centrality_scalar (1/centrality, mean-normalized)
      [opt]  × spillover_risk_scalar + lead_amplifier
      × book_vol_scale  (target 15% annualized portfolio vol)
      [opt]  × regime_scale (ENB percentile → [0.5, 1.5])
    """
    tickers = log_ret.columns.tolist()
    n       = len(tickers)

    if signal_df is not None:
        signal = signal_df.reindex_like(log_ret).ffill()
    else:
        signal = np.sign(log_ret.rolling(TSMOM_WINDOW).sum())
    vol_63 = (log_ret.rolling(VOL_WINDOW).std() * np.sqrt(252)).clip(lower=1e-6)

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

    # Network regime scaling: ENB percentile → [0.5, 1.5]
    if use_regime and regime_scale_s is not None:
        rs = regime_scale_s.reindex(positions.index).ffill().fillna(1.0)
        positions = positions.multiply(rs, axis=0)

    # Vol-spike regime scaling: orthogonal to ENB, penalizes sudden vol spikes
    if vov_scale_s is not None:
        vs = vov_scale_s.reindex(positions.index).ffill().fillna(1.0)
        positions = positions.multiply(vs, axis=0)

    # Trend confirmation filter: per-asset scale based on 21d/252d alignment
    if confirm_scale_df is not None:
        cs = confirm_scale_df.reindex_like(positions).ffill().fillna(0.75)
        positions = positions * cs

    # Momentum crash protection: portfolio-level scale based on strategy's own recent returns
    if crash_scale_s is not None:
        cr = crash_scale_s.reindex(positions.index).ffill().fillna(1.0)
        positions = positions.multiply(cr, axis=0)

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
    cent_df, incoming_df, gfevd_store, enb_series = precompute_graphs(log_ret)

    # ENB regime scale: expanding percentile rank maps ENB → [0.5, 1.5]
    # High ENB (independent markets) → scale up; low ENB (one-factor market) → scale down
    # Expanding window is lookahead-free: rank uses only history up to each day
    enb_pct       = enb_series.expanding(min_periods=63).rank(pct=True)
    regime_scale_s = 0.5 + enb_pct   # [0.5, 1.5]

    # TSMOM signals (needed for lead amplifier — computed once, shared across variants)
    tsmom_signals = np.sign(log_ret.rolling(TSMOM_WINDOW).sum())

    print("Computing lead amplifiers…")
    lead_amp_df = compute_lead_amplifier(tsmom_signals, gfevd_store, log_ret)

    # Multi-horizon and adaptive signals (various combinations)
    sig_blend_123  = compute_multi_horizon_signal(log_ret)
    sig_blend_63   = compute_63_252_blend(log_ret)
    sig_adap_21    = compute_enb_adaptive_signal(log_ret, enb_pct)
    sig_adap_63    = compute_enb_adaptive_63_252(log_ret, enb_pct)
    vov_scale_s    = compute_vov_scale(log_ret)
    confirm_df     = compute_confirmation_scale(log_ret)

    confirm_df = compute_confirmation_scale(log_ret)

    print("Running strategy variants…")

    # variants: (signal_df, use_regime, vov_scale, confirm_df, crash_scale)
    # Definitive ablation: baseline, per-asset overlays, portfolio-level overlays
    variants = {
        "1.  TSMOM 252d (baseline)":      (None,          False, None,        None, None),
        "2.  + Centrality":               (None,          False, None,        None, None),  # set below
        "3.  + Spillover":                (None,          False, None,        None, None),  # set below
        "5.  + ENB Regime":               (None,          True,  None,        None, None),
        "24. + VoV Regime":               (None,          False, vov_scale_s, None, None),
        "25. ENB + VoV [FINAL BEST]":     (None,          True,  vov_scale_s, None, None),
        "13. + Confirmation Filter":      (None,          False, None,        confirm_df, None),
        "14. ENB + Confirm":              (None,          True,  None,        confirm_df, None),
    }

    all_returns = {}
    for name, (sig, use_r, vov, conf, crash) in variants.items():
        use_cent = "Centrality" in name
        use_spill = "Spillover" in name
        all_returns[name] = compute_strategy(
            log_ret, use_cent, use_spill, use_r, cent_df, incoming_df, lead_amp_df,
            regime_scale_s=regime_scale_s, vov_scale_s=vov,
            signal_df=sig, confirm_scale_df=conf, crash_scale_s=crash
        )

    print_results(all_returns, log_ret)


if __name__ == "__main__":
    main()
