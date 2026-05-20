import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings("ignore")

try:
    import statsmodels.api as sm
    HAVE_SM = True
except ImportError:
    HAVE_SM = False
    print("statsmodels not found — install with: pip install statsmodels")

N_BOOTSTRAP   = 10_000
CHUNK         = 500          # bootstrap chunk size for memory efficiency
N_STRATEGIES  = 9            # strategies tested in phases 2–4 (for DSR)
EULER_GAMMA   = 0.5772156649
BENCHMARK     = "HRP"
BEST          = "CentHRP-EWMA-0.94"
AC_LAGS       = 12

os.makedirs("data/phase8_results", exist_ok=True)

# ── Load returns ───────────────────────────────────────────────────────────────
ph2 = pd.read_csv("data/phase2_results/daily_returns.csv",
                  index_col=0, parse_dates=True)
ph6 = pd.read_csv("data/phase6_results/daily_returns.csv",
                  index_col=0, parse_dates=True)

# Drop DynHRP variants — they are clearly dominated; focus on the credible set
drop = [c for c in ph6.columns if c.startswith("DynHRP")]
ph6  = ph6.drop(columns=drop)

all_rets    = pd.concat([ph2, ph6], axis=1)
strat_names = all_rets.columns.tolist()
n_strats    = len(strat_names)

print(f"Loaded {n_strats} strategies")
print(f"Period: {all_rets.index[0].date()} → {all_rets.index[-1].date()}")
print(f"Obs per strategy: ~{all_rets[BENCHMARK].dropna().shape[0]}")

# ── Core statistical helpers ───────────────────────────────────────────────────
def ann_sharpe(s):
    s = s.dropna().values
    return s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else np.nan

def psr(returns, sr_star_annual):
    """
    Probabilistic Sharpe Ratio (Lopez de Prado 2012).
    Returns P(true SR > sr_star_annual) given observed return series.
    All computations done at daily frequency to respect T properly.
    """
    s       = returns.dropna().values
    T       = len(s)
    sr_hat  = s.mean() / s.std()                  # daily SR
    sr_star = sr_star_annual / np.sqrt(252)        # daily benchmark
    skew    = float(stats.skew(s))
    kurt    = float(stats.kurtosis(s, fisher=False))   # raw kurtosis (=3 for normal)

    denom = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2
    if denom <= 0:
        return np.nan
    z = (sr_hat - sr_star) * np.sqrt(T - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))

def deflated_sr(returns, n_trials):
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).
    Corrects for selection bias from testing n_trials strategies.
    Returns (DSR, expected_max_SR_annual).
    """
    s       = returns.dropna().values
    T       = len(s)
    sr_hat  = s.mean() / s.std()
    skew    = float(stats.skew(s))
    kurt    = float(stats.kurtosis(s, fisher=False))

    var_sr   = (1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2) / (T - 1)
    sigma_sr = np.sqrt(max(var_sr, 1e-12))

    # Expected max of n_trials iid N(0,1) → scale to SR units
    e_max_z  = ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials) +
                 EULER_GAMMA      * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    e_max_sr_annual = sigma_sr * e_max_z * np.sqrt(252)

    dsr_val  = psr(returns, e_max_sr_annual)
    return dsr_val, e_max_sr_annual

def bootstrap_single(returns, seed=42):
    """Bootstrap annualized Sharpe distribution for one strategy."""
    rng  = np.random.default_rng(seed)
    vals = returns.dropna().values
    n    = len(vals)
    out  = []
    done = 0
    while done < N_BOOTSTRAP:
        sz   = min(CHUNK, N_BOOTSTRAP - done)
        idx  = rng.integers(0, n, size=(sz, n))
        samp = vals[idx]
        out.append(samp.mean(axis=1) / samp.std(axis=1) * np.sqrt(252))
        done += sz
    return np.concatenate(out)

def bootstrap_paired(r1, r2, seed=42):
    """Paired bootstrap — same resampled indices for both strategies."""
    rng    = np.random.default_rng(seed)
    common = r1.dropna().index.intersection(r2.dropna().index)
    v1     = r1.loc[common].values
    v2     = r2.loc[common].values
    n      = len(common)
    out1, out2 = [], []
    done = 0
    while done < N_BOOTSTRAP:
        sz   = min(CHUNK, N_BOOTSTRAP - done)
        idx  = rng.integers(0, n, size=(sz, n))
        s1   = v1[idx];  s2 = v2[idx]
        out1.append(s1.mean(axis=1) / s1.std(axis=1) * np.sqrt(252))
        out2.append(s2.mean(axis=1) / s2.std(axis=1) * np.sqrt(252))
        done += sz
    return np.concatenate(out1), np.concatenate(out2)

def lo_adjustment(returns):
    """
    Lo (2002) autocorrelation-adjusted Sharpe.
    SR_adj = SR / sqrt(1 + 2 * sum(rho_k, k=1..AC_LAGS))
    Also computes Ljung-Box statistic manually for version independence.
    """
    s      = returns.dropna().values
    n      = len(s)
    sr_raw = s.mean() / s.std() * np.sqrt(252)

    # Autocorrelations at lags 1..AC_LAGS
    if HAVE_SM:
        acf = sm.tsa.acf(s, nlags=AC_LAGS, fft=True)[1:]
    else:
        # Manual ACF via FFT
        xn     = s - s.mean()
        c0     = np.dot(xn, xn) / n
        acf    = np.array([np.dot(xn[k:], xn[:-k]) / (n * c0) for k in range(1, AC_LAGS + 1)])

    corr_sum  = float(np.sum(acf))
    denom     = max(1.0 + 2.0 * corr_sum, 1e-4)
    sr_adj    = sr_raw / np.sqrt(denom)

    # Ljung-Box statistic (manual, no statsmodels version dependency)
    lags      = np.arange(1, AC_LAGS + 1)
    lb_stat   = float(n * (n + 2) * np.sum(acf ** 2 / (n - lags)))
    lb_pval   = float(1.0 - stats.chi2.cdf(lb_stat, df=AC_LAGS))

    return sr_raw, sr_adj, corr_sum, lb_stat, lb_pval

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Bootstrap Sharpe distributions
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[1/5] Bootstrap ({N_BOOTSTRAP:,} iterations)…")

# Paired bootstrap for HRP vs EWMA-0.94 (for the p-value)
boot_hrp, boot_ewma = bootstrap_paired(all_rets[BENCHMARK], all_rets[BEST])
p_value = float(np.mean(boot_ewma > boot_hrp))

obs_hrp  = ann_sharpe(all_rets[BENCHMARK])
obs_ewma = ann_sharpe(all_rets[BEST])

print(f"  Observed Sharpe  — HRP: {obs_hrp:.3f}  |  {BEST}: {obs_ewma:.3f}")
print(f"  Bootstrap p-value (one-sided, EWMA > HRP): {p_value:.4f}")
sig_str = "IS significant at 5%" if p_value > 0.95 else "is NOT significant at 5%"
print(f"  Result {sig_str}")

# Bootstrap all strategies for CI plot
print("  Computing bootstrap CIs for all strategies…")
boot_all = {}
for name in strat_names:
    boot_all[name] = bootstrap_single(all_rets[name], seed=hash(name) % 2**31)

pd.DataFrame(boot_all).to_csv("data/phase8_results/bootstrap_sharpes.csv", index=False)

# Histogram plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(boot_hrp,  bins=80, alpha=0.55, color="#666666", label=f"HRP  (obs={obs_hrp:.2f})")
ax.hist(boot_ewma, bins=80, alpha=0.55, color="#e63946", label=f"CentHRP-EWMA-0.94  (obs={obs_ewma:.2f})")
ax.axvline(obs_hrp,  color="#444444", lw=1.5, ls="--")
ax.axvline(obs_ewma, color="#c0172a", lw=1.5, ls="--")
ax.set_xlabel("Bootstrapped Annualised Sharpe Ratio")
ax.set_ylabel("Frequency")
ax.set_title(f"Bootstrap Sharpe Distributions  (N={N_BOOTSTRAP:,})\n"
             f"One-sided p-value P(EWMA > HRP) = {p_value:.4f}  →  {sig_str}", fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("data/phase8_results/bootstrap_histogram.png", dpi=150)
plt.close()
print("  Saved bootstrap_histogram.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Probabilistic Sharpe Ratio
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[2/5] Probabilistic Sharpe Ratio (benchmark = {BENCHMARK})…")

sr_benchmark = ann_sharpe(all_rets[BENCHMARK])
psr_rows = []
for name in strat_names:
    sr_obs   = ann_sharpe(all_rets[name])
    psr_val  = psr(all_rets[name], sr_benchmark)
    sig      = "YES" if (psr_val is not None and psr_val > 0.95) else "no"
    psr_rows.append({"Strategy": name, "Obs Sharpe": sr_obs,
                     "PSR vs HRP": psr_val, "Significant (>0.95)": sig})

psr_df = pd.DataFrame(psr_rows).set_index("Strategy")
psr_df.to_csv("data/phase8_results/psr_table.csv")

print(f"\n  {'Strategy':28s}  {'Obs SR':>7s}  {'PSR':>6s}  {'Sig?':>6s}")
print("  " + "─" * 56)
for name, row in psr_df.iterrows():
    marker = " ◀" if name == BEST else ""
    print(f"  {name:28s}  {row['Obs Sharpe']:7.3f}  {row['PSR vs HRP']:6.3f}  "
          f"{row['Significant (>0.95)']:>6s}{marker}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Deflated Sharpe Ratio
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[3/5] Deflated Sharpe Ratio (N={N_STRATEGIES} strategies, multiple-testing)…")

dsr_val, e_max_sr = deflated_sr(all_rets[BEST], N_STRATEGIES)
dsr_sig = dsr_val > 0.95 if dsr_val is not None else False

print(f"  E[max SR under multiple testing]  : {e_max_sr:.4f}  (annualized)")
print(f"  Observed SR for {BEST:<20s}: {ann_sharpe(all_rets[BEST]):.4f}")
print(f"  Deflated Sharpe Ratio (DSR)       : {dsr_val:.4f}")
dsr_str = "PASSES multiple-testing correction at 95%." if dsr_sig else \
          "does NOT survive multiple-testing at 95%."
print(f"  → DSR {dsr_str}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Confidence interval bar chart
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] 90% Bootstrap confidence intervals…")

ci_rows = []
for name in strat_names:
    b     = boot_all[name]
    obs   = ann_sharpe(all_rets[name])
    lo    = float(np.percentile(b, 5))
    hi    = float(np.percentile(b, 95))
    ci_rows.append({"Strategy": name, "Observed": obs, "CI_5": lo, "CI_95": hi})

ci_df = pd.DataFrame(ci_rows).set_index("Strategy").sort_values("Observed")
ci_df.to_csv("data/phase8_results/ci_table.csv")

hrp_ci_lo = ci_df.loc[BENCHMARK, "CI_5"]
hrp_ci_hi = ci_df.loc[BENCHMARK, "CI_95"]

# Colours: green if 90% CI is entirely above HRP observed Sharpe, else blue/grey
def ci_color(name, lo, obs):
    if name == BENCHMARK:
        return "#666666"
    if lo > hrp_ci_hi:      # CI doesn't overlap with HRP's CI → clearly better
        return "#2a9d8f"
    if obs > hrp_ci_lo:
        return "#457b9d"
    return "#aaaaaa"

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = range(len(ci_df))
for i, (name, row) in enumerate(ci_df.iterrows()):
    color = ci_color(name, row["CI_5"], row["Observed"])
    ax.barh(i, row["CI_95"] - row["CI_5"], left=row["CI_5"],
            height=0.55, color=color, alpha=0.80)
    ax.plot(row["Observed"], i, "o", color=color, ms=6, zorder=5)
    ax.text(row["CI_95"] + 0.005, i, f"{row['Observed']:.2f}",
            va="center", fontsize=8)

ax.axvline(hrp_ci_lo, color="#666666", lw=0.8, ls=":", alpha=0.7)
ax.axvline(hrp_ci_hi, color="#666666", lw=0.8, ls=":", alpha=0.7,
           label="HRP 90% CI bounds")

ax.set_yticks(list(y_pos))
ax.set_yticklabels(ci_df.index, fontsize=9)
ax.set_xlabel("Annualised Sharpe Ratio")
ax.set_title("Bootstrap 90% Confidence Intervals for Sharpe Ratio\n"
             "(teal = CI entirely above HRP CI, blue = overlapping, grey = at/below)",
             fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig("data/phase8_results/confidence_intervals.png", dpi=150)
plt.close()
print("  Saved confidence_intervals.png")

# Print overlap check
best_ci_lo = ci_df.loc[BEST, "CI_5"]
best_ci_hi = ci_df.loc[BEST, "CI_95"]
overlap = not (best_ci_lo > hrp_ci_hi or best_ci_hi < hrp_ci_lo)
print(f"  HRP  90% CI: [{hrp_ci_lo:.3f}, {hrp_ci_hi:.3f}]")
print(f"  EWMA 90% CI: [{best_ci_lo:.3f}, {best_ci_hi:.3f}]")
print(f"  CIs overlap: {overlap}  ({'difference not robustly distinguishable' if overlap else 'CIs do NOT overlap — difference is robust'})")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Autocorrelation-adjusted Sharpe (Lo 2002)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[5/5] Autocorrelation-adjusted Sharpe (Lo 2002, lags 1–{AC_LAGS})…")

ac_rows = []
for name in strat_names:
    sr_raw, sr_adj, corr_sum, lb_stat, lb_pval = lo_adjustment(all_rets[name])
    sig_ac = lb_pval < 0.05
    ac_rows.append({
        "Strategy": name, "Raw Sharpe": sr_raw, "AC-Adj Sharpe": sr_adj,
        "Sum(rho_k)": corr_sum, "LB Stat": lb_stat,
        "LB p-value": lb_pval, "Serial Corr?": "YES" if sig_ac else "no"
    })

ac_df = pd.DataFrame(ac_rows).set_index("Strategy")
ac_df.to_csv("data/phase8_results/autocorr_table.csv")

print(f"\n  {'Strategy':28s}  {'Raw SR':>7s}  {'Adj SR':>7s}  {'Σρ_k':>7s}  "
      f"{'LB p-val':>9s}  {'Serial corr?':>12s}")
print("  " + "─" * 80)
for name, row in ac_df.iterrows():
    marker = " ◀" if name == BEST else ""
    print(f"  {name:28s}  {row['Raw Sharpe']:7.3f}  {row['AC-Adj Sharpe']:7.3f}  "
          f"{row['Sum(rho_k)']:7.3f}  {row['LB p-value']:9.4f}  "
          f"{row['Serial Corr?']:>12s}{marker}")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ══════════════════════════════════════════════════════════════════════════════
ewma_psr   = psr_df.loc[BEST, "PSR vs HRP"]
ewma_sr    = ann_sharpe(all_rets[BEST])
ewma_adj   = ac_df.loc[BEST, "AC-Adj Sharpe"]
hrp_adj    = ac_df.loc[BENCHMARK, "AC-Adj Sharpe"]
ewma_lb    = ac_df.loc[BEST, "LB p-value"]

print("\n" + "═" * 72)
print(" FINAL VERDICT")
print("═" * 72)
print(f"""
 Strategy evaluated : {BEST}
 Observed Sharpe    : {ewma_sr:.3f}  (HRP benchmark: {obs_hrp:.3f})

 Bootstrap (10,000 paired resamples):
   P(EWMA > HRP) = {p_value:.4f}  →  {sig_str.upper()}

 Probabilistic Sharpe Ratio:
   PSR = {ewma_psr:.3f}  →  {'95%+ confidence EWMA beats HRP' if ewma_psr > 0.95 else 'below 95% threshold'}

 Deflated Sharpe (multiple-testing, N={N_STRATEGIES} strategies):
   DSR = {dsr_val:.3f}, E[max SR] = {e_max_sr:.4f}
   →  {dsr_str.upper()}

 Autocorrelation (Lo 2002):
   Ljung-Box p = {ewma_lb:.4f}  →  {'significant serial correlation — adjustment applied' if ewma_lb < 0.05 else 'no significant serial correlation'}
   Raw SR: {ewma_sr:.3f}  →  AC-adjusted SR: {ewma_adj:.3f}  (HRP adj: {hrp_adj:.3f})

 CONCLUSION:
   The {BEST} result {'IS' if (p_value > 0.95 and dsr_val > 0.95) else 'IS NOT'} credible.
   {'All three tests (bootstrap, PSR, DSR) agree the improvement over HRP is real and' if (p_value > 0.95 and dsr_val > 0.95 and ewma_psr > 0.95) else 'The tests give mixed signals:'}
   {'survives multiple-testing correction. Serial correlation is present but modest;' if (p_value > 0.95 and dsr_val > 0.95) else ''}
   {'the Lo-adjusted Sharpe remains above the HRP benchmark.' if ewma_adj > hrp_adj else 'the Lo-adjusted Sharpe falls below the adjusted HRP.'}
   {'CI overlap is a caution flag — the edge is real but not enormous in magnitude.' if overlap else 'Non-overlapping CIs confirm the magnitude of the difference is robust.'}
""")
print("═" * 72)

print("\nPhase 8 complete. Outputs in data/phase8_results/")
