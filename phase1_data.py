import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

TICKERS = {
    "Energy":      ["USO", "XLE", "UNG", "BNO"],
    "Metals":      ["GLD", "SLV", "COPX"],
    "Equities":    ["SPY", "EEM", "EWJ"],
    "Bonds":       ["TLT", "IEF", "HYG"],
    "Agriculture": ["DBA", "WEAT", "CORN"],
    "FX":          ["UUP", "FXE"],
}
ALL_TICKERS = [t for group in TICKERS.values() for t in group]

START = "2010-01-01"
END   = pd.Timestamp.today().strftime("%Y-%m-%d")
MISSING_THRESHOLD = 0.05

os.makedirs("data", exist_ok=True)

# --- Download ---
print(f"Downloading {len(ALL_TICKERS)} ETFs from {START} to {END}...")
raw = yf.download(ALL_TICKERS, start=START, end=END, auto_adjust=True, progress=False)["Close"]
raw.index = pd.to_datetime(raw.index)

# --- Missing data audit ---
total_days = len(raw)
missing_frac = raw.isna().mean()
dropped = missing_frac[missing_frac > MISSING_THRESHOLD].index.tolist()
kept   = missing_frac[missing_frac <= MISSING_THRESHOLD].index.tolist()

if dropped:
    print(f"\nWARNING: Dropping {len(dropped)} ETF(s) with >{MISSING_THRESHOLD:.0%} missing data:")
    for t in dropped:
        print(f"  {t}: {missing_frac[t]:.1%} missing")
else:
    print("\nNo ETFs exceeded the missing data threshold.")

prices = raw[kept].copy()

# Forward-fill remaining gaps (max 5 consecutive days)
filled = prices.ffill(limit=5)
remaining_na = filled.isna().sum().sum()
if remaining_na:
    print(f"WARNING: {remaining_na} NaN(s) remain after forward-fill — dropping those rows.")
    filled = filled.dropna()

prices = filled

# --- Log returns ---
log_returns = np.log(prices / prices.shift(1)).dropna()

# --- Save ---
prices.to_csv("data/prices.csv")
log_returns.to_csv("data/log_returns.csv")
print(f"\nSaved data/prices.csv and data/log_returns.csv")

# --- Basic stats ---
print(f"\n--- Summary ---")
print(f"Date range : {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"Trading days: {len(prices)}")
print(f"Assets kept : {len(kept)}  {kept}")
if dropped:
    print(f"Assets dropped: {dropped}")

# --- Plot 1: Normalized price history ---
fig, ax = plt.subplots(figsize=(14, 6))
normalized = prices / prices.iloc[0] * 100

colors = plt.cm.tab20.colors
color_map = {}
i = 0
for group, tickers in TICKERS.items():
    for t in tickers:
        if t in normalized.columns:
            color_map[t] = colors[i % len(colors)]
        i += 1

for group, tickers in TICKERS.items():
    for t in tickers:
        if t in normalized.columns:
            ax.plot(normalized.index, normalized[t], label=t, color=color_map[t], linewidth=0.9)

ax.set_title("ETF Price History (Normalized to 100)", fontsize=13)
ax.set_ylabel("Indexed Price")
ax.set_xlabel("")
ax.legend(ncol=4, fontsize=7, loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/price_history.png", dpi=150)
plt.close()
print("Saved data/price_history.png")

# --- Plot 2: Correlation heatmap ---
corr = log_returns.corr()

# Order tickers by group
ordered = [t for group in TICKERS.values() for t in group if t in corr.columns]
corr = corr.loc[ordered, ordered]

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    corr,
    ax=ax,
    cmap="RdYlGn",
    vmin=-1, vmax=1,
    annot=True, fmt=".2f",
    annot_kws={"size": 7},
    linewidths=0.4,
    square=True,
)
ax.set_title("Log Return Correlation Matrix", fontsize=13)
plt.tight_layout()
plt.savefig("data/correlation_heatmap.png", dpi=150)
plt.close()
print("Saved data/correlation_heatmap.png")

print("\nPhase 1 complete.")
