import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("results/trades.csv")
df["time"] = pd.to_datetime(df["time"], utc=True)

token = "BERA"
bio_df = (
    df[df["symbol"].str.upper() == token]
    .sort_values("time")
    .copy()
)

if bio_df.empty:
    print("Nessun trade trovato per BIO")
else:
    bio_df["cum_pnl"] = bio_df["realized_pnl"].cumsum()

    plt.figure(figsize=(10, 5))
    plt.plot(bio_df["time"], bio_df["cum_pnl"], marker="o", linewidth=1.8)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("BIO — PnL cumulato")
    plt.xlabel("Data")
    plt.ylabel("PnL cumulato")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plt.show()

