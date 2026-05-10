"""Funding rate distribution and sign-run duration analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = Path("data/all_perps_hourly_funding.csv")
FIGURES_DIR = Path("analysis/figures")

PERP_SUBSET: list[str] = ["BTC", "HYPE", "ETH"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    df = df.sort_values(["perp", "time"]).reset_index(drop=True)
    return df


def compute_runs(rates: pd.Series) -> tuple[list[int], list[int]]:
    """Return (positive_run_lengths, negative_run_lengths) in hours."""
    pos_runs: list[int] = []
    neg_runs: list[int] = []

    sign = np.sign(rates.values)
    sign[sign == 0] = 1  # zero treated as positive

    current_sign = sign[0]
    count = 1
    for s in sign[1:]:
        if s == current_sign:
            count += 1
        else:
            (pos_runs if current_sign == 1 else neg_runs).append(count)
            current_sign = s
            count = 1
    (pos_runs if current_sign == 1 else neg_runs).append(count)

    return pos_runs, neg_runs


def plot_funding_distribution(rates: pd.Series, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(rates, bins=200, color="steelblue", edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("Funding rate")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_run_durations(
    pos_runs: list[int], neg_runs: list[int], title_prefix: str, out_path: Path
) -> None:
    fig, (ax_pos, ax_neg) = plt.subplots(1, 2, figsize=(12, 5))

    max_hours = max(max(pos_runs, default=1), max(neg_runs, default=1))
    bins = np.arange(1, min(max_hours + 2, 201))

    ax_pos.hist(pos_runs, bins=bins, color="seagreen", edgecolor="none")
    ax_pos.set_xlabel("Duration (hours)")
    ax_pos.set_ylabel("Count")
    ax_pos.set_title(f"{title_prefix} — positive runs")

    ax_neg.hist(neg_runs, bins=bins, color="firebrick", edgecolor="none")
    ax_neg.set_xlabel("Duration (hours)")
    ax_neg.set_ylabel("Count")
    ax_neg.set_title(f"{title_prefix} — negative runs")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_stats(pos_runs: list[int], neg_runs: list[int], label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    avg_pos = np.mean(pos_runs) if pos_runs else float("nan")
    avg_neg = np.mean(neg_runs) if neg_runs else float("nan")
    print(f"{prefix}Average duration of positive funding runs : {avg_pos:.2f} h")
    print(f"{prefix}Average duration of negative funding runs : {avg_neg:.2f} h")
    print(f"{prefix}Average time + → - transition             : {avg_pos:.2f} h")
    print(f"{prefix}Average time - → + transition             : {avg_neg:.2f} h")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    # --- pooled across all perps ---
    all_pos_runs: list[int] = []
    all_neg_runs: list[int] = []
    for _, perp_df in df.groupby("perp"):
        pos, neg = compute_runs(perp_df["fundingRate"].reset_index(drop=True))
        all_pos_runs.extend(pos)
        all_neg_runs.extend(neg)

    plot_funding_distribution(
        df["fundingRate"],
        "Distribution of hourly funding rate (all perps pooled)",
        FIGURES_DIR / "funding_rate_dist.png",
    )
    plot_run_durations(
        all_pos_runs,
        all_neg_runs,
        "All perps pooled",
        FIGURES_DIR / "funding_run_durations.png",
    )
    print_stats(all_pos_runs, all_neg_runs, label="ALL")

    # --- per-perp subset ---
    available = set(df["perp"].unique())
    print()
    for perp in PERP_SUBSET:
        if perp not in available:
            print(f"[WARN] perp '{perp}' not found in data — skipping")
            continue

        perp_dir = FIGURES_DIR / perp
        perp_dir.mkdir(parents=True, exist_ok=True)

        perp_rates = df.loc[df["perp"] == perp, "fundingRate"].reset_index(drop=True)
        pos_runs, neg_runs = compute_runs(perp_rates)

        plot_funding_distribution(
            perp_rates,
            f"Distribution of hourly funding rate — {perp}",
            perp_dir / "funding_rate_dist.png",
        )
        plot_run_durations(
            pos_runs,
            neg_runs,
            perp,
            perp_dir / "funding_run_durations.png",
        )
        print_stats(pos_runs, neg_runs, label=perp)


if __name__ == "__main__":
    main()
