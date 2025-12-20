import pandas as pd
from datetime import datetime, timezone, timedelta
from crowdcent_challenge import ChallengeClient


# =====================
# LAZY CLIENT (Lambda-safe)
# =====================
_client = None


def get_client():
    global _client
    if _client is None:
        _client = ChallengeClient(challenge_slug="hyperliquid-ranking")
    return _client


# =====================
# PUBLIC API
# =====================
def get_long_short_ids(n: int = 3):
    client = get_client()

    # Lambda allows only /tmp for writes
    parquet_path = "/tmp/meta_model.parquet"

    client.download_meta_model(dest_path=parquet_path)

    df = pd.read_parquet(parquet_path)
    df["release_date"] = pd.to_datetime(df["release_date"], utc=True)

    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)

    # Try today first, fallback to yesterday
    df_today = df[df["release_date"].dt.date == today]

    if df_today.empty:
        print("[INFO] No predictions for today, falling back to yesterday")
        df_today = df[df["release_date"].dt.date == yesterday]

    if df_today.empty:
        raise RuntimeError("No predictions available for today or yesterday")

    longs = (
        df_today.sort_values("pred_30d", ascending=False)
        .head(n)["id"]
        .tolist()
    )

    shorts = (
        df_today.sort_values("pred_30d", ascending=True)
        .head(n)["id"]
        .tolist()
    )

    return longs, shorts
