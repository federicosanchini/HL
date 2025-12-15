import pandas as pd
from datetime import datetime, timezone, timedelta
from crowdcent_challenge import ChallengeClient

from datetime import timedelta, timezone, datetime

def get_long_short_ids(n=3):
    client = ChallengeClient(challenge_slug="hyperliquid-ranking")
    try:
        client.download_meta_model(dest_path="/tmp/meta_model.parquet")

        df = pd.read_parquet("/tmp/meta_model.parquet")
        df["release_date"] = pd.to_datetime(df["release_date"])

        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        df_today = df[df["release_date"].dt.date == yesterday]

        if df_today.empty:
            raise RuntimeError("No predictions available for today")

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

    finally:
        try:
            client.close()
        except Exception:
            pass

