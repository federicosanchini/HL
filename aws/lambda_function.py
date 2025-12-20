from signals import get_long_short_ids
from trades import execute_trades


def lambda_handler(event, context):
    """
    AWS Lambda entry point.
    Triggered by EventBridge (cron).
    """

    # ---- CONFIG FROM EVENT (optional) ----
    n = event.get("n", 3)
    close_after_days = event.get("close_after_days", 29.9)

    print(
        f"[START] Lambda execution | "
        f"n={n} | close_after_days={close_after_days}"
    )

    # ---- GET SIGNALS ----
    long_ids, short_ids = get_long_short_ids(n=n)

    print(f"[SIGNALS] LONG={long_ids} SHORT={short_ids}")

    # ---- EXECUTE TRADES ----
    execute_trades(
        long_ids=long_ids,
        short_ids=short_ids,
        close_after_days=close_after_days,
    )

    print("[END] Lambda execution completed successfully")

    return {
        "statusCode": 200,
        "body": {
            "long_ids": long_ids,
            "short_ids": short_ids,
        },
    }
