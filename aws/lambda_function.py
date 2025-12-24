# lambda_function.py
import json
import logging

from aws_signals import get_long_short_ids
from aws_trades import execute_trades, shutdown
from aws_close_trades import close_net_in_time_window

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# =====================
# DEFAULT CONFIG (as in main.py)
# =====================
DEFAULT_N = 3

DEFAULT_CLOSE_MIN_AGE = 29.9
DEFAULT_CLOSE_MAX_AGE = 30.1
DEFAULT_CLOSE_UNIT = "days"
DEFAULT_CLOSE_CLAMP = True


def lambda_handler(event, context):
    """
    AWS Lambda entry point (e.g., triggered by EventBridge cron).

    Optional event overrides:
    {
      "n": 3,
      "close_min_age": 29.9,
      "close_max_age": 30.1,
      "close_unit": "days",
      "close_clamp": true
    }
    """
    event = event or {}

    # ---- CONFIG (defaults match your main.py) ----
    n = int(event.get("n", DEFAULT_N))

    close_min_age = float(event.get("close_min_age", DEFAULT_CLOSE_MIN_AGE))
    close_max_age = float(event.get("close_max_age", DEFAULT_CLOSE_MAX_AGE))
    close_unit = event.get("close_unit", DEFAULT_CLOSE_UNIT)
    close_clamp = bool(event.get("close_clamp", DEFAULT_CLOSE_CLAMP))

    logger.info(
        "START lambda | n=%s | close=[%s,%s] %s | clamp=%s",
        n, close_min_age, close_max_age, close_unit, close_clamp
    )

    long_ids = short_ids = None
    close_result = None

    try:
        # 1) Open/manage trades
        long_ids, short_ids = get_long_short_ids(n=n)
        logger.info("SIGNALS | long=%s short=%s", long_ids, short_ids)

        execute_trades(long_ids, short_ids)

        # 2) Close sizes in the configured time window
        close_result = close_net_in_time_window(
            min_age=close_min_age,
            max_age=close_max_age,
            unit=close_unit,
            clamp_to_position=close_clamp,
        )

        logger.info("CLOSE completed | orders=%s", (close_result or {}).get("orders"))

        # Return (API Gateway proxy-compatible)
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(
                {
                    "ok": True,
                    "long_ids": long_ids,
                    "short_ids": short_ids,
                    "close_window": {
                        "min_age": close_min_age,
                        "max_age": close_max_age,
                        "unit": close_unit,
                        "clamp": close_clamp,
                    },
                    "close_result": close_result,
                },
                default=str,
            ),
        }

    except Exception as e:
        logger.exception("Lambda failed")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(
                {"ok": False, "error": str(e), "long_ids": long_ids, "short_ids": short_ids},
                default=str,
            ),
        }

    finally:
        # 3) Always shutdown (matches your main.py semantics)
        try:
            shutdown()
        except Exception:
            logger.exception("shutdown() failed")
