import os
import math
from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta



BASE_URL = "https://api.hyperliquid.xyz"
TARGET_USD = 10.0
MIN_NOTIONAL = 10.0
DRY_RUN = True


# --- KEY ---

load_dotenv()
pk = os.getenv("HYPERLIQUID_PRIVATE_KEY")

if pk is None:
    raise RuntimeError("Missing HYPERLIQUID_PRIVATE_KEY")

account = Account.from_key(pk)

# --- CLIENTS ---
info = Info(BASE_URL)
exchange = Exchange(account, BASE_URL)

def get_open_positions():
    state = info.user_state(account.address)
    return state.get("assetPositions", [])

def close_position(symbol: str, size: float):
    is_long = size > 0
    side = "CLOSE LONG" if is_long else "CLOSE SHORT"

    if DRY_RUN:
        print(f"[DRY-RUN] {side} {symbol} | size={abs(size)}")
        return

    exchange.order(
        name=symbol,
        is_buy=not is_long,   # opposite side
        sz=abs(size),
        px=None,
        order_type={"market": {}},
        reduce_only=True
    )

    print(f"{side} {symbol} | size={abs(size)}")

def close_positions_older_than(days: int = 30):
    cutoff_ms = int(
        (datetime.utcnow() - timedelta(days=days)).timestamp() * 1000
    )

    positions = get_open_positions()

    print(f"\n--- CHECKING POSITIONS OLDER THAN {days} DAYS ---")

    for p in positions:
        pos = p["position"]

        size = float(pos["szi"])
        if size == 0:
            continue

        entry_time = pos.get("entryTime")
        symbol = pos["coin"]

        if entry_time is None:
            continue

        if entry_time < cutoff_ms:
            age_days = (time.time() * 1000 - entry_time) / (1000 * 86400)
            print(
                f"Found old position: {symbol} | "
                f"size={size} | age={age_days:.1f} days"
            )
            close_position(symbol, size)


def get_size_decimals(symbol: str) -> int:
    universe = info.meta()["universe"]
    asset = next(a for a in universe if a["name"] == symbol)
    return asset["szDecimals"]


def ceil_size(raw_size: float, decimals: int) -> float:
    factor = 10 ** decimals
    return math.ceil(raw_size * factor) / factor


def open_position(symbol: str, usd: float, is_long: bool):
    # --- Price ---
    price = float(info.all_mids()[symbol])

    # --- Size precision ---
    sz_decimals = get_size_decimals(symbol)

    # --- Raw size with buffer ---
    raw_size = (usd) / price

    # --- Ceiling to allowed precision ---
    size = ceil_size(raw_size, sz_decimals)

    # --- Estimated notional ---
    est_notional = size * price

    side = "LONG" if is_long else "SHORT"

    if est_notional < MIN_NOTIONAL:
        print(
            f"[DRY-RUN] SKIP {symbol}: "
            f"size={size} | est ${est_notional:.2f} < ${MIN_NOTIONAL}"
        )
        return

    # --- DRY RUN ---
    if DRY_RUN:
        print(
            f"[DRY-RUN] {side} {symbol} | "
            f"price={price:.6f} | "
            f"szDecimals={sz_decimals} | "
            f"size={size} | "
            f"est_notional=${est_notional:.2f}"
        )
        return

    # --- REAL ORDER ---
    exchange.order(
        name=symbol,
        is_buy=is_long,
        sz=size,
        px=None,
        order_type={"market": {}},
        reduce_only=False
    )

    print(
        f"{side} {symbol} | size={size} | est ${est_notional:.2f}"
    )


long_ids = ["PAXG", "BCH", "BNB"]
short_ids = ["ZK", "WCT", "XPL"]


def main():
    close_positions_older_than(30)

    print("\n--- DRY-RUN LONG POSITIONS ---")
    for symbol in long_ids:
        open_position(symbol, TARGET_USD, is_long=True)

    print("\n--- DRY-RUN SHORT POSITIONS ---")
    for symbol in short_ids:
        open_position(symbol, TARGET_USD, is_long=False)


def shutdown():
    print("\nShutting down...")

    try:
        exchange.close()
    except Exception:
        pass

    try:
        info.close()
    except Exception:
        pass

    print("Shutdown complete.")
    os._exit(0)


if __name__ == "__main__":
    main()
    shutdown()
