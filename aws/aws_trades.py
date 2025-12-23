import os
import math
import time
from datetime import datetime, timedelta

from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange


# =====================
# CONFIG
# =====================
BASE_URL = "https://api.hyperliquid.xyz"
TARGET_USD = 10.0
MIN_NOTIONAL = 10.0
VAULT_ADDRESS = "0x9e72aabff75fe7c02cb12112c8ea8eb80b0b51b6"
DRY_RUN = True  # make env-configurable later if you want


# =====================
# LAZY CLIENTS (Lambda-safe)
# =====================
_account = None
_info = None
_exchange = None


def get_clients():
    global _account, _info, _exchange

    if _account is None:
        pk = os.environ["HYPERLIQUID_PRIVATE_KEY"]
        _account = Account.from_key(pk)

    if _info is None:
        _info = Info(BASE_URL)

    if _exchange is None:
        _exchange = Exchange(_account, BASE_URL)

    return _info, _exchange


# =====================
# HELPERS
# =====================
def get_open_positions():
    info, _ = get_clients()
    state = info.user_state(VAULT_ADDRESS)
    return state.get("assetPositions", [])


def close_position(symbol: str, size: float):
    info, exchange = get_clients()

    is_long = size > 0
    side = "CLOSE LONG" if is_long else "CLOSE SHORT"

    if DRY_RUN:
        print(f"[DRY-RUN] {side} {symbol} | size={abs(size)}")
        return

    exchange.order(
        name=symbol,
        is_buy=not is_long,
        sz=abs(size),
        px=None,
        order_type={"market": {}},
        reduce_only=True,
        vaultAddress=VAULT_ADDRESS,
    )

    print(f"{side} {symbol} | size={abs(size)}")


def close_positions_older_than(days: float):
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
    info, _ = get_clients()
    universe = info.meta()["universe"]
    asset = next(a for a in universe if a["name"] == symbol)
    return asset["szDecimals"]


def ceil_size(raw_size: float, decimals: int) -> float:
    factor = 10 ** decimals
    return math.ceil(raw_size * factor) / factor


def open_position(symbol: str, usd: float, is_long: bool):
    info, exchange = get_clients()

    price = float(info.all_mids()[symbol])
    sz_decimals = get_size_decimals(symbol)

    raw_size = usd / price
    size = ceil_size(raw_size, sz_decimals)
    est_notional = size * price

    side = "LONG" if is_long else "SHORT"

    if est_notional < MIN_NOTIONAL:
        print(
            f"[DRY-RUN] SKIP {symbol}: "
            f"est ${est_notional:.2f} < ${MIN_NOTIONAL}"
        )
        return

    if DRY_RUN:
        print(
            f"[DRY-RUN] {side} {symbol} | "
            f"price={price:.6f} | "
            f"size={size} | "
            f"est=${est_notional:.2f}"
        )
        return

    exchange.order(
        name=symbol,
        is_buy=is_long,
        sz=size,
        px=None,
        order_type={"market": {}},
        reduce_only=False,
        vaultAddress=VAULT_ADDRESS,
    )

    print(f"{side} {symbol} | size={size} | est ${est_notional:.2f}")


# =====================
# PUBLIC API
# =====================
def execute_trades(
    long_ids: list[str],
    short_ids: list[str],
    close_after_days: float = 29.9,
):
    close_positions_older_than(close_after_days)

    print("\n--- OPENING LONG POSITIONS ---")
    for symbol in long_ids:
        open_position(symbol, TARGET_USD, is_long=True)

    print("\n--- OPENING SHORT POSITIONS ---")
    for symbol in short_ids:
        open_position(symbol, TARGET_USD, is_long=False)
