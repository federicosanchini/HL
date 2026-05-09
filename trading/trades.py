# trades.py
import os
import math
import time
from datetime import datetime, timedelta

from dotenv import load_dotenv
from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants


# =====================
# CONFIG
# =====================
VAULT_ADDRESS = "0x9e72aabff75fe7c02cb12112c8ea8eb80b0b51b6"

TARGET_USD = 11.0
MIN_NOTIONAL = 10.0

# Metti False per tradare davvero
DRY_RUN = False


# =====================
# SETUP (come il tuo script funzionante)
# =====================
load_dotenv()

pk = os.getenv("HYPERLIQUID_PRIVATE_KEY")
if pk is None or pk.strip() == "":
    raise RuntimeError("Missing HYPERLIQUID_PRIVATE_KEY (set it in .env or export it in your shell)")
pk = pk.strip()

account = Account.from_key(pk)

# usa la stessa base_url dello script funzionante
info = Info(constants.MAINNET_API_URL)

# IMPORTANTISSIMO: passa vault_address al costruttore (come nel tuo script)
exchange = Exchange(
    wallet=account,
    base_url=constants.MAINNET_API_URL,
    vault_address=VAULT_ADDRESS,
)


# =====================
# HELPERS
# =====================
def get_open_positions():
    state = info.user_state(VAULT_ADDRESS)
    return state.get("assetPositions", [])


def get_size_decimals(symbol: str) -> int:
    universe = info.meta()["universe"]
    asset = next(a for a in universe if a["name"] == symbol)
    return int(asset["szDecimals"])


def ceil_size(raw_size: float, decimals: int) -> float:
    factor = 10 ** decimals
    return math.ceil(raw_size * factor) / factor


def _get_mid(symbol: str) -> float | None:
    mids = info.all_mids()
    v = mids.get(symbol) if isinstance(mids, dict) else None
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# =====================
# TRADING
# =====================

'''
def close_position(symbol: str, size: float):
    """
    size signed:
      >0 => long aperto (per chiudere bisogna vendere)
      <0 => short aperto (per chiudere bisogna comprare)
    """
    is_long = size > 0
    side = "CLOSE LONG" if is_long else "CLOSE SHORT"
    qty = abs(float(size))

    if qty == 0:
        return

    if DRY_RUN:
        print(f"[DRY-RUN] {side} {symbol} | size={qty}")
        return

    # Usa le API "market_*" che nel tuo script funzionano
    try:
        if is_long:
            resp = exchange.market_close(name=symbol, sz=qty)  # chiude long
        else:
            resp = exchange.market_close(name=symbol, sz=qty)  # chiude short (SDK gestisce lato)
        print(f"{side} {symbol} | size={qty} | resp={resp}")
    except Exception as e:
        print(f"[ERROR] {side} {symbol} | size={qty} | {type(e).__name__}: {e}")


def close_positions_older_than(days: float):
    cutoff_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    positions = get_open_positions()

    print(f"\n--- CHECKING POSITIONS OLDER THAN {days} DAYS ---")

    for p in positions:
        pos = p.get("position", {})
        symbol = pos.get("coin")
        entry_time = pos.get("entryTime")

        if not symbol or entry_time is None:
            continue

        try:
            size = float(pos.get("szi", 0.0))
        except Exception:
            continue

        if size == 0:
            continue

        if entry_time < cutoff_ms:
            age_days = (time.time() * 1000 - entry_time) / (1000 * 86400)
            print(f"Found old position: {symbol} | size={size} | age={age_days:.1f} days")
            close_position(symbol, size)

'''


def open_position(symbol: str, usd: float, is_long: bool):
    """
    Apre una posizione a market usando market_open come nel tuo script.
    """
    mid = _get_mid(symbol)
    side = "LONG" if is_long else "SHORT"

    if mid is None:
        print(f"[SKIP] {side} {symbol}: no mid price available from info.all_mids()")
        return

    sz_decimals = get_size_decimals(symbol)

    raw_sz = usd / mid
    sz = ceil_size(raw_sz, sz_decimals)
    est_notional = sz * mid

    if est_notional < MIN_NOTIONAL:
        print(f"[SKIP] {side} {symbol}: est ${est_notional:.2f} < ${MIN_NOTIONAL:.2f}")
        return

    if DRY_RUN:
        print(f"[DRY-RUN] {side} {symbol} | price={mid:.6f} | size={sz} | est=${est_notional:.2f}")
        return

    try:
        resp = exchange.market_open(
            name=symbol,
            is_buy=is_long,  # True => long, False => short
            sz=sz,
        )
        print(f"{side} {symbol} | price={mid:.6f} | size={sz} | est=${est_notional:.2f} | resp={resp}")
    except Exception as e:
        print(f"[ERROR] {side} {symbol} | size={sz} | {type(e).__name__}: {e}")


# =====================
# PUBLIC API
# =====================
def execute_trades(
    long_ids: list[str],
    short_ids: list[str],
    # close_after_days: float = 29.9,
):
    # close_positions_older_than(close_after_days)

    print("\n--- OPENING LONG POSITIONS ---")
    for symbol in long_ids:
        exchange.update_leverage(1, symbol, is_cross=False)  
        open_position(symbol, TARGET_USD, is_long=True)

    print("\n--- OPENING SHORT POSITIONS ---")
    for symbol in short_ids:
        exchange.update_leverage(1, symbol, is_cross=False) 
        open_position(symbol, TARGET_USD, is_long=False)


def shutdown():
    try:
        exchange.close()
    except Exception:
        pass
    try:
        info.close()
    except Exception:
        pass
