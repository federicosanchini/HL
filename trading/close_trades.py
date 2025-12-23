# close_window_net.py
import os
import math
from datetime import datetime, timedelta, timezone
from typing import Literal, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

# =====================
# CONFIG
# =====================
VAULT = "0x9e72aabff75fe7c02cb12112c8ea8eb80b0b51b6"
DEFAULT_TZ = timezone.utc

# =====================
# INIT CLIENTS (MAINNET)
# =====================
load_dotenv()
pk = (os.getenv("HYPERLIQUID_PRIVATE_KEY") or "").strip()
if not pk:
    raise RuntimeError("Missing HYPERLIQUID_PRIVATE_KEY")

account = Account.from_key(pk)

info = Info(constants.MAINNET_API_URL)
exchange = Exchange(wallet=account, base_url=constants.MAINNET_API_URL, vault_address=VAULT)

# =====================
# HELPERS
# =====================
#ritorna il timestamp corrente in UTC
def _now_utc() -> datetime:
    return datetime.now(tz=DEFAULT_TZ)

#converte un datetime in millisecondi
def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

# legge il numero di decimali per la size di un asset
def get_size_decimals(symbol: str) -> int:
    universe = info.meta()["universe"]
    asset = next(a for a in universe if a["name"] == symbol)
    return int(asset["szDecimals"])

# arrotonda verso il basso la size in base ai decimali supportati, serve per evitare di superare la size disponibile
def quantize_down(sz: float, decimals: int) -> float:
    """Round down so we don't overshoot and accidentally flip position."""
    factor = 10**decimals
    return math.floor(sz * factor) / factor

# ottiene il prezzo medio (mid) di un asset
def _get_mid(symbol: str) -> float:
    mids = info.all_mids()
    if symbol not in mids:
        raise RuntimeError(f"No mid price for symbol {symbol}")
    return float(mids[symbol])

# Estrae il simbolo da un fill
def _extract_symbol(fill: dict) -> str:
    return fill.get("coin") or fill.get("name") or fill.get("symbol") or ""

# estrae il timestamp in ms da un fill
def _extract_ts_ms(fill: dict) -> int:
    t = fill.get("time")
    if t is None:
        t = fill.get("timestamp")
    if t is None:
        raise RuntimeError(f"Fill missing timestamp fields: keys={list(fill.keys())}")
    return int(t)

# estrae la size da un fill
def _extract_sz(fill: dict) -> float:
    v = fill.get("sz")
    if v is None:
        v = fill.get("size")
    if v is None:
        v = fill.get("qty")
    if v is None:
        raise RuntimeError(f"Fill missing size fields: keys={list(fill.keys())}")
    return float(v)

# determina se un fill è un acquisto o una vendita
def _extract_side_is_buy(fill: dict) -> bool:
    """
    Returns True if buy, False if sell.
    Supports multiple HL schemas:
      - isBuy: bool
      - side: "B"/"S" or "BUY"/"SELL"
      - side: "A"/"B" (Ask/Bid)
      - dir/direction: "Open Long", "Close Long", "Open Short", "Close Short"
    """
    if "isBuy" in fill:
        return bool(fill["isBuy"])

    side = fill.get("side")
    if isinstance(side, str):
        s = side.strip().upper()
        if s in ("B", "BUY"):
            return True
        if s in ("S", "SELL"):
            return False
        if s in ("A", "ASK"):
            return False
        if s in ("BID",):
            return True

    d = fill.get("dir") or fill.get("direction")
    if isinstance(d, str):
        ds = d.strip().lower()
        if "open long" in ds:
            return True
        if "close long" in ds:
            return False
        if "open short" in ds:
            return False
        if "close short" in ds:
            return True
        if ds == "long":
            return True
        if ds == "short":
            return False

    raise RuntimeError(f"Cannot infer buy/sell from fill fields: {fill}")

# Ottiene i fill di un utente in un intervallo di tempo
def _get_fills(address: str, start_ms: int, end_ms: int) -> List[dict]:
    if hasattr(info, "user_fills_by_time"):
        return info.user_fills_by_time(address, start_ms, end_ms)

    if hasattr(info, "user_fills"):
        try:
            return info.user_fills(address, start_ms, end_ms)
        except TypeError:
            fills = info.user_fills(address)
            out = []
            for f in fills:
                t = _extract_ts_ms(f)
                if start_ms <= t <= end_ms:
                    out.append(f)
            return out

    raise RuntimeError("Info SDK has no user_fills/user_fills_by_time method.")

# Serve a tradurre un input semantico (10 minuti, 1 ora, 10 giorni) in un oggetto timedelta che Python può usare per fare aritmetica temporale.
def _age_to_timedelta(x: float, unit: Literal["minutes", "hours", "days"]) -> timedelta:
    if x < 0:
        raise ValueError("age must be >= 0")
    if unit == "minutes":
        return timedelta(minutes=x)
    if unit == "hours":
        return timedelta(hours=x)
    if unit == "days":
        return timedelta(days=x)
    raise ValueError("unit must be one of: minutes, hours, days")

# Ottiene la posizione corrente in unità base (positive long, negative short, zero flat)
def _get_current_position_base(symbol: str) -> float:
    """
    Returns current position size in base units:
      > 0 long, < 0 short, 0 flat
    Robust to a few HL schemas.
    """
    st = info.user_state(VAULT)
    aps = st.get("assetPositions") or st.get("positions") or []
    sym = symbol.upper()

    for ap in aps:
        # common schema: {"position": {...}}
        pos = ap.get("position") if isinstance(ap, dict) else None
        if pos is None and isinstance(ap, dict):
            pos = ap  # sometimes already the position dict

        if not isinstance(pos, dict):
            continue

        coin = (pos.get("coin") or pos.get("symbol") or pos.get("name") or "").upper()
        if coin != sym:
            continue

        # common field: szi (string) is signed size
        for k in ("szi", "sz", "size", "positionSize"):
            if k in pos and pos[k] is not None:
                try:
                    return float(pos[k])
                except Exception:
                    pass

        # sometimes nested as {"szi": "..."} etc; if missing, treat as flat
        return 0.0

    return 0.0

# =====================
# CORE
# =====================
def close_net_in_time_window(
    min_age: float,
    max_age: float,
    unit: Literal["minutes", "hours", "days"] = "minutes",
    clamp_to_position: bool = True,
) -> Dict[str, dict]:
    """
    Window defined as:
      now - max_age  <= fill_time <=  now - min_age

    1) Finds ALL coins traded in that window (vault fills)
    2) Computes net_base per coin in that window:
         buys add, sells subtract
    3) Closes that net_base NOW:
         net_base > 0  -> SELL net_base
         net_base < 0  -> BUY  abs(net_base)

    If clamp_to_position=True, it will NOT exceed current position size per coin
    (prevents flipping if your current position is smaller than what the window implies).
    """
    if max_age <= min_age:
        raise ValueError("max_age must be > min_age (e.g., 10.1 and 9.9).")

    now = _now_utc()
    window_start = now - _age_to_timedelta(max_age, unit)
    window_end = now - _age_to_timedelta(min_age, unit)

    start_ms = _to_ms(window_start)
    end_ms = _to_ms(window_end)

    fills = _get_fills(VAULT, start_ms, end_ms)
    if not fills:
        print(f"[OK] No fills in window {window_start.isoformat()} -> {window_end.isoformat()}.")
        return {}

    # group net_base per symbol
    net_by_sym: Dict[str, float] = {}
    count_by_sym: Dict[str, int] = {}

    for f in fills:
        sym = _extract_symbol(f).upper()
        if not sym:
            continue
        sz = _extract_sz(f)
        is_buy = _extract_side_is_buy(f)
        net_by_sym[sym] = net_by_sym.get(sym, 0.0) + (sz if is_buy else -sz)
        count_by_sym[sym] = count_by_sym.get(sym, 0) + 1

    # execute closes
    results: Dict[str, dict] = {}
    print(f"\n--- WINDOW {window_start.isoformat()} -> {window_end.isoformat()} | fills={len(fills)} ---")

    for sym, net_base in sorted(net_by_sym.items()):
        if abs(net_base) < 1e-12:
            continue

        try:
            sz_dec = get_size_decimals(sym)
        except StopIteration:
            print(f"[SKIP] {sym}: not found in universe.")
            continue
        except Exception as e:
            print(f"[SKIP] {sym}: cannot read szDecimals ({e}).")
            continue

        desired_close = abs(net_base)

        if clamp_to_position:
            pos = _get_current_position_base(sym)
            if abs(pos) < 1e-12:
                print(f"[SKIP] {sym}: current position is flat; nothing to close safely.")
                continue

            # We close in the direction that reduces current exposure.
            # If current position direction disagrees with net_base direction, closing "net_base window" might increase risk.
            # Keep it conservative: only close up to current position in the direction implied by net_base.
            # Example: net_base>0 implies SELL. If you're currently short (pos<0), selling increases short -> skip.
            if net_base > 0 and pos <= 0:
                print(f"[SKIP] {sym}: window implies SELL, but current pos is not long (pos={pos}).")
                continue
            if net_base < 0 and pos >= 0:
                print(f"[SKIP] {sym}: window implies BUY, but current pos is not short (pos={pos}).")
                continue

            desired_close = min(desired_close, abs(pos))

        close_sz = quantize_down(desired_close, sz_dec)
        if close_sz <= 0:
            print(f"[SKIP] {sym}: quantized close size is 0 (szDecimals={sz_dec}).")
            continue

        try:
            mid = _get_mid(sym)
        except Exception as e:
            print(f"[SKIP] {sym}: no mid ({e}).")
            continue

        is_buy_to_close = net_base < 0  # net short in window => buy to cover; net long => sell
        side_txt = "BUY (cover)" if is_buy_to_close else "SELL (close)"

        print(
            f"[ACTION] {sym}: fills_count={count_by_sym.get(sym, 0)} "
            f"net_base_window={net_base:.8f} close_sz={close_sz} mid={mid:.8f} "
            f"est_notional=${close_sz * mid:.2f} -> {side_txt}"
        )

        try:
            res = exchange.market_open(name=sym, is_buy=is_buy_to_close, sz=close_sz)
            print("[RESULT]", res)
            results[sym] = res
        except Exception as e:
            print(f"[ERROR] {sym}: order failed ({e})")
            results[sym] = {"error": str(e)}

    return results

# =====================
# EXAMPLE CLI
# =====================
if __name__ == "__main__":
    # Example: trades between 10.1 and 9.9 minutes ago
    close_net_in_time_window(min_age=1.1, max_age=10.1, unit="minutes", clamp_to_position=True)

