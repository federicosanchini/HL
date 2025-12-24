# lambda_function.py
import os
import math
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Literal, Dict, List, Optional, Tuple, Any

from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TZ = timezone.utc

# Defaults (override via env)
DEFAULT_VAULT = "0x9e72aabff75fe7c02cb12112c8ea8eb80b0b51b6"
DEFAULT_BASE_URL = constants.MAINNET_API_URL

# Cached clients (Lambda warm starts)
_INFO: Optional[Info] = None
_EXCHANGE: Optional[Exchange] = None
_VAULT: Optional[str] = None


def _now_utc() -> datetime:
    return datetime.now(tz=TZ)


def _dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def quantize_down(sz: float, decimals: int) -> float:
    factor = 10 ** decimals
    return math.floor(sz * factor) / factor


def _get_clients() -> Tuple[Info, Exchange, str]:
    global _INFO, _EXCHANGE, _VAULT

    if _INFO is not None and _EXCHANGE is not None and _VAULT is not None:
        return _INFO, _EXCHANGE, _VAULT

    pk = (os.getenv("HYPERLIQUID_PRIVATE_KEY") or "").strip()
    if not pk:
        raise RuntimeError("Missing env var HYPERLIQUID_PRIVATE_KEY")

    vault = (os.getenv("VAULT_ADDRESS") or DEFAULT_VAULT).strip()
    base_url = (os.getenv("HYPERLIQUID_BASE_URL") or DEFAULT_BASE_URL).strip()

    account = Account.from_key(pk)
    _INFO = Info(base_url)
    _EXCHANGE = Exchange(wallet=account, base_url=base_url, vault_address=vault)
    _VAULT = vault

    logger.info("Initialized clients base_url=%s vault=%s", base_url, vault)
    return _INFO, _EXCHANGE, _VAULT


def _extract_symbol(fill: dict) -> str:
    return fill.get("coin") or ""


def _extract_side(fill: dict) -> Literal["B", "S"]:
    side = fill.get("side")
    if side in ("B", "S"):
        return side
    raise RuntimeError(f"Cannot extract side from fill: {fill}")


def _extract_size(fill: dict) -> float:
    sz = fill.get("sz")
    if sz is None:
        raise RuntimeError(f"Cannot extract sz from fill: {fill}")
    return float(sz)


def _extract_time_ms(fill: dict) -> int:
    t = fill.get("time")
    if t is None:
        raise RuntimeError(f"Cannot extract time from fill: {fill}")
    return int(t)


def _positions_map(info: Info, address: str) -> Dict[str, dict]:
    st = info.user_state(address)
    ap = st.get("assetPositions") or []
    out: Dict[str, dict] = {}
    for item in ap:
        pos = item.get("position") if isinstance(item, dict) else None
        if not pos:
            continue
        coin = pos.get("coin")
        if coin:
            out[str(coin)] = pos
    return out


def _signed_pos_size(pos: dict) -> float:
    # In Hyperliquid user_state position dict usually has "szi" as signed size
    if "szi" in pos:
        return float(pos["szi"])
    raise RuntimeError(f"Cannot extract signed size (szi) from position: {pos}")


def close_net_in_time_window(
    *,
    min_age: float,
    max_age: float,
    unit: Literal["seconds", "minutes", "hours", "days"] = "minutes",
    clamp_to_position: bool = True,
    dry_run: bool = False,
    coins: Optional[List[str]] = None,
) -> Dict[str, Any]:
    info, exchange, vault_addr = _get_clients()

    mult = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}[unit]
    now = _now_utc()

    min_td = timedelta(seconds=min_age * mult)
    max_td = timedelta(seconds=max_age * mult)

    newer_than = now - min_td
    older_than = now - max_td
    if older_than > newer_than:
        older_than, newer_than = newer_than, older_than

    t_min = _dt_to_ms(older_than)
    t_max = _dt_to_ms(newer_than)

    fills = info.user_fills(vault_addr) or []
    window_fills = [f for f in fills if t_min <= _extract_time_ms(f) <= t_max]

    if coins:
        wanted = {c.upper() for c in coins}
        window_fills = [f for f in window_fills if _extract_symbol(f).upper() in wanted]

    net_by_coin: Dict[str, float] = {}
    for f in window_fills:
        coin = _extract_symbol(f)
        side = _extract_side(f)
        sz = _extract_size(f)
        net_by_coin.setdefault(coin, 0.0)
        net_by_coin[coin] += sz if side == "B" else -sz

    # decimals
    meta = info.meta()
    universe = meta.get("universe") or []
    decimals_map: Dict[str, int] = {}
    for u in universe:
        name = u.get("name")
        if name:
            decimals_map[str(name)] = int(u.get("szDecimals", 0))

    pos_map = _positions_map(info, vault_addr)

    results: Dict[str, Any] = {
        "window": {
            "older_than": older_than.isoformat(),
            "newer_than": newer_than.isoformat(),
            "t_min_ms": t_min,
            "t_max_ms": t_max,
            "unit": unit,
            "min_age": min_age,
            "max_age": max_age,
        },
        "fills_in_window": len(window_fills),
        "net_by_coin": net_by_coin,
        "orders": {},
        "dry_run": dry_run,
    }

    for sym, net in net_by_coin.items():
        if net == 0:
            continue

        # If net>0 you are net-long from fills => sell to close; if net<0 => buy to close
        is_buy = net < 0
        close_sz = abs(net)

        if clamp_to_position and sym in pos_map:
            try:
                signed_pos = _signed_pos_size(pos_map[sym])
                close_sz = min(close_sz, abs(signed_pos))
            except Exception as e:
                logger.warning("Clamp failed for %s: %s", sym, e)

        dec = decimals_map.get(sym, 0)
        close_sz_q = quantize_down(close_sz, dec)

        if close_sz_q <= 0:
            results["orders"][sym] = {"skipped": True, "reason": "size<=0 after clamp/quantize", "decimals": dec}
            continue

        order_info = {"coin": sym, "is_buy": is_buy, "size": close_sz_q, "decimals": dec}

        if dry_run:
            results["orders"][sym] = {"dry_run": True, "order": order_info}
            continue

        # THIS MATCHES YOUR CODE: exchange.market_open(sym, is_buy, sz)
        try:
            res = exchange.market_open(sym, is_buy, close_sz_q)
            results["orders"][sym] = {"ok": True, "order": order_info, "response": res}
        except Exception as e:
            logger.exception("Order failed for %s", sym)
            results["orders"][sym] = {"ok": False, "order": order_info, "error": str(e)}

    return results


def _parse_bool(v: Any, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "t", "yes", "y", "on")
    return default


def lambda_handler(event: dict, context) -> dict:
    """
    Handler: lambda_function.lambda_handler
    Example event:
    {
      "min_age": 20,
      "max_age": 60,
      "unit": "minutes",
      "clamp_to_position": true,
      "dry_run": false,
      "coins": ["BTC","ETH"]
    }
    """
    try:
        event = event or {}

        min_age = float(event.get("min_age", 20))
        max_age = float(event.get("max_age", 60))
        unit = event.get("unit", "minutes")
        if unit not in ("seconds", "minutes", "hours", "days"):
            raise ValueError("unit must be one of: seconds, minutes, hours, days")

        clamp = _parse_bool(event.get("clamp_to_position"), True)
        dry_run = _parse_bool(event.get("dry_run"), False)

        coins = event.get("coins")
        if coins is not None and not isinstance(coins, list):
            raise ValueError("coins must be a list like ['BTC','ETH']")

        result = close_net_in_time_window(
            min_age=min_age,
            max_age=max_age,
            unit=unit,
            clamp_to_position=clamp,
            dry_run=dry_run,
            coins=coins,
        )

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result, default=str),
        }

    except Exception as e:
        logger.exception("Lambda failed")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
