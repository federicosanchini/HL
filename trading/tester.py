import os
import math
from dotenv import load_dotenv
from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

# =====================
# CONFIG
# =====================
VAULT = "0x9e72aabff75fe7c02cb12112c8ea8eb80b0b51b6"
USD_NOTIONAL = 10.0
SYMBOL = "ETH"  # Target asset
LEVERAGE = 1   # Scegli la leva desiderata

# =====================
# LOAD KEY
# =====================
load_dotenv()
pk = os.getenv("HYPERLIQUID_PRIVATE_KEY")
if pk is None or pk.strip() == "":
    raise RuntimeError("Missing HYPERLIQUID_PRIVATE_KEY")
pk = pk.strip()

account = Account.from_key(pk)

# =====================
# CLIENTS (MAINNET)
# =====================
info = Info(constants.MAINNET_API_URL)
exchange = Exchange(
    wallet=account,
    base_url=constants.MAINNET_API_URL,
    vault_address=VAULT
)

def get_size_decimals(symbol: str) -> int:
    universe = info.meta()["universe"]
    asset = next(a for a in universe if a["name"] == symbol)
    return asset["szDecimals"]

def ceil_size(raw_size: float, decimals: int) -> float:
    factor = 10 ** decimals
    return math.ceil(raw_size * factor) / factor

# =====================
# 1. SET ISOLATED MARGIN
# =====================
# Questo comando forza il margine a ISOLATED (is_cross=False)
print(f"Configuring {SYMBOL} to ISOLATED margin with {LEVERAGE}x leverage...")
exchange.update_leverage(LEVERAGE, SYMBOL, is_cross=False)

# =====================
# 2. CHECK VAULT BALANCE
# =====================
print("Signer (master) address:", account.address)
user_state = info.user_state(VAULT)
print("Vault withdrawable:", user_state["withdrawable"])

# =====================
# 3. SIZE + MARKET ORDER (ETH)
# =====================
# Prendiamo il prezzo medio di ETH
mid = float(info.all_mids()[SYMBOL])
raw_sz = USD_NOTIONAL / mid

# Calcoliamo la size corretta per ETH
sz = ceil_size(raw_sz, get_size_decimals(SYMBOL))

print(f"Opening Market Order on {SYMBOL} | Size: {sz} | Price: {mid}")

order = exchange.market_open(
    name=SYMBOL,
    is_buy=True,
    sz=sz
)

print("Order result:")
print(order)