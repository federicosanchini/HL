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

# =====================
# LOAD KEY
# =====================
load_dotenv()  # loads .env from current working directory (and parents)

pk = os.getenv("HYPERLIQUID_PRIVATE_KEY")
if pk is None or pk.strip() == "":
    raise RuntimeError("Missing HYPERLIQUID_PRIVATE_KEY (set it in .env or export it in your shell)")
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



print(get_size_decimals("BTC"))

# =====================
# CHECK VAULT BALANCE
# =====================
print("Signer (master) address:", account.address)
print("Vault withdrawable:", info.user_state(VAULT)["withdrawable"])

# =====================
# SIZE + MARKET ORDER
# =====================
mid = float(info.all_mids()["BTC"])
raw_sz = USD_NOTIONAL / mid

sz = ceil_size(raw_sz, get_size_decimals("BTC"))

order = exchange.market_open(
    name="BTC",
    is_buy=True,
    sz=sz
)

print(order)
