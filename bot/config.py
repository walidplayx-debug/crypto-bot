"""Central settings for the bot. Change values here, not inside the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STATE_DIR = ROOT / "state"
DOCS_DIR = ROOT / "docs"

# Five coins of different kinds
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"]

# First month to download (coins listed later simply start later)
START_MONTH = "2017-08"

# Timeframe variants the bot tries
#   1h      -> 1-hour candles, 1-hour indicators
#   4h      -> 4-hour candles, 4-hour indicators
#   1h+4h   -> 1-hour candles, 1-hour indicators + 4-hour trend indicators
VARIANTS = ["1h", "4h", "1h+4h"]

# Data split. The bot learns on TRAIN, picks its champion on VALIDATION,
# and TEST is hidden: it is never used for any decision, only reported.
TRAIN_END = "2023-01-01"
VALID_END = "2024-07-01"

# Realistic trading costs (Binance futures taker + slippage), per side
FEE = 0.0005
SLIPPAGE = 0.0003
# Funding / holding cost per hour held (≈0.01% every 8h)
FUNDING_PER_HOUR = 0.0001 / 8

# Leverage suggestion: risk this fraction of the account per trade,
# so leverage = RISK_PER_TRADE / stop-loss distance, capped.
RISK_PER_TRADE = 0.02
MAX_LEVERAGE = 10

# Genetic algorithm
POPULATION = 120
ELITE = 12
MAX_RULES = 3
MIN_TRADES_PER_YEAR = 6       # fewer trades than this per coin is penalised
