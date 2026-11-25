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
MAX_RULES = 2                 # fewer conditions = less room to memorise history
# Minimum trades per coin per year (fewer is penalised hard), by candle size in hours.
# 75 a year ≈ one trade every 5 days per coin -> across 5 coins ≈ one signal a day.
MIN_TRADES_PER_YEAR = {1: 75, 4: 75}
# Maximum time a trade may stay open: 5 days.
MAX_HOLD_HOURS = 120

# Anti-luck scoring
DROP_TOP_WINNERS = 0.05       # ignore the best 5% of winning trades when scoring
WORST_YEARS = 2               # a coin is judged on the average of its 2 worst years
COMPLEXITY_PENALTY = 0.01     # per condition
CHAMPION_MARGIN = 0.01        # a new champion must beat the old one by this much

# Independent searches per strategy. If they agree, the pattern is probably real.
SEEDS = 3
# A search stops after this many runs (every 6h) without a better champion: 8 = 2 days
STOP_AFTER_RUNS = 8
