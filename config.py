"""System-wide constraints and watchlist configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).parent

MAX_RISK_PER_TRADE = 10.0
MAX_HOLDING_DAYS = 14
MIN_TRUST_SCORE_FOR_SCALE_UP = 60.0

# Trust-based risk tiers (max dollars at risk per trade)
RISK_TIERS = [
    {"min_trust": 0, "max_risk": 10.0, "label": "Sandbox"},
    {"min_trust": 30, "max_risk": 25.0, "label": "Learning"},
    {"min_trust": 60, "max_risk": 50.0, "label": "Trusted"},
    {"min_trust": 80, "max_risk": 100.0, "label": "Proven"},
]

WATCHLIST = {
    "Mega/Large Cap": ["AAPL", "NVDA", "AMD", "PLTR", "TSLA", "MSFT", "AMZN", "META", "GOOGL"],
    "Mid Cap": ["CELH", "ELF", "DUOL", "APP", "VKTX", "PATH", "CRWD", "NET"],
    "Small Cap": ["SOUN", "BBAI", "RIG", "HIMS", "IONQ", "RKLB", "SOFI"],
}

SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Consumer Cyclical": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
}

DATA_PERIOD = "60d"
DATA_INTERVAL = "1d"

SCALE_OUT_LEVELS = [
    {"r_multiple": 1.0, "pct_to_sell": 0.33, "label": "Target 1"},
    {"r_multiple": 2.0, "pct_to_sell": 0.33, "label": "Target 2"},
]
TRAILING_STOP_ATR_MULT = 2.0
TIME_STOP_DAYS = 10
MIN_PROFIT_BY_DAY = {7: 0.15, 10: 0.25}

MIN_CONTEXT_SCORE_FOR_TRADE = 50
MIN_REGIME_SCORE = 40
MIN_QA_SCORE = 60

STATE_FILE = str(BASE_DIR / "trade_state.json")
REPORTS_DIR = str(BASE_DIR / "reports")
UNIVERSE_FILE = str(BASE_DIR / "universe.txt")

# Training data (mass backtest / calibration)
TRAINING_YEARS = 10
TRAINING_USE_MAX_HISTORY = True
TRAINING_MIN_BARS = 252
TRAINING_BATCH_SIZE = 12
TRAINING_BATCH_DELAY_SEC = 2.0
TRAINING_CACHE_TTL_HOURS = 168
TRAINING_UNIVERSE_DIR = BASE_DIR / "training" / "universes"
SP500_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies"
    "/master/data/constituents.csv"
)

# Checkpoint vs final training profiles
CHECKPOINT_WALK_STEP = 3
CHECKPOINT_MAX_SETUPS_PER_TICKER = 300
CHECKPOINT_EARLY_STOP_SETUPS = 10_000
CHECKPOINT_CHUNK_SIZE = 24

FULL_WALK_STEP = 1
FULL_MAX_SETUPS_PER_TICKER = None
FULL_EARLY_STOP_SETUPS = None