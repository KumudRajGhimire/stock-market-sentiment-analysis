"""
config.py — Central configuration for the BDA-AAT pipeline.
Loads settings from environment variables (via .env file) with sensible defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── MongoDB ──────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "stock_sentiment_db")

# Collection names
COL_TWEETS_RAW = "tweets_raw"
COL_TWEETS_PROCESSED = "tweets_processed"
COL_DAILY_FEATURES = "daily_features"
COL_MODEL_RESULTS = "model_results"

# ── Twitter / X API ──────────────────────────────────────────────────────────
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

# Data mode: "csv" (default), "simulated", or "live"
DATA_MODE = os.getenv("DATA_MODE", "csv")

# ── CSV Dataset ──────────────────────────────────────────────────────────────
CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), "stock_tweets.csv")

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_TICKER = "AAPL"
DEFAULT_DAYS = 365

# ── FinBERT ──────────────────────────────────────────────────────────────────
FINBERT_MODEL = "ProsusAI/finbert"
SENTIMENT_BATCH_SIZE = 32
