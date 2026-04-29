"""
data_collection.py — Collect tweets (from CSV, simulated, or live) and stock data.
"""

import os
import random
import datetime
import pymongo
import yfinance as yf
import pandas as pd
from tqdm import tqdm

import config


# ── MongoDB helper ───────────────────────────────────────────────────────────
def get_db():
    """Return a pymongo Database handle."""
    client = pymongo.MongoClient(config.MONGO_URI)
    return client[config.MONGO_DB_NAME]


# ── CSV dataset helpers ──────────────────────────────────────────────────────

def get_available_stocks():
    """
    Read the CSV file and return a list of dicts with stock info.
    Each dict: {'ticker': ..., 'company_name': ..., 'tweet_count': ...}
    """
    csv_path = config.CSV_FILE_PATH
    if not os.path.exists(csv_path):
        print(f"[!] CSV file not found: {csv_path}")
        return []

    df = pd.read_csv(csv_path, usecols=["Stock Name", "Company Name"])
    # Build summary per ticker
    grouped = df.groupby("Stock Name").agg(
        company_name=("Company Name", "first"),
        tweet_count=("Stock Name", "count"),
    ).reset_index()

    stocks = []
    for _, row in grouped.sort_values("tweet_count", ascending=False).iterrows():
        stocks.append({
            "ticker": row["Stock Name"],
            "company_name": row["company_name"],
            "tweet_count": int(row["tweet_count"]),
        })
    return stocks


def _collect_csv_tweets(ticker: str):
    """Load tweets for a ticker from stock_tweets.csv."""
    csv_path = config.CSV_FILE_PATH
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"[*] Loading tweets for {ticker} from CSV ...")
    df = pd.read_csv(csv_path)
    df = df[df["Stock Name"] == ticker].copy()

    if df.empty:
        print(f"[!] No tweets found for {ticker} in CSV.")
        return []

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], utc=True)

    tweets = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="   Loading CSV"):
        tweets.append({
            "text": str(row["Tweet"]),
            "created_at": row["Date"].to_pydatetime(),
            "ticker": ticker,
            "company_name": str(row["Company Name"]),
            "source": "csv",
        })

    print(f"    [OK] {len(tweets)} tweets loaded from CSV.")
    return tweets


# ── Synthetic tweet generator ────────────────────────────────────────────────

BULLISH_TEMPLATES = [
    "{ticker} is looking strong today! Time to buy",
    "Just bought more {ticker}. Earnings looking great!",
    "{ticker} breaking out! Next stop: all-time highs",
    "Very bullish on {ticker} this quarter. Revenue growth is impressive.",
    "${{ticker}} calls are printing! Love this stock",
    "Analysts upgrading {ticker} - price target raised significantly",
    "{ticker} fundamentals are solid. Long-term hold for me.",
    "Big institutional buying in {ticker} today. Smart money moving in.",
    "{ticker} beat expectations again. Incredible management team.",
    "Loading up on {ticker} at these levels. Undervalued IMO.",
]

BEARISH_TEMPLATES = [
    "{ticker} is tanking. Glad I sold early",
    "Avoid {ticker} right now. Earnings miss incoming.",
    "{ticker} looking weak. Support levels breaking down.",
    "Short {ticker}. Overvalued at these prices.",
    "${{ticker}} puts are the play. This stock is going lower.",
    "{ticker} facing serious headwinds. Supply chain issues mounting.",
    "Insider selling at {ticker} - not a good sign.",
    "{ticker} losing market share fast. Competitors eating their lunch.",
    "Downgrading {ticker}. Growth story is over.",
    "{ticker} guidance was terrible. Expecting further decline.",
]

NEUTRAL_TEMPLATES = [
    "{ticker} trading sideways today. Waiting for a catalyst.",
    "Not sure about {ticker}. Mixed signals from the market.",
    "Holding {ticker} for now. Need to see next earnings.",
    "{ticker} volume is low today. No clear direction.",
    "Watching {ticker} closely. Could go either way from here.",
    "{ticker} consolidating. Breakout or breakdown soon.",
    "Interesting price action on {ticker} but I'm staying neutral.",
    "{ticker} at fair value according to my analysis.",
    "Market is uncertain about {ticker}. Waiting for more data.",
    "No strong opinion on {ticker} today. Sitting this one out.",
]


def _generate_synthetic_tweets(ticker: str, start_date: datetime.date,
                               end_date: datetime.date,
                               tweets_per_day: tuple = (30, 80)):
    """Generate realistic synthetic tweets for a date range."""
    tweets = []
    current = start_date
    delta = datetime.timedelta(days=1)

    while current <= end_date:
        # Skip weekends (markets closed — fewer tweets)
        if current.weekday() >= 5:
            n_tweets = random.randint(5, 20)
        else:
            n_tweets = random.randint(*tweets_per_day)

        for _ in range(n_tweets):
            sentiment_roll = random.random()
            if sentiment_roll < 0.35:
                template = random.choice(BULLISH_TEMPLATES)
            elif sentiment_roll < 0.70:
                template = random.choice(BEARISH_TEMPLATES)
            else:
                template = random.choice(NEUTRAL_TEMPLATES)

            text = template.format(ticker=ticker)
            # Random time during market hours (extended)
            hour = random.randint(6, 22)
            minute = random.randint(0, 59)
            created_at = datetime.datetime.combine(
                current,
                datetime.time(hour, minute),
            )

            tweets.append({
                "text": text,
                "created_at": created_at,
                "ticker": ticker,
                "source": "simulated",
            })

        current += delta

    return tweets


# ── Real tweet collection (Tweepy) ──────────────────────────────────────────

def _collect_real_tweets(ticker: str, max_results: int = 100):
    """Collect tweets via Twitter API v2 (requires valid credentials)."""
    import tweepy

    client = tweepy.Client(
        bearer_token=config.TWITTER_BEARER_TOKEN,
        consumer_key=config.TWITTER_API_KEY,
        consumer_secret=config.TWITTER_API_SECRET,
        access_token=config.TWITTER_ACCESS_TOKEN,
        access_token_secret=config.TWITTER_ACCESS_SECRET,
    )

    query = f"{ticker} OR ${ticker} lang:en -is:retweet"
    tweets_data = []

    try:
        response = client.search_recent_tweets(
            query=query,
            max_results=min(max_results, 100),
            tweet_fields=["created_at", "text"],
        )
        if response.data:
            for tweet in response.data:
                tweets_data.append({
                    "text": tweet.text,
                    "created_at": tweet.created_at,
                    "ticker": ticker,
                    "source": "twitter_api",
                })
    except Exception as e:
        print(f"[!] Twitter API error: {e}")
        print("    Falling back to simulated data.")
        return None

    return tweets_data


# ── Stock data ───────────────────────────────────────────────────────────────

def collect_stock_data(ticker: str, days: int = 365,
                       start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Download historical stock data via yfinance.
    If start_date/end_date are given, use those; otherwise compute from 'days'.
    """
    if start_date and end_date:
        print(f"[*] Downloading stock data for {ticker} ({start_date} -> {end_date}) ...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    else:
        end = datetime.date.today()
        start = end - datetime.timedelta(days=days)
        print(f"[*] Downloading stock data for {ticker} ({start} -> {end}) ...")
        df = yf.download(ticker, start=str(start), end=str(end), progress=False)

    if df.empty:
        raise ValueError(f"No stock data returned for {ticker}")

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.rename(columns={"Date": "date"}, inplace=True)
    print(f"    [OK] {len(df)} trading days downloaded.")
    return df


# -- Main collection entry point ----------------------------------------------

def collect_tweets(ticker: str, days: int = 365, mode: str = None):
    """Collect tweets and store them in MongoDB."""
    mode = mode or config.DATA_MODE
    db = get_db()
    col = db[config.COL_TWEETS_RAW]

    # Clear previous data for this ticker
    deleted = col.delete_many({"ticker": ticker})
    if deleted.deleted_count:
        print(f"[*] Cleared {deleted.deleted_count} old tweets for {ticker}.")

    tweets = []

    if mode == "csv":
        tweets = _collect_csv_tweets(ticker)

    elif mode == "live":
        tweets = _collect_real_tweets(ticker)
        if tweets is None:
            # Fallback to simulated
            mode = "simulated"

    if mode == "simulated":
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=days)
        print(f"[*] Generating simulated tweets for {ticker} ({start_date} -> {end_date}) ...")
        tweets = _generate_synthetic_tweets(ticker, start_date, end_date)

    if tweets:
        col.insert_many(tweets)
        print(f"    [OK] {len(tweets)} tweets stored in '{config.COL_TWEETS_RAW}'.")
    else:
        print("[!] No tweets collected.")

    return len(tweets) if tweets else 0


if __name__ == "__main__":
    # Show available stocks
    stocks = get_available_stocks()
    print("Available stocks in CSV:")
    for s in stocks:
        print(f"  {s['ticker']:6s}  {s['company_name']:30s}  ({s['tweet_count']} tweets)")

    collect_tweets(config.DEFAULT_TICKER, config.DEFAULT_DAYS)
    df = collect_stock_data(config.DEFAULT_TICKER, config.DEFAULT_DAYS)
    print(df.head())
