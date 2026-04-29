"""
feature_engineering.py — Aggregate daily sentiment from MongoDB,
merge with stock data, and create prediction targets.
"""

import pymongo
import pandas as pd
import numpy as np
import datetime

import config
from data_collection import collect_stock_data


def get_db():
    """Return a pymongo Database handle."""
    client = pymongo.MongoClient(config.MONGO_URI)
    return client[config.MONGO_DB_NAME]


def aggregate_daily_sentiment(ticker: str) -> pd.DataFrame:
    """
    Use MongoDB aggregation pipeline to compute daily avg sentiment
    and tweet count for a ticker.
    """
    db = get_db()
    col = db[config.COL_TWEETS_PROCESSED]

    pipeline = [
        {"$match": {"ticker": ticker, "sentiment": {"$ne": None}}},
        {"$group": {
            "_id": {"date": "$date", "ticker": "$ticker"},
            "avg_sentiment": {"$avg": "$sentiment"},
            "tweet_count": {"$sum": 1},
            "positive_count": {
                "$sum": {"$cond": [{"$gt": ["$sentiment", 0.05]}, 1, 0]}
            },
            "negative_count": {
                "$sum": {"$cond": [{"$lt": ["$sentiment", -0.05]}, 1, 0]}
            },
        }},
        {"$project": {
            "_id": 0,
            "date": "$_id.date",
            "ticker": "$_id.ticker",
            "avg_sentiment": {"$round": ["$avg_sentiment", 4]},
            "tweet_count": 1,
            "positive_count": 1,
            "negative_count": 1,
        }},
        {"$sort": {"date": 1}},
    ]

    results = list(col.aggregate(pipeline))
    if not results:
        print(f"[!] No aggregated sentiment data for {ticker}.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["date"])
    print(f"[*] Aggregated sentiment for {len(df)} trading days.")
    return df


def _get_tweet_date_range(ticker: str) -> tuple:
    """
    Determine the date range of tweets for a ticker in MongoDB.
    Returns (min_date_str, max_date_str) or (None, None).
    """
    db = get_db()
    col = db[config.COL_TWEETS_PROCESSED]

    pipeline = [
        {"$match": {"ticker": ticker, "sentiment": {"$ne": None}}},
        {"$group": {
            "_id": None,
            "min_date": {"$min": "$date"},
            "max_date": {"$max": "$date"},
        }},
    ]
    results = list(col.aggregate(pipeline))
    if not results:
        return None, None

    return results[0]["min_date"], results[0]["max_date"]


def build_features(ticker: str, days: int = 365):
    """
    Build the daily features dataset:
      1. Aggregate tweet sentiment per day
      2. Fetch stock data (matching the tweet date range for CSV mode)
      3. Merge on date
      4. Create target variable (next-day price movement)
      5. Store in MongoDB
    """
    print(f"\n{'='*60}")
    print(f"  Feature Engineering - {ticker}")
    print(f"{'='*60}\n")

    # Step 1: Aggregate daily sentiment
    sentiment_df = aggregate_daily_sentiment(ticker)
    if sentiment_df.empty:
        raise ValueError("No sentiment data available. Run sentiment analysis first.")

    # Step 2: Fetch stock data
    # Determine date range from tweets for accurate stock data fetch
    min_date, max_date = _get_tweet_date_range(ticker)
    if min_date and max_date:
        # Add buffer days for stock data (weekends, holidays)
        start_dt = pd.to_datetime(min_date) - pd.Timedelta(days=7)
        end_dt = pd.to_datetime(max_date) + pd.Timedelta(days=7)
        stock_df = collect_stock_data(
            ticker, days,
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=end_dt.strftime("%Y-%m-%d"),
        )
    else:
        stock_df = collect_stock_data(ticker, days)

    stock_df["date"] = pd.to_datetime(stock_df["date"])

    # Step 3: Merge on date
    merged = pd.merge(
        stock_df[["date", "Close"]],
        sentiment_df,
        on="date",
        how="inner",
    )
    merged.rename(columns={"Close": "close_price"}, inplace=True)

    if merged.empty:
        raise ValueError("No overlapping dates between stock and sentiment data.")

    # Step 4: Create target — next-day movement
    merged["next_close"] = merged["close_price"].shift(-1)
    merged["target"] = (merged["next_close"] > merged["close_price"]).astype(int)
    # Drop last row (no next-day data)
    merged = merged.dropna(subset=["next_close"]).copy()
    merged.drop(columns=["next_close"], inplace=True)

    # Step 5: Compute sentiment ratio
    merged["sentiment_ratio"] = merged["positive_count"] / (
        merged["positive_count"] + merged["negative_count"] + 1e-6
    )

    print(f"    [OK] {len(merged)} feature rows created.")
    print(f"    Target distribution: UP={merged['target'].sum()}, "
          f"DOWN={len(merged) - merged['target'].sum()}")

    # Step 6: Store in MongoDB
    db = get_db()
    col = db[config.COL_DAILY_FEATURES]
    col.delete_many({"ticker": ticker})

    records = merged.to_dict("records")
    # Convert timestamps to strings for clean storage
    for r in records:
        r["date"] = r["date"].strftime("%Y-%m-%d")
    col.insert_many(records)

    # Create indexes
    col.create_index([("date", 1), ("ticker", 1)])
    print(f"    [OK] Features stored in '{config.COL_DAILY_FEATURES}'.")

    return merged


if __name__ == "__main__":
    df = build_features(config.DEFAULT_TICKER, config.DEFAULT_DAYS)
    print("\nSample features:")
    print(df.head(10).to_string())
