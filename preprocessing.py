"""
preprocessing.py — Clean raw tweets and prepare them for sentiment analysis.
"""

import re
import pymongo
from tqdm import tqdm

import config


def get_db():
    """Return a pymongo Database handle."""
    client = pymongo.MongoClient(config.MONGO_URI)
    return client[config.MONGO_DB_NAME]


def clean_text(text: str) -> str:
    """Clean a tweet's text for NLP processing."""
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove mentions
    text = re.sub(r"@\w+", "", text)
    # Remove hashtag symbol but keep word
    text = re.sub(r"#(\w+)", r"\1", text)
    # Remove special characters (keep alphanumeric, spaces, basic punctuation)
    text = re.sub(r"[^\w\s.,!?'-]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Lowercase
    text = text.lower()
    return text


def preprocess_tweets(ticker: str):
    """Load raw tweets, clean them, and store in the processed collection."""
    db = get_db()
    raw_col = db[config.COL_TWEETS_RAW]
    proc_col = db[config.COL_TWEETS_PROCESSED]

    # Clear previous processed data for this ticker
    deleted = proc_col.delete_many({"ticker": ticker})
    if deleted.deleted_count:
        print(f"[*] Cleared {deleted.deleted_count} old processed tweets for {ticker}.")

    raw_tweets = list(raw_col.find({"ticker": ticker}))
    if not raw_tweets:
        print(f"[!] No raw tweets found for {ticker}.")
        return 0

    print(f"[*] Preprocessing {len(raw_tweets)} tweets for {ticker} ...")

    processed = []
    for tweet in tqdm(raw_tweets, desc="   Cleaning"):
        cleaned = clean_text(str(tweet["text"]))
        if len(cleaned) < 5:
            continue  # skip very short / empty tweets

        # Handle both datetime objects and strings for created_at
        created_at = tweet["created_at"]
        if hasattr(created_at, "strftime"):
            date_str = created_at.strftime("%Y-%m-%d")
        else:
            # Parse string date (e.g., from CSV)
            date_str = str(created_at)[:10]

        processed.append({
            "text": cleaned,
            "original_text": tweet["text"],
            "date": date_str,
            "ticker": ticker,
            "sentiment": None,  # will be filled by sentiment.py
        })

    if processed:
        proc_col.insert_many(processed)
        print(f"    [OK] {len(processed)} cleaned tweets stored in '{config.COL_TWEETS_PROCESSED}'.")
    else:
        print("[!] No tweets survived preprocessing.")

    return len(processed)


if __name__ == "__main__":
    preprocess_tweets(config.DEFAULT_TICKER)
