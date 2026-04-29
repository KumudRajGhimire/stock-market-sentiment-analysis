"""
run_pipeline.py — End-to-end orchestrator for the stock sentiment pipeline.

Usage:
    python run_pipeline.py                          # defaults (AAPL, 365 days)
    python run_pipeline.py --ticker MSFT --days 180
"""

import argparse
import time
import config
from data_collection import collect_tweets, collect_stock_data
from preprocessing import preprocess_tweets
from sentiment import analyze_sentiment
from feature_engineering import build_features
from model import run_experiments


def run_pipeline(ticker: str, days: int, mode: str = None):
    """Execute the full pipeline in sequence."""
    mode = mode or config.DATA_MODE
    start = time.time()

    print("\n" + "=" * 60)
    print("  BDA-AAT: Stock Movement Prediction Pipeline")
    print(f"  Ticker: {ticker}  |  Days: {days}  |  Mode: {mode}")
    print("=" * 60)

    # Step 1: Collect tweets
    print("\n[STEP 1/5] Collecting tweets ...")
    n_tweets = collect_tweets(ticker, days, mode)

    # Step 2: Preprocess tweets
    print("\n[STEP 2/5] Preprocessing tweets ...")
    n_processed = preprocess_tweets(ticker)

    # Step 3: Sentiment analysis
    print("\n[STEP 3/5] Running FinBERT sentiment analysis ...")
    n_scored = analyze_sentiment(ticker)

    # Step 4: Feature engineering
    print("\n[STEP 4/5] Building features ...")
    features_df = build_features(ticker, days)

    # Step 5: Model training & evaluation
    print("\n[STEP 5/5] Training and evaluating models ...")
    results = run_experiments(ticker)

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"  [DONE] Pipeline complete in {elapsed:.1f}s")
    print(f"  [DATA] Tweets: {n_tweets} -> Processed: {n_processed} -> Scored: {n_scored}")
    print(f"  [DATA] Feature rows: {len(features_df)}")
    print(f"  [DATA] Models trained: {len(results)}")
    print("=" * 60)
    print("\n  Launch dashboard:  streamlit run app.py\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Sentiment Prediction Pipeline")
    parser.add_argument("--ticker", default=config.DEFAULT_TICKER, help="Stock ticker (default: AAPL)")
    parser.add_argument("--days", type=int, default=config.DEFAULT_DAYS, help="Days of history (default: 365)")
    parser.add_argument("--mode", default=None, choices=["csv", "simulated", "live"], help="Data mode")
    args = parser.parse_args()

    run_pipeline(args.ticker, args.days, args.mode)
