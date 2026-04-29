"""
sentiment.py — Run FinBERT sentiment analysis on preprocessed tweets.
"""

import pymongo
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

import config


def get_db():
    """Return a pymongo Database handle."""
    client = pymongo.MongoClient(config.MONGO_URI)
    return client[config.MONGO_DB_NAME]


def load_finbert():
    """Load the FinBERT sentiment analysis pipeline."""
    print(f"[*] Loading FinBERT model ({config.FINBERT_MODEL}) ...")
    tokenizer = AutoTokenizer.from_pretrained(config.FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(config.FINBERT_MODEL)
    nlp = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=512,
    )
    print("    [OK] FinBERT loaded.")
    return nlp


def sentiment_to_score(result: dict) -> float:
    """
    Convert FinBERT output to a numeric score.
      positive -> +confidence
      negative -> -confidence
      neutral  -> 0.0
    """
    label = result["label"].lower()
    score = result["score"]
    if label == "positive":
        return score
    elif label == "negative":
        return -score
    else:
        return 0.0


def analyze_sentiment(ticker: str):
    """Run FinBERT on all unscored tweets and update MongoDB."""
    db = get_db()
    col = db[config.COL_TWEETS_PROCESSED]

    # Find tweets without sentiment
    unscored = list(col.find({"ticker": ticker, "sentiment": None}))
    if not unscored:
        print(f"[*] No unscored tweets for {ticker}. Skipping sentiment analysis.")
        return 0

    print(f"[*] Analyzing sentiment for {len(unscored)} tweets ...")

    # Load model
    nlp = load_finbert()

    # Process in batches
    batch_size = config.SENTIMENT_BATCH_SIZE
    operations = []

    for i in tqdm(range(0, len(unscored), batch_size), desc="   FinBERT"):
        batch = unscored[i : i + batch_size]
        texts = [t["text"] for t in batch]

        try:
            results = nlp(texts, batch_size=batch_size)
        except Exception as e:
            print(f"[!] Error in batch {i}: {e}")
            continue

        for tweet, result in zip(batch, results):
            score = sentiment_to_score(result)
            operations.append(
                pymongo.UpdateOne(
                    {"_id": tweet["_id"]},
                    {"$set": {
                        "sentiment": score,
                        "sentiment_label": result["label"],
                        "sentiment_confidence": result["score"],
                    }},
                )
            )

    if operations:
        result = col.bulk_write(operations)
        print(f"    [OK] {result.modified_count} tweets updated with sentiment scores.")
    else:
        print("[!] No sentiment updates performed.")

    return len(operations)


if __name__ == "__main__":
    analyze_sentiment(config.DEFAULT_TICKER)
