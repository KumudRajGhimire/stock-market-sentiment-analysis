"""
model.py — Train and evaluate ML models for stock movement prediction.
Compares performance with and without sentiment features.
"""

import pymongo
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler
import config


def get_db():
    client = pymongo.MongoClient(config.MONGO_URI)
    return client[config.MONGO_DB_NAME]


def load_features(ticker):
    db = get_db()
    col = db[config.COL_DAILY_FEATURES]
    data = list(col.find({"ticker": ticker}, {"_id": 0}))
    if not data:
        raise ValueError(f"No features for {ticker}.")
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def time_based_split(df, ratio=0.8):
    idx = int(len(df) * ratio)
    return df.iloc[:idx].copy(), df.iloc[idx:].copy()


def train_eval(X_tr, X_te, y_tr, y_te, model, name):
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    model.fit(X_tr_s, y_tr)
    y_pred = model.predict(X_te_s)
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_te, y_pred), 4),
        "precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_te, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_te, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        "classification_report": classification_report(y_te, y_pred, output_dict=True),
    }, y_pred


def run_experiments(ticker):
    print(f"\n{'='*60}")
    print(f"  Model Training & Evaluation - {ticker}")
    print(f"{'='*60}\n")

    df = load_features(ticker)
    train, test = time_based_split(df)
    print(f"[*] Dataset: {len(df)} | Train: {len(train)} | Test: {len(test)}")

    feat_no_sent = ["tweet_count"]
    feat_with_sent = ["avg_sentiment", "tweet_count", "sentiment_ratio"]
    all_results = []

    for fs_name, fcols in [("Without Sentiment", feat_no_sent), ("With Sentiment", feat_with_sent)]:
        print(f"\n--- {fs_name} (features: {fcols}) ---")
        X_tr, X_te = train[fcols].values, test[fcols].values
        y_tr, y_te = train["target"].values, test["target"].values

        for mdl, mname in [
            (LogisticRegression(max_iter=1000, random_state=42), "Logistic Regression"),
            (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"),
        ]:
            label = f"{mname} ({fs_name})"
            metrics, _ = train_eval(X_tr, X_te, y_tr, y_te, mdl, label)
            metrics.update({"feature_set": fs_name, "features_used": fcols,
                            "ticker": ticker, "train_size": len(train), "test_size": len(test)})
            all_results.append(metrics)
            print(f"    {label} - Acc: {metrics['accuracy']}  F1: {metrics['f1_score']}")

    db = get_db()
    col = db[config.COL_MODEL_RESULTS]
    col.delete_many({"ticker": ticker})
    col.insert_many(all_results)
    print(f"\n    [OK] Results stored in '{config.COL_MODEL_RESULTS}'.")

    comp = pd.DataFrame(all_results)[["model", "accuracy", "precision", "recall", "f1_score"]]
    print(f"\n{'='*60}\n  COMPARISON\n{'='*60}")
    print(comp.to_string(index=False))
    return all_results


if __name__ == "__main__":
    run_experiments(config.DEFAULT_TICKER)
