"""
visualize.py — Generate Plotly charts for the Streamlit dashboard.
"""

import pymongo
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import config


def get_db():
    client = pymongo.MongoClient(config.MONGO_URI)
    return client[config.MONGO_DB_NAME]


def load_features_df(ticker):
    db = get_db()
    data = list(db[config.COL_DAILY_FEATURES].find({"ticker": ticker}, {"_id": 0}))
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_model_results(ticker):
    db = get_db()
    data = list(db[config.COL_MODEL_RESULTS].find({"ticker": ticker}, {"_id": 0}))
    return data


def chart_sentiment_vs_price(df):
    """Dual-axis chart: closing price + average sentiment."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df["date"], y=df["close_price"], name="Close Price",
                   line=dict(color="#6366f1", width=2)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df["date"], y=df["avg_sentiment"], name="Avg Sentiment",
                   line=dict(color="#f97316", width=2), opacity=0.8),
        secondary_y=True,
    )
    fig.update_layout(
        title="Stock Price vs Average Sentiment",
        template="plotly_dark",
        hovermode="x unified",
        height=450,
    )
    fig.update_yaxes(title_text="Close Price ($)", secondary_y=False)
    fig.update_yaxes(title_text="Avg Sentiment Score", secondary_y=True)
    return fig


def chart_tweet_volume(df):
    """Bar chart of daily tweet volume with color by sentiment."""
    colors = df["avg_sentiment"].apply(
        lambda s: "#22c55e" if s > 0.05 else ("#ef4444" if s < -0.05 else "#94a3b8")
    )
    fig = go.Figure(go.Bar(x=df["date"], y=df["tweet_count"], marker_color=colors, name="Tweets"))
    fig.update_layout(
        title="Daily Tweet Volume (colored by sentiment)",
        template="plotly_dark", height=350,
        xaxis_title="Date", yaxis_title="Tweet Count",
    )
    return fig


def chart_confusion_matrix(cm, model_name):
    """Heatmap confusion matrix."""
    labels = ["DOWN (0)", "UP (1)"]
    fig = px.imshow(
        cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
        x=labels, y=labels, color_continuous_scale="Purples",
    )
    fig.update_layout(title=f"Confusion Matrix - {model_name}", template="plotly_dark", height=400)
    return fig


def chart_model_comparison(results):
    """Grouped bar chart comparing all models."""
    df = pd.DataFrame(results)[["model", "accuracy", "precision", "recall", "f1_score"]]
    df_melted = df.melt(id_vars="model", var_name="Metric", value_name="Score")
    fig = px.bar(
        df_melted, x="Metric", y="Score", color="model", barmode="group",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        title="Model Performance Comparison",
        template="plotly_dark", height=450,
        yaxis=dict(range=[0, 1]),
    )
    return fig


def chart_target_distribution(df):
    """Pie chart of UP vs DOWN distribution."""
    counts = df["target"].value_counts()
    fig = px.pie(
        values=counts.values,
        names=["DOWN" if i == 0 else "UP" for i in counts.index],
        color_discrete_sequence=["#ef4444", "#22c55e"],
    )
    fig.update_layout(title="Target Distribution (UP vs DOWN)", template="plotly_dark", height=350)
    return fig
