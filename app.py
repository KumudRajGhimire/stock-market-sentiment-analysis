"""
app.py — Streamlit dashboard for the Stock Sentiment Prediction pipeline.
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import config
from visualize import (
    load_features_df, load_model_results,
    chart_sentiment_vs_price, chart_tweet_volume,
    chart_confusion_matrix, chart_model_comparison,
    chart_target_distribution,
)
from data_collection import collect_tweets, collect_stock_data, get_available_stocks
from preprocessing import preprocess_tweets
from sentiment import analyze_sentiment
from feature_engineering import build_features
from model import run_experiments


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Sentiment Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #334155);
        border-radius: 12px; padding: 20px;
        border: 1px solid #475569; text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #6366f1; }
    .metric-label { font-size: 0.85rem; color: #94a3b8; }
    h1 { color: #e2e8f0 !important; }
    h2, h3 { color: #cbd5e1 !important; }

    /* Stock card styling */
    .stock-card {
        background: linear-gradient(135deg, #1e293b, #334155);
        border-radius: 10px; padding: 12px 16px;
        border: 1px solid #475569;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }
    .stock-card:hover {
        border-color: #6366f1;
        transform: translateX(4px);
    }
    .stock-ticker {
        font-size: 1.1rem; font-weight: 700;
        color: #6366f1; display: inline-block;
        min-width: 60px;
    }
    .stock-company {
        font-size: 0.85rem; color: #94a3b8;
    }
    .stock-tweets {
        font-size: 0.75rem; color: #64748b;
        float: right; margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load available stocks ────────────────────────────────────────────────────
available_stocks = get_available_stocks()
stock_tickers = [s["ticker"] for s in available_stocks]
stock_map = {s["ticker"]: s for s in available_stocks}

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    # Data mode selector
    mode = st.selectbox(
        "Data Source",
        ["csv", "simulated", "live"],
        index=0,
        help="'csv' uses stock_tweets.csv, 'simulated' generates fake tweets, 'live' uses Twitter API",
    )

    if mode == "csv" and available_stocks:
        # Stock selector dropdown
        default_idx = stock_tickers.index(config.DEFAULT_TICKER) if config.DEFAULT_TICKER in stock_tickers else 0
        ticker = st.selectbox(
            "Select Stock",
            stock_tickers,
            index=default_idx,
            format_func=lambda t: f"{t} — {stock_map[t]['company_name']}",
        )
        days = 365  # CSV has ~1 year of data, not configurable
        st.caption(f"📊 {stock_map[ticker]['tweet_count']:,} tweets available")
    else:
        ticker = st.text_input("Stock Ticker", value=config.DEFAULT_TICKER)
        days = st.slider("Historical Days", 30, 730, config.DEFAULT_DAYS)

    st.divider()

    if st.button("🚀 Run Full Pipeline", use_container_width=True, type="primary"):
        with st.status("Running pipeline...", expanded=True) as status:
            st.write("📡 Collecting tweets...")
            n_tweets = collect_tweets(ticker, days, mode)
            st.write(f"   ✓ {n_tweets} tweets collected")

            st.write("🧹 Preprocessing tweets...")
            n_proc = preprocess_tweets(ticker)
            st.write(f"   ✓ {n_proc} tweets cleaned")

            st.write("🧠 Running FinBERT sentiment analysis...")
            n_sent = analyze_sentiment(ticker)
            st.write(f"   ✓ {n_sent} tweets scored")

            st.write("🔧 Building features...")
            build_features(ticker, days)
            st.write("   ✓ Features built")

            st.write("📊 Training models...")
            run_experiments(ticker)
            st.write("   ✓ Models trained")

            status.update(label="Pipeline complete! ✅", state="complete")

    st.divider()

    # Show available stocks list
    if available_stocks:
        with st.expander("📋 Available Stocks in Dataset", expanded=False):
            for s in available_stocks:
                st.markdown(
                    f'<div class="stock-card">'
                    f'<span class="stock-ticker">{s["ticker"]}</span>'
                    f'<span class="stock-tweets">{s["tweet_count"]:,} tweets</span>'
                    f'<br><span class="stock-company">{s["company_name"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.caption("BDA-AAT | Big Data Analytics")

# ── Main content ─────────────────────────────────────────────────────────────
st.title("📈 Stock Movement Prediction with Sentiment")

company_label = stock_map[ticker]["company_name"] if ticker in stock_map else ticker
st.caption(f"Analyzing **{ticker}** ({company_label}) — {'CSV dataset' if mode == 'csv' else f'{days} days of data'}")

# Load data
df = load_features_df(ticker)
results = load_model_results(ticker)

if df.empty:
    st.info("👆 Use the sidebar to select a stock and run the pipeline first, then the dashboard will populate.")

    # Show available stocks in the main area as a nice grid
    if available_stocks:
        st.subheader("🏢 Available Stocks for Analysis")
        st.markdown("The following stocks are available in the `stock_tweets.csv` dataset:")

        # Display as a nicely formatted table
        stocks_df = pd.DataFrame(available_stocks)
        stocks_df.columns = ["Ticker", "Company", "Tweets"]
        stocks_df = stocks_df.sort_values("Tweets", ascending=False).reset_index(drop=True)
        stocks_df.index += 1

        # Show in columns for a nicer layout
        col1, col2 = st.columns(2)
        half = len(stocks_df) // 2 + 1
        with col1:
            st.dataframe(
                stocks_df.iloc[:half],
                use_container_width=True,
                hide_index=False,
            )
        with col2:
            st.dataframe(
                stocks_df.iloc[half:],
                use_container_width=True,
                hide_index=False,
            )

    st.stop()

# ── Metrics row ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Trading Days", len(df))
with c2:
    st.metric("Avg Sentiment", f"{df['avg_sentiment'].mean():.3f}")
with c3:
    st.metric("Total Tweets", f"{df['tweet_count'].sum():,}")
with c4:
    latest_price = df["close_price"].iloc[-1]
    price_change = df["close_price"].iloc[-1] - df["close_price"].iloc[0]
    st.metric("Latest Close", f"${latest_price:.2f}", f"{price_change:+.2f}")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🤖 Predictions", "⚖️ Comparison"])

with tab1:
    st.plotly_chart(chart_sentiment_vs_price(df), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(chart_tweet_volume(df), use_container_width=True)
    with col2:
        st.plotly_chart(chart_target_distribution(df), use_container_width=True)

    with st.expander("📋 Raw Feature Data"):
        st.dataframe(df, use_container_width=True)

with tab2:
    if not results:
        st.warning("No model results yet. Run the pipeline first.")
    else:
        # Show metrics for best model
        best = max(results, key=lambda r: r["f1_score"])
        st.success(f"**Best Model:** {best['model']} — F1: {best['f1_score']}")

        cols = st.columns(2)
        for i, r in enumerate(results):
            with cols[i % 2]:
                st.subheader(r["model"])
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Accuracy", r["accuracy"])
                mc2.metric("Precision", r["precision"])
                mc3.metric("Recall", r["recall"])
                mc4.metric("F1", r["f1_score"])
                cm = np.array(r["confusion_matrix"])
                st.plotly_chart(chart_confusion_matrix(cm, r["model"]), use_container_width=True)

with tab3:
    if not results:
        st.warning("No model results yet. Run the pipeline first.")
    else:
        st.plotly_chart(chart_model_comparison(results), use_container_width=True)

        # Comparison table
        comp_df = pd.DataFrame(results)[["model", "accuracy", "precision", "recall", "f1_score"]]
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Insight
        with_sent = [r for r in results if "With" in r["feature_set"]]
        without_sent = [r for r in results if "Without" in r["feature_set"]]
        if with_sent and without_sent:
            avg_with = np.mean([r["f1_score"] for r in with_sent])
            avg_without = np.mean([r["f1_score"] for r in without_sent])
            diff = avg_with - avg_without
            if diff > 0:
                st.info(f"🔍 Adding sentiment features **improved** average F1 by **{diff:.4f}**")
            else:
                st.info(f"🔍 Sentiment features **did not improve** average F1 (Δ = {diff:.4f})")
