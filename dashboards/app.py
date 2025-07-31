import streamlit as st
import pandas as pd
import os
import time
import sys

# Add the parent directory to sys.path so Python can locate pipeline.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Import your pipeline functions
from pipeline import fetch_esg_news_for_portfolio, calculate_market_features
from models.sentiment_model import score_news_sentiment
from models.regression_model import run_regression

st.set_page_config(page_title="ESG Pulse", layout="wide")
st.title("🌿 ESG Pulse – Sentiment & Stock Impact Tracker")

# --- Portfolio Input ---
st.sidebar.header("Portfolio Setup")
ticker_input = st.sidebar.text_input("Enter up to 3 comma-separated stock tickers", "AAPL,MSFT,TSLA")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

if len(tickers) > 3:
    st.sidebar.error("Please enter no more than 3 tickers.")
    tickers = tickers[:3]

if st.sidebar.button("Save Portfolio"):
    os.makedirs("data", exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv("data/user_portfolio.csv", index=False)
    st.sidebar.success("Portfolio saved to data/user_portfolio.csv!")

st.sidebar.header("Settings")
run_analysis = st.sidebar.button("Run ESG Analysis")

# --- Main Analysis ---
if run_analysis:
    if len(tickers) == 0:
        st.error("Please enter at least one ticker.")
    elif len(tickers) > 3:
        st.error("Please enter no more than 3 tickers.")
    else:
        st.success("Running full ESG sentiment analysis pipeline...")

        # 1. Save portfolio
        os.makedirs("data", exist_ok=True)
        pd.DataFrame({"ticker": tickers}).to_csv("data/user_portfolio.csv", index=False)
        st.info("Portfolio saved.")

        # 2. Fetch ESG news
        with st.spinner("Fetching ESG news..."):
            fetch_esg_news_for_portfolio()
            st.success("News fetched!")

        # 3. Score sentiment
        with st.spinner("Scoring sentiment..."):
            score_news_sentiment("data/sample_news.csv", "data/sample_news_scored.csv")
            st.success("Sentiment scored!")

        # 4. Generate market features
        with st.spinner("Generating market features..."):
            calculate_market_features("data/sample_news_scored.csv", "data/market_features.csv")
            st.success("Market features generated!")

        # 5. Run regression
        with st.spinner("Running regression analysis..."):
            model, results = run_regression()
            if results:
                st.subheader("📊 Regression Summary")
                st.text(results['model_summary'])

                # Show key findings
                st.markdown("### Key Findings")
                st.write(f"**Sentiment Impact:** {results['sentiment_coef']:.6f} (p={results['sentiment_pvalue']:.4f})")
                st.write(f"**R-squared:** {results['r_squared']:.4f}")
                st.write(f"**Observations:** {results['n_observations']}")

                # Show analysis by sentiment category
                df = pd.read_csv("data/market_features.csv")
                df['sentiment_category'] = pd.cut(df['sentiment_score'], bins=[-float('inf'), 0.3, 0.7, float('inf')], labels=['Negative', 'Neutral', 'Positive'])
                st.markdown("### Abnormal Returns by Sentiment Category")
                st.write(df.groupby('sentiment_category')['abnormal_return'].agg(['mean', 'std', 'count']))

                # Plots
                import matplotlib.pyplot as plt
                st.markdown("### Distribution of Abnormal Returns")
                fig1, ax1 = plt.subplots()
                df['abnormal_return'].hist(bins=30, ax=ax1)
                ax1.set_xlabel('Abnormal Return')
                ax1.set_ylabel('Frequency')
                st.pyplot(fig1)

                st.markdown("### Abnormal Returns by Sentiment Category")
                fig2, ax2 = plt.subplots()
                df.boxplot(column='abnormal_return', by='sentiment_category', ax=ax2)
                plt.suptitle('')
                st.pyplot(fig2)

                # Download button
                st.download_button(
                    label="Download Market Features as CSV",
                    data=df.to_csv(index=False),
                    file_name='market_features.csv',
                    mime='text/csv'
                )
            else:
                st.error("No regression results available. Please ensure you have valid data.")

else:
    st.info("Use the sidebar to input up to 3 tickers and run analysis.")

# --- Optionally, show raw data ---
if os.path.exists('data/market_features.csv'):
    with st.expander("Show raw market features data"):
        df = pd.read_csv('data/market_features.csv')
        st.dataframe(df)