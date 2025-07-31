import pandas as pd
import requests
import time
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
from models.sentiment_model import score_news_sentiment
from models.regression_model import run_regression

# ESG News Ingestion
ESG_KEYWORDS = [
    "sustainability", "emissions", "diversity", "governance", "climate", "ESG",
    "carbon", "renewable", "green", "social", "responsibility", "inclusion"
]
GNEWS_API_KEY = st.secrets["GNEWS_API_KEY"]  # Set this in Streamlit Cloud secrets
GNEWS_ENDPOINT = "https://gnews.io/api/v4/search"
SP500_TICKER = "^GSPC"
VIX_TICKER = "^VIX"

def fetch_esg_news_for_portfolio():
    df = pd.read_csv("data/user_portfolio.csv")
    all_news = []
    for ticker in df["ticker"]:
        query = f"{ticker} (" + " OR ".join(ESG_KEYWORDS) + ")"
        params = {
            "q": query,
            "token": GNEWS_API_KEY,
            "lang": "en",
            "max": 10,
            "sort_by": "publishedAt"
        }
        response = requests.get(GNEWS_ENDPOINT, params=params)
        if response.status_code != 200:
            continue
        articles = response.json().get("articles", [])
        for a in articles:
            text = (a.get("title", "") + " " + a.get("description", "")).lower()
            if any(kw.lower() in text for kw in ESG_KEYWORDS):
                all_news.append({
                    "ticker": ticker,
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "publishedAt": a.get("publishedAt", ""),
                    "url": a.get("url", "")
                })
        time.sleep(1)
    news_df = pd.DataFrame(all_news)
    news_df.to_csv("data/sample_news.csv", index=False)

def get_price_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date, progress=False)

def calculate_market_features(news_csv="data/sample_news_scored.csv", output_csv="data/market_features.csv"):
    news_df = pd.read_csv(news_csv)
    features = []
    def extract_scalar(value):
        if hasattr(value, 'iloc'):
            if len(value) > 0:
                val = value.iloc[0]
                return float(val) if not pd.isna(val) else None
            else:
                return None
        elif hasattr(value, 'item'):
            try:
                val = value.item()
                return float(val) if not pd.isna(val) else None
            except:
                return None
        else:
            try:
                if pd.isna(value):
                    return None
                return float(value)
            except:
                return None
    for idx, row in news_df.iterrows():
        ticker = row["ticker"]
        event_date = row["publishedAt"][:10]
        try:
            event_dt = datetime.strptime(event_date, "%Y-%m-%d")
            if event_dt > datetime.now():
                continue
        except Exception:
            continue
        estimation_start = (event_dt - timedelta(days=80)).strftime("%Y-%m-%d")
        estimation_end = (event_dt - timedelta(days=6)).strftime("%Y-%m-%d")
        event_window_start = (event_dt - timedelta(days=5)).strftime("%Y-%m-%d")
        event_window_end = (event_dt + timedelta(days=5)).strftime("%Y-%m-%d")
        try:
            stock_prices = get_price_data(ticker, estimation_start, event_window_end)
            sp500_prices = get_price_data(SP500_TICKER, estimation_start, event_window_end)
            vix_prices = get_price_data(VIX_TICKER, event_window_start, event_window_end)
        except Exception:
            continue
        close_col = "Close" if "Close" in stock_prices.columns else ("Close", ticker)
        sp500_close_col = "Close" if "Close" in sp500_prices.columns else ("Close", SP500_TICKER)
        stock_returns = stock_prices[close_col].pct_change().dropna()
        sp500_returns = sp500_prices[sp500_close_col].pct_change().dropna()
        est_stock = stock_returns.loc[estimation_start:estimation_end]
        est_sp500 = sp500_returns.loc[estimation_start:estimation_end]
        if len(est_stock) < 10 or len(est_sp500) < 10:
            continue
        import statsmodels.api as sm
        X = sm.add_constant(est_sp500.values)
        y = est_stock.values
        model = sm.OLS(y, X).fit()
        alpha, beta = model.params
        for offset in range(-3, 4):
            day = (event_dt + timedelta(days=offset)).strftime("%Y-%m-%d")
            if day not in stock_returns.index or day not in sp500_returns.index:
                continue
            actual_return = stock_returns.loc[day]
            expected_return = alpha + beta * sp500_returns.loc[day]
            actual_scalar = extract_scalar(actual_return)
            expected_scalar = extract_scalar(expected_return)
            abnormal_return = actual_scalar - expected_scalar if actual_scalar is not None and expected_scalar is not None else None
            try:
                momentum = (stock_prices.loc[day, close_col] / stock_prices.loc[day - 1, close_col]) - 1
            except Exception:
                momentum = None
            momentum_val = float(momentum) 
            if isinstance(momentum, pd.Series):
                momentum_val = momentum.item() if len(momentum) == 1 else None
            elif pd.notna(momentum):
                momentum_val = float(momentum)

            # Append processed features
            features.append({
                "date": row["date"],
                "ticker": row["ticker"],
                "actual_return": actual_scalar,
                "expected_return": expected_scalar,
                "abnormal_return": abnormal_return,
                "momentum": momentum_val,
                "sentiment_label": row["sentiment_label"],
                "sentiment_score": float(row["sentiment_score"]) if pd.notna(row["sentiment_score"]) else None,
                "title": row["title"]
            })
    features_df = pd.DataFrame(features)
    features_df.to_csv(output_csv, index=False) 