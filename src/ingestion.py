import pandas as pd
import requests
import time
import yfinance as yf
from datetime import datetime, timedelta

ESG_KEYWORDS = [
    "sustainability", "emissions", "diversity", "governance", "climate", "ESG",
    "carbon", "renewable", "green", "social", "responsibility", "inclusion"
]
GNEWS_API_KEY = "f7f254dfc760cd3c0cf3691cd5b2f494"
GNEWS_ENDPOINT = "https://gnews.io/api/v4/search"

SP500_TICKER = "^GSPC"
VIX_TICKER = "^VIX"


def fetch_esg_news_for_ticker(ticker, max_articles=10):
    # Build query string with ESG keywords
    query = f"{ticker} (" + " OR ".join(ESG_KEYWORDS) + ")"
    params = {
        "q": query,
        "token": GNEWS_API_KEY,
        "lang": "en",
        "max": max_articles,
        "sort_by": "publishedAt"
    }
    response = requests.get(GNEWS_ENDPOINT, params=params)
    if response.status_code != 200:
        print(f"Failed to fetch news for {ticker}: {response.text}")
        return []
    articles = response.json().get("articles", [])
    # Filter articles for ESG keywords in title/description
    filtered = []
    for a in articles:
        text = (a.get("title", "") + " " + a.get("description", "")).lower()
        if any(kw.lower() in text for kw in ESG_KEYWORDS):
            filtered.append({
                "ticker": ticker,
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "publishedAt": a.get("publishedAt", ""),
                "url": a.get("url", "")
            })
    return filtered

def fetch_esg_news_for_portfolio():
    df = pd.read_csv("data/user_portfolio.csv")
    all_news = []
    for ticker in df["ticker"]:
        print(f"Fetching news for {ticker}...")
        news = fetch_esg_news_for_ticker(ticker)
        all_news.extend(news)
        time.sleep(1)  # avoid rate limits
    news_df = pd.DataFrame(all_news)
    news_df.to_csv("data/sample_news.csv", index=False)
    print(f"Saved {len(news_df)} news articles to data/sample_news.csv")

def get_price_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date, progress=False)

def calculate_market_features(news_csv="data/sample_news_scored.csv", output_csv="data/market_features.csv"):
    news_df = pd.read_csv(news_csv)
    features = []
    print(f"Processing {len(news_df)} news events...")
    
    # Extract scalar values from Series
    def extract_scalar(value):
        # Handle pandas Series
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
            # Handle scalar values
            try:
                if pd.isna(value):
                    return None
                return float(value)
            except:
                return None
    
    for idx, row in news_df.iterrows():
        ticker = row["ticker"]
        event_date = row["publishedAt"][:10]  # YYYY-MM-DD
        print(f"Processing {ticker} on {event_date}...")
        try:
            event_dt = datetime.strptime(event_date, "%Y-%m-%d")
            # Skip future dates
            if event_dt > datetime.now():
                print(f"  Skipping future date: {event_date}")
                continue
        except Exception as e:
            print(f"  Error parsing date {event_date}: {e}")
            continue
        # Define windows
        estimation_start = (event_dt - timedelta(days=80)).strftime("%Y-%m-%d")
        estimation_end = (event_dt - timedelta(days=6)).strftime("%Y-%m-%d")
        event_window_start = (event_dt - timedelta(days=5)).strftime("%Y-%m-%d")
        event_window_end = (event_dt + timedelta(days=5)).strftime("%Y-%m-%d")
        print(f"  Windows: estimation={estimation_start} to {estimation_end}, event={event_window_start} to {event_window_end}")
        # Fetch prices
        try:
            stock_prices = get_price_data(ticker, estimation_start, event_window_end)
            sp500_prices = get_price_data(SP500_TICKER, estimation_start, event_window_end)
            vix_prices = get_price_data(VIX_TICKER, event_window_start, event_window_end)
            print(f"  Got {len(stock_prices)} stock prices, {len(sp500_prices)} S&P500 prices, {len(vix_prices)} VIX prices")
        except Exception as e:
            print(f"  Error fetching data for {ticker}: {e}")
            continue
        # Calculate daily returns
        # Handle multi-level column names from yfinance
        close_col = None
        if "Close" in stock_prices.columns:
            close_col = "Close"
        elif ("Close", ticker) in stock_prices.columns:
            close_col = ("Close", ticker)
        else:
            print(f"  Ticker {ticker}: 'Close' column missing. Columns: {stock_prices.columns.tolist()}. Skipping.")
            continue
            
        sp500_close_col = None
        if "Close" in sp500_prices.columns:
            sp500_close_col = "Close"
        elif ("Close", SP500_TICKER) in sp500_prices.columns:
            sp500_close_col = ("Close", SP500_TICKER)
        else:
            print(f"  S&P500: 'Close' column missing. Columns: {sp500_prices.columns.tolist()}. Skipping.")
            continue
            
        stock_returns = stock_prices[close_col].pct_change().dropna()
        sp500_returns = sp500_prices[sp500_close_col].pct_change().dropna()
        # Market model regression (estimation window)
        est_stock = stock_returns.loc[estimation_start:estimation_end]
        est_sp500 = sp500_returns.loc[estimation_start:estimation_end]
        if len(est_stock) < 10 or len(est_sp500) < 10:
            print(f"  Insufficient data for regression: {len(est_stock)} stock returns, {len(est_sp500)} S&P500 returns")
            continue
        try:
            import statsmodels.api as sm
            X = sm.add_constant(est_sp500.values)
            y = est_stock.values
            model = sm.OLS(y, X).fit()
            alpha, beta = model.params
            print(f"  Regression successful: alpha={alpha:.4f}, beta={beta:.4f}")
        except Exception as e:
            print(f"  Regression error for {ticker}: {e}")
            continue
        # Event window actual/expected/abnormal returns
        event_features = 0
        for offset in range(-3, 4):  # T-3 to T+3
            day = (event_dt + timedelta(days=offset)).strftime("%Y-%m-%d")
            if day not in stock_returns.index or day not in sp500_returns.index:
                continue
            actual_return = stock_returns.loc[day]
            expected_return = alpha + beta * sp500_returns.loc[day]
            
            # Extract scalar values first
            actual_scalar = extract_scalar(actual_return)
            expected_scalar = extract_scalar(expected_return)
            
            # Calculate abnormal return
            if actual_scalar is not None and expected_scalar is not None:
                abnormal_return = actual_scalar - expected_scalar
            else:
                abnormal_return = None
            
            # Debug: print the values
            actual_str = f"{actual_scalar:.6f}" if actual_scalar is not None else "None"
            expected_str = f"{expected_scalar:.6f}" if expected_scalar is not None else "None"
            abnormal_str = f"{abnormal_return:.6f}" if abnormal_return is not None else "None"
            print(f"    Day {day}: actual={actual_str}, expected={expected_str}, abnormal={abnormal_str}")
            
            # Momentum: % change over last 5 days
            try:
                momentum = (stock_prices.loc[day, close_col] / stock_prices.loc[:day].iloc[-6][close_col]) - 1
            except Exception:
                momentum = None
            # VIX value
            vix_close_col = None
            if "Close" in vix_prices.columns:
                vix_close_col = "Close"
            elif ("Close", VIX_TICKER) in vix_prices.columns:
                vix_close_col = ("Close", VIX_TICKER)
            else:
                print(f"    VIX columns available: {vix_prices.columns.tolist()}")
                vix_value = None
            if vix_close_col:
                vix_value = vix_prices[vix_close_col].get(day, None)
                print(f"    VIX for {day}: {vix_value}")
            else:
                vix_value = None
                
            features.append({
                "ticker": ticker,
                "event_date": event_date,
                "window_day": offset,
                "actual_return": extract_scalar(actual_return),
                "expected_return": extract_scalar(expected_return),
                "abnormal_return": extract_scalar(abnormal_return),
                "momentum": extract_scalar(momentum) if momentum is not None else None,
                "vix": extract_scalar(vix_value) if vix_value is not None else None,
                "sentiment_label": row["sentiment_label"],
                "sentiment_score": float(row["sentiment_score"]) if not pd.isna(row["sentiment_score"]) else None,
                "title": row["title"]
            })
            event_features += 1
        print(f"  Generated {event_features} features for this event")
    features_df = pd.DataFrame(features)
    features_df.to_csv(output_csv, index=False)
    print(f"Saved {len(features_df)} market features to {output_csv}")

if __name__ == "__main__":
    # Uncomment to run this step
    # fetch_esg_news_for_portfolio()
    calculate_market_features()
    pass
