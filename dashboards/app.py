
import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="ESG Pulse", layout="wide")

st.title("🌿 ESG Pulse – Sentiment & Stock Impact Tracker")

st.sidebar.header("Portfolio Setup")
ticker_input = st.sidebar.text_input("Enter comma-separated stock tickers", "AAPL,MSFT,TSLA")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# Add Save Portfolio button
if st.sidebar.button("Save Portfolio"):
    import os
    os.makedirs("data", exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv("data/user_portfolio.csv", index=False)
    st.sidebar.success("Portfolio saved to data/user_portfolio.csv!")

st.sidebar.header("Settings")
run_analysis = st.sidebar.button("Run ESG Analysis")

if run_analysis:
    st.success("Running ESG sentiment analysis...")

    # Placeholder: simulate sentiment fetch & analysis
    sentiment_scores = [-0.6, 0.1, -0.8]
    abnormal_returns = [-0.03, 0.02, -0.05]
    vix = [19.3, 19.3, 19.3]
    momentum = [0.02, -0.01, 0.03]
    sector_dummies = [1, 0, 1]  # Simulated sector sensitivity

    payload = {
        "abnormal_returns": abnormal_returns,
        "sentiment_scores": sentiment_scores,
        "sector_dummies": sector_dummies,
        "vix_values": vix,
        "momentums": momentum
    }

    response = requests.post("http://localhost:5000/run-analysis", json=payload)
    if response.status_code == 200:
        st.subheader("📊 Regression Summary")
        st.text(response.json().get("model_summary"))
    else:
        st.error("Failed to fetch regression output.")

    # Alerts
    st.subheader("⚠️ ESG Sentiment Alerts")
    alert_payload = {
        "records": [
            {"ticker": "AAPL", "sentiment_score": -0.6, "sector": "consumer_goods"},
            {"ticker": "MSFT", "sentiment_score": 0.1, "sector": "technology"},
            {"ticker": "TSLA", "sentiment_score": -0.8, "sector": "energy"},
        ]
    }

    alert_response = requests.post("http://localhost:5000/generate-alerts", json=alert_payload)
    if alert_response.status_code == 200:
        alerts = alert_response.json()
        for alert in alerts:
            st.warning(f"{alert['ticker']}: {alert['alert']}")
    else:
        st.error("Alert generation failed.")
else:
    st.info("Use the sidebar to input tickers and run analysis.")
