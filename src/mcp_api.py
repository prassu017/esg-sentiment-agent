
from flask import Flask, request, jsonify
from transformers import pipeline
import numpy as np
import statsmodels.api as sm

app = Flask(__name__)

# Sentiment pipeline using FinBERT (placeholder)
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

@app.route('/sentiment-score', methods=['POST'])
def sentiment_score():
    data = request.json
    texts = data.get("texts", [])
    results = sentiment_pipeline(texts)
    return jsonify(results)

@app.route('/run-analysis', methods=['POST'])
def run_analysis():
    data = request.json
    y = np.array(data["abnormal_returns"])
    X = np.column_stack([
        data["sentiment_scores"],
        data["sector_dummies"],
        data["vix_values"],
        data["momentums"]
    ])
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    summary = model.summary().as_text()
    return jsonify({"model_summary": summary})

@app.route('/generate-alerts', methods=['POST'])
def generate_alerts():
    data = request.json
    alerts = []
    for record in data["records"]:
        if record["sentiment_score"] < -0.5 and record["sector"] in ["energy", "consumer_goods"]:
            alerts.append({
                "ticker": record["ticker"],
                "alert": "⚠️ Negative ESG sentiment detected. Monitor closely."
            })
    return jsonify(alerts)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
