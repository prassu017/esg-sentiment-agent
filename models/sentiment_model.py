from transformers import pipeline
import pandas as pd

# Load FinBERT sentiment pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def score_news_sentiment(input_csv="data/sample_news.csv", output_csv="data/sample_news_scored.csv"):
    df = pd.read_csv(input_csv)
    # Score each headline (title)
    df["sentiment_label"] = df["title"].apply(lambda x: sentiment_pipeline(x)[0]["label"])
    df["sentiment_score"] = df["title"].apply(lambda x: sentiment_pipeline(x)[0]["score"])
    df.to_csv(output_csv, index=False)
    print(f"Saved sentiment-scored news to {output_csv}")

if __name__ == "__main__":
    score_news_sentiment()
