# 🌿 ESG Pulse - Sentiment & Stock Impact Tracker

A quantitative analysis system that tracks ESG-related news sentiment and analyzes its impact on short-term stock returns.

## 🎯 What It Does

ESG Pulse is an automated agent that:
- **Tracks ESG news** for user-defined stock portfolios
- **Analyzes sentiment** using FinBERT (financial sentiment analysis)
- **Calculates abnormal returns** using market model regression
- **Quantifies impact** through multi-factor regression analysis
- **Provides insights** on ESG sentiment's effect on stock performance

## 🏗️ Architecture

```
Portfolio Input → News Ingestion → Sentiment Analysis → Market Data → Regression → Insights
```

### Key Components:
- **Frontend**: Streamlit dashboard for portfolio input and results visualization
- **Backend**: Python scripts for data processing and analysis
- **NLP**: FinBERT for ESG sentiment scoring
- **Analytics**: Multi-factor regression (sentiment + momentum + market factors)
- **Automation**: n8n workflows for end-to-end orchestration

## 📊 Current Results

Our analysis of 118 observations shows:
- **Momentum Impact**: Highly significant (p = 0.001) - strong positive effect on returns
- **ESG Sentiment**: Not statistically significant in current sample
- **Model Performance**: R-squared = 9.2%, F-statistic = 5.827 (p = 0.00389)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/esg-sentiment-agent.git
cd esg-sentiment-agent

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Required API Keys
- **GNews API**: For ESG news fetching
- **Yahoo Finance**: For stock price data (free)

## 🚀 Quick Start

1. **Run the Streamlit Dashboard:**
   ```bash
   streamlit run dashboards/app.py
   ```

2. **Input Your Portfolio:**
   - Enter stock tickers (e.g., AAPL, MSFT, TSLA)
   - Click "Save Portfolio"

3. **Run Analysis:**
   ```bash
   # Fetch ESG news
   python src/ingestion.py
   
   # Score sentiment
   python models/sentiment_model.py
   
   # Generate market features
   python src/ingestion.py  # (switch to market features mode)
   
   # Run regression analysis
   python models/regression_model.py
   ```

## 📁 Project Structure

```
esg-sentiment-agent/
├── dashboards/          # Streamlit dashboard
├── data/               # CSV files (portfolio, news, features)
├── models/             # ML models (sentiment, regression)
├── src/                # Core processing scripts
├── n8n/               # Automation workflows
├── notebooks/         # Jupyter notebooks for analysis
└── requirements.txt   # Python dependencies
```

## 🔧 Configuration

### Environment Variables
```bash
GNEWS_API_KEY=your_gnews_api_key
```

### Portfolio Setup
- Edit `data/user_portfolio.csv` or use the Streamlit interface
- Supported tickers: Any stock available on Yahoo Finance

## 📈 Features

### Current Features
- ✅ Portfolio management
- ✅ ESG news ingestion
- ✅ Sentiment analysis (FinBERT)
- ✅ Market data collection
- ✅ Abnormal return calculation
- ✅ Multi-factor regression
- ✅ Results visualization

### Planned Features
- 🔄 Real-time alerts
- 📊 Interactive dashboards
- 🤖 Automated scheduling
- 📱 Mobile notifications
- 🌐 Web API endpoints

## 📊 Analysis Methodology

### Market Model
- **Estimation Window**: T-80 to T-6 days
- **Event Window**: T-3 to T+3 days
- **Benchmark**: S&P 500 returns
- **Model**: OLS regression with CAPM-style market model

### Sentiment Analysis
- **Model**: FinBERT (ProsusAI/finbert)
- **Categories**: Positive, Neutral, Negative
- **Features**: Sentiment score, sentiment label

### Regression Model
```
abnormal_return ~ sentiment_score + momentum + vix + sector_dummies
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FinBERT**: For financial sentiment analysis
- **Yahoo Finance**: For market data
- **GNews**: For ESG news aggregation
- **Streamlit**: For the interactive dashboard

## 📞 Contact

- **Project Link**: [https://github.com/yourusername/esg-sentiment-agent](https://github.com/yourusername/esg-sentiment-agent)
- **Issues**: [https://github.com/yourusername/esg-sentiment-agent/issues](https://github.com/yourusername/esg-sentiment-agent/issues)

---

**Built with ❤️ for ESG-conscious investors**
