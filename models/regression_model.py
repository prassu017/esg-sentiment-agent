import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def run_regression(input_csv="data/market_features.csv"):
    """
    Run multi-factor regression to analyze ESG sentiment impact on abnormal returns
    """
    print("Loading market features data...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} observations")
    
    # Data preprocessing
    print("\nData preprocessing...")
    # Convert sentiment_score to numeric (handle string values)
    df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
    
    # Check for missing values before dropping
    print("\nMissing values in each column:")
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count} missing values")
    
    # Show sample of data
    print("\nSample of data (first 5 rows):")
    print(df[["abnormal_return", "sentiment_score", "vix", "momentum"]].head())
    
    # Drop rows with missing values
    df_clean = df.dropna(subset=["abnormal_return", "sentiment_score", "vix", "momentum"])
    print(f"After dropping missing values: {len(df_clean)} observations")
    
    if len(df_clean) == 0:
        print("\nTrying regression without VIX...")
        df_clean = df.dropna(subset=["abnormal_return", "sentiment_score", "momentum"])
        print(f"After dropping missing values (without VIX): {len(df_clean)} observations")
        
        if len(df_clean) == 0:
            print("\nERROR: No data left after dropping missing values!")
            print("This means all rows have at least one missing value in the required columns.")
            return None, None
        
        # Run regression without VIX
        X = df_clean[["sentiment_score", "momentum"]]
        y = df_clean["abnormal_return"]
        X = sm.add_constant(X)
        
        # Create sentiment categories for analysis
        df_clean['sentiment_category'] = pd.cut(df_clean['sentiment_score'], 
                                              bins=[-np.inf, 0.3, 0.7, np.inf], 
                                              labels=['Negative', 'Neutral', 'Positive'])
        
        print("\nRunning regression analysis (without VIX)...")
        print("Model: abnormal_return ~ sentiment_score + momentum")
        
        model = sm.OLS(y, X).fit()
        
        # Print results
        print("\n" + "="*60)
        print("REGRESSION RESULTS (WITHOUT VIX)")
        print("="*60)
        print(model.summary())
        
        # Interpret key results
        print("\n" + "="*60)
        print("KEY FINDINGS")
        print("="*60)
        
        # Sentiment impact
        sentiment_coef = model.params['sentiment_score']
        sentiment_pvalue = model.pvalues['sentiment_score']
        print(f"Sentiment Impact: {sentiment_coef:.6f}")
        print(f"P-value: {sentiment_pvalue:.6f}")
        
        if sentiment_pvalue < 0.05:
            if sentiment_coef > 0:
                print("✅ SIGNIFICANT: Positive ESG sentiment is associated with higher abnormal returns")
            else:
                print("✅ SIGNIFICANT: Negative ESG sentiment is associated with lower abnormal returns")
        else:
            print("❌ NOT SIGNIFICANT: No clear relationship between ESG sentiment and abnormal returns")
        
        # Momentum impact
        momentum_coef = model.params['momentum']
        momentum_pvalue = model.pvalues['momentum']
        print(f"\nMomentum Impact: {momentum_coef:.6f}")
        print(f"P-value: {momentum_pvalue:.6f}")
        
        # Model fit
        print(f"\nR-squared: {model.rsquared:.4f}")
        print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
        
        # Additional analysis by sentiment category
        print("\n" + "="*60)
        print("ANALYSIS BY SENTIMENT CATEGORY")
        print("="*60)
        
        sentiment_analysis = df_clean.groupby('sentiment_category')['abnormal_return'].agg(['mean', 'std', 'count'])
        print(sentiment_analysis)
        
        # Save results
        results = {
            'model_summary': str(model.summary()),
            'sentiment_coef': sentiment_coef,
            'sentiment_pvalue': sentiment_pvalue,
            'r_squared': model.rsquared,
            'n_observations': len(df_clean),
            'model_type': 'without_vix'
        }
        
        return model, results
    
    # Create sentiment categories for analysis
    df_clean['sentiment_category'] = pd.cut(df_clean['sentiment_score'], 
                                          bins=[-np.inf, 0.3, 0.7, np.inf], 
                                          labels=['Negative', 'Neutral', 'Positive'])
    
    # Prepare features and target
    X = df_clean[["sentiment_score", "vix", "momentum"]]
    y = df_clean["abnormal_return"]
    
    # Add constant for intercept
    X = sm.add_constant(X)
    
    print("\nRunning regression analysis...")
    print("Model: abnormal_return ~ sentiment_score + vix + momentum")
    
    # Fit the model
    model = sm.OLS(y, X).fit()
    
    # Print results
    print("\n" + "="*60)
    print("REGRESSION RESULTS")
    print("="*60)
    print(model.summary())
    
    # Interpret key results
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    # Sentiment impact
    sentiment_coef = model.params['sentiment_score']
    sentiment_pvalue = model.pvalues['sentiment_score']
    print(f"Sentiment Impact: {sentiment_coef:.6f}")
    print(f"P-value: {sentiment_pvalue:.6f}")
    
    if sentiment_pvalue < 0.05:
        if sentiment_coef > 0:
            print("✅ SIGNIFICANT: Positive ESG sentiment is associated with higher abnormal returns")
        else:
            print("✅ SIGNIFICANT: Negative ESG sentiment is associated with lower abnormal returns")
    else:
        print("❌ NOT SIGNIFICANT: No clear relationship between ESG sentiment and abnormal returns")
    
    # VIX impact
    vix_coef = model.params['vix']
    vix_pvalue = model.pvalues['vix']
    print(f"\nVIX Impact: {vix_coef:.6f}")
    print(f"P-value: {vix_pvalue:.6f}")
    
    # Momentum impact
    momentum_coef = model.params['momentum']
    momentum_pvalue = model.pvalues['momentum']
    print(f"\nMomentum Impact: {momentum_coef:.6f}")
    print(f"P-value: {momentum_pvalue:.6f}")
    
    # Model fit
    print(f"\nR-squared: {model.rsquared:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    
    # Additional analysis by sentiment category
    print("\n" + "="*60)
    print("ANALYSIS BY SENTIMENT CATEGORY")
    print("="*60)
    
    sentiment_analysis = df_clean.groupby('sentiment_category')['abnormal_return'].agg(['mean', 'std', 'count'])
    print(sentiment_analysis)
    
    # Save results
    results = {
        'model_summary': str(model.summary()),
        'sentiment_coef': sentiment_coef,
        'sentiment_pvalue': sentiment_pvalue,
        'r_squared': model.rsquared,
        'n_observations': len(df_clean)
    }
    
    return model, results

def create_visualizations(df, model_results):
    """
    Create visualizations for the regression results
    """
    try:
        # Sentiment vs Abnormal Returns scatter plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.scatter(df['sentiment_score'], df['abnormal_return'], alpha=0.6)
        plt.xlabel('Sentiment Score')
        plt.ylabel('Abnormal Return')
        plt.title('Sentiment vs Abnormal Returns')
        
        # Add regression line
        x_range = np.linspace(df['sentiment_score'].min(), df['sentiment_score'].max(), 100)
        y_pred = model_results['sentiment_coef'] * x_range
        plt.plot(x_range, y_pred, 'r-', alpha=0.8)
        
        # Distribution of abnormal returns
        plt.subplot(2, 2, 2)
        plt.hist(df['abnormal_return'], bins=30, alpha=0.7)
        plt.xlabel('Abnormal Return')
        plt.ylabel('Frequency')
        plt.title('Distribution of Abnormal Returns')
        
        # Sentiment distribution
        plt.subplot(2, 2, 3)
        plt.hist(df['sentiment_score'], bins=30, alpha=0.7)
        plt.xlabel('Sentiment Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Sentiment Scores')
        
        # Box plot by sentiment category
        plt.subplot(2, 2, 4)
        df.boxplot(column='abnormal_return', by='sentiment_category', ax=plt.gca())
        plt.title('Abnormal Returns by Sentiment Category')
        plt.suptitle('')  # Remove default title
        
        plt.tight_layout()
        plt.savefig('regression_results.png', dpi=300, bbox_inches='tight')
        print("\nVisualizations saved as 'regression_results.png'")
        
    except Exception as e:
        print(f"Could not create visualizations: {e}")

if __name__ == "__main__":
    # Run the regression
    model, results = run_regression()
    
    # Create visualizations
    df = pd.read_csv("data/market_features.csv")
    df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
    df_clean = df.dropna(subset=["abnormal_return", "sentiment_score", "vix", "momentum"])
    create_visualizations(df_clean, results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
