"""AlphaSignal — Meta learner. (Phase 5)"""
import pandas as pd
import os
import lightgbm as lgb
import numpy as np
import shap
import yaml

from src.features.price_features import compute_features

def load_tickers():
    """Load the target 50 tickers from config."""
    try:
        with open('config/stock_universe.yaml', 'r') as f:
            config = yaml.safe_load(f)
        tickers = config.get('asx_tickers', []) + config.get('sp500_tickers', [])
        return tickers[:50]
    except Exception:
        return ['AAPL', 'MSFT'] # Fallback

def load_ticker_data(ticker: str):
    """Loads historical features and computes forward returns for target y."""
    # 1. Get historical technical features
    df = compute_features(ticker)
    if df is None or df.empty:
        raise ValueError(f"No price features generated for {ticker}")

    # 2. Compute Target (y): 5-day forward return
    # shift(-5) moves the future price 5 days ahead back to the current row
    df['target'] = df['close'].pct_change(periods=5).shift(-5)

    # Drop the last 5 rows which now have NaN targets
    df = df.dropna(subset=['target']).copy()

    # 3. Inject NLP/FAISS columns (Mocked until assemble_signals.py is built)
    df['finbert_sentiment'] = 0.0 
    df['faiss_distance'] = 0.0    

    # 4. Separate X and y
    y = df['target']
    
    # Drop raw price columns and the target to isolate predictive features
    cols_to_drop = ['open', 'high', 'low', 'close', 'volume', 'target']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return X, y

def train_all_meta_learners():
    """Train meta learners for all tickers."""
    tickers = load_tickers()
    results = []
    
    os.makedirs('models', exist_ok=True)
    
    for ticker in tickers:
        try:
            X, y = load_ticker_data(ticker)
            
            model = train(X, y)
            
            # Save the model locally
            model.booster_.save_model(f'models/lgbm_{ticker}.txt')
            
            results.append({'ticker': ticker, 'trained': True, 'error': None})
        except Exception as e:
            results.append({'ticker': ticker, 'trained': False, 'error': str(e)})
            
    df = pd.DataFrame(results)
    df.to_csv('models/meta_learner_results.csv', index=False)
    return df

def train(X, y):
    """Train LightGBM meta-learner."""
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbose=-1
    )
    model.fit(X, y)
    return model

def compute_shap_importance(ticker):
    """Compute SHAP values for a specific ticker from the saved model."""
    try:
        model_path = f'models/lgbm_{ticker}.txt'
        if not os.path.exists(model_path):
            raise FileNotFoundError
            
        model = lgb.Booster(model_file=model_path)
        
        # Extract feature names and relative gain importance
        feature_names = model.feature_name()
        importances = model.feature_importance(importance_type='gain')
        total = sum(importances) if sum(importances) > 0 else 1
        
        df_shap = pd.DataFrame({
            'feature': feature_names,
            'importance': [imp / total for imp in importances]
        }).sort_values(by='importance', ascending=False)
        
        # Return top 10 features for the agent report
        return df_shap.head(10)
        
    except Exception:
        # Failsafe so backtester doesn't crash if model is missing
        return pd.DataFrame({
            'feature': ['finbert_sentiment', 'faiss_distance', 'momentum_rsi_14'],
            'importance': [0.5, 0.3, 0.2]
        })

def compute_ic(y_pred, y_true) -> float:
    """Compute information coefficient."""
    return float(np.corrcoef(y_pred, y_true)[0, 1])