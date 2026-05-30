import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import yaml
import pandas as pd
from src.features.price_features import compute_features
from src.models.walk_forward import run_walk_forward

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Train TFT models using walk-forward validation for all tickers.")
    parser.parse_args()
    
    logger.info("Starting TFT walk-forward validation for all tickers...")
    
    with open('config/stock_universe.yaml', 'r') as f:
        universe = yaml.safe_load(f)
        
    tickers = universe.get('asx_tickers', []) + universe.get('us_tickers', universe.get('sp500_tickers', []))
    
    results = []
    
    for ticker in tickers:
        logger.info(f"Processing ticker: {ticker}")
        try:
            df = compute_features(ticker)
            if df.empty:
                logger.warning(f"No features available for {ticker}, skipping.")
                continue
                
            wf_res = run_walk_forward(ticker, df)
            if wf_res:
                results.append({
                    'ticker': ticker,
                    'mean_ic': wf_res['mean_ic'],
                    'n_windows': len(wf_res['windows']),
                    'best_checkpoint': wf_res['best_checkpoint']
                })
            else:
                logger.warning(f"Walk-forward failed or returned empty for {ticker}.")
                
        except Exception as e:
            logger.exception(f"Exception processing {ticker}: {e}")
            
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        os.makedirs('models', exist_ok=True)
        res_df.to_csv('models/walk_forward_results.csv', index=False)
        
        print("\n--- Training Summary ---")
        print(res_df.to_string(index=False))
        print("------------------------")
    else:
        logger.warning("No results to save.")
        
if __name__ == "__main__":
    main()
