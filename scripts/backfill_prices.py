import sys
import os
import time
import logging
import yaml
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.price_fetcher import fetch_historical, insert_price_history

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        with open('config/stock_universe.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        tickers = []
        if 'asx_tickers' in config:
            tickers.extend(config['asx_tickers'])
        if 'sp500_tickers' in config:
            tickers.extend(config['sp500_tickers'])
            
        success_count = 0
        failure_count = 0
        total_rows = 0
        
        batch_size = 10
        
        pbar = tqdm(total=len(tickers), desc="Tickers")
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            for ticker in batch:
                try:
                    df = fetch_historical(ticker, years=5)
                    if not df.empty:
                        inserted = insert_price_history(ticker, df)
                        total_rows += inserted
                        success_count += 1
                    else:
                        failure_count += 1
                except Exception as e:
                    logger.error(f"Error processing ticker {ticker}: {e}")
                    failure_count += 1
                pbar.update(1)
                    
            if i + batch_size < len(tickers):
                time.sleep(5)
        pbar.close()
                
        logger.info(f"Backfill Complete. Total tickers: {len(tickers)}, Success: {success_count}, Failures: {failure_count}, Total rows inserted: {total_rows}")
    except Exception as e:
        logger.error(f"backfill_prices main failed: {e}")

if __name__ == "__main__":
    main()
