import sys
import os
import time
import logging
import yaml
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.edgar_fetcher import get_recent_filings, insert_filings
from src.data.asx_fetcher import fetch_announcements, insert_asx_filings

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
            
        total_asx_inserted = 0
        total_us_inserted = 0
        
        for ticker in tqdm(tickers, desc="Backfilling filings"):
            try:
                if ticker.endswith('.AX'):
                    filings = fetch_announcements(ticker)
                    if filings:
                        inserted = insert_asx_filings(filings)
                        total_asx_inserted += inserted
                else:
                    filings = get_recent_filings(ticker, days_back=730)
                    if filings:
                        inserted = insert_filings(filings)
                        total_us_inserted += inserted
            except Exception as e:
                logger.error(f"Error processing ticker {ticker}: {e}")
                    
            time.sleep(2)
            
        logger.info(f"Backfill Complete. Total ASX filings inserted: {total_asx_inserted}, Total US filings inserted: {total_us_inserted}")
    except Exception as e:
        logger.error(f"backfill_filings main failed: {e}")

if __name__ == "__main__":
    main()
