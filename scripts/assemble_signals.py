"""
AlphaSignal — assemble_signals.py
Runs all four signals for all 50 tickers and writes to signal_scores.
Run this after TFT training completes to populate signal_scores fully.
"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import logging
import yaml
import pandas as pd
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    with open('config/stock_universe.yaml', 'r') as f:
        universe = yaml.safe_load(f)
    tickers = universe.get('asx_tickers', []) + universe.get('sp500_tickers', [])
    logger.info(f"Processing {len(tickers)} tickers...")

    from src.signals.price_signal import compute_price_signal
    from src.signals.sentiment_signal import compute_sentiment_signal, write_signals_to_db
    from src.signals.filing_signal import compute_filing_signal
    from src.signals.social_signal import compute_social_signal

    results = []
    failed = []

    for i, ticker in enumerate(tickers):
        try:
            price = compute_price_signal(ticker)
            sentiment = compute_sentiment_signal(ticker)
            filing = compute_filing_signal(ticker)
            social = compute_social_signal(ticker)
            # Only pass price if non-zero (i.e. model exists)
            price_arg = price if price != 0.0 else None
            write_signals_to_db(ticker, sentiment=sentiment, filing=filing, social=social, price=price_arg)
            logger.info(f"[{i+1}/{len(tickers)}] {ticker}: price={price:.3f} sent={sentiment:.3f} fil={filing:.3f} soc={social:.3f}")
            results.append({'ticker': ticker, 'price': price, 'sentiment': sentiment, 'filing': filing, 'social': social})
        except Exception as e:
            logger.error(f"[{i+1}/{len(tickers)}] {ticker}: FAILED - {e}")
            failed.append(ticker)

    logger.info(f"Done. {len(results)} succeeded, {len(failed)} failed.")
    if failed:
        logger.warning(f"Failed tickers: {failed}")

    df = pd.DataFrame(results)
    df.to_csv('models/assembled_signals.csv', index=False)
    logger.info("Saved to models/assembled_signals.csv")

if __name__ == "__main__":
    main()
