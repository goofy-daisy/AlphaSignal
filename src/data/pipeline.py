import os
import yaml
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

from src.data.price_fetcher import fetch_all_tickers, insert_price_history
from src.data.news_fetcher import fetch_news_for_ticker, insert_news_items
from src.data.reddit_fetcher import fetch_posts_for_ticker, insert_reddit_posts
from src.data.asx_fetcher import fetch_announcements
from src.data.edgar_fetcher import get_recent_filings, insert_filings

load_dotenv()
logger = logging.getLogger(__name__)

def load_tickers():
    try:
        with open('config/stock_universe.yaml', 'r') as f:
            config = yaml.safe_load(f)
        tickers = []
        if 'asx_tickers' in config:
            tickers.extend(config['asx_tickers'])
        if 'sp500_tickers' in config:
            tickers.extend(config['sp500_tickers'])
        return tickers[:50]
    except Exception as e:
        logger.error(f"load_tickers failed: {e}")
        return []

def run_price_job() -> None:
    try:
        tickers = load_tickers()
        results = fetch_all_tickers(tickers)
        total_inserted = 0
        for ticker, df in results.items():
            inserted = insert_price_history(ticker, df)
            total_inserted += inserted
        logger.info(f"run_price_job: Total rows inserted: {total_inserted}")
    except Exception as e:
        logger.error(f"run_price_job failed: {e}")

def run_news_job() -> None:
    try:
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            logger.error("run_news_job: NEWS_API_KEY missing.")
            return
            
        tickers = load_tickers()
        total_inserted = 0
        for ticker in tickers:
            articles = fetch_news_for_ticker(ticker, api_key)
            if articles:
                inserted = insert_news_items(ticker, articles)
                total_inserted += inserted
        logger.info(f"run_news_job: Total articles inserted: {total_inserted}")
    except Exception as e:
        logger.error(f"run_news_job failed: {e}")

def run_social_job() -> None:
    try:
        tickers = load_tickers()
        total_inserted = 0
        for ticker in tickers:
            posts = fetch_posts_for_ticker(ticker)
            if posts:
                inserted = insert_reddit_posts(ticker, posts)
                total_inserted += inserted
        logger.info(f"run_social_job: Total posts inserted: {total_inserted}")
    except Exception as e:
        logger.error(f"run_social_job failed: {e}")

def run_filings_job() -> None:
    try:
        tickers = load_tickers()
        total_inserted = 0
        for ticker in tickers:
            if ticker.endswith('.AX'):
                filings = fetch_announcements(ticker)
                if filings:
                    inserted = insert_filings(filings) # Changed to generic insert
                    total_inserted += inserted
            else:
                filings = get_recent_filings(ticker, days_back=90)
                if filings:
                    inserted = insert_filings(filings)
                    total_inserted += inserted
        logger.info(f"run_filings_job: Total filings inserted: {total_inserted}")
    except Exception as e:
        logger.error(f"run_filings_job failed: {e}")

def run_signal_job():
    """Assembles all signals for all tickers and writes to signal_scores."""
    import yaml
    from src.signals.price_signal import compute_price_signal
    from src.signals.sentiment_signal import compute_sentiment_signal, write_signals_to_db
    from src.signals.filing_signal import compute_filing_signal
    from src.signals.social_signal import compute_social_signal
    
    with open('config/stock_universe.yaml', 'r') as f:
        universe = yaml.safe_load(f)
    tickers = universe.get('asx_tickers', []) + universe.get('sp500_tickers', [])
    
    for ticker in tickers:
        try:
            price = compute_price_signal(ticker)
            sentiment = compute_sentiment_signal(ticker)
            filing = compute_filing_signal(ticker)
            social = compute_social_signal(ticker)
            write_signals_to_db(ticker, sentiment=sentiment, filing=filing, social=social, price=price if price != 0.0 else None)
        except Exception as e:
            logger.error(f"Signal job failed for {ticker}: {e}")

def run_faiss_job():
    """Rebuilds FAISS indexes for all tickers."""
    import yaml
    from src.embeddings.faiss_store import build_faiss_index
    
    with open('config/stock_universe.yaml', 'r') as f:
        universe = yaml.safe_load(f)
    tickers = universe.get('asx_tickers', []) + universe.get('sp500_tickers', [])
    
    for ticker in tickers:
        try:
            index, meta = build_faiss_index(ticker)
            logger.info(f"FAISS rebuilt for {ticker}: {index.ntotal} vectors")
        except Exception as e:
            logger.error(f"FAISS job failed for {ticker}: {e}")

from apscheduler.triggers.cron import CronTrigger

scheduler = BackgroundScheduler()
scheduler.add_job(run_price_job, CronTrigger(hour=18, minute=0))
scheduler.add_job(run_news_job, CronTrigger(hour=19, minute=0))
scheduler.add_job(run_social_job, CronTrigger(hour=19, minute=30))
scheduler.add_job(run_signal_job, CronTrigger(hour=20, minute=0))
scheduler.add_job(run_faiss_job, CronTrigger(hour=21, minute=0))

def start_scheduler():
    """Starts the background scheduler."""
    scheduler.start()
    logger.info("APScheduler started with 5 jobs")

def stop_scheduler():
    """Stops the background scheduler."""
    scheduler.shutdown()
    logger.info("APScheduler stopped")