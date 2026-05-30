import os
import time
import logging
import requests
from typing import List, Dict
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert
from src.data.database import engine, reddit_posts
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Module-level variable to track calls
_calls_today = 0
_last_call_date = None

def fetch_posts_for_ticker(ticker: str, limit: int = 50) -> List[Dict]:
    """
    Calls Alpha Vantage NEWS_SENTIMENT endpoint.
    """
    global _calls_today, _last_call_date
    
    today = datetime.now().date()
    if _last_call_date != today:
        _calls_today = 0
        _last_call_date = today
        
    if _calls_today >= 25:
        logger.error("fetch_posts_for_ticker: Alpha Vantage daily limit (25) reached.")
        return []

    api_key = os.getenv("ALPHA_VANTAGE_KEY")
    if not api_key:
        logger.error("fetch_posts_for_ticker: ALPHA_VANTAGE_KEY is missing.")
        return []
        
    clean_ticker = ticker.replace(".AX", "")
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={clean_ticker}&apikey={api_key}&limit={limit}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        _calls_today += 1
        time.sleep(12) # Respect rate limit
        
        feed = data.get("feed", [])
        posts = []
        for item in feed:
            # Parse time_published
            time_pub_str = item.get("time_published", "")
            try:
                created_utc = datetime.strptime(time_pub_str, '%Y%m%dT%H%M%S')
            except ValueError:
                created_utc = datetime.utcnow()
                
            # Scale sentiment score -100 to 100
            score_float = item.get("overall_sentiment_score", 0.0)
            score_int = int(score_float * 100)
            
            posts.append({
                "ticker": ticker,
                "title": item.get("title", ""),
                "body": item.get("summary", ""),
                "score": score_int,
                "created_utc": created_utc,
                "subreddit": "alpha_vantage"
            })
            
        return posts
        
    except requests.exceptions.RequestException as e:
        logger.error(f"fetch_posts_for_ticker: Network error fetching for {ticker}: {e}")
        time.sleep(12)
        return []
    except Exception as e:
        logger.error(f"fetch_posts_for_ticker: Error processing data for {ticker}: {e}")
        time.sleep(12)
        return []

def insert_reddit_posts(ticker: str, posts: List[Dict]) -> int:
    """
    Inserts into reddit_posts table.
    """
    if not posts:
        return 0
        
    records = []
    for p in posts:
        records.append({
            "ticker": p.get("ticker", ticker),
            "published_at": p.get("created_utc"),
            "title": p.get("title"),
            "body": p.get("body"),
            "source": p.get("subreddit"),
            "sentiment_score": float(p.get("score", 0)) / 100.0
        })
        
    if not records:
        return 0

    stmt = pg_insert(reddit_posts).values(records)
    stmt = stmt.on_conflict_do_nothing()
    
    try:
        with Session(engine) as session:
            result = session.execute(stmt)
            session.commit()
            return result.rowcount
    except Exception as e:
        logger.error(f"insert_reddit_posts: Failed to insert posts for {ticker}: {e}")
        return 0