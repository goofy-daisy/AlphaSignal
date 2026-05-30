import os
import time
import logging
from typing import List, Dict
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert
from src.data.database import engine, news_items, count_api_calls_today
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def fetch_news_for_ticker(ticker: str, api_key: str, days_back: int = 7) -> List[Dict]:
    """
    Fetches news from NewsAPI for a given ticker.
    """
    if not api_key:
        logger.error("fetch_news_for_ticker: NewsAPI key is missing.")
        return []
        
    calls_today = count_api_calls_today('NewsAPI')
    if calls_today >= 95:
        logger.warning(f"fetch_news_for_ticker: Approaching NewsAPI daily limit (100). Current calls today: {calls_today}")
    if calls_today >= 100:
        logger.error("fetch_news_for_ticker: NewsAPI daily limit reached.")

    clean_ticker = ticker.replace(".AX", "")
    
    newsapi = NewsApiClient(api_key=api_key)
    
    from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')
    
    articles_out = []
    
    try:
        response = newsapi.get_everything(
            q=clean_ticker,
            from_param=from_date,
            to=to_date,
            language='en',
            sort_by='publishedAt'
        )
        
        if response.get('status') != 'ok':
            logger.error(f"fetch_news_for_ticker: API error for {ticker}: {response}")
            return []
            
        articles = response.get('articles', [])
        for a in articles:
            published_at_str = a.get('publishedAt')
            try:
                published_at = datetime.strptime(published_at_str, '%Y-%m-%dT%H:%M:%SZ') if published_at_str else datetime.utcnow()
            except ValueError:
                published_at = datetime.utcnow()
            
            articles_out.append({
                "ticker": ticker,
                "headline": a.get('title', ''),
                "body": a.get('description', ''),
                "published_at": published_at,
                "source": a.get('source', {}).get('name', 'NewsAPI'),
                "url": a.get('url', '')
            })
            
        return articles_out
    except Exception as e:
        logger.error(f"fetch_news_for_ticker: Failed to fetch news for {ticker}: {e}")
        return []

def insert_news_items(ticker: str, articles: List[Dict]) -> int:
    """
    Inserts into news_items table, ignoring duplicate URLs.
    """
    if not articles:
        return 0
        
    stmt = pg_insert(news_items).values(articles)
    stmt = stmt.on_conflict_do_nothing(index_elements=['url'])
    
    try:
        with Session(engine) as session:
            result = session.execute(stmt)
            session.commit()
            return result.rowcount
    except Exception as e:
        logger.error(f"insert_news_items: Failed to insert news for {ticker}: {e}")
        return 0
