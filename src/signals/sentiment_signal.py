import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import logging
from datetime import datetime, timedelta, timezone, date
import pandas as pd
import numpy as np
from sqlalchemy import text, MetaData
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.data.database import get_engine
from src.models.finbert_pipeline import load_finbert, score_texts_batch

logger = logging.getLogger(__name__)

_tokenizer = None
_model = None
_device = None

def _get_finbert():
    """Lazy loader for FinBERT model."""
    global _tokenizer, _model, _device
    if _model is None:
        from src.models.finbert_pipeline import load_finbert
        _tokenizer, _model, _device = load_finbert()
    return _tokenizer, _model, _device

def compute_sentiment_signal(ticker: str, lookback_days: int = 30) -> float:
    """
    Computes sentiment signal for ticker from recent news and reddit posts.
    lookback_days: only consider items published in the last N days.
    Weights: news_items score weighted 0.6, reddit_posts score weighted 0.4.
    Returns float in [-1, 1]. Returns 0.0 if no data in lookback window.
    """
    engine = get_engine()
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    
    # 1. News items
    query_news = """
        SELECT headline, body FROM news_items
        WHERE ticker = :ticker AND published_at >= :cutoff
    """
    with engine.connect() as conn:
        df_news = pd.read_sql(text(query_news), conn, params={'ticker': ticker, 'cutoff': cutoff_date})
    
    mean_news_score = None
    if not df_news.empty:
        df_news['headline'] = df_news['headline'].fillna('')
        df_news['body'] = df_news['body'].fillna('')
        texts_news = (df_news['headline'] + ' ' + df_news['body']).tolist()
        
        tokenizer, model, device = _get_finbert()
        scores_news = score_texts_batch(texts_news, tokenizer, model, device)
        if scores_news:
            mean_news_score = np.mean(scores_news)
            
    # 2. Reddit posts
    query_reddit = """
        SELECT title, body FROM reddit_posts
        WHERE ticker = :ticker AND published_at >= :cutoff
    """
    with engine.connect() as conn:
        df_reddit = pd.read_sql(text(query_reddit), conn, params={'ticker': ticker, 'cutoff': cutoff_date})
        
    mean_reddit_score = None
    if not df_reddit.empty:
        df_reddit['title'] = df_reddit['title'].fillna('')
        df_reddit['body'] = df_reddit['body'].fillna('')
        texts_reddit = (df_reddit['title'] + ' ' + df_reddit['body']).tolist()
        
        tokenizer, model, device = _get_finbert()
        scores_reddit = score_texts_batch(texts_reddit, tokenizer, model, device)
        if scores_reddit:
            mean_reddit_score = np.mean(scores_reddit)
            
    # Combine
    if mean_news_score is not None and mean_reddit_score is not None:
        signal = 0.6 * mean_news_score + 0.4 * mean_reddit_score
    elif mean_news_score is not None:
        signal = mean_news_score
    elif mean_reddit_score is not None:
        signal = mean_reddit_score
    else:
        return 0.0
        
    return float(np.clip(signal, -1.0, 1.0))

def compute_all_sentiment_signals(tickers: list = None, lookback_days: int = 30) -> pd.DataFrame:
    """
    Returns DataFrame with columns: ticker, sentiment_signal, computed_at.
    """
    if not tickers:
        import yaml
        with open('config/stock_universe.yaml', 'r') as f:
            config = yaml.safe_load(f)
            tickers = config.get('asx_tickers', []) + config.get('sp500_tickers', [])
            
    results = []
    now = datetime.now(timezone.utc)
    for ticker in tickers:
        sig = compute_sentiment_signal(ticker, lookback_days)
        results.append({
            'ticker': ticker,
            'sentiment_signal': sig,
            'computed_at': now
        })
        
    return pd.DataFrame(results)

def _get_signal_scores_table(engine):
    meta = MetaData()
    meta.reflect(bind=engine, only=['signal_scores'])
    return meta.tables['signal_scores']

def write_signals_to_db(ticker: str, sentiment: float, filing: float, social: float, price: float = None):
    """
    Upserts all signals for ticker into signal_scores table for today's date.
    """
    today = date.today()
    now = datetime.now(timezone.utc)
    
    # Compute composite
    signals = {}
    if price is not None:
        signals['price'] = (price, 0.4)
    signals['sentiment'] = (sentiment, 0.3)
    signals['filing'] = (filing, 0.15)
    signals['social'] = (social, 0.15)
    
    total_weight = sum(w for _, w in signals.values())
    if total_weight > 0:
        composite = sum(v * w for v, w in signals.values()) / total_weight
        composite = float(np.clip(composite, -1.0, 1.0))
    else:
        composite = 0.0
    
    engine = get_engine()
    signal_scores_table = _get_signal_scores_table(engine)
    
    with engine.begin() as conn:
        stmt = pg_insert(signal_scores_table).values(
            ticker=ticker,
            date=today,
            price_signal=price,
            sentiment_signal=float(sentiment),
            filing_signal=float(filing),
            social_signal=float(social),
            composite_score=composite, # db column is composite_score
            created_at=now
        ).on_conflict_do_update(
            index_elements=['ticker', 'date'],
            set_=dict(
                price_signal=price,
                sentiment_signal=float(sentiment),
                filing_signal=float(filing),
                social_signal=float(social),
                composite_score=composite, # db column is composite_score
                created_at=now
            )
        )
        conn.execute(stmt)
