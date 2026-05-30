import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import logging
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from sqlalchemy import text

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
        _tokenizer, _model, _device = load_finbert()
    return _tokenizer, _model, _device

def fetch_yfinance_news(ticker: str) -> list:
    """Fetches recent news from yfinance for a ticker. Returns list of dicts."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        news = t.news
        if not news:
            return []
        results = []
        for item in news:
            content_dict = item.get('content', {})
            title = content_dict.get('title', '') or item.get('title', '') or ''
            summary = content_dict.get('summary', '') or ''
            results.append({'title': title, 'summary': summary})
        return results
    except Exception as e:
        logger.warning(f"yfinance news fetch failed for {ticker}: {e}")
        return []

def compute_social_signal(ticker: str) -> float:
    """
    Computes social signal for ticker.
    For all tickers: scores yfinance .news with FinBERT (primary source).
    For US tickers additionally: loads all reddit_posts from DB and blends.
    Blend weights: yfinance 0.7, reddit 0.3 for US tickers.
    Returns float in [-1, 1]. Returns 0.0 if no data.
    """
    tokenizer, model, device = _get_finbert()
    is_asx = ticker.upper().endswith('.AX')

    # yfinance news — primary for all tickers
    yf_news = fetch_yfinance_news(ticker)
    mean_yf_score = None
    if yf_news:
        texts = [(item['title'] + ' ' + item['summary']).strip() for item in yf_news]
        scores = score_texts_batch(texts, tokenizer, model, device)
        if scores:
            mean_yf_score = float(np.mean(scores))

    # Reddit posts — US tickers only, no date filter (use all available)
    mean_reddit_score = None
    if not is_asx:
        engine = get_engine()
        query = "SELECT title, body FROM reddit_posts WHERE ticker = :ticker"
        with engine.connect() as conn:
            df_reddit = pd.read_sql(text(query), conn, params={'ticker': ticker})
        if not df_reddit.empty:
            df_reddit['title'] = df_reddit['title'].fillna('')
            df_reddit['body'] = df_reddit['body'].fillna('')
            texts_r = (df_reddit['title'] + ' ' + df_reddit['body']).tolist()
            scores_r = score_texts_batch(texts_r, tokenizer, model, device)
            if scores_r:
                mean_reddit_score = float(np.mean(scores_r))

    # Combine
    if mean_yf_score is not None and mean_reddit_score is not None:
        signal = 0.7 * mean_yf_score + 0.3 * mean_reddit_score
    elif mean_yf_score is not None:
        signal = mean_yf_score
    elif mean_reddit_score is not None:
        signal = mean_reddit_score
    else:
        logger.warning(f"No social data found for {ticker}, returning 0.0")
        return 0.0

    return float(np.clip(signal, -1.0, 1.0))

def compute_all_social_signals(tickers: list = None) -> pd.DataFrame:
    """
    Returns DataFrame with columns: ticker, social_signal, computed_at.
    If tickers is None, loads from config/stock_universe.yaml.
    """
    if not tickers:
        import yaml
        with open('config/stock_universe.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
            tickers = cfg.get('asx_tickers', []) + cfg.get('sp500_tickers', [])
    results = []
    now = datetime.now(timezone.utc)
    for t in tickers:
        sig = compute_social_signal(t)
        results.append({'ticker': t, 'social_signal': sig, 'computed_at': now})
    return pd.DataFrame(results)