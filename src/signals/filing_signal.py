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

def compute_filing_signal(ticker: str, lookback_days: int = 90) -> float:
    """Compute filing signal for a ticker.

    For ASX tickers (ending with '.AX') we only consider filings where ``source='asx'``.
    For US tickers we consider filings where ``source='edgar'``.
    Only filings whose ``filing_date`` is within the last ``lookback_days`` are used.
    The score is the mean FinBERT sentiment score across all selected filings.
    Returns ``0.0`` if no filings are found.
    """
    engine = get_engine()
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    source_filter = 'asx' if ticker.upper().endswith('.AX') else 'edgar'

    query = """
        SELECT headline, body_text FROM filings
        WHERE ticker = :ticker AND source = :source AND filing_date >= :cutoff
    """
    params = {'ticker': ticker, 'source': source_filter, 'cutoff': cutoff_date}
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
    if df.empty:
        logger.info(f"No filings found for {ticker} within lookback; returning 0.0 signal.")
        return 0.0
    df['headline'] = df['headline'].fillna('')
    df['body_text'] = df['body_text'].fillna('')
    texts = (df['headline'] + ' ' + df['body_text']).tolist()
    tokenizer, model, device = _get_finbert()
    scores = score_texts_batch(texts, tokenizer, model, device)
    if not scores:
        return 0.0
    mean_score = float(np.mean(scores))
    return float(np.clip(mean_score, -1.0, 1.0))

def compute_all_filing_signals(tickers: list = None, lookback_days: int = 90) -> pd.DataFrame:
    """Compute filing signals for a list of tickers.

    Returns a DataFrame with columns ``ticker``, ``filing_signal`` and ``computed_at``.
    If ``tickers`` is ``None`` the universe is loaded from ``config/stock_universe.yaml``.
    """
    if not tickers:
        import yaml
        with open('config/stock_universe.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
            tickers = cfg.get('asx_tickers', []) + cfg.get('sp500_tickers', [])
    results = []
    now = datetime.now(timezone.utc)
    for t in tickers:
        sig = compute_filing_signal(t, lookback_days)
        results.append({'ticker': t, 'filing_signal': sig, 'computed_at': now})
    return pd.DataFrame(results)