import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import logging
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from sqlalchemy import text

from src.data.database import get_engine

logger = logging.getLogger(__name__)

_tokenizer = None
_model = None
_device = None

def _get_finbert():
    """Lazy loader — loads FinBERT once per process."""
    global _tokenizer, _model, _device
    if _model is None:
        _tokenizer, _model, _device = load_finbert()
    return _tokenizer, _model, _device


MODEL_NAME = "ProsusAI/finbert"

def load_finbert():
    """Load FinBERT model and tokenizer."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    return tokenizer, model, device

def score_text(text: str, tokenizer, model, device) -> float:
    """
    Runs FinBERT on a single text string.
    Returns float in [-1, 1]. Returns 0.0 on empty or error.
    Truncates text to 512 tokens.
    """
    if not text or not text.strip():
        return 0.0
    
    text = text.strip()
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            # index 0 = positive, index 1 = negative, index 2 = neutral
            score = probs[0, 0].item() - probs[0, 1].item()
            
        return max(-1.0, min(1.0, score))
    except Exception as e:
        logger.warning(f"Error scoring text: {e}")
        return 0.0

def score_texts_batch(texts: list, tokenizer, model, device, batch_size: int = 16) -> list:
    """
    Runs FinBERT on a list of texts in batches.
    Returns list of floats in [-1, 1], same length as input.
    Empty strings get score 0.0.
    """
    if not texts:
        return []

    scores = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Keep track of valid texts vs empty texts
        valid_indices = []
        valid_texts = []
        
        batch_scores = [0.0] * len(batch_texts)
        
        for j, text_val in enumerate(batch_texts):
            if text_val and text_val.strip():
                valid_indices.append(j)
                valid_texts.append(text_val.strip())
                
        if not valid_texts:
            scores.extend(batch_scores)
            continue
            
        try:
            inputs = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                
                for idx, valid_idx in enumerate(valid_indices):
                    score = probs[idx, 0].item() - probs[idx, 1].item()
                    batch_scores[valid_idx] = max(-1.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Error scoring batch on MPS. Falling back to CPU for this batch: {e}")
            try:
                cpu_device = torch.device('cpu')
                model_cpu = model.to(cpu_device)
                inputs = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = model_cpu(**inputs)
                    probs = F.softmax(outputs.logits, dim=-1)
                    
                    for idx, valid_idx in enumerate(valid_indices):
                        score = probs[idx, 0].item() - probs[idx, 1].item()
                        batch_scores[valid_idx] = max(-1.0, min(1.0, score))
                # Put model back to original device
                model.to(device)
            except Exception as e2:
                logger.warning(f"Error scoring batch on CPU fallback: {e2}")
                # Leave as 0.0
        
        scores.extend(batch_scores)
        
    return scores

def score_news_items(ticker: str = None) -> pd.DataFrame:
    """
    Loads news_items from DB for given ticker (or all if ticker=None).
    Runs FinBERT on title + ' ' + description (mapped to headline + ' ' + body based on schema).
    Returns DataFrame with columns: id, ticker, score, published_at.
    """
    engine = get_engine()
    
    query = "SELECT id, ticker, published_at, headline, body FROM news_items"
    params = {}
    if ticker:
        query += " WHERE ticker = :ticker"
        params['ticker'] = ticker
        
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
        
    if df.empty:
        return pd.DataFrame(columns=['id', 'ticker', 'score', 'published_at'])
        
    df['headline'] = df['headline'].fillna('')
    df['body'] = df['body'].fillna('')
    texts = (df['headline'] + ' ' + df['body']).tolist()
    
    tokenizer, model, device = _get_finbert()
    scores = score_texts_batch(texts, tokenizer, model, device)
    
    df['score'] = scores
    
    return df[['id', 'ticker', 'score', 'published_at']]

def score_filings(ticker: str = None) -> pd.DataFrame:
    """
    Loads filings from DB for given ticker (or all if ticker=None).
    Runs FinBERT on headline + ' ' + body_text.
    Returns DataFrame with columns: id, ticker, score, filing_date, source.
    """
    engine = get_engine()
    
    query = "SELECT id, ticker, filing_date, source, headline, body_text FROM filings"
    params = {}
    if ticker:
        query += " WHERE ticker = :ticker"
        params['ticker'] = ticker
        
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
        
    if df.empty:
        return pd.DataFrame(columns=['id', 'ticker', 'score', 'filing_date', 'source'])
        
    df['headline'] = df['headline'].fillna('')
    df['body_text'] = df['body_text'].fillna('')
    texts = (df['headline'] + ' ' + df['body_text']).tolist()
    
    tokenizer, model, device = _get_finbert()
    scores = score_texts_batch(texts, tokenizer, model, device)
    
    df['score'] = scores
    
    return df[['id', 'ticker', 'score', 'filing_date', 'source']]

def score_reddit_posts(ticker: str = None) -> pd.DataFrame:
    """
    Loads reddit_posts (Alpha Vantage data) from DB for given ticker (or all).
    Runs FinBERT on title + ' ' + body.
    Returns DataFrame with columns: id, ticker, score, published_at.
    """
    engine = get_engine()
    
    query = "SELECT id, ticker, published_at, title, body FROM reddit_posts"
    params = {}
    if ticker:
        query += " WHERE ticker = :ticker"
        params['ticker'] = ticker
        
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
        
    if df.empty:
        return pd.DataFrame(columns=['id', 'ticker', 'score', 'published_at'])
        
    df['title'] = df['title'].fillna('')
    df['body'] = df['body'].fillna('')
    texts = (df['title'] + ' ' + df['body']).tolist()
    
    tokenizer, model, device = _get_finbert()
    scores = score_texts_batch(texts, tokenizer, model, device)
    
    df['score'] = scores
    
    return df[['id', 'ticker', 'score', 'published_at']]