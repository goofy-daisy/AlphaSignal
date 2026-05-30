import os
import logging
import pickle
import faiss
import numpy as np 
import pandas as pd
# from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from src.data.database import get_engine
from src.models.finbert_pipeline import score_news_items
 
logger = logging.getLogger(__name__)

# Lazy loading of embedding model
_embedding_model = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _embedding_model

def build_faiss_index(ticker: str = None) -> tuple:
    """
    Builds FAISS index from news_items in DB.
    If ticker provided, builds index for that ticker only.
    If ticker=None, builds index for all tickers.
    Returns (index, metadata_list) where:
        index: faiss.IndexFlatL2 with dimension 384
        metadata_list: list of dicts with keys: id, ticker, title, published_at
    Saves index to models/faiss_{ticker}.index or models/faiss_all.index
    Saves metadata to models/faiss_{ticker}_meta.pkl or models/faiss_all_meta.pkl
    """
    engine = get_engine()
    
    query = "SELECT id, ticker, published_at, headline as title, body FROM news_items"
    params = {}
    if ticker:
        query += " WHERE ticker = :ticker"
        params['ticker'] = ticker
        
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
        
    suffix = ticker if ticker else "all"
    index_path = f"models/faiss_{suffix}.index"
    meta_path = f"models/faiss_{suffix}_meta.pkl"
    
    dimension = 384
    index = faiss.IndexFlatL2(dimension)
    metadata_list = []
    
    if df.empty:
        logger.warning(f"No news items found for {suffix}. Returning empty index.")
        faiss.write_index(index, index_path)
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata_list, f)
        return index, metadata_list
        
    df['title'] = df['title'].fillna('')
    df['body'] = df['body'].fillna('')
    texts_to_embed = (df['title'] + ' ' + df['body']).tolist()
    
    model = _get_embedding_model()
    embeddings = model.encode(texts_to_embed, convert_to_numpy=True)
    
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    for _, row in df.iterrows():
        metadata_list.append({
            'id': row['id'],
            'ticker': row['ticker'],
            'title': row['title'],
            'published_at': row['published_at']
        })
        
    faiss.write_index(index, index_path)
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata_list, f)
        
    return index, metadata_list

def load_faiss_index(ticker: str = None) -> tuple:
    """
    Loads saved FAISS index and metadata from disk.
    Returns (index, metadata_list) or (None, []) if not found.
    """
    suffix = ticker if ticker else "all"
    index_path = f"models/faiss_{suffix}.index"
    meta_path = f"models/faiss_{suffix}_meta.pkl"
    
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return None, []
        
    try:
        index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            metadata_list = pickle.load(f)
        return index, metadata_list
    except Exception as e:
        logger.warning(f"Failed to load FAISS index for {suffix}: {e}")
        return None, []

def retrieve_similar_news(query_text: str, ticker: str = None, top_k: int = 5) -> list:
    """
    Embeds query_text and retrieves top_k most similar news items.
    Returns list of dicts with keys: ticker, title, published_at, distance.
    Loads index from disk if not already loaded.
    """
    if not query_text or not query_text.strip():
        return []
        
    index, metadata_list = load_faiss_index(ticker)
    if index is None or index.ntotal == 0:
        return []
        
    model = _get_embedding_model()
    query_emb = model.encode([query_text], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    
    # We want top_k
    k = min(top_k, index.ntotal)
    distances, indices = index.search(query_emb, k)
    
    results = []
    for i in range(k):
        idx = indices[0][i]
        if idx < 0 or idx >= len(metadata_list):
            continue
        meta = metadata_list[idx]
        results.append({
            'ticker': meta['ticker'],
            'title': meta['title'],
            'published_at': meta['published_at'],
            'distance': float(distances[0][i])
        })
        
    return results

def get_ticker_news_embedding_score(ticker: str) -> float:
    """
    Returns average embedding-based sentiment score for a ticker.
    Uses the FinBERT scores from news already stored, not raw embeddings.
    This is a helper for the sentiment signal.
    Returns 0.0 if no data.
    """
    try:
        # Since FinBERT scores are not persistently stored, we compute them on the fly
        df = score_news_items(ticker)
        if df.empty:
            return 0.0
        return float(df['score'].mean())
    except Exception as e:
        logger.warning(f"Error computing ticker news embedding score for {ticker}: {e}")
        return 0.0