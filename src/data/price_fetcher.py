import time
import logging
from typing import Dict, List
from datetime import date, timedelta
import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert
from src.data.database import engine, price_history
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def validate_ohlcv(df: pd.DataFrame) -> bool:
    """
    Returns True if DataFrame has all required columns, no NaN in Close, at least 100 rows.
    """
    required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_cols.issubset(set(df.columns)):
        return False
        
    if df['Close'].isna().any():
        return False
        
    if len(df) < 100:
        return False
        
    return True

def fetch_historical(ticker: str, years: int = 5) -> pd.DataFrame:
    """
    Downloads `years` years of daily OHLCV data for ticker using yfinance.
    """
    retries = 3
    period = f"{years}y"
    for attempt in range(retries):
        try:
            df = yf.download(ticker, period=period, interval="1d", auto_adjust=True)
            if df.empty:
                logger.warning(f"fetch_historical: No data found for {ticker}.")
                return pd.DataFrame()
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            cols_map = {c: c.capitalize() for c in df.columns if c.lower() in ['open', 'high', 'low', 'close', 'volume']}
            df = df.rename(columns=cols_map)
            
            df = df.dropna(subset=['Close'])
            
            if not validate_ohlcv(df):
                logger.warning(f"fetch_historical: Data validation failed for {ticker} (less than 100 rows or missing cols).")
            
            return df
            
        except Exception as e:
            if "YFRateLimitError" in str(type(e).__name__) or "RateLimit" in str(e):
                logger.warning(f"Rate limit error for {ticker}. Waiting 60 seconds (attempt {attempt + 1}/{retries}).")
                time.sleep(60)
            else:
                logger.error(f"Error fetching historical data for {ticker}: {e}")
                break
                
    logger.error(f"fetch_historical: Failed to fetch data for {ticker} after retries.")
    return pd.DataFrame()

def fetch_latest(ticker: str) -> pd.DataFrame:
    """
    Downloads last 5 days of data for ticker.
    """
    try:
        df = yf.download(ticker, period="5d", interval="1d", auto_adjust=True)
        if df.empty:
            logger.warning(f"fetch_latest: No data found for {ticker}.")
            return pd.DataFrame()
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        cols_map = {c: c.capitalize() for c in df.columns if c.lower() in ['open', 'high', 'low', 'close', 'volume']}
        df = df.rename(columns=cols_map)
        df = df.dropna(subset=['Close'])
        
        return df
    except Exception as e:
        logger.error(f"Error fetching latest data for {ticker}: {e}")
        return pd.DataFrame()

def fetch_all_tickers(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Calls fetch_historical for each ticker in list in batches of 10.
    """
    results = {}
    success_count = 0
    failure_count = 0
    
    batch_size = 10
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        for ticker in batch:
            df = fetch_historical(ticker)
            if not df.empty:
                results[ticker] = df
                success_count += 1
            else:
                failure_count += 1
        
        if i + batch_size < len(tickers):
            time.sleep(2)
            
    logger.info(f"fetch_all_tickers: {success_count} succeeded, {failure_count} failed.")
    return results

def insert_price_history(ticker: str, df: pd.DataFrame) -> int:
    """
    Inserts DataFrame rows into price_history table.
    """
    if df.empty:
        return 0
        
    records = []
    for date_val, row in df.iterrows():
        records.append({
            "ticker": ticker,
            "date": date_val.date() if isinstance(date_val, pd.Timestamp) else date_val,
            "open": float(row['Open']),
            "high": float(row['High']),
            "low": float(row['Low']),
            "close": float(row['Close']),
            "volume": int(row['Volume'])
        })
        
    if not records:
        return 0
        
    stmt = pg_insert(price_history).values(records)
    stmt = stmt.on_conflict_do_nothing(constraint="uq_price_ticker_date")
    
    try:
        with Session(engine) as session:
            result = session.execute(stmt)
            session.commit()
            return result.rowcount
    except Exception as e:
        logger.error(f"insert_price_history: Failed to insert rows for {ticker}: {e}")
        return 0
