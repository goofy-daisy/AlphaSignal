import time
import logging
import requests
from typing import List, Dict
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert
from src.data.database import engine, filings
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def get_cik_for_ticker(ticker: str) -> str | None:
    """
    Fetches CIK number from SEC EDGAR company search.
    """
    url = f'https://efts.sec.gov/LATEST/search-index?q="{ticker}"&forms=10-K'
    headers = {'User-Agent': 'AlphaSignal research@alphasignal.com'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        time.sleep(0.12)
        
        data = response.json()
        hits = data.get("hits", {}).get("hits", [])
        if hits:
            cik_list = hits[0].get("_source", {}).get("ciks", [])
            if cik_list:
                return str(cik_list[0]).zfill(10)
        return None
    except Exception as e:
        logger.error(f"get_cik_for_ticker: Failed for {ticker}: {e}")
        time.sleep(0.12)
        return None

def get_recent_filings(ticker: str, days_back: int = 90) -> List[Dict]:
    """
    Fetches recent 10-K, 10-Q, 8-K filings for US ticker.
    """
    cik = get_cik_for_ticker(ticker)
    if not cik:
        return []
        
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {'User-Agent': 'AlphaSignal research@alphasignal.com'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        time.sleep(0.12)
        
        data = response.json()
        recent_filings = data.get("filings", {}).get("recent", {})
        if not recent_filings:
            return []
            
        forms = recent_filings.get("form", [])
        dates = recent_filings.get("filingDate", [])
        accessions = recent_filings.get("accessionNumber", [])
        docs = recent_filings.get("primaryDocument", [])
        
        from datetime import datetime, timedelta
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        results = []
        for i in range(len(forms)):
            form = forms[i]
            if form not in ["10-K", "10-Q", "8-K"]:
                continue
                
            filing_date = dates[i]
            if filing_date < cutoff_date:
                continue
                
            accession = str(accessions[i]).replace("-", "")
            doc = docs[i]
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
            
            results.append({
                "ticker": ticker,
                "filing_date": datetime.strptime(filing_date, "%Y-%m-%d").date(),
                "filing_type": form,
                "full_text": f"Filing {form} accessed at {filing_url}"[:8000],
                "url": filing_url
            })
            
        return results
    except Exception as e:
        logger.error(f"get_recent_filings: Failed for {ticker}: {e}")
        time.sleep(0.12)
        return []

def insert_filings(filings_list: List[Dict], source: str = 'edgar') -> int:
    """
    Inserts into filings table.
    """
    if not filings_list:
        return 0
        
    db_records = []
    for r in filings_list:
        db_records.append({
            "ticker": r.get("ticker"),
            "filing_date": r.get("filing_date"),
            "filing_type": r.get("filing_type"),
            "source": source,
            "headline": f"{r.get('filing_type')} filing",
            "body_text": r.get("full_text", ""),
            "url": r.get("url")
        })
        
    stmt = pg_insert(filings).values(db_records)
    stmt = stmt.on_conflict_do_nothing(index_elements=['ticker', 'filing_date', 'filing_type'])
    
    try:
        with Session(engine) as session:
            result = session.execute(stmt)
            session.commit()
            return result.rowcount
    except Exception as e:
        logger.error(f"insert_filings: Failed to insert: {e}")
        return 0