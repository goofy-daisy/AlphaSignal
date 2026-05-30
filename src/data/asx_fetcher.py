"""
AlphaSignal — asx_fetcher module.

Fetches ASX company announcements using the Markit Digital API
(the same API that powers asx.com.au company pages).

API pattern: https://asx.api.markitdigital.com/asx-research/1.0/companies/{TICKER}/announcements
Ticker format: strip .AX suffix (BHP.AX -> BHP, CBA.AX -> CBA)
Document text: headlines only — PDFs are behind auth, not fetched.

Implemented in Phase 2 (updated with working endpoint).
"""

import logging
import time
import requests

logger = logging.getLogger(__name__)

# Working headers required by the Markit Digital API
_HEADERS = {
    "Accept": "application/json",
    "Origin": "https://www.asx.com.au",
    "Referer": "https://www.asx.com.au/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

_BASE_URL = "https://asx.api.markitdigital.com/asx-research/1.0/companies/{ticker}/announcements"


def fetch_announcements(ticker: str, items_per_page: int = 20) -> list[dict]:
    """Fetch recent ASX announcements for a given ticker.

    Parameters
    ----------
    ticker:
        ASX ticker with or without .AX suffix (e.g. 'BHP.AX' or 'BHP').
    items_per_page:
        Number of announcements to fetch (default 20).

    Returns
    -------
    list[dict]
        List of announcement dicts compatible with the filings table schema:
        ticker, filing_date, filing_type, source, headline, body_text, url.
        Returns empty list on any error.
    """
    # Strip .AX suffix for the API call
    clean_ticker = ticker.replace(".AX", "").upper()
    url = _BASE_URL.format(ticker=clean_ticker)

    try:
        response = requests.get(
            url,
            headers=_HEADERS,
            params={"itemsPerPage": items_per_page},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        items = data.get("data", {}).get("items", [])
        results = []

        for item in items:
            results.append({
                "ticker": ticker,  # store with .AX suffix for consistency
                "filing_date": item.get("date", "")[:10],  # YYYY-MM-DD
                "filing_type": item.get("announcementType", "ASX_ANNOUNCEMENT"),
                "source": "asx",
                "headline": item.get("headline", ""),
                "body_text": item.get("headline", ""),  # headline only, PDF not accessible
                "url": f"https://www.asx.com.au/asx/1/file?id={item.get('documentKey', '')}",
            })

        logger.info(f"fetch_announcements: {ticker} — {len(results)} announcements fetched")
        return results

    except requests.exceptions.RequestException as e:
        logger.warning(f"fetch_announcements: {ticker} — request failed: {e}")
        return []
    except Exception as e:
        logger.warning(f"fetch_announcements: {ticker} — unexpected error: {e}")
        return []


def fetch_announcements_for_universe(tickers: list[str], sleep_seconds: float = 0.5) -> dict[str, int]:
    """Fetch and return announcements for all ASX tickers.

    Parameters
    ----------
    tickers:
        List of ASX tickers with .AX suffix.
    sleep_seconds:
        Sleep between requests to be polite.

    Returns
    -------
    dict[str, int]
        Mapping of ticker -> number of announcements fetched.
    """
    results = {}
    for ticker in tickers:
        announcements = fetch_announcements(ticker)
        results[ticker] = len(announcements)
        time.sleep(sleep_seconds)
    return results