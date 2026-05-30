"""AlphaSignal — Agent tools. (Phase 6)

Provides LangChain tool functions for the AlphaSignal agent.
"""

import logging
from typing import List

from langchain.tools import tool

from src.signals.price_signal import compute_price_signal
from src.signals.sentiment_signal import compute_sentiment_signal
from src.signals.filing_signal import compute_filing_signal
from src.signals.social_signal import compute_social_signal
from src.embeddings.faiss_store import (
    retrieve_similar_news,
    build_faiss_index,
    load_faiss_index,
)
from src.data.database import get_engine
from sqlalchemy import text
import pandas as pd

logger = logging.getLogger(__name__)


def _format_signal(value: float, pos_label: str = "bullish", neg_label: str = "bearish") -> str:
    """Helper to format a signal value with direction label."""
    if value > 0.1:
        direction = pos_label
    elif value < -0.1:
        direction = neg_label
    else:
        direction = "neutral"
    return f"{value:.3f} ({direction})"


@tool
def get_price_signal(ticker: str) -> str:
    """Returns the price signal for a ticker based on TFT model predictions.
    Input: ticker symbol like 'BHP.AX' or 'AAPL'.
    Returns a string describing the price signal direction and strength.
    """
    try:
        score = compute_price_signal(ticker)
        formatted = _format_signal(score, "bullish", "bearish")
        return f"Price signal for {ticker}: {formatted}"
    except Exception as e:
        logger.error(f"Error in get_price_signal for {ticker}: {e}")
        return f"Error retrieving price signal for {ticker}: {str(e)}"


@tool
def get_sentiment_signal(ticker: str) -> str:
    """Returns the sentiment signal for a ticker based on FinBERT analysis of recent news articles and social media posts.
    Input: ticker symbol like 'BHP.AX' or 'AAPL'.
    Returns a string describing the sentiment score and direction.
    """
    try:
        score = compute_sentiment_signal(ticker)
        formatted = _format_signal(score, "positive", "negative")
        return f"Sentiment signal for {ticker}: {formatted}"
    except Exception as e:
        logger.error(f"Error in get_sentiment_signal for {ticker}: {e}")
        return f"Error retrieving sentiment signal for {ticker}: {str(e)}"


@tool
def get_filing_signal(ticker: str) -> str:
    """Returns the filing signal for a ticker based on FinBERT analysis of recent regulatory filings and announcements.
    Input: ticker symbol like 'BHP.AX' or 'AAPL'.
    Returns a string describing the filing sentiment.
    """
    try:
        score = compute_filing_signal(ticker)
        formatted = _format_signal(score, "positive", "negative")
        return f"Filing signal for {ticker}: {formatted}"
    except Exception as e:
        logger.error(f"Error in get_filing_signal for {ticker}: {e}")
        return f"Error retrieving filing signal for {ticker}: {str(e)}"


@tool
def get_social_signal(ticker: str) -> str:
    """Returns the social signal for a ticker based on yfinance news and social media data scored with FinBERT.
    Input: ticker symbol like 'BHP.AX' or 'AAPL'.
    Returns a string describing the social sentiment.
    """
    try:
        score = compute_social_signal(ticker)
        formatted = _format_signal(score, "positive", "negative")
        return f"Social signal for {ticker}: {formatted}"
    except Exception as e:
        logger.error(f"Error in get_social_signal for {ticker}: {e}")
        return f"Error retrieving social signal for {ticker}: {str(e)}"


@tool
def get_composite_score(ticker: str) -> str:
    """
    Returns the latest composite score from signal_scores table for a ticker.
    Input: ticker symbol like 'BHP.AX' or 'AAPL'.
    Returns a string with the composite score and its components.
    """
    try:
        from src.data.database import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT price_signal, sentiment_signal, filing_signal,
                           social_signal, composite_score, date
                    FROM signal_scores
                    WHERE ticker = :ticker
                    ORDER BY date DESC LIMIT 1
                """),
                {"ticker": ticker}
            ).fetchone()
        if result is None:
            return f"No composite score available for {ticker}"
        price = result[0]
        sentiment = result[1]
        filing = result[2]
        social = result[3]
        composite = result[4]
        date = result[5]
        price_str = f"{price:.3f}" if price is not None else "N/A"
        sentiment_str = f"{sentiment:.3f}" if sentiment is not None else "N/A"
        filing_str = f"{filing:.3f}" if filing is not None else "N/A"
        social_str = f"{social:.3f}" if social is not None else "N/A"
        composite_str = f"{composite:.3f}" if composite is not None else "N/A"
        return (
            f"Composite score for {ticker} as of {date}: {composite_str} | "
            f"price={price_str}, sentiment={sentiment_str}, "
            f"filing={filing_str}, social={social_str}"
        )
    except Exception as e:
        logger.error(f"Error in get_composite_score for {ticker}: {e}")
        return f"Error retrieving composite score for {ticker}: {str(e)}"


@tool
def retrieve_relevant_news(ticker: str) -> str:
    """Retrieves the most relevant recent news headlines for a ticker using FAISS semantic search on the news database.
    Input: ticker symbol like 'BHP.AX' or 'AAPL'.
    Returns a string with up to 5 relevant news headlines.
    """
    try:
        # Ensure FAISS index exists
        index = load_faiss_index(ticker)
        if index is None:
            build_faiss_index(ticker)
        # Perform retrieval
        query = f"{ticker} stock performance outlook"
        results = retrieve_similar_news(query, ticker=ticker, top_k=5)
        if not results:
            return f"No recent news found for {ticker}"
        lines = []
        for i, item in enumerate(results, 1):
            title = item.get("title") or item.get("headline") or "(no title)"
            published = item.get("published_at") or item.get("date") or ""
            lines.append(f"{i}. {title} ({published})")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Error in retrieve_relevant_news for {ticker}: {e}")
        return f"Error retrieving news for {ticker}: {str(e)}"


@tool
def get_recent_filings(ticker: str) -> str:
    """Returns recent regulatory filings and announcements for a ticker from the filings database table.
    Input: ticker symbol like 'BHP.AX' or 'AAPL'.
    Returns a string listing recent filing headlines.
    """
    try:
        engine = get_engine()
        query = """
            SELECT filing_date, headline
            FROM filings
            WHERE ticker = :ticker
            ORDER BY filing_date DESC
            LIMIT 5
        """
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"ticker": ticker})
        if df.empty:
            return f"No recent filings found for {ticker}"
        lines = []
        for _, row in df.iterrows():
            date_str = row["filing_date"].strftime("%Y-%m-%d") if hasattr(row["filing_date"], "strftime") else str(row["filing_date"]) 
            lines.append(f"{date_str}: {row['headline']}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Error in get_recent_filings for {ticker}: {e}")
        return f"Error retrieving filings for {ticker}: {str(e)}"

# Note: get_tools and get_price_data remain unimplemented as they are not required for Phase 6.