"""
AlphaSignal — database module.

Provides SQLAlchemy 2.0 engine, table definitions, and all CRUD helpers
used across the platform. This is the only file with real logic in Phase 1.
"""

import os
import logging
from datetime import date, datetime, timezone
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

_engine: Engine | None = None

DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "postgresql://localhost/alphasignal"
)

metadata = MetaData()

# ---------------------------------------------------------------------------
# Table definitions
# ---------------------------------------------------------------------------

stocks = Table(
    "stocks",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ticker", String(20), unique=True, nullable=False),
    Column("exchange", String(10)),
    Column("sector", String(50)),
    Column("created_at", DateTime, default=datetime.utcnow),
)

price_history = Table(
    "price_history",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ticker", String(20), nullable=False),
    Column("date", Date, nullable=False),
    Column("open", Float),
    Column("high", Float),
    Column("low", Float),
    Column("close", Float),
    Column("volume", BigInteger),
    UniqueConstraint("ticker", "date", name="uq_price_ticker_date"),
)

signal_scores = Table(
    "signal_scores",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ticker", String(20), nullable=False),
    Column("date", Date, nullable=False),
    Column("price_signal", Float),
    Column("sentiment_signal", Float),
    Column("filing_signal", Float),
    Column("social_signal", Float),
    Column("composite_score", Float),
    Column("created_at", DateTime, default=datetime.utcnow),
    UniqueConstraint("ticker", "date", name="uq_signal_ticker_date"),
)

filings = Table(
    "filings",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ticker", String(20), nullable=False),
    Column("filing_date", Date),
    Column("filing_type", String(20)),
    Column("source", String(20)),
    Column("headline", Text),
    Column("body_text", Text),
    Column("url", String(500)),
    Column("created_at", DateTime, default=datetime.utcnow),
)

news_items = Table(
    "news_items",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ticker", String(20), nullable=False),
    Column("published_at", DateTime),
    Column("headline", Text),
    Column("body", Text),
    Column("source", String(100)),
    Column("url", String(500)),
    Column("sentiment_score", Float),
    Column("created_at", DateTime, default=datetime.utcnow),
)

reddit_posts = Table(
    "reddit_posts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ticker", String(20), nullable=False),
    Column("published_at", DateTime),
    Column("title", Text),
    Column("body", Text),
    Column("source", String(50)),
    Column("sentiment_score", Float),
    Column("created_at", DateTime, default=datetime.utcnow),
)

reports = Table(
    "reports",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ticker", String(20), nullable=False),
    Column("generated_at", DateTime, default=datetime.utcnow),
    Column("report_markdown", Text),
    Column("model_used", String(50)),
)

_ALL_TABLES = [
    "stocks",
    "price_history",
    "signal_scores",
    "filings",
    "news_items",
    "reddit_posts",
    "reports",
]

# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------


def get_engine() -> Engine:
    """Return (or create) the shared SQLAlchemy engine.

    Returns
    -------
    Engine
        Configured SQLAlchemy engine connected to DATABASE_URL.
    """
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            future=True,
        )
        logger.info("SQLAlchemy engine created: %s", DATABASE_URL)
    return _engine


# Module-level engine instance for direct imports
engine = get_engine()


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------


def create_all_tables() -> None:
    """Create all 7 AlphaSignal tables if they do not already exist."""
    engine = get_engine()
    metadata.create_all(engine)
    for name in _ALL_TABLES:
        logger.info("Table ensured: %s", name)
    logger.info("Schema creation complete — %d tables", len(_ALL_TABLES))


# ---------------------------------------------------------------------------
# Price history
# ---------------------------------------------------------------------------


def insert_price_history(records: list[dict]) -> int:
    """Bulk-insert OHLCV records, skipping duplicates.

    Parameters
    ----------
    records:
        List of dicts with keys: ticker, date, open, high, low, close, volume.

    Returns
    -------
    int
        Number of rows actually inserted.
    """
    if not records:
        return 0
    engine = get_engine()
    stmt = pg_insert(price_history).values(records)
    stmt = stmt.on_conflict_do_nothing(
        constraint="uq_price_ticker_date"
    )
    try:
        with engine.begin() as conn:
            result = conn.execute(stmt)
            inserted = result.rowcount
        logger.info(
            "insert_price_history: %d/%d rows inserted",
            inserted,
            len(records),
        )
        return inserted
    except SQLAlchemyError as exc:
        logger.exception("insert_price_history failed: %s", exc)
        return 0


# ---------------------------------------------------------------------------
# Filings
# ---------------------------------------------------------------------------


def insert_filing(record: dict) -> bool:
    """Insert a single filing record.

    Parameters
    ----------
    record:
        Dict with keys matching the filings table columns.

    Returns
    -------
    bool
        True on success, False on failure.
    """
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(insert(filings).values(**record))
        logger.info(
            "Filing inserted: ticker=%s type=%s",
            record.get("ticker"),
            record.get("filing_type"),
        )
        return True
    except SQLAlchemyError as exc:
        logger.exception("insert_filing failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# News items
# ---------------------------------------------------------------------------


def insert_news_item(record: dict) -> bool:
    """Insert a single news item.

    Parameters
    ----------
    record:
        Dict with keys matching the news_items table columns.

    Returns
    -------
    bool
        True on success, False on failure.
    """
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(insert(news_items).values(**record))
        logger.info(
            "News item inserted: ticker=%s source=%s",
            record.get("ticker"),
            record.get("source"),
        )
        return True
    except SQLAlchemyError as exc:
        logger.exception("insert_news_item failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Reddit / Alpha Vantage social posts
# ---------------------------------------------------------------------------


def insert_reddit_post(record: dict) -> bool:
    """Insert a single social post record (Alpha Vantage NEWS_SENTIMENT).

    Parameters
    ----------
    record:
        Dict with keys matching the reddit_posts table columns.

    Returns
    -------
    bool
        True on success, False on failure.
    """
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(insert(reddit_posts).values(**record))
        logger.info(
            "Social post inserted: ticker=%s source=%s",
            record.get("ticker"),
            record.get("source"),
        )
        return True
    except SQLAlchemyError as exc:
        logger.exception("insert_reddit_post failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Signal scores
# ---------------------------------------------------------------------------


def insert_signal_scores(record: dict) -> bool:
    """Upsert signal scores for a ticker + date combination.

    If a row for (ticker, date) already exists it is updated in-place;
    otherwise a new row is inserted.

    Parameters
    ----------
    record:
        Dict with keys matching the signal_scores table columns.

    Returns
    -------
    bool
        True on success, False on failure.
    """
    engine = get_engine()
    stmt = pg_insert(signal_scores).values(**record)
    update_cols = {
        c.name: c
        for c in stmt.excluded
        if c.name
        not in ("id", "ticker", "date")
    }
    stmt = stmt.on_conflict_do_update(
        constraint="uq_signal_ticker_date",
        set_=update_cols,
    )
    try:
        with engine.begin() as conn:
            conn.execute(stmt)
        logger.info(
            "Signal scores upserted: ticker=%s date=%s composite=%.4f",
            record.get("ticker"),
            record.get("date"),
            record.get("composite_score", float("nan")),
        )
        return True
    except SQLAlchemyError as exc:
        logger.exception("insert_signal_scores failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def get_price_history(
    ticker: str, start_date: str, end_date: str
) -> list[dict]:
    """Fetch OHLCV rows for a ticker within an inclusive date range.

    Parameters
    ----------
    ticker:
        Ticker symbol (e.g. "AAPL").
    start_date:
        ISO date string "YYYY-MM-DD".
    end_date:
        ISO date string "YYYY-MM-DD".

    Returns
    -------
    list[dict]
        List of OHLCV dicts ordered by date ascending.
    """
    engine = get_engine()
    stmt = (
        select(price_history)
        .where(price_history.c.ticker == ticker)
        .where(price_history.c.date >= start_date)
        .where(price_history.c.date <= end_date)
        .order_by(price_history.c.date.asc())
    )
    try:
        with engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]
    except SQLAlchemyError as exc:
        logger.exception("get_price_history failed: %s", exc)
        return []


def get_latest_signal(ticker: str) -> dict | None:
    """Return the most recent signal_scores row for a ticker.

    Parameters
    ----------
    ticker:
        Ticker symbol.

    Returns
    -------
    dict | None
        Row dict or None if not found.
    """
    engine = get_engine()
    stmt = (
        select(signal_scores)
        .where(signal_scores.c.ticker == ticker)
        .order_by(signal_scores.c.date.desc())
        .limit(1)
    )
    try:
        with engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
        return dict(row) if row else None
    except SQLAlchemyError as exc:
        logger.exception("get_latest_signal failed: %s", exc)
        return None


def get_all_latest_signals() -> list[dict]:
    """Return the latest signal row for every ticker, sorted by composite_score desc.

    Uses a subquery to get the max date per ticker then joins back.

    Returns
    -------
    list[dict]
        List of signal rows ordered by composite_score descending.
    """
    engine = get_engine()
    subq = (
        select(
            signal_scores.c.ticker,
            func.max(signal_scores.c.date).label("max_date"),
        )
        .group_by(signal_scores.c.ticker)
        .subquery()
    )
    stmt = (
        select(signal_scores)
        .join(
            subq,
            (signal_scores.c.ticker == subq.c.ticker)
            & (signal_scores.c.date == subq.c.max_date),
        )
        .order_by(signal_scores.c.composite_score.desc())
    )
    try:
        with engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]
    except SQLAlchemyError as exc:
        logger.exception("get_all_latest_signals failed: %s", exc)
        return []


def get_recent_filings(ticker: str, limit: int = 5) -> list[dict]:
    """Return the most recent filing records for a ticker.

    Parameters
    ----------
    ticker:
        Ticker symbol.
    limit:
        Maximum number of rows to return.

    Returns
    -------
    list[dict]
        Filings ordered by filing_date descending.
    """
    engine = get_engine()
    stmt = (
        select(filings)
        .where(filings.c.ticker == ticker)
        .order_by(filings.c.filing_date.desc())
        .limit(limit)
    )
    try:
        with engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]
    except SQLAlchemyError as exc:
        logger.exception("get_recent_filings failed: %s", exc)
        return []


def get_recent_news(ticker: str, hours_back: int = 48) -> list[dict]:
    """Return news items published within the last N hours for a ticker.

    Parameters
    ----------
    ticker:
        Ticker symbol.
    hours_back:
        How many hours back to look from now.

    Returns
    -------
    list[dict]
        News items ordered by published_at descending.
    """
    from datetime import timedelta

    engine = get_engine()
    cutoff = datetime.utcnow() - timedelta(hours=hours_back)
    stmt = (
        select(news_items)
        .where(news_items.c.ticker == ticker)
        .where(news_items.c.published_at >= cutoff)
        .order_by(news_items.c.published_at.desc())
    )
    try:
        with engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]
    except SQLAlchemyError as exc:
        logger.exception("get_recent_news failed: %s", exc)
        return []


def count_api_calls_today(source: str) -> int:
    """Count news_items rows for a given source created since midnight UTC.

    Parameters
    ----------
    source:
        Source name to filter on (e.g. "newsapi", "alpha_vantage").

    Returns
    -------
    int
        Row count for the source today.
    """
    engine = get_engine()
    today_midnight = datetime.utcnow().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    stmt = (
        select(func.count())
        .select_from(news_items)
        .where(news_items.c.source == source)
        .where(news_items.c.created_at >= today_midnight)
    )
    try:
        with engine.connect() as conn:
            count = conn.execute(stmt).scalar_one_or_none() or 0
        return int(count)
    except SQLAlchemyError as exc:
        logger.exception("count_api_calls_today failed: %s", exc)
        return 0
