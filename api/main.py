"""
AlphaSignal — FastAPI application entry-point.

Serves the REST API on port 8000. All heavy model logic lives in src/;
this module wires up middleware, routers, and lifecycle hooks only.
"""

import logging
import os
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import reports, signals

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AlphaSignal API",
    description="Local stock intelligence platform REST API.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every incoming HTTP request and its response status."""
    logger.info(
        "REQUEST  %s %s  client=%s",
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
    )
    response = await call_next(request)
    logger.info(
        "RESPONSE %s %s  status=%d",
        request.method,
        request.url.path,
        response.status_code,
    )
    return response


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(signals.router, prefix="/signals", tags=["signals"])
app.include_router(reports.router, prefix="/report", tags=["reports"])

# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """Return API liveness status."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
from src.data.database import get_engine
from sqlalchemy import text
import pandas as pd
from src.signals.price_signal import compute_price_signal
from src.signals.sentiment_signal import compute_sentiment_signal
from src.signals.filing_signal import compute_filing_signal
from src.signals.social_signal import compute_social_signal
from src.agent.agent import generate_report
import asyncio

# Phase 6 Endpoints

@app.get("/signals/{ticker}", tags=["signals"]) 
async def get_signals(ticker: str) -> dict:
    """Return all four signals and composite score for a ticker.
    If today's composite score exists in DB, return cached (source='cached'), else compute live (source='computed').
    """
    ticker = ticker.upper()
    engine = get_engine()
    today = datetime.now(timezone.utc).date()
    query = """
        SELECT price_signal, sentiment_signal, filing_signal, social_signal, composite_score, date
        FROM signal_scores
        WHERE ticker = :ticker
        ORDER BY date DESC
        LIMIT 1
    """
    loop = asyncio.get_event_loop()
    def fetch_cached():
        with engine.connect() as conn:
            return conn.execute(text(query), {"ticker": ticker}).fetchone()
    cached = await loop.run_in_executor(None, fetch_cached)
    if cached and cached.date == today:
        price, sentiment, filing, social, composite, _ = cached
        source = "cached"
    else:
        price = await loop.run_in_executor(None, compute_price_signal, ticker)
        sentiment = await loop.run_in_executor(None, compute_sentiment_signal, ticker)
        filing = await loop.run_in_executor(None, compute_filing_signal, ticker)
        social = await loop.run_in_executor(None, compute_social_signal, ticker)
        composite = sentiment * 0.3 + filing * 0.15 + social * 0.15 + price * 0.4
        source = "computed"
    return {
        "ticker": ticker,
        "price_signal": round(price, 3),
        "sentiment_signal": round(sentiment, 3),
        "filing_signal": round(filing, 3),
        "social_signal": round(social, 3),
        "composite_score": round(composite, 3),
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

from fastapi import HTTPException

@app.get("/reports/{ticker}")
async def get_reports_for_ticker(ticker: str):
    """Returns all reports for a ticker from reports table."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT id, ticker, report_type, generated_at, content, report_markdown
                FROM reports WHERE ticker = :ticker
                ORDER BY generated_at DESC LIMIT 20
            """),
            {"ticker": ticker}
        ).fetchall()
    if not result:
        return []
    return [
        {
            "id": row[0],
            "ticker": row[1],
            "report_type": row[2],
            "generated_at": str(row[3]),
            "content": row[4],
            "report_markdown": row[5]
        }
        for row in result
    ]

@app.post("/analyze/{ticker}")
async def analyze_ticker(ticker: str):
    """Runs agent analysis for a ticker and returns report dict."""
    import asyncio
    from functools import partial
    from src.agent.agent import generate_report
    loop = asyncio.get_event_loop()
    try:
        report = await loop.run_in_executor(None, generate_report, ticker)
        return {
            "ticker": report["ticker"],
            "response": report["response"],
            "signals": report["signals"],
            "generated_at": report["generated_at"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
