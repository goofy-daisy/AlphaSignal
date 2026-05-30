"""
AlphaSignal — signals API router.

Provides endpoint for listing all tickers signals (stub) and leaves detailed ticker endpoint to main app.
"""

import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("", response_model=list, status_code=200)
async def list_signals_stub():
    """Return empty list as placeholder; detailed ticker signal is implemented at /signals/{ticker} in main app."""
    logger.info("GET /signals – stub placeholder returning empty list")
    return []
