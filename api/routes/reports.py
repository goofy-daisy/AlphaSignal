"""
AlphaSignal — reports API router.

Provides endpoints to generate analytical reports for stock tickers.
"""

import logging
import uuid
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.agent.agent import generate_report as agent_generate_report

logger = logging.getLogger(__name__)

router = APIRouter()

# Response schemas
class ReportJobResponse(BaseModel):
    """Response returned when a report generation job is queued."""
    job_id: str
    status: str
    ticker: str

class JobStatusResponse(BaseModel):
    """Response for a job status query."""
    job_id: str
    status: str
    message: str

class ReportResultResponse(BaseModel):
    """Response containing a completed report."""
    job_id: str
    status: str
    ticker: str
    report_markdown: str | None = None

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/{ticker}", response_model=ReportResultResponse, status_code=200)
async def generate_report(ticker: str) -> ReportResultResponse:
    """Generate a report synchronously for the given ticker and return it."""
    ticker = ticker.upper()
    loop = asyncio.get_event_loop()
    try:
        report = await loop.run_in_executor(None, agent_generate_report, ticker)
    except Exception as e:
        logger.exception("Report generation failed for %s", ticker)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")
    job_id = str(uuid.uuid4())
    return ReportResultResponse(
        job_id=job_id,
        status="completed",
        ticker=ticker,
        report_markdown=report.get("response", ""),
    )

@router.get("/status/{job_id}", response_model=JobStatusResponse, status_code=200)
async def get_report_status(job_id: str) -> JobStatusResponse:
    """Placeholder status endpoint – always completed for sync implementation."""
    return JobStatusResponse(
        job_id=job_id,
        status="completed",
        message="Report generated synchronously; no pending jobs.",
    )

@router.get("/result/{job_id}", response_model=ReportResultResponse, status_code=200)
async def get_report_result(job_id: str) -> ReportResultResponse:
    """Placeholder result endpoint – not used in sync mode."""
    raise HTTPException(status_code=404, detail="Result not found; reports are generated synchronously via POST.")
