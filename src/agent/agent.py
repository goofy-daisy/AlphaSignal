"""AlphaSignal — Agent. (Phase 6)"""

import logging
import os
from datetime import datetime, timezone
from typing import List

from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.agent.tools import (
    get_price_signal,
    get_sentiment_signal,
    get_filing_signal,
    get_social_signal,
    get_composite_score,
    retrieve_relevant_news,
    get_recent_filings,
)
from src.signals.price_signal import compute_price_signal
from src.signals.sentiment_signal import compute_sentiment_signal
from src.signals.filing_signal import compute_filing_signal
from src.signals.social_signal import compute_social_signal
from src.backtest.backtester import write_report

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are AlphaSignal, an AI investment research assistant specializing in ASX and S&P500 stocks.

You have access to tools that provide quantitative signals derived from price data, news sentiment, regulatory filings, and social media analysis.

When analyzing a stock, you should:
1. Use get_composite_score to get the overall signal picture
2. Use get_price_signal for technical/model-based outlook
3. Use retrieve_relevant_news to understand recent news context
4. Use get_sentiment_signal and get_social_signal for market sentiment
5. Use get_filing_signal and get_recent_filings for regulatory context

Always base your analysis on the tool outputs. Be concise, factual, and highlight the most important signals. Mention both bullish and bearish factors. End with a clear summary of the overall outlook.

CRITICAL INSTRUCTION: You must strictly use the provided tool-calling functionality to gather data. DO NOT output raw JSON strings, code blocks, or describe the tools you are going to call in your final response. Your final output must ONLY be the synthesized Markdown analyst report. Do not print internal tool execution formats or JSON to the user."""


def get_llm() -> ChatOllama:
    """Returns a configured ChatOllama instance using environment variables."""
    return ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        temperature=0.1,
    )


def get_all_tools() -> List:
    """Collects all tool functions defined in src.agent.tools."""
    return [
        get_price_signal,
        get_sentiment_signal,
        get_filing_signal,
        get_social_signal,
        get_composite_score,
        retrieve_relevant_news,
        get_recent_filings,
    ]


def build_agent() -> AgentExecutor:
    """Builds and returns a LangChain AgentExecutor with all tools."""
    llm = get_llm()
    tools = get_all_tools()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=6, handle_parsing_errors=True)


def run_agent(query: str, ticker: str = None) -> str:
    """Runs the agent with the given query.

    If ``ticker`` is provided, it is prefixed to the query.
    Returns the agent's textual response or an error string.
    """
    try:
        executor = build_agent()
        full_query = f"Analyze {ticker}: {query}" if ticker else query
        result = executor.invoke({"input": full_query})
        return result.get("output", "No response generated")
    except Exception as e:
        logger.error(f"Agent run failed: {e}")
        return f"Agent error: {str(e)}"


def generate_report(ticker: str) -> dict:
    """Generates a comprehensive analysis report for a ticker.

    Returns a dictionary containing the ticker, original query, agent response,
    raw signal values, a computed composite score, and a generation timestamp.
    The report is also persisted to the ``reports`` table via ``write_report``.
    """
    query = (
        f"Provide a comprehensive investment analysis for {ticker}. "
        "Use all available tools to gather signals and news. "
        "Summarize the price outlook, sentiment, filing signals, and social signals. "
        "Conclude with an overall assessment."
    )
    response = run_agent(query, ticker=ticker)

    # Live signal calculations
    price = compute_price_signal(ticker)
    sentiment = compute_sentiment_signal(ticker)
    filing = compute_filing_signal(ticker)
    social = compute_social_signal(ticker)
    composite = sentiment * 0.3 + filing * 0.15 + social * 0.15 + price * 0.4

    report = {
        "ticker": ticker,
        "query": query,
        "response": response,
        "signals": {
            "price": price,
            "sentiment": sentiment,
            "filing": filing,
            "social": social,
            "composite": composite,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        write_report(ticker, "agent_report", {"response": response, "signals": report["signals"]})
    except Exception as e:
        logger.error(f"Failed to write report to DB for {ticker}: {e}")

    return report