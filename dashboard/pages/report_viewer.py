"""AlphaSignal — Report Viewer page. Implemented in Phase 7."""
import streamlit as st
import pandas as pd
import yaml
import requests
from sqlalchemy import text
from src.data.database import get_engine

@st.cache_data(ttl=300)
def load_universe():
    with open('config/stock_universe.yaml', 'r') as f:
        universe = yaml.safe_load(f)
    return universe.get('asx_tickers', []) + universe.get('sp500_tickers', [])

def render():
    st.title("📋 AI Research Reports")
    
    all_tickers = load_universe()
    ticker = st.selectbox("Select Ticker", all_tickers)
    
    if not ticker:
        return
        
    tab1, tab2 = st.tabs(["View Reports", "Generate New Report"])
    
    with tab1:
        engine = get_engine()
        with engine.connect() as conn:
            reports = pd.read_sql(
                text("""
                    SELECT id, ticker, report_type, generated_at,
                           report_markdown, content
                    FROM reports
                    WHERE ticker = :ticker
                    ORDER BY generated_at DESC
                    LIMIT 10
                """),
                conn, params={"ticker": ticker}
            )
            
        if not reports.empty:
            for _, row in reports.iterrows():
                with st.expander(f"{row['report_type']} — {row['generated_at']}"):
                    if pd.notna(row['report_markdown']) and row['report_markdown']:
                        st.markdown(row['report_markdown'])
                    else:
                        st.json(row['content'])
        else:
            st.info("No reports generated yet for this ticker.")
            
    with tab2:
        if st.button("🤖 Generate Analysis Report"):
            with st.spinner("Running AI analysis (30-60 seconds)..."):
                try:
                    resp = requests.post(f"http://127.0.0.1:8000/analyze/{ticker}", timeout=120)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success("Report generated!")
                        st.markdown(data.get('response', ''))
                        st.json(data.get('signals', {}))
                    else:
                        st.error(f"API returned {resp.status_code}")
                except Exception as e:
                    st.error(f"Could not reach API: {e}")
