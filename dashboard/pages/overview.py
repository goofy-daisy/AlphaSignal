"""AlphaSignal — Overview page. Implemented in Phase 7."""
import streamlit as st
import pandas as pd
import yaml
from sqlalchemy import text
from src.data.database import get_engine
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data(ttl=300)
def load_signal_data() -> pd.DataFrame:
    """
    Loads latest signal scores for all tickers from signal_scores table.
    Joins with stock universe to include tickers with no signals yet (showing 0.0).
    Returns DataFrame with columns: ticker, price_signal, sentiment_signal,
    filing_signal, social_signal, composite_score, date, market.
    market column is 'ASX' for .AX tickers else 'US'.
    """
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT DISTINCT ON (ticker) ticker, date, price_signal,
                       sentiment_signal, filing_signal, social_signal, composite_score
                FROM signal_scores
                ORDER BY ticker, date DESC
            """),
            conn
        )
    
    with open('config/stock_universe.yaml', 'r') as f:
        universe = yaml.safe_load(f)
    asx = universe.get('asx_tickers', [])
    us = universe.get('sp500_tickers', [])
    all_tickers = asx + us
    
    full_df = pd.DataFrame({'ticker': all_tickers})
    full_df['market'] = full_df['ticker'].apply(lambda x: 'ASX' if str(x).endswith('.AX') else 'US')
    
    if df.empty:
        df = pd.DataFrame(columns=['ticker', 'date', 'price_signal', 'sentiment_signal', 'filing_signal', 'social_signal', 'composite_score'])
    
    df = pd.merge(full_df, df, on='ticker', how='left')
    signals = ['price_signal', 'sentiment_signal', 'filing_signal', 'social_signal', 'composite_score']
    df[signals] = df[signals].fillna(0.0)
    
    return df

def render():
    st.title("📊 Market Overview")
    
    df = load_signal_data()
    
    total_tickers = len(df)
    bullish = len(df[df['composite_score'] > 0.1])
    bearish = len(df[df['composite_score'] < -0.1])
    neutral = total_tickers - bullish - bearish
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tickers", total_tickers)
    c2.metric("Bullish (>0.1)", bullish)
    c3.metric("Bearish (<-0.1)", bearish)
    c4.metric("Neutral", neutral)
    
    st.write("---")
    
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        market_filter = st.selectbox("Market", ["All", "ASX", "US"])
    with col_filter2:
        sort_by = st.selectbox("Sort By", ["composite_score", "price_signal", "sentiment_signal"])
        
    filtered_df = df.copy()
    if market_filter != "All":
        filtered_df = filtered_df[filtered_df['market'] == market_filter]
        
    filtered_df = filtered_df.sort_values(by=sort_by, ascending=False)
    
    st.dataframe(
        filtered_df.style.background_gradient(subset=['composite_score'], cmap='RdYlGn').format({
            'price_signal': "{:.3f}",
            'sentiment_signal': "{:.3f}",
            'filing_signal': "{:.3f}",
            'social_signal': "{:.3f}",
            'composite_score': "{:.3f}"
        }),
        use_container_width=True
    )
    
    st.subheader(f"Composite Scores ({market_filter})")
    
    colors = ['#00CC44' if v > 0 else '#FF3333' for v in filtered_df['composite_score']]
    fig = go.Figure(go.Bar(
        x=filtered_df['composite_score'],
        y=filtered_df['ticker'],
        orientation='h',
        marker_color=colors
    ))
    fig.update_layout(height=max(400, len(filtered_df) * 20), yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
