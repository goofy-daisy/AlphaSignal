"""AlphaSignal — Stock Detail page. Implemented in Phase 7."""
import streamlit as st
import pandas as pd
import yaml
from sqlalchemy import text
from src.data.database import get_engine
import plotly.express as px
import plotly.graph_objects as go
from src.embeddings.faiss_store import retrieve_similar_news
from src.models.meta_learner import compute_shap_importance

@st.cache_data(ttl=300)
def load_universe():
    with open('config/stock_universe.yaml', 'r') as f:
        universe = yaml.safe_load(f)
    return universe.get('asx_tickers', []) + universe.get('sp500_tickers', [])

def render():
    st.title("🔍 Stock Detail")
    
    all_tickers = load_universe()
    ticker = st.selectbox("Select Ticker", all_tickers)
    
    if not ticker:
        return
        
    engine = get_engine()
    
    c_left, c_right = st.columns([0.6, 0.4])
    
    with c_left:
        # Signal breakdown bar chart
        with engine.connect() as conn:
            latest_sig = pd.read_sql(
                text("SELECT price_signal, sentiment_signal, filing_signal, social_signal, composite_score FROM signal_scores WHERE ticker = :ticker ORDER BY date DESC LIMIT 1"),
                conn, params={"ticker": ticker}
            )
        
        if not latest_sig.empty:
            sig_vals = latest_sig.iloc[0]
            st.subheader("Signal Breakdown")
            x_vals = ['Price', 'Sentiment', 'Filing', 'Social']
            y_vals = [sig_vals['price_signal'], sig_vals['sentiment_signal'], sig_vals['filing_signal'], sig_vals['social_signal']]
            colors = ['#00CC44' if v > 0 else '#FF3333' for v in y_vals]
            
            fig_bar = go.Figure(go.Bar(
                x=x_vals, y=y_vals, marker_color=colors
            ))
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Signal history line chart
            with engine.connect() as conn:
                hist = pd.read_sql(
                    text("SELECT date, composite_score, sentiment_signal, price_signal FROM signal_scores WHERE ticker = :ticker ORDER BY date ASC"),
                    conn, params={"ticker": ticker}
                )
            
            st.subheader("Signal History")
            if len(hist) > 1:
                fig_hist = px.line(hist, x='date', y=['composite_score', 'sentiment_signal', 'price_signal'])
                st.plotly_chart(fig_hist, use_container_width=True)
            elif len(hist) == 1:
                fig_hist = px.scatter(hist, x='date', y=['composite_score', 'sentiment_signal', 'price_signal'])
                st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No signal data available.")
            sig_vals = {'price_signal': 0.0, 'sentiment_signal': 0.0, 'filing_signal': 0.0, 'social_signal': 0.0, 'composite_score': 0.0}
            
        # Price history chart
        with engine.connect() as conn:
            prices = pd.read_sql(
                text("SELECT date, close FROM price_history WHERE ticker = :ticker ORDER BY date DESC LIMIT 90"),
                conn, params={"ticker": ticker}
            )
        
        st.subheader("Price History (Last 90 Days)")
        if not prices.empty:
            fig_price = px.line(prices, x='date', y='close')
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.info("No price data available.")
            
    with c_right:
        st.subheader("Signal Scorecard")
        sc1, sc2 = st.columns(2)
        sc1.metric("Price Signal", f"{sig_vals['price_signal']:.3f}")
        sc2.metric("Sentiment Signal", f"{sig_vals['sentiment_signal']:.3f}")
        sc3, sc4 = st.columns(2)
        sc3.metric("Filing Signal", f"{sig_vals['filing_signal']:.3f}")
        sc4.metric("Social Signal", f"{sig_vals['social_signal']:.3f}")
        
        st.metric("Composite Score", f"{sig_vals['composite_score']:.3f}")
        
        st.subheader("Recent News (FAISS)")
        try:
            news = retrieve_similar_news(ticker + " stock outlook performance", ticker=ticker, top_k=5)
            if news:
                for item in news:
                    st.markdown(f"- **{item.get('published_at', '')}**: {item.get('title', '')}")
            else:
                st.info("No news index built yet.")
        except Exception:
            st.info("No news index built yet.")
            
        st.subheader("Recent Filings")
        with engine.connect() as conn:
            filings = pd.read_sql(
                text("SELECT filing_date, headline, filing_type FROM filings WHERE ticker = :ticker ORDER BY filing_date DESC LIMIT 5"),
                conn, params={"ticker": ticker}
            )
        
        if not filings.empty:
            for _, row in filings.iterrows():
                st.markdown(f"- **{row['filing_date']}**: {row['headline']} ({row['filing_type']})")
        else:
            st.info("No filings found.")
            
        st.subheader("SHAP Importance")
        try:
            shap_df = compute_shap_importance(ticker)
            if shap_df is not None and not shap_df.empty:
                fig_shap = px.bar(
                    shap_df, 
                    x='importance', 
                    y='feature', 
                    orientation='h'
                )
                fig_shap.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_shap, use_container_width=True)
            else:
                st.info("SHAP not available.")
        except Exception:
            st.info("SHAP not available.")
