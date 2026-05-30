"""AlphaSignal — Streamlit dashboard entry point."""
import os
import sys
import multiprocessing

# Force safe thread initialization for macOS
try:
    multiprocessing.set_start_method("spawn", force=True)
except (RuntimeError, RuntimeWarning):
    pass

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from datetime import datetime, timezone

# MUST BE THE ABSOLUTE FIRST st. COMMAND
st.set_page_config(
    page_title="AlphaSignal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide native Streamlit pages menu to prevent blank page clicks
st.markdown("""
    <style>
    [data-testid="stSidebarNav"] {display: none;}
    </style>
    """, unsafe_allow_html=True)

from dashboard.pages import overview, stock_detail, report_viewer

st.sidebar.title("AlphaSignal")
st.sidebar.caption("Multi-Signal Intelligence Platform")
st.sidebar.write(f"**UTC Time:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

page = st.sidebar.radio("Navigation", ["📊 Overview", "🔍 Stock Detail", "📋 Report Viewer"])

if page == "📊 Overview":
    overview.render()
elif page == "🔍 Stock Detail":
    stock_detail.render()
elif page == "📋 Report Viewer":
    report_viewer.render()