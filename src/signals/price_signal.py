"""AlphaSignal — Price signal. (Phase 3)"""
import os
import logging
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from src.features.price_features import compute_features

logger = logging.getLogger(__name__)

def compute_price_signal(ticker: str) -> float:
    """
    Returns price signal in [-1, 1] for given ticker.
    Returns 0.0 if no model available.
    """
    results_path = 'models/walk_forward_results.csv'
    if not os.path.exists(results_path):
        logger.warning(f"No walk_forward_results.csv found.")
        return 0.0
        
    results_df = pd.read_csv(results_path)
    ticker_res = results_df[results_df['ticker'] == ticker]
    if ticker_res.empty:
        logger.warning(f"No model available for {ticker}.")
        return 0.0
        
    checkpoint_path = ticker_res.iloc[0]['best_checkpoint']
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint {checkpoint_path} not found.")
        return 0.0
        
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load checkpoint for {ticker}: {e}")
        return 0.0
        
    df = compute_features(ticker)
    if df.empty or len(df) < 60:
        logger.warning(f"Not enough features for {ticker} (need at least 60).")
        return 0.0
        
    # Get last 60 days
    df_infer = df.iloc[-60:].copy()
    
    # Needs to look exactly like training data structure
    df_infer['target'] = '0'  # Dummy categorical target
    df_infer['time_idx'] = np.arange(len(df_infer))
    df_infer['ticker'] = ticker
    
    last_idx = df_infer['time_idx'].max()
    future_rows = pd.DataFrame({
        'time_idx': np.arange(last_idx + 1, last_idx + 21),
        'ticker': ticker,
        'target': '0'
    })
    for col in df_infer.columns:
        if col not in future_rows.columns:
            future_rows[col] = 0.0
            
    df_infer = pd.concat([df_infer, future_rows], ignore_index=True)
    
    try:
        test_ds = TimeSeriesDataSet.from_parameters(
            model.dataset_parameters, df_infer, predict=True, stop_randomization=True
        )
        test_dl = test_ds.to_dataloader(train=False, batch_size=1, num_workers=0)
    except Exception as e:
        logger.error(f"Failed to build inference dataset for {ticker}: {e}")
        return 0.0
        
    try:
        with torch.no_grad():
            out = model.predict(test_dl, mode="raw", return_x=False, trainer_kwargs=dict(accelerator='cpu'))
            logits = out.prediction[:, 0, :]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    except Exception as e:
        logger.error(f"Failed to run inference for {ticker}: {e}")
        return 0.0
        
    signal = probs[0]*(-1.0) + probs[1]*(-0.5) + probs[2]*0.0 + probs[3]*0.5 + probs[4]*1.0
    signal = float(np.clip(signal, -1.0, 1.0))
    return signal

def compute_all_price_signals(tickers: list = None) -> pd.DataFrame:
    """
    Returns DataFrame with columns: ticker, price_signal, computed_at
    """
    from datetime import datetime, timezone
    
    if tickers is None:
        import yaml
        universe = yaml.safe_load(open('config/stock_universe.yaml'))
        tickers = universe.get('asx_tickers', []) + universe.get('us_tickers', universe.get('sp500_tickers', []))
        
    results = []
    now = datetime.now(timezone.utc)
    for ticker in tickers:
        try:
            signal = compute_price_signal(ticker)
            results.append({
                'ticker': ticker,
                'price_signal': signal,
                'computed_at': now
            })
        except Exception as e:
            logger.exception(f"Error computing signal for {ticker}: {e}")
            
    return pd.DataFrame(results)