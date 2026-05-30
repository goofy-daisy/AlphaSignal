"""AlphaSignal — Walk forward validation. (Phase 3)"""
import os
import shutil
import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pytorch_forecasting import TimeSeriesDataSet
from src.models.tft_model import build_tft_dataset, train_tft, predict_tft, make_target

logger = logging.getLogger(__name__)

def compute_ic(predicted_scores: np.ndarray, actual_returns: np.ndarray) -> float:
    mask = ~np.isnan(predicted_scores) & ~np.isnan(actual_returns)
    if mask.sum() < 10:
        return 0.0
    ic, _ = spearmanr(predicted_scores[mask], actual_returns[mask])
    return float(ic) if not np.isnan(ic) else 0.0

def run_walk_forward(ticker: str, features_df: pd.DataFrame) -> dict:
    """
    Runs walk-forward validation for one ticker.
    Returns dict with keys:
        windows: list of dicts with train_start, train_end, val_start, val_end, ic
        mean_ic: float
        best_window: int (index of window with highest IC)
        best_checkpoint: str (path to best checkpoint file)
    """
    n_days = len(features_df)
    train_days = 756
    val_days = 126
    step_days = 63
    
    effective_days = n_days - 20
    if effective_days < train_days + val_days:
        logger.warning(f"Not enough data for {ticker}: {n_days} days.")
        return {}
        
    val_starts = list(range(train_days, effective_days - val_days + 1, step_days))
    if len(val_starts) < 2:
        logger.warning(f"Not enough windows for {ticker}: {len(val_starts)} windows.")
        return {}
        
    os.makedirs('models/checkpoints', exist_ok=True)
    
    windows_results = []
    
    for i, val_start_idx in enumerate(val_starts):
        logger.info(f"Running window {i} for {ticker}")
        train_start_idx = val_start_idx - train_days
        val_end_idx = val_start_idx + val_days
        
        train_slice = features_df.iloc[train_start_idx:val_start_idx].copy()
        
        val_slice_end = min(val_end_idx + 40, n_days)
        val_slice_raw = features_df.iloc[val_start_idx - 60 : val_slice_end].copy()
        
        train_ds, _, train_dl, val_dl = build_tft_dataset(train_slice, ticker)
        
        model = train_tft(train_dl, val_dl, ticker)
        
        val_slice_raw['target'] = make_target(val_slice_raw['close'])
        val_slice_raw['target_float'] = val_slice_raw['close'].shift(-20) / val_slice_raw['close'] - 1
        
        val_slice = val_slice_raw.dropna(subset=['target']).copy()
        if len(val_slice) < 60 + 1:
            logger.warning(f"Window {i} validation slice too short.")
            continue
            
        val_slice['target'] = val_slice['target'].astype(int).astype(str)
        val_slice = val_slice.sort_index()
        val_slice['time_idx'] = np.arange(len(val_slice))
        val_slice['ticker'] = ticker
        
        try:
            test_ds = TimeSeriesDataSet.from_dataset(train_ds, val_slice, predict=False, stop_randomization=True)
            test_dl = test_ds.to_dataloader(train=False, batch_size=32, num_workers=0)
        except Exception as e:
            logger.warning(f"Failed to create test dataloader for window {i}: {e}")
            continue
            
        probs = predict_tft(model, test_dl)
        if probs.shape[0] == 0:
            logger.warning(f"No predictions generated for window {i}")
            continue
            
        scores = probs[:, 0]*(-1.0) + probs[:, 1]*(-0.5) + probs[:, 2]*0.0 + probs[:, 3]*0.5 + probs[:, 4]*1.0
        
        predicted_time_indices = test_ds.index['time'].values + 60
        actual_returns = val_slice['target_float'].iloc[predicted_time_indices].values
        
        ic = compute_ic(scores, actual_returns)
        logger.info(f"Window {i} IC: {ic:.4f}")
        
        checkpoint_path = f"models/checkpoints/tft_{ticker}_window_{i}.ckpt"
        if hasattr(model, 'best_model_path') and model.best_model_path and os.path.exists(model.best_model_path):
            shutil.copy(model.best_model_path, checkpoint_path)
        else:
            import lightning.pytorch as pl
            trainer = pl.Trainer()
            trainer.strategy.connect(model)
            trainer.save_checkpoint(checkpoint_path)
            
        windows_results.append({
            'train_start': str(train_slice.index[0]),
            'train_end': str(train_slice.index[-1]),
            'val_start': str(val_slice.index[60] if len(val_slice) > 60 else val_slice.index[-1]),
            'val_end': str(val_slice.index[-1]),
            'ic': ic,
            'checkpoint': checkpoint_path,
            'window_idx': i
        })
        
    if not windows_results:
        return {}
        
    mean_ic = np.mean([w['ic'] for w in windows_results])
    best_window_idx = np.argmax([w['ic'] for w in windows_results])
    best_checkpoint = windows_results[best_window_idx]['checkpoint']
    
    return {
        'windows': windows_results,
        'mean_ic': float(mean_ic),
        'best_window': int(best_window_idx),
        'best_checkpoint': best_checkpoint
    }

def run_walk_forward_all(tickers: list = None) -> pd.DataFrame:
    """
    Runs walk-forward for all tickers or provided list.
    Returns DataFrame with columns: ticker, mean_ic, n_windows, best_checkpoint
    Saves results to models/walk_forward_results.csv
    """
    from src.features.price_features import compute_features
    if tickers is None:
        import yaml
        universe = yaml.safe_load(open('config/stock_universe.yaml'))
        tickers = universe.get('asx_tickers', []) + universe.get('us_tickers', universe.get('sp500_tickers', []))
        
    results = []
    for ticker in tickers:
        try:
            logger.info(f"Processing {ticker}...")
            df = compute_features(ticker)
            if df.empty:
                continue
            wf_res = run_walk_forward(ticker, df)
            if wf_res:
                results.append({
                    'ticker': ticker,
                    'mean_ic': wf_res['mean_ic'],
                    'n_windows': len(wf_res['windows']),
                    'best_checkpoint': wf_res['best_checkpoint']
                })
        except Exception as e:
            logger.exception(f"Error processing {ticker}: {e}")
            
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        os.makedirs('models', exist_ok=True)
        res_df.to_csv('models/walk_forward_results.csv', index=False)
    return res_df