"""AlphaSignal — TFT model. (Phase 3)"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import logging
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import CrossEntropy

logger = logging.getLogger(__name__)

def make_target(close: pd.Series) -> pd.Series:
    """Computes 5-bucket label from 20-day forward returns."""
    fwd_return = close.shift(-20) / close - 1
    bins = [-np.inf, -0.05, -0.01, 0.01, 0.05, np.inf]
    labels = [0, 1, 2, 3, 4]
    return pd.cut(fwd_return, bins=bins, labels=labels).astype(float)

def build_tft_dataset(features_df: pd.DataFrame, ticker: str) -> tuple:
    """Returns (train_dataset, val_dataset, train_dataloader, val_dataloader)"""
    df = features_df.copy()
    df['target'] = make_target(df['close'])
    
    # Drop rows where target is NaN (the last 20 days)
    df = df.dropna(subset=['target']).copy()
    
    # Target must be string for CrossEntropy classification in pytorch-forecasting
    df['target'] = df['target'].astype(int).astype(str)
    
    df = df.sort_index()
    df['time_idx'] = np.arange(len(df))
    df['ticker'] = ticker
    
    feature_cols = [c for c in df.columns if c not in ['target', 'time_idx', 'ticker']]
    
    # Use last 100 days of the training slice for validation (EarlyStopping)
    max_time = df['time_idx'].max()
    val_cutoff = max_time - 100
    
    df_train = df[df['time_idx'] <= val_cutoff].copy()
    
    target_encoder = NaNLabelEncoder(add_nan=False)
    target_encoder.fit(np.array(["0", "1", "2", "3", "4"]))
    
    train_dataset = TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",
        target="target",
        group_ids=["ticker"],
        min_encoder_length=60,
        max_encoder_length=60,
        min_prediction_length=20,
        max_prediction_length=20,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=feature_cols,
        target_normalizer=target_encoder,
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
    )
    
    # Validation dataset for EarlyStopping
    df_val = df[df['time_idx'] > val_cutoff - 60 - 20].copy()
    val_dataset = TimeSeriesDataSet.from_dataset(train_dataset, df_val, predict=False, stop_randomization=True)
    
    batch_size = 32
    train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    
    return train_dataset, val_dataset, train_dataloader, val_dataloader

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

def train_tft(train_dataloader, val_dataloader, ticker: str) -> pl.LightningModule:
    """Trains TFT and returns best model checkpoint"""
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename=f"tft-{ticker}-" + "{epoch}-{val_loss:.2f}",
        dirpath="models/checkpoints/tmp"
    )
    
    accelerator = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=False,
        enable_model_summary=False,
    )
    
    tft = TemporalFusionTransformer.from_dataset(
        train_dataloader.dataset,
        learning_rate=0.001,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=5,
        loss=CrossEntropy(),
        log_interval=0,
        reduce_on_plateau_patience=4,
    )
    
    logger.info(f"Training TFT for {ticker} on {accelerator}...")
    try:
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
    except Exception as e:
        logger.warning(f"Training on {accelerator} failed: {e}. Falling back to CPU.")
        if accelerator != 'cpu':
            logger.info("Initializing new trainer and callbacks for CPU fallback...")
            cpu_early_stop = EarlyStopping(
                monitor="val_loss",
                patience=5,
                mode="min",
            )
            cpu_checkpoint = ModelCheckpoint(
                dirpath="models/checkpoints/tmp_cpu",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            )
            trainer = pl.Trainer(
                max_epochs=50,
                accelerator='cpu',
                devices=1,
                gradient_clip_val=0.1,
                enable_progress_bar=True,
                callbacks=[cpu_early_stop, cpu_checkpoint],
                logger=False,
                enable_model_summary=False,
            )
            tft = TemporalFusionTransformer.from_dataset(
                train_dataloader.dataset,
                learning_rate=0.001,
                hidden_size=16,
                attention_head_size=1,
                dropout=0.1,
                hidden_continuous_size=8,
                output_size=5,
                loss=CrossEntropy(),
                log_interval=0,
                reduce_on_plateau_patience=4,
            )
            trainer.fit(
                tft,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )
            
            best_model_path = cpu_checkpoint.best_model_path
        else:
            raise e
    else:
        best_model_path = checkpoint_callback.best_model_path
        
    if best_model_path:
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        best_tft.best_model_path = best_model_path
        return best_tft
    
    trainer.model.best_model_path = ""
    return trainer.model

def predict_tft(model, val_dataloader) -> np.ndarray:
    """Returns array of shape (N, 5) with class probabilities"""
    model.eval()
    with torch.no_grad():
        out = model.predict(val_dataloader, mode="raw", return_x=False, trainer_kwargs=dict(accelerator='cpu'))
        logits = out.prediction[:, 0, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs