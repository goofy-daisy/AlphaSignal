"""AlphaSignal — Price features. (Phase 3)
Uses ta library not pandas-ta.
"""
import logging
import numpy as np
import pandas as pd
from sqlalchemy import text
import ta

from src.data.database import get_engine

logger = logging.getLogger(__name__)

def load_price_data(ticker: str) -> pd.DataFrame:
    """Load raw OHLCV data for a ticker from the database."""
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT date, open, high, low, close, volume FROM price_history WHERE ticker = :ticker ORDER BY date ASC"),
            conn,
            params={"ticker": ticker}
        )
    if df.empty:
        return df

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df

def compute_features(ticker: str) -> pd.DataFrame:
    """
    Returns DataFrame with date index and exactly 40 feature columns plus
    the original OHLCV columns. All NaN rows dropped. Minimum 200 rows.
    """
    df = load_price_data(ticker)
    if df.empty:
        logger.warning(f"No price data found for {ticker}")
        return df
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Trend - 8 features
    df['trend_sma_20'] = ta.trend.sma_indicator(close, window=20)
    df['trend_sma_50'] = ta.trend.sma_indicator(close, window=50)
    df['trend_ema_12'] = ta.trend.ema_indicator(close, window=12)
    df['trend_ema_26'] = ta.trend.ema_indicator(close, window=26)
    macd = ta.trend.MACD(close)
    df['trend_macd_line'] = macd.macd()
    df['trend_macd_signal'] = macd.macd_signal()
    df['trend_macd_diff'] = macd.macd_diff()
    df['trend_adx_14'] = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    
    # Momentum - 8 features
    df['momentum_rsi_14'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14)
    df['momentum_stoch_k'] = stoch.stoch()
    df['momentum_stoch_d'] = stoch.stoch_signal()
    df['momentum_williams_r'] = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()
    df['momentum_cci_20'] = ta.trend.CCIIndicator(high, low, close, window=20).cci()
    df['momentum_roc_10'] = ta.momentum.ROCIndicator(close, window=10).roc()
    df['momentum_ultimate_osc'] = ta.momentum.UltimateOscillator(high, low, close).ultimate_oscillator()
    df['momentum_mfi_14'] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index()
    
    # Volatility - 7 features
    bollinger = ta.volatility.BollingerBands(close, window=20)
    df['volatility_bb_high'] = bollinger.bollinger_hband()
    df['volatility_bb_low'] = bollinger.bollinger_lband()
    df['volatility_bb_width'] = bollinger.bollinger_wband()
    df['volatility_atr_14'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    keltner = ta.volatility.KeltnerChannel(high, low, close, window=20)
    df['volatility_kc_high'] = keltner.keltner_channel_hband()
    df['volatility_kc_low'] = keltner.keltner_channel_lband()
    df['volatility_dc_width'] = ta.volatility.DonchianChannel(high, low, close, window=20).donchian_channel_wband()
    
    # Volume - 5 features
    df['volume_obv'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df['volume_vwap'] = ta.volume.VolumeWeightedAveragePrice(high, low, close, volume).volume_weighted_average_price()
    df['volume_cmf'] = ta.volume.ChaikinMoneyFlowIndicator(high, low, close, volume, window=20).chaikin_money_flow()
    df['volume_force_index'] = ta.volume.ForceIndexIndicator(close, volume, window=13).force_index()
    df['volume_eom'] = ta.volume.EaseOfMovementIndicator(high, low, volume, window=14).ease_of_movement()
    
    # Price derived - 12 features
    df['price_log_ret_1d'] = np.log(close / close.shift(1))
    df['price_log_ret_5d'] = np.log(close / close.shift(5))
    df['price_log_ret_20d'] = np.log(close / close.shift(20))
    pct_change = close.pct_change()
    df['price_roll_std_10d'] = pct_change.rolling(10).std()
    df['price_roll_std_20d'] = pct_change.rolling(20).std()
    df['price_roll_skew_20d'] = pct_change.rolling(20).skew()
    df['price_52w_high_ratio'] = close / close.rolling(252).max()
    df['price_52w_low_ratio'] = close / close.rolling(252).min()
    df['price_vs_sma20'] = close / df['trend_sma_20']
    df['price_vs_sma50'] = close / df['trend_sma_50']
    df['price_volume_sma_ratio'] = volume / volume.rolling(20).mean()
    df['price_roll_beta_60d'] = pct_change.rolling(60).std() / pct_change.rolling(252).std()
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how='any')
    
    if len(df) < 200:
        logger.warning(f"{ticker} has fewer than 200 rows after dropping NaNs ({len(df)}).")
        return pd.DataFrame()
        
    return df

def compute_features_all(tickers: list) -> dict:
    """Returns dict of {ticker: DataFrame} for all tickers that succeed."""
    results = {}
    for ticker in tickers:
        try:
            df = compute_features(ticker)
            if not df.empty:
                results[ticker] = df
        except Exception as e:
            logger.error(f"Error computing features for {ticker}: {e}")
            
    return results
