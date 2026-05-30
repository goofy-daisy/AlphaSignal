# AlphaSignal Methodology

## Walk-Forward Validation
All model evaluation uses strictly walk-forward validation windows with no lookahead bias — train/test splits are never used.

## Information Coefficient
The Information Coefficient (IC) measures the rank correlation between predicted composite scores and realized next-period returns across all walk-forward windows.

## Signal Design
Four heterogeneous signals (price, sentiment, filing, social) are combined by a LightGBM meta-learner into a composite score in [-1, 1].

## Known Limitations
All inference is local on a MacBook Air M5 and model capacity is constrained by 16GB unified memory and MPS acceleration availability.
