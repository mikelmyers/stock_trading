"""Machine-learning layer for the trading research agent.

This package reframes the existing rule-based backtest as a supervised-learning
problem:

    features at decision time  ->  forward trade outcome (from the simulator)

The rule-based setups (``agents/setups``) act as the *candidate generator* and
``simulate_trade_forward`` acts as the *labeler*. The ML model learns to rank /
filter those candidates by predicted edge instead of relying solely on the
hand-coded ``confidence_score``.

Modules
-------
features : point-in-time, causal feature engineering from OHLCV.
dataset  : walk the price cache, emit (features, label) rows, save to disk.
splits   : purged, embargoed walk-forward cross-validation (no look-ahead).
model    : train + honestly validate a gradient-boosted model.

Nothing here is imported by the live trading/calibration path, so it cannot
affect a training run in progress.
"""
