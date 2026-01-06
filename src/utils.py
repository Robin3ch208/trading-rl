"""Backward-compatible re-exports.

Historically this project imported `src.utils`. We now keep the source of truth in
`forex_rl_trading.features`, but re-export here so existing scripts keep working.
"""

from forex_rl_trading.features import calculate_ema_atr_fast, prepare_features

__all__ = ["prepare_features", "calculate_ema_atr_fast"]
