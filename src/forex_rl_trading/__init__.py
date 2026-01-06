"""forex_rl_trading

Core package for the R&D project: FX trading environment + feature engineering helpers.
"""

from forex_rl_trading.envs.forex import ForexTradingEnv
from forex_rl_trading.features import prepare_features

__all__ = ["ForexTradingEnv", "prepare_features"]
