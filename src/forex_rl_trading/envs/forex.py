from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd


class ForexTradingEnv(gym.Env[np.ndarray, int]):
    """Gymnasium FX tick trading environment (MVP).

    Actions:
    - 0: HOLD
    - 1: BUY  (open long if flat)
    - 2: SELL (open short if flat)

    Simplifications (intentional for the R&D MVP):
    - execution at mid price
    - fixed $ costs per side (commission + spread proxy)

    TP/SL are calculated dynamically at trade entry: TP = tp_atr_multiplier * ATR, SL = sl_atr_multiplier * ATR.
    
    Reward (current MVP):
    - close by TP/SL: +tp_atr_multiplier/sl_atr_multiplier if pnl>0 else -1 if pnl<0 else 0
    - if flat and HOLD: neg_reward_for_waiting (to avoid "never trade" collapse)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10_000.0,
        tp_atr_multiplier: float = 10.0,
        sl_atr_multiplier: float = 10.0,
        commission_per_std_lot_per_side: float = 0.0,
        spread_cost_per_std_lot_per_side: float = 0.0,
        position_size: int = 100_000,
        lookback_window: int = 1024,
        neg_reward_for_waiting: float = -0.001,
    ) -> None:
        super().__init__()

        self.df = df.copy()
        self.df["datetime"] = pd.to_datetime(self.df["datetime"])
        if "hour" not in self.df.columns:
            self.df["hour"] = self.df["datetime"].dt.hour

        # ATR is required to calculate TP/SL dynamically, but not in features
        required_columns = ["mid_diff/atr", "hour_sin", "hour_cos", "mid", "datetime", "spread", "atr"]
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(
                f"Colonnes manquantes dans df: {missing}. "
                f"Appliquer forex_rl_trading.features.prepare_features() avant."
            )

        self.initial_balance = float(initial_balance)
        self.lookback_window = int(lookback_window)
        self.tp_atr_multiplier = float(tp_atr_multiplier)
        self.sl_atr_multiplier = float(sl_atr_multiplier)
        self.position_size = int(position_size)
        self.neg_reward_for_waiting = float(neg_reward_for_waiting)

        self.commission_per_std_lot_per_side = float(commission_per_std_lot_per_side)
        self.spread_cost_per_std_lot_per_side = float(spread_cost_per_std_lot_per_side)

        # Base features from dataframe (vary over time)
        self.features_columns = ["mid_diff/atr", "hour_sin", "hour_cos"]
        # Total features: 3 base + 3 window-based (d_low, d_high, pos_range)
        self.n_features = 6

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback_window, self.n_features),
            dtype=np.float32,
        )

        self.current_idx = 0
        self.position = 0  # -1 short, 0 flat, 1 long
        self.entry_price = 0.0
        self.entry_idx = -1
        self.entry_tp = 0.0  # TP calculated at entry time (tp_atr_multiplier * ATR)
        self.entry_sl = 0.0  # SL calculated at entry time (sl_atr_multiplier * ATR)
        self.balance = self.initial_balance
        self.trades_history: list[dict[str, Any]] = []
        self.total_costs = 0.0
        self.total_pnl = 0.0

        self.reset()

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        _ = options

        self.current_idx = 0
        self.position = 0
        self.entry_price = 0.0
        self.entry_idx = -1
        self.entry_tp = 0.0
        self.entry_sl = 0.0
        self.balance = self.initial_balance
        self.trades_history = []
        self.total_costs = 0.0
        self.total_pnl = 0.0

        return self._get_observation(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        terminated = False
        truncated = False
        reward = 0.0
        info: dict[str, Any] = {"trade": None}

        # 1) If position open, check TP/SL
        close_trade = False
        if self.position != 0:
            current_price = self._get_current_price()
            if self._check_tp_sl(current_price):
                pnl = self._close_position(current_price, "tp_sl")
                self.total_pnl += pnl
                reward = self._calculate_trade_reward(pnl)
                info["trade"] = {"action": "close", "reason": "tp_sl"}
                close_trade = True

        # 2) If flat and not closing this step, optionally open a trade
        if not close_trade and self.position == 0 and action in (1, 2):
            current_price = self._get_current_price()
            current_atr = self._get_current_atr()
            
            # Calculate TP/SL dynamically based on current ATR
            self.entry_tp = self.tp_atr_multiplier * current_atr
            self.entry_sl = self.sl_atr_multiplier * current_atr
            
            cost = self._get_cost_per_side()
            self.total_costs += cost
            self.balance -= cost

            if action == 1:
                self.position = 1
                self.entry_price = current_price
                self.entry_idx = self.current_idx
                info["trade"] = {
                    "action": "buy",
                    "price": self.entry_price,
                    "tp": self.entry_tp,
                    "sl": self.entry_sl,
                    "atr": current_atr,
                }
            else:
                self.position = -1
                self.entry_price = current_price
                self.entry_idx = self.current_idx
                info["trade"] = {
                    "action": "sell",
                    "price": self.entry_price,
                    "tp": self.entry_tp,
                    "sl": self.entry_sl,
                    "atr": current_atr,
                }

        # 3) Holding flat gets a small penalty
        if not close_trade and self.position == 0 and action == 0:
            reward += self.neg_reward_for_waiting

        # 4) Advance tick
        self.current_idx += 1

        # 5) End of data
        if self.current_idx >= len(self.df) - 1:
            truncated = True
            if self.position != 0:
                current_price = self._get_current_price()
                _ = self._close_position(current_price, "data_end")

        return self._get_observation(), float(reward), terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Build observation with base features + window-based features.
        
        Base features (vary over time): mid_diff/atr, hour_sin, hour_cos
        Window features (constant over window): d_low, d_high, pos_range
        
        Window: [current_idx - lookback_window + 1, current_idx] (inclusive both ends)
        This gives exactly lookback_window elements including current_idx.
        """
        # Determine window bounds: [start, current_idx] inclusive
        if self.current_idx < self.lookback_window:
            start = 0
            end = self.current_idx + 1  # +1 because iloc is exclusive
            window_size = end - start
        else:
            # start = current_idx - lookback_window + 1 to get exactly lookback_window elements
            start = self.current_idx - self.lookback_window + 1
            end = self.current_idx + 1  # +1 because iloc is exclusive
            window_size = self.lookback_window
        
        # Get base features (vary over time)
        base_features = self.df.iloc[start:end][self.features_columns].values  # (window_size, 3)
        
        # Calculate window-based features for current tick (constant over window)
        d_low, d_high, pos_range = self._calculate_window_features(start, end)
        
        # Build full observation
        obs = np.zeros((self.lookback_window, self.n_features), dtype=np.float32)
        
        if self.current_idx < self.lookback_window:
            # Padding case: fill from the end
            obs[-window_size:, :3] = base_features
            obs[-window_size:, 3] = d_low
            obs[-window_size:, 4] = d_high
            obs[-window_size:, 5] = pos_range
        else:
            # Normal case: fill entire window
            obs[:, :3] = base_features
            obs[:, 3] = d_low
            obs[:, 4] = d_high
            obs[:, 5] = pos_range
        
        return obs
    
    def _calculate_window_features(self, start: int, end: int) -> tuple[float, float, float]:
        """Calculate d_low, d_high, pos_range for the current window.
        
        Args:
            start: Start index of the window (inclusive)
            end: End index of the window (exclusive, so end-1 is the current tick)
            
        Returns:
            Tuple of (d_low, d_high, pos_range) for the current tick (end-1)
        """
        # Get mid prices in the window [start, end) (end is exclusive)
        window_mids = np.asarray(self.df.iloc[start:end]["mid"].values, dtype=np.float64)
        current_idx = end - 1  # Current tick is the last one in the window
        current_mid = float(self.df.iloc[current_idx]["mid"])
        current_atr = float(self.df.iloc[current_idx]["atr"])
        
        min_mid = float(np.min(window_mids))
        max_mid = float(np.max(window_mids))
        range_mid = max_mid - min_mid
        
        # d_low: distance from current price to minimum, normalized by ATR
        d_low = (current_mid - min_mid) / (current_atr + 1e-8)
        
        # d_high: distance from maximum to current price, normalized by ATR
        d_high = (max_mid - current_mid) / (current_atr + 1e-8)
        
        # pos_range: relative position in the range [0, 1]
        eps = 1e-8
        pos_range = (current_mid - min_mid) / (range_mid + eps)
        
        return (d_low, d_high, pos_range)

    def _get_current_price(self) -> float:
        return float(self.df.iloc[self.current_idx]["mid"])

    def _get_current_atr(self) -> float:
        """Get current ATR value from dataframe."""
        return float(self.df.iloc[self.current_idx]["atr"])

    def _check_tp_sl(self, current_price: float) -> bool:
        """Check if TP or SL is hit using dynamically calculated values."""
        if self.position == 0:
            return False

        if self.position == 1:  # LONG
            tp_price = self.entry_price + self.entry_tp
            sl_price = self.entry_price - self.entry_sl
            return current_price >= tp_price or current_price <= sl_price

        # SHORT
        tp_price = self.entry_price - self.entry_tp
        sl_price = self.entry_price + self.entry_sl
        return current_price <= tp_price or current_price >= sl_price

    def _calculate_pnl(self, exit_price: float) -> float:
        pnl_before_cost = self.position * (exit_price - self.entry_price) * self.position_size
        return float(pnl_before_cost - self._get_cost_per_side())

    def _close_position(self, exit_price: float, reason: str) -> float:
        pnl = self._calculate_pnl(exit_price)

        self.trades_history.append(
            {
                "entry_idx": self.entry_idx,
                "exit_idx": self.current_idx,
                "position": "LONG" if self.position == 1 else "SHORT",
                "entry_price": float(self.entry_price),
                "exit_price": float(exit_price),
                "entry_tp": float(self.entry_tp),  # Store TP used for this trade
                "entry_sl": float(self.entry_sl),  # Store SL used for this trade
                "pnl": float(pnl),
                "reason": reason,
                "duration": int(self.current_idx - self.entry_idx),
            }
        )

        self.balance += pnl

        self.position = 0
        self.entry_price = 0.0
        self.entry_idx = -1
        self.entry_tp = 0.0
        self.entry_sl = 0.0

        return float(pnl)

    def _calculate_trade_reward(self, pnl: float) -> float:
        """Calculate reward using TP/SL ratio (based on ATR multipliers)."""
        if pnl > 0:
            return float(self.tp_atr_multiplier / self.sl_atr_multiplier)
        if pnl < 0:
            return -1.0
        return 0.0

    def _get_cost_per_side(self) -> float:
        commission_cost = self.commission_per_std_lot_per_side * self.position_size / 100_000
        spread_cost = self.spread_cost_per_std_lot_per_side * self.position_size / 100_000
        return float(commission_cost + spread_cost)

    def get_stats(self) -> dict[str, Any]:
        closed = [t for t in self.trades_history if "exit_price" in t]
        total_pnl_closed = float(sum(t["pnl"] for t in closed)) if closed else 0.0
        win_rate = float(sum(1 for t in closed if t["pnl"] > 0) / len(closed)) if closed else 0.0
        return {
            "n_trades": len(self.trades_history),
            "n_closed": len(closed),
            "total_pnl_closed": total_pnl_closed,
            "balance": float(self.balance),
            "profit": float(self.balance - self.initial_balance),
            "win_rate": win_rate,
            "trades": closed[-5:],
        }
