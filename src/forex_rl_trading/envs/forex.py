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

    Reward (current MVP):
    - close by TP/SL: +tp/sl if pnl>0 else -1 if pnl<0 else 0
    - if flat and HOLD: neg_reward_for_waiting (to avoid "never trade" collapse)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10_000.0,
        tp: float = 0.0002,
        sl: float = 0.0002,
        commission_per_std_lot_per_side: float = 0.0,
        spread_cost_per_std_lot_per_side: float = 0.0,
        position_size: int = 100_000,
        lookback_window: int = 1000,
        neg_reward_for_waiting: float = -0.001,
    ) -> None:
        super().__init__()

        self.df = df.copy()
        self.df["datetime"] = pd.to_datetime(self.df["datetime"])
        if "hour" not in self.df.columns:
            self.df["hour"] = self.df["datetime"].dt.hour

        required_columns = ["mid_diff/atr", "hour_sin", "hour_cos", "mid", "datetime", "spread"]
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(
                f"Colonnes manquantes dans df: {missing}. "
                f"Appliquer forex_rl_trading.features.prepare_features() avant."
            )

        self.initial_balance = float(initial_balance)
        self.lookback_window = int(lookback_window)
        self.base_tp = float(tp)
        self.base_sl = float(sl)
        self.position_size = int(position_size)
        self.neg_reward_for_waiting = float(neg_reward_for_waiting)

        self.commission_per_std_lot_per_side = float(commission_per_std_lot_per_side)
        self.spread_cost_per_std_lot_per_side = float(spread_cost_per_std_lot_per_side)

        self.features_columns = ["mid_diff/atr", "hour_sin", "hour_cos"]
        self.n_features = len(self.features_columns)

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
            cost = self._get_cost_per_side()
            self.total_costs += cost
            self.balance -= cost

            if action == 1:
                self.position = 1
                self.entry_price = current_price
                self.entry_idx = self.current_idx
                info["trade"] = {"action": "buy", "price": self.entry_price}
            else:
                self.position = -1
                self.entry_price = current_price
                self.entry_idx = self.current_idx
                info["trade"] = {"action": "sell", "price": self.entry_price}

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
        obs = np.zeros((self.lookback_window, self.n_features), dtype=np.float32)
        if self.current_idx < self.lookback_window:
            if self.current_idx > 0:
                real_data = self.df.iloc[: self.current_idx][self.features_columns].values
                obs[-self.current_idx :, :] = real_data
        else:
            start = self.current_idx - self.lookback_window
            end = self.current_idx
            obs[:, :] = self.df.iloc[start:end][self.features_columns].values
        return obs

    def _get_current_price(self) -> float:
        return float(self.df.iloc[self.current_idx]["mid"])

    def _check_tp_sl(self, current_price: float) -> bool:
        if self.position == 0:
            return False

        if self.position == 1:
            tp_price = self.entry_price + self.base_tp
            sl_price = self.entry_price - self.base_sl
            return current_price >= tp_price or current_price <= sl_price

        tp_price = self.entry_price - self.base_tp
        sl_price = self.entry_price + self.base_sl
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
                "pnl": float(pnl),
                "reason": reason,
                "duration": int(self.current_idx - self.entry_idx),
            }
        )

        self.balance += pnl

        self.position = 0
        self.entry_price = 0.0
        self.entry_idx = -1

        return float(pnl)

    def _calculate_trade_reward(self, pnl: float) -> float:
        if pnl > 0:
            return float(self.base_tp / self.base_sl)
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
