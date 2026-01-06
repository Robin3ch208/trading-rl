from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def prepare_features(
    df: pd.DataFrame, high_spread_hours: Iterable[int] = (15, 16, 17, 18, 19)
) -> pd.DataFrame:
    """Prepare features for the RL environment / policy.

    Input expected (minimum):
    - datetime (parseable)
    - mid (float)
    - spread (float)

    Output adds:
    - hour
    - atr (smoothed)
    - mid_diff
    - mid_diff/atr (winsorized)
    - hour_sin, hour_cos

    It also filters out hours typically associated with high spread (default 15h-19h).
    """
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    if "hour" not in out.columns:
        out["hour"] = out["datetime"].dt.hour

    out["atr"] = calculate_ema_atr_fast(out, atr_period=10_000, ema_period=20_000)
    out["mid_diff"] = out["mid"].diff()

    # Identify artificial gaps between <15h and >=20h (market close/open) and neutralize.
    out["prev_hour"] = out["hour"].shift(1)
    is_gap = (out["prev_hour"] < 15) & (out["hour"] >= 20)
    out.loc[is_gap, "mid_diff"] = 0.0

    out["mid_diff/atr"] = out["mid_diff"] / out["atr"]
    out["mid_diff/atr"] = out["mid_diff/atr"].clip(-10, 10)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    out = out[~out["hour"].isin(list(high_spread_hours))].copy()

    out.drop(columns=["prev_hour"], inplace=True)
    out.dropna(inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def calculate_ema_atr_fast(df: pd.DataFrame, atr_period: int, ema_period: int) -> pd.Series:
    """Fast ATR proxy for tick data.

    Uses abs(mid.diff()) and smooths with rolling mean + EWM.
    Robust on short slices via min_periods=1.
    """
    tr = df["mid"].diff().abs()
    sma = tr.rolling(window=atr_period, min_periods=1).mean()
    ema_atr = sma.ewm(span=ema_period, adjust=False).mean()
    return ema_atr.clip(lower=1e-6)
