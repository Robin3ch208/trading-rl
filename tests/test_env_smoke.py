import pandas as pd

from forex_rl_trading.envs.forex import ForexTradingEnv


def _make_minimal_featured_df(n: int = 50) -> pd.DataFrame:
    base = pd.Timestamp("2024-01-02 10:00:00")
    dt = [base + pd.Timedelta(seconds=i) for i in range(n)]
    return pd.DataFrame(
        {
            "datetime": dt,
            "hour": [10] * n,
            "mid": [1.10000 + 1e-5 * i for i in range(n)],
            "spread": [0.00002] * n,
            "atr": [0.00001] * n,  # ATR required for dynamic TP/SL calculation
            "mid_diff/atr": [0.0] * n,
            "hour_sin": [0.0] * n,
            "hour_cos": [1.0] * n,
        }
    )


def test_env_reset_and_step_hold() -> None:
    df = _make_minimal_featured_df()
    env = ForexTradingEnv(df=df, lookback_window=10, neg_reward_for_waiting=-0.123)

    obs, info = env.reset()
    assert obs.shape == (10, 6)  # lookback window * n_features (3 base + 3 window-based)
    assert info == {}

    obs2, reward, terminated, truncated, info2 = env.step(0)  # 0 = HOLD
    assert obs2.shape == (10, 6)  # lookback window * n_features (3 base + 3 window-based)
    assert reward == -0.123
    assert terminated is False
    assert isinstance(truncated, bool)
    assert "trade" in info2
