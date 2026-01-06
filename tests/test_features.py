import pandas as pd

from forex_rl_trading.features import prepare_features


def test_prepare_features_adds_expected_columns_and_filters_hours() -> None:
    df = pd.DataFrame(
        {
            "datetime": [
                "2024-01-02 14:59:00",
                "2024-01-02 20:00:00",  # gap open
                "2024-01-02 20:00:01",
            ],
            "mid": [1.10000, 1.10100, 1.10110],
            "spread": [0.00002, 0.00002, 0.00002],
        }
    )

    out = prepare_features(df, high_spread_hours=(15, 16, 17, 18, 19))

    for col in ["atr", "mid_diff", "mid_diff/atr", "hour_sin", "hour_cos", "hour", "datetime"]:
        assert col in out.columns

    # Should keep only tradable hours (not 15-19). Our sample uses 14 and 20, so both ok.
    assert not out["hour"].isin([15, 16, 17, 18, 19]).any()


def test_prepare_features_neutralizes_14_to_20_gap_mid_diff() -> None:
    df = pd.DataFrame(
        {
            "datetime": ["2024-01-02 14:59:00", "2024-01-02 20:00:00", "2024-01-02 20:00:01"],
            "mid": [1.0000, 1.5000, 1.5001],
            "spread": [0.00002, 0.00002, 0.00002],
        }
    )
    out = prepare_features(df)

    # The first row is dropped by diff/NaNs, so "20:00:00" becomes index 0 in output.
    # At that gap-open tick, mid_diff should have been neutralized to 0.0 (before normalization).
    assert float(out.loc[0, "mid_diff"]) == 0.0
