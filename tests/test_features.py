import numpy as np
import pandas as pd
import pytest

from ettpoc.data import FEATURE_COLS, TARGET_COL
from ettpoc.features import (
    build_features,
    build_targets,
    valid_rows,
)

HORIZONS = [1, 6, 24]


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=500, freq="h", name="date")
    data = {c: rng.normal(size=500).cumsum() for c in FEATURE_COLS}
    data[TARGET_COL] = 20 + rng.normal(size=500).cumsum()
    return pd.DataFrame(data, index=idx)


def test_no_future_leakage(df):
    """Changing anything strictly after t must not change the features at t."""
    X = build_features(df)
    t = df.index[300]
    df2 = df.copy()
    df2.loc[df2.index > t] += 1000.0  # corrupt the future
    X2 = build_features(df2)
    pd.testing.assert_frame_equal(X.loc[:t], X2.loc[:t])


def test_lag_values_are_exact(df):
    X = build_features(df, ot_lags=[1, 3], load_lags=[2])
    t = df.index[100]
    assert X.loc[t, "OT_lag1"] == df.OT.iloc[99]
    assert X.loc[t, "OT_lag3"] == df.OT.iloc[97]
    assert X.loc[t, "OT_diff3"] == df.OT.iloc[100] - df.OT.iloc[97]
    assert X.loc[t, "HUFL_lag2"] == df.HUFL.iloc[98]
    assert X.loc[t, "OT"] == df.OT.iloc[100]


def test_rolling_uses_past_window_including_now(df):
    X = build_features(df, roll_windows=[24])
    t = df.index[100]
    assert X.loc[t, "OT_rmean24"] == pytest.approx(df.OT.iloc[77:101].mean())
    assert X.loc[t, "OT_rmax24"] == pytest.approx(df.OT.iloc[77:101].max())


def test_time_features(df):
    X = build_features(df)
    six_am = X[X.index.hour == 6].iloc[0]
    assert six_am["hour_sin"] == pytest.approx(1.0)
    assert six_am["hour_cos"] == pytest.approx(0.0, abs=1e-12)
    assert set(["hour_sin", "hour_cos", "doy_sin", "doy_cos"]) <= set(X.columns)


def test_targets_are_future_deltas(df):
    Y = build_targets(df, HORIZONS)
    t = df.index[100]
    assert Y.loc[t, "delta_6"] == pytest.approx(df.OT.iloc[106] - df.OT.iloc[100])
    assert Y.loc[t, "delta_1"] == pytest.approx(df.OT.iloc[101] - df.OT.iloc[100])
    assert Y["delta_6"].iloc[-6:].isna().all()
    assert Y["delta_6"].iloc[:-6].notna().all()
    assert Y["delta_24"].iloc[-24:].isna().all()


def test_no_nan_after_warmup(df):
    X = build_features(df, ot_lags=[1, 24], load_lags=[1], roll_windows=[24, 168])
    assert X.iloc[168:].notna().all().all()
    assert X.iloc[:167].isna().any(axis=1).all()


def test_valid_rows_excludes_flagged_now_or_future(df):
    flags = pd.Series(False, index=df.index)
    flags.iloc[50] = True
    ok = valid_rows(flags, horizon=6)
    assert not ok.iloc[50]        # flagged at t
    assert not ok.iloc[44]        # t + 6 is flagged
    assert ok.iloc[43] and ok.iloc[51]
    assert not ok.iloc[-6:].any()  # no target available
