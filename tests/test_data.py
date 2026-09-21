import pandas as pd
import pytest

from ettpoc.data import FEATURE_COLS, TARGET_COL, load_ett


def test_columns_and_index():
    df = load_ett("ETTh1")
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "date"
    assert list(df.columns) == FEATURE_COLS + [TARGET_COL]


def test_etth1_is_hourly_and_complete():
    df = load_ett("ETTh1")
    assert len(df) == 17420
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates
    # no missing timestamps: every consecutive diff is exactly 1 hour
    assert (df.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


def test_ettm1_is_15min():
    df = load_ett("ETTm1")
    assert len(df) == 69680
    assert (df.index.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all()


def test_no_missing_values():
    for name in ["ETTh1", "ETTh2"]:
        assert not load_ett(name).isna().any().any(), name


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        load_ett("ETTx9")


# ---- artifact flagging -------------------------------------------------------
from ettpoc.data import flag_artifacts


def _toy():
    idx = pd.date_range("2020-01-01", periods=5, freq="h", name="date")
    df = pd.DataFrame(
        {c: [1.0, 2.0, 2.0, 3.0, 4.0] for c in FEATURE_COLS} | {"OT": [10.0, 11.0, 11.0, 0.0, 12.0]},
        index=idx,
    )
    return df


def test_flag_artifacts_on_toy():
    flags = flag_artifacts(_toy())
    # row 2 is an exact copy of row 1 -> frozen; row 3 has OT == 0 -> dropout
    assert flags.tolist() == [False, False, True, True, False]
    assert flags.name == "is_artifact"
    assert flags.dtype == bool


def test_flag_artifacts_catches_day31_in_etth1():
    df = load_ett("ETTh1")
    flags = flag_artifacts(df)
    assert flags.loc["2016-07-31 00:00"] is not None  # index alignment
    assert not flags.loc["2016-07-31 00:00"]          # first hour is real
    assert flags.loc["2016-07-31 01:00":"2016-07-31 23:00"].all()
    assert not flags.loc["2016-08-01 00:00"]
    # 14 months with 31 days in the span, 23 forward-filled hours each
    assert flags[flags.index.day == 31].sum() == 14 * 23
    # sensor dropout: OT exactly 0.0 with normal load
    assert flags.loc["2018-01-05 03:00"]
