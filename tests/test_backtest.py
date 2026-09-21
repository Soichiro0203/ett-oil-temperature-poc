import pandas as pd

from ettpoc.backtest import monthly_folds


def test_monthly_folds_cover_last_12_months_without_overlap():
    idx = pd.date_range("2016-07-01", "2018-06-26 19:00", freq="h")
    folds = monthly_folds(idx, n_test_months=12)
    assert len(folds) == 12
    assert folds[0].test_start == pd.Timestamp("2017-07-01")
    assert folds[-1].test_start == pd.Timestamp("2018-06-01")
    assert folds[-1].test_end == pd.Timestamp("2018-07-01")
    for a, b in zip(folds, folds[1:]):
        assert a.test_end == b.test_start  # contiguous
    for f in folds:
        assert f.train_end == f.test_start  # train is strictly before test
        assert f.train_end - idx[0] >= pd.Timedelta(days=365)


def test_fold_masks_are_disjoint():
    idx = pd.date_range("2016-07-01", "2018-06-26 19:00", freq="h")
    f = monthly_folds(idx, 12)[3]
    tr, te = f.train_mask(idx), f.test_mask(idx)
    assert not (tr & te).any()
    assert idx[tr].max() < idx[te].min()
    assert te.sum() == 31 * 24  # October 2017
