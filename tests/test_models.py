import numpy as np
import pandas as pd
import pytest

from ettpoc.models import LGBMDeltaModel, persistence_delta, seasonal_naive_delta


def _ot():
    idx = pd.date_range("2020-01-01", periods=200, freq="h", name="date")
    return pd.Series(np.arange(200, dtype=float) ** 1.1, index=idx, name="OT")


def test_persistence_predicts_no_change():
    ot = _ot()
    assert (persistence_delta(ot, 6) == 0).all()


def test_seasonal_naive_uses_same_hour_yesterday():
    ot = _ot()
    d = seasonal_naive_delta(ot, 6)
    t = 100
    # predicted OT[t+6] = OT[t+6-24] -> delta = OT[t-18] - OT[t]
    assert d.iloc[t] == pytest.approx(ot.iloc[t - 18] - ot.iloc[t])
    assert (seasonal_naive_delta(ot, 24).dropna() == 0).all()


def test_lgbm_learns_simple_relation():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(2000, 3)), columns=list("abc"))
    y = 2 * X["a"] - X["b"] + rng.normal(scale=0.05, size=2000)
    m = LGBMDeltaModel(n_estimators=300).fit(X.iloc[:1500], y.iloc[:1500])
    pred = m.predict(X.iloc[1500:])
    assert np.sqrt(np.mean((pred - y.iloc[1500:]) ** 2)) < 0.3
    assert set(m.feature_importance().index) == {"a", "b", "c"}
    assert m.feature_importance().iloc[0] > m.feature_importance()["c"]  # sorted, informative first


def test_lgbm_accepts_nan_features():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(500, 2)), columns=["a", "b"])
    X.iloc[:50, 0] = np.nan
    y = X["a"].fillna(0) + X["b"]
    m = LGBMDeltaModel(n_estimators=50).fit(X, y)
    assert np.isfinite(m.predict(X)).all()
