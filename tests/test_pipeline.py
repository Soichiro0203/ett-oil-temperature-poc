import numpy as np
import pandas as pd

from ettpoc.data import FEATURE_COLS, TARGET_COL
from ettpoc.pipeline import run_backtest


def _synthetic():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2016-07-01", "2018-06-26 19:00", freq="h", name="date")
    n = len(idx)
    hour = idx.hour.to_numpy()
    ot = 20 + 5 * np.sin(2 * np.pi * hour / 24) + rng.normal(scale=0.5, size=n).cumsum() * 0.05
    data = {c: rng.normal(size=n) for c in FEATURE_COLS}
    data[TARGET_COL] = ot
    return pd.DataFrame(data, index=idx)


def test_run_backtest_schema_and_baselines():
    df = _synthetic()
    preds = run_backtest(df, horizons=[1, 6], n_test_months=2, variants=["lgbm"], lgbm_kwargs={"n_estimators": 20})
    assert set(preds.columns) >= {"date", "horizon", "model", "ot_now", "delta_true", "delta_pred"}
    assert set(preds.model) == {"persistence", "seasonal_naive", "lgbm"}
    assert set(preds.horizon) == {1, 6}
    # predictions exist only inside the two test months
    assert preds.date.min() >= pd.Timestamp("2018-05-01")
    assert preds.date.max() < pd.Timestamp("2018-07-01")
    assert (preds.loc[preds.model == "persistence", "delta_pred"] == 0).all()
    # within a horizon every model covers the same timestamps
    counts = preds.groupby(["horizon", "model"]).size()
    assert (counts.groupby("horizon").nunique() == 1).all()
    # delta_true is consistent with the raw series
    row = preds[(preds.model == "lgbm") & (preds.horizon == 6)].iloc[10]
    t = row.date
    assert row.delta_true == df.OT[t + pd.Timedelta(hours=6)] - df.OT[t]
    assert row.ot_now == df.OT[t]
