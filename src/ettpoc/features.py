"""Feature and target construction for multi-horizon oil-temperature forecasting.

Every feature at time ``t`` is a function of observations at ``t`` or earlier
only: lagged values, rolling statistics with trailing windows, and calendar
features of ``t`` itself.  Future load is never used (it is unknown at
prediction time).  Targets are the *change* in oil temperature from ``t`` to
``t + h``, which removes the slowly drifting level from the learning problem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURE_COLS, TARGET_COL

DEFAULT_OT_LAGS = [1, 2, 3, 4, 5, 6, 9, 12, 18, 24, 48, 168]
DEFAULT_LOAD_LAGS = [0, 1, 3, 6, 12, 24]
DEFAULT_ROLL_WINDOWS = [6, 24, 168]
DEFAULT_HORIZONS = [1, 3, 6, 12, 24]


def build_features(
    df: pd.DataFrame,
    ot_lags: list[int] = DEFAULT_OT_LAGS,
    load_lags: list[int] | None = DEFAULT_LOAD_LAGS,
    roll_windows: list[int] = DEFAULT_ROLL_WINDOWS,
) -> pd.DataFrame:
    """Return the feature matrix aligned on the prediction time ``t``.

    ``load_lags=None`` builds a temperature-only model (used to measure the
    contribution of load information).
    """
    ot = df[TARGET_COL]
    out: dict[str, pd.Series] = {"OT": ot}
    for k in ot_lags:
        out[f"OT_lag{k}"] = ot.shift(k)
        out[f"OT_diff{k}"] = ot - ot.shift(k)
    for w in roll_windows:
        r = ot.rolling(w, min_periods=w)
        out[f"OT_rmean{w}"] = r.mean()
        out[f"OT_rstd{w}"] = r.std()
        out[f"OT_rmin{w}"] = r.min()
        out[f"OT_rmax{w}"] = r.max()
    if load_lags:
        for c in FEATURE_COLS:
            for k in load_lags:
                out[f"{c}_lag{k}"] = df[c].shift(k)
    idx = df.index
    out["hour_sin"] = pd.Series(np.sin(2 * np.pi * idx.hour / 24), index=idx)
    out["hour_cos"] = pd.Series(np.cos(2 * np.pi * idx.hour / 24), index=idx)
    out["doy_sin"] = pd.Series(np.sin(2 * np.pi * idx.dayofyear / 365.25), index=idx)
    out["doy_cos"] = pd.Series(np.cos(2 * np.pi * idx.dayofyear / 365.25), index=idx)
    return pd.DataFrame(out, index=idx)


def build_targets(df: pd.DataFrame, horizons: list[int] = DEFAULT_HORIZONS) -> pd.DataFrame:
    """``delta_h[t] = OT[t+h] - OT[t]``; NaN where ``t+h`` is beyond the data."""
    ot = df[TARGET_COL]
    return pd.DataFrame({f"delta_{h}": ot.shift(-h) - ot for h in horizons}, index=df.index)


def valid_rows(flags: pd.Series, horizon: int) -> pd.Series:
    """Rows usable for training/evaluation at ``horizon``.

    A row is dropped when the observation at ``t`` or at ``t + h`` is a data
    artifact, or when ``t + h`` does not exist.
    """
    future_flag = flags.shift(-horizon)
    ok = (~flags) & (~future_flag.fillna(True).astype(bool))
    return ok.rename("valid")
