"""Forecasters. All of them predict the *change* in OT over the horizon."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd


def persistence_delta(ot: pd.Series, horizon: int) -> pd.Series:
    """'Tomorrow is like now': predicted change is zero."""
    return pd.Series(0.0, index=ot.index, name=f"persistence_{horizon}")


def seasonal_naive_delta(ot: pd.Series, horizon: int, period: int = 24) -> pd.Series:
    """'Same as this hour yesterday': OT[t+h] := OT[t+h-24]."""
    if horizon > period:
        raise ValueError("seasonal naive only defined for horizon <= period")
    return (ot.shift(period - horizon) - ot).rename(f"seasonal_naive_{horizon}")


class LGBMDeltaModel:
    """Gradient-boosted trees on lag/calendar features, one model per horizon."""

    def __init__(self, n_estimators: int = 600, learning_rate: float = 0.03, num_leaves: int = 31,
                 min_child_samples: int = 50, subsample: float = 0.8, colsample_bytree: float = 0.8,
                 reg_lambda: float = 1.0, seed: int = 0):
        self.params = dict(
            n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
            min_child_samples=min_child_samples, subsample=subsample, subsample_freq=1,
            colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, random_state=seed,
            objective="regression_l1", verbose=-1,
        )
        self.model: lgb.LGBMRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame | None = None,
            y_val: pd.Series | None = None) -> "LGBMDeltaModel":
        self.model = lgb.LGBMRegressor(**self.params)
        if X_val is not None:
            self.model.fit(X, y, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        else:
            self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "call fit first"
        return self.model.predict(X)

    def feature_importance(self) -> pd.Series:
        assert self.model is not None
        imp = pd.Series(self.model.booster_.feature_importance("gain"), index=self.model.feature_name_)
        return imp.sort_values(ascending=False)
