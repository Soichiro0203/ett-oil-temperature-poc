"""Rolling-origin (expanding window) monthly backtest.

For each of the last ``n_test_months`` calendar months, the model is trained on
all data strictly before that month and evaluated on the month.  This mimics a
"retrain monthly" operation and yields an evaluation that spans every season.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    train_end: pd.Timestamp  # exclusive
    test_start: pd.Timestamp  # inclusive
    test_end: pd.Timestamp  # exclusive

    def train_mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray(index < self.train_end)

    def test_mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray((index >= self.test_start) & (index < self.test_end))


def monthly_folds(index: pd.DatetimeIndex, n_test_months: int = 12) -> list[Fold]:
    last_month = index.max().to_period("M")
    months = pd.period_range(last_month - (n_test_months - 1), last_month, freq="M")
    return [
        Fold(train_end=m.start_time, test_start=m.start_time, test_end=(m + 1).start_time)
        for m in months
    ]
