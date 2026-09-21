"""Metrics: standard regression errors plus operational event metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> dict:
    err = np.asarray(y_true) - np.asarray(y_pred)
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err**2))), "n": int(len(err))}


def event_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """Detection metrics for a binary risk event.

    recall   = share of true events that were warned in advance (検知率)
    precision= share of warnings that were true (1 - 誤報率)
    """
    a = np.asarray(actual, dtype=bool); p = np.asarray(predicted, dtype=bool)
    tp = int((a & p).sum()); fp = int((~a & p).sum()); fn = int((a & ~p).sum())
    recall = tp / (tp + fn) if tp + fn else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    return {"n_events": tp + fn, "detected": tp, "recall": recall, "false_alarms": fp,
            "precision": precision, "f1": f1, "n": int(len(a))}


def threshold_event(ot_now: pd.Series, delta_true: pd.Series, delta_pred: pd.Series, thr: float):
    """Event: OT at t+h exceeds ``thr``. Returns (actual, predicted) boolean series."""
    return (ot_now + delta_true > thr), (ot_now + delta_pred > thr)


def rise_event(delta_true: pd.Series, delta_pred: pd.Series, rise: float):
    """Event: OT rises by at least ``rise`` degrees within the horizon."""
    return (delta_true >= rise), (delta_pred >= rise)
