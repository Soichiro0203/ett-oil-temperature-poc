import numpy as np
import pandas as pd
import pytest

from ettpoc.evaluate import event_metrics, regression_metrics, rise_event, threshold_event


def test_regression_metrics_match_hand_computation():
    y = pd.Series([1.0, 2.0, 4.0]); p = pd.Series([1.0, 3.0, 2.0])
    m = regression_metrics(y, p)
    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert m["n"] == 3


def test_event_metrics_counts():
    actual = pd.Series([True, True, False, False, True, False])
    pred = pd.Series([True, False, True, False, True, False])
    m = event_metrics(actual, pred)
    assert m["n_events"] == 3
    assert m["detected"] == 2
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["false_alarms"] == 1
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_event_metrics_no_events_gives_nan_recall():
    m = event_metrics(pd.Series([False, False]), pd.Series([False, True]))
    assert m["n_events"] == 0 and np.isnan(m["recall"]) and m["false_alarms"] == 1


def test_threshold_and_rise_events():
    ot_now = pd.Series([30.0, 40.0, 44.0])
    delta_true = pd.Series([2.0, 6.0, -1.0])
    delta_pred = pd.Series([6.0, 4.0, 2.0])
    a, p = threshold_event(ot_now, delta_true, delta_pred, thr=45)
    assert a.tolist() == [False, True, False]
    assert p.tolist() == [False, False, True]
    a, p = rise_event(delta_true, delta_pred, rise=5)
    assert a.tolist() == [False, True, False]
    assert p.tolist() == [True, False, False]
