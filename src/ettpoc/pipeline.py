"""End-to-end rolling backtest producing a long table of predictions."""

from __future__ import annotations

import pandas as pd

from .backtest import monthly_folds
from .data import FEATURE_COLS, TARGET_COL, flag_artifacts
from .features import DEFAULT_HORIZONS, build_features, build_targets, valid_rows
from .models import LGBMDeltaModel, persistence_delta, seasonal_naive_delta

VARIANTS = ["lgbm", "lgbm_no_load", "lgbm_oracle_load"]


def _features_for_variant(df: pd.DataFrame, variant: str, horizon: int) -> pd.DataFrame:
    if variant == "lgbm_no_load":
        return build_features(df, load_lags=None)
    X = build_features(df)
    if variant == "lgbm_oracle_load":
        # NOT available in operation: load at t+h. Upper bound "if a load plan existed".
        fut = {f"{c}_future": df[c].shift(-horizon) for c in FEATURE_COLS}
        X = pd.concat([X, pd.DataFrame(fut, index=df.index)], axis=1)
    return X


def run_backtest(
    df: pd.DataFrame,
    horizons: list[int] = DEFAULT_HORIZONS,
    n_test_months: int = 12,
    variants: list[str] = VARIANTS,
    lgbm_kwargs: dict | None = None,
    val_days: int = 30,
    importances: dict | None = None,
) -> pd.DataFrame:
    """Rolling monthly backtest. Returns one row per (date, horizon, model)."""
    lgbm_kwargs = lgbm_kwargs or {}
    flags = flag_artifacts(df)
    ot = df[TARGET_COL]
    Y = build_targets(df, horizons)
    folds = monthly_folds(df.index, n_test_months)
    feats = {(v, h): _features_for_variant(df, v, h) for v in variants for h in horizons}
    rows = []
    for fold in folds:
        tr_all, te_all = fold.train_mask(df.index), fold.test_mask(df.index)
        for h in horizons:
            ok = valid_rows(flags, h).to_numpy()
            te = te_all & ok
            y_true = Y[f"delta_{h}"][te]
            base = {"date": df.index[te], "horizon": h, "ot_now": ot[te].to_numpy(), "delta_true": y_true.to_numpy()}
            for name, pred in [("persistence", persistence_delta(ot, h)), ("seasonal_naive", seasonal_naive_delta(ot, h))]:
                rows.append(pd.DataFrame(base | {"model": name, "delta_pred": pred[te].to_numpy()}))
            for v in variants:
                X = feats[(v, h)]
                tr = tr_all & ok & X.notna().all(axis=1).to_numpy()
                val_start = fold.train_end - pd.Timedelta(days=val_days)
                is_val = tr & (df.index >= val_start)
                is_fit = tr & ~is_val
                m = LGBMDeltaModel(**lgbm_kwargs).fit(X[is_fit], Y[f"delta_{h}"][is_fit], X[is_val], Y[f"delta_{h}"][is_val])
                rows.append(pd.DataFrame(base | {"model": v, "delta_pred": m.predict(X[te])}))
                if importances is not None:
                    importances.setdefault((v, h), []).append(m.feature_importance())
    return pd.concat(rows, ignore_index=True)
