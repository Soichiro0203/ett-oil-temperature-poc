"""Loading the ETT (Electricity Transformer Temperature) dataset."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
FEATURE_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
TARGET_COL = "OT"


def load_ett(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load one ETT csv as a DataFrame indexed by timestamp.

    Columns are the six load variables followed by the oil temperature ``OT``.
    """
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")
    df = pd.read_csv(data_dir / f"{name}.csv", parse_dates=["date"], index_col="date")
    df = df.sort_index()
    return df[FEATURE_COLS + [TARGET_COL]]


def flag_artifacts(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows that look like data-collection artifacts, not physics.

    Two patterns found in EDA:
    * *frozen* rows: every column identical to the previous row. The source data
      is missing for the 31st of each month and was forward-filled (23 h blocks).
    * *dropouts*: oil temperature recorded as exactly 0.0 while loads look normal.
    """
    frozen = (df == df.shift()).all(axis=1)
    dropout = df[TARGET_COL] == 0.0
    return (frozen | dropout).rename("is_artifact").astype(bool)
