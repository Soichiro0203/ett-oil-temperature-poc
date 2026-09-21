import pickle, sys, time
import pandas as pd
from ettpoc.data import load_ett
from ettpoc.pipeline import run_backtest

for name in sys.argv[1:]:
    t0 = time.time()
    imps = {}
    preds = run_backtest(load_ett(name), importances=imps)
    preds["dataset"] = name
    preds.to_parquet(f"outputs/predictions_{name}.parquet", index=False)
    pickle.dump(imps, open(f"outputs/importances_{name}.pkl", "wb"))
    print(name, preds.shape, f"{time.time()-t0:.0f}s", flush=True)
