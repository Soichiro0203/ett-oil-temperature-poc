"""Slide-optimised versions of the analysis figures (larger type, deck palette)."""

import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from ettpoc.data import FEATURE_COLS, flag_artifacts, load_ett
from ettpoc.evaluate import event_metrics, regression_metrics, rise_event, threshold_event

warnings.filterwarnings("ignore")

INK = "#1F2933"
ACCENT = "#E8833A"
BASE = "#9AA5B1"
TEAL = "#2F7A8C"
MUTED = "#CBD2D9"
FIG = "reports/figures"

plt.rcParams.update({
    "font.family": ["Hiragino Sans", "DejaVu Sans"],
    "figure.dpi": 200, "font.size": 13, "axes.grid": True, "grid.alpha": 0.25,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
})

data = {n: load_ett(n) for n in ["ETTh1", "ETTh2"]}
flags = {n: flag_artifacts(df) for n, df in data.items()}
preds = pd.concat([pd.read_parquet(f"outputs/predictions_{n}.parquet") for n in data], ignore_index=True)
preds["month"] = preds.date.dt.to_period("M")
DS_COLOR = {"ETTh1": TEAL, "ETTh2": ACCENT}


def save(name):
    plt.savefig(f"{FIG}/slide_{name}.png", facecolor="white")
    plt.close()


# --- 1. two transformers, two behaviours ------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True)
for ax, (n, df) in zip(axes, data.items()):
    ax.plot(df.index, df.OT, lw=0.35, alpha=0.55, color=DS_COLOR[n])
    mm = df.OT.resample("MS").mean()
    ax.plot(mm.index + pd.Timedelta(days=15), mm, "-", color=INK, lw=2.4)
    ax.set_ylabel("OT [°C]")
    ax.set_title(f"{n}", loc="left")
axes[0].set_ylim(-8, 50); axes[1].set_ylim(-8, 65)
axes[0].annotate("2 年で平均 34°C → 10°C", xy=(pd.Timestamp("2018-03-01"), 33), color=INK,
                 fontsize=12, fontweight="bold", ha="center")
axes[1].annotate("毎夏 50°C 超が再現", xy=(pd.Timestamp("2017-10-01"), 58), color=INK,
                 fontsize=12, fontweight="bold", ha="center")
save("ot_overview")

# --- 2. predictability: ACF of first difference + daily profile -------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for n, df in data.items():
    ad = acf(df.OT.diff().dropna(), nlags=72)
    axes[0].plot(ad, color=DS_COLOR[n], lw=2, label=n)
    dev = df.OT - df.OT.groupby(df.index.date).transform("mean")
    axes[1].plot(dev.groupby(df.index.hour).mean(), "o-", color=DS_COLOR[n], lw=2, ms=5, label=n)
axes[0].axhline(0, color=INK, lw=0.8)
axes[0].set_title("1 時間差分の自己相関", loc="left"); axes[0].set_xlabel("ラグ [h]")
axes[0].annotate("ETTh1 はほぼ構造なし\n(ランダムウォークに近い)", xy=(7, 0.72), fontsize=11, color=TEAL, fontweight="bold")
axes[1].set_title("日内プロファイル(日平均からの偏差)", loc="left")
axes[1].set_xlabel("時刻"); axes[1].set_ylabel("[°C]"); axes[1].set_xticks([0, 6, 12, 18, 23])
axes[1].legend(frameon=False)
save("predictability")

# --- 3. load is not a leading indicator -------------------------------------
lags = [0, 1, 2, 3, 6, 12, 24]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.3), sharey=True)
for ax, (n, df) in zip(axes, data.items()):
    for col in FEATURE_COLS:
        ax.plot(lags, [df.OT.corr(df[col].shift(k)) for k in lags], "o-", lw=1.6, ms=4, label=col)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_title(f"{n}", loc="left"); ax.set_xlabel("負荷のラグ k [h]"); ax.set_ylim(-0.35, 0.65)
axes[0].set_ylabel("corr(OT$_t$, load$_{t-k}$)")
axes[1].legend(frameon=False, ncol=2, fontsize=10)
save("load_xcorr")

# --- 4. data quality --------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
df = data["ETTh1"]
s = df.loc["2016-07-29":"2016-08-02"]
axes[0].plot(s.index, s.OT, color=INK, lw=2, label="OT")
axes[0].plot(s.index, s.HUFL, color=BASE, lw=1.6, label="HUFL")
axes[0].axvspan(pd.Timestamp("2016-07-31 01:00"), pd.Timestamp("2016-07-31 23:00"), color=ACCENT, alpha=0.25)
axes[0].set_title("毎月 31 日は全列が前方補完(両設備で 322 行)", loc="left", fontsize=12)
s = df.loc["2018-01-03":"2018-01-07"]
axes[1].plot(s.index, s.OT, color=INK, lw=2, label="OT")
axes[1].plot(s.index, s.HUFL, color=BASE, lw=1.6, label="HUFL")
axes[1].axvspan(pd.Timestamp("2018-01-05 00:00"), pd.Timestamp("2018-01-05 08:00"), color=ACCENT, alpha=0.25)
axes[1].set_title("OT = 0.00 の欠測(負荷は正常、ETTh1 で 111 行)", loc="left", fontsize=12)
for ax in axes:
    ax.tick_params(axis="x", labelrotation=25, labelsize=10); ax.legend(frameon=False, fontsize=10)
save("artifacts")

# --- 5. accuracy ------------------------------------------------------------
rows = [{"dataset": d, "horizon": h, "model": m, **regression_metrics(g.delta_true, g.delta_pred)}
        for (d, h, m), g in preds.groupby(["dataset", "horizon", "model"])]
mae = pd.DataFrame(rows).pivot_table(index=["dataset", "model"], columns="horizon", values="mae")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for ax, d in zip(axes, data):
    ax.plot(mae.columns, mae.loc[(d, "persistence")], "o--", color=BASE, lw=2, ms=6, label="persistence(現状相当)")
    ax.plot(mae.columns, mae.loc[(d, "lgbm")], "o-", color=ACCENT, lw=2.6, ms=7, label="LightGBM")
    for h in mae.columns:
        imp = 1 - mae.loc[(d, "lgbm"), h] / mae.loc[(d, "persistence"), h]
        off, ha = ((18, -10), "left") if h == 1 else ((0, -18), "center")
        ax.annotate(f"−{imp:.0%}", (h, mae.loc[(d, 'lgbm'), h]), textcoords="offset points",
                    xytext=off, ha=ha, fontsize=10.5, color=ACCENT, fontweight="bold")
    ax.set_title(f"{d}", loc="left"); ax.set_xlabel("予測ホライズン [h]"); ax.set_ylabel("MAE [°C]")
    ax.set_xticks(mae.columns)
    ax.set_ylim(-0.13 * mae.loc[(d, "persistence")].max(), None)
    ax.spines["bottom"].set_position(("data", 0))
axes[0].legend(frameon=False, fontsize=11, loc="upper left")
save("mae_by_horizon")

# --- 6. business metrics (ETTh2) --------------------------------------------
EV = {"ETTh1": dict(thr=20, rise=3), "ETTh2": dict(thr=45, rise=5)}
rows = []
for (d, h, m), g in preds.groupby(["dataset", "horizon", "model"]):
    c = EV[d]
    a, p = threshold_event(g.ot_now, g.delta_true, g.delta_pred, c["thr"])
    rows.append({"dataset": d, "event": "threshold", "horizon": h, "model": m, **event_metrics(a, p)})
    a, p = rise_event(g.delta_true, g.delta_pred, c["rise"])
    rows.append({"dataset": d, "event": "rise", "horizon": h, "model": m, **event_metrics(a, p)})
ev = pd.DataFrame(rows)
ev.to_csv("outputs/table_events.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
hs = [3, 6, 12]
x = np.arange(len(hs)); w = 0.34
titles = {"threshold": "45°C 超過を事前警告(498 件)", "rise": "+5°C 急上昇を事前警告(6h: 1,519 件)"}
for ax, evname in zip(axes, ["threshold", "rise"]):
    for i, (m, c, lab) in enumerate([("persistence", BASE, "現状相当"), ("lgbm", ACCENT, "LightGBM")]):
        s = ev[(ev.dataset == "ETTh2") & (ev.event == evname) & (ev.model == m) & ev.horizon.isin(hs)].sort_values("horizon")
        v = s.recall.fillna(0).to_numpy()
        b = ax.bar(x + (i - 0.5) * w, v, w, color=c, label=lab)
        ax.bar_label(b, labels=[f"{y:.0%}" for y in v], fontsize=11, fontweight="bold", padding=2)
    ax.set_xticks(x); ax.set_xticklabels([f"{h} 時間前" for h in hs]); ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0%", "50%", "100%"])
    ax.set_title(titles[evname], loc="left", fontsize=12.5)
axes[0].set_ylabel("検知率 (recall)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=12, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.16))
save("event_recall")

# --- 7. example week --------------------------------------------------------
g = preds[(preds.dataset == "ETTh2") & (preds.horizon == 6)]
g = g[(g.date >= "2017-07-17") & (g.date < "2017-07-25")].sort_values("date").set_index("date")
fig, ax = plt.subplots(figsize=(11.5, 3.3))
act = g[g.model == "lgbm"]
t = act.index + pd.Timedelta(hours=6)
ax.plot(t, act.ot_now + act.delta_true, color=INK, lw=2.4, label="実測")
s = g[g.model == "persistence"]
ax.plot(s.index + pd.Timedelta(hours=6), s.ot_now + s.delta_pred, color=BASE, lw=1.6, label="現状相当(6h 先)")
ax.plot(t, act.ot_now + act.delta_pred, color=ACCENT, lw=2, label="LightGBM(6h 先)")
ax.axhline(45, color="#B3261E", ls="--", lw=1.4)
ax.annotate("45°C 閾値", (t[3], 45.6), color="#B3261E", fontsize=11, fontweight="bold")
ax.set_ylabel("OT [°C]"); ax.legend(frameon=False, ncol=3, fontsize=11, loc="upper left")
ax.tick_params(axis="x", labelsize=10)
save("example_week")

# --- 8. feature importance --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, d in zip(axes, data):
    imps = pickle.load(open(f"outputs/importances_{d}.pkl", "rb"))[("lgbm", 6)]
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
    imp = imp / imp.sum()
    top = imp.head(10)[::-1]
    is_load = [any(k in i for k in FEATURE_COLS) for i in top.index]
    ax.barh(top.index, top.values, color=[ACCENT if L else TEAL for L in is_load])
    share = imp[[i for i in imp.index if any(k in i for k in FEATURE_COLS)]].sum()
    ax.set_title(f"{d}  —  負荷特徴の寄与合計 {share:.0%}", loc="left", fontsize=12.5)
    ax.tick_params(axis="y", labelsize=10.5)
    ax.set_xlabel("gain importance(正規化)")
save("importance")

# --- 9. monthly MAE ---------------------------------------------------------
h6 = preds[preds.horizon == 6]
mm = h6.groupby(["dataset", "month", "model"]).apply(lambda g: np.mean(np.abs(g.delta_true - g.delta_pred))).unstack("model")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharex=True)
for ax, d in zip(axes, data):
    t = mm.loc[d]
    ax.plot(t.index.to_timestamp(), t["persistence"], "o--", color=BASE, lw=2, ms=5, label="persistence")
    ax.plot(t.index.to_timestamp(), t["lgbm"], "o-", color=ACCENT, lw=2.4, ms=5, label="LightGBM")
    ax.set_title(f"{d}", loc="left"); ax.set_ylabel("6h 先 MAE [°C]")
    ax.tick_params(axis="x", labelrotation=30, labelsize=10)
axes[0].legend(frameon=False, fontsize=11)
save("mae_by_month")

print("slide figures written")
