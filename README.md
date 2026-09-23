# 変圧器オイル温度の予測 PoC — ETT データセット

電力用変圧器のオイル温度を数時間先まで予測し、**高温リスクの事前警告**として運用に価値を出せるかを検証した PoC です。
公開ベンチマーク [ETT (Electricity Transformer Temperature)](https://github.com/zhouhaoyi/ETDataset) を使用しています。

📄 **報告スライド: [`reports/poc_report.pdf`](reports/poc_report.pdf)**(クライアント向け・全 17 枚)

## 結論

1. **設備によって予測の価値が大きく異なる。** 日内変動が大きい ETTh2 では 6〜12 時間先の誤差を
   persistence(現状の閾値監視に相当)比で **56〜59% 削減**。変動が緩やかな ETTh1 では 17〜21% にとどまる。
2. **ETTh2 では 6 時間前の高温警告が実用水準。** 45℃ 超過を検知率 78% / 適合率 85% で事前警告できる
   (現状相当では 42% / 43%)。「6 時間以内に +5℃ 上昇」は現状では原理的に検知できないが、本手法では 79% を検知。
3. **24 時間先はどの設備・どのモデルも persistence と同等。** 外気温などの外部データが不可欠。
4. **負荷 6 変数の寄与は小さい。** 負荷ラグの有無で精度はほぼ変わらず、未来の負荷を与えても改善しない。

| 6 時間先 MAE [℃] | ETTh1 | ETTh2 |
|---|---|---|
| persistence(現状相当) | 1.32 | 4.39 |
| **LightGBM** | **1.04** (−21%) | **1.82** (−58%) |

根拠・図表・設計判断の詳細はスライド、または下記ノートブックを参照してください。

## 検証の設定

| 項目 | 内容 |
|---|---|
| 予測ホライズン | 1 / 3 / 6 / 12 / 24 時間先 |
| 予測対象 | 現在温度からの**変化量** `OT[t+h] − OT[t]` |
| 入力 | 温度と負荷 6 変数の**過去値のみ** + 時刻特徴(未来の負荷は使わない) |
| 評価 | 2017-07〜2018-06 の 12 ヶ月を月次ローリング(各月、その月より前の全データで再学習) |
| 対象設備 | ETTh1・ETTh2(1 時間粒度、2016-07〜2018-06、各 17,420 時間) |

主要な設計判断は 3 つです。いずれもスライド 8〜9 枚目で詳述しています。

- **変化量を予測する** — レベルを直接予測すると、ETTh1 のように学習期間と評価期間で温度帯が変わる設備で木モデルが破綻するため
- **未来の負荷を使わない** — 予測時点で未知の値を入力に含めるのはリークにあたるため(参考として「未来の負荷が分かる場合」も測定)
- **テストを実装より先に書く** — 時系列パイプラインで最も危険なリークと時刻ずれを、「t 以降のデータを改変しても t の特徴量が変わらない」等のテストで機械的に検出する

## リポジトリ構成

```
.
├── data/                     # ETT データ(課題で提供されたもの)
├── src/ettpoc/
│   ├── data.py               # 読み込み + データ品質フラグ
│   ├── features.py           # ラグ・ローリング・時刻特徴、変化量ターゲット
│   ├── models.py             # persistence / seasonal naive / LightGBM
│   ├── backtest.py           # 月次ローリング分割
│   ├── evaluate.py           # 回帰指標 + イベント検知指標
│   └── pipeline.py           # バックテスト実行
├── tests/                    # 実装より先に書いたテスト(25 件)
├── notebooks/
│   ├── 01_eda.ipynb          # EDA と、そこから導いた設計方針
│   └── 02_results.ipynb      # バックテスト結果の評価
├── scripts/
│   ├── run_backtest.py       # バックテスト実行
│   ├── make_slide_figures.py # スライド用の図を生成
│   └── make_slides.py        # 報告スライド (.pptx) を生成
├── reports/poc_report.pdf    # クライアント向け報告スライド
└── outputs/                  # 予測結果・集計表(実行時に生成)
```

## 再現手順

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest                                       # テスト 25 件
python scripts/run_backtest.py ETTh1 ETTh2   # 約 7 分。outputs/ に予測を出力
jupyter lab notebooks/                       # 01_eda → 02_results
```

macOS で LightGBM を使う場合は `brew install libomp` が必要です。

## 出典

- ETT データセット: Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*, AAAI 2021.
  <https://github.com/zhouhaoyi/ETDataset>
- 追加データは使用していません。
