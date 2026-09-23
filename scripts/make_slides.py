"""Build the client-facing PoC report deck (message & body layout).

Text blocks are measured before they are drawn (`n_lines`), so cards and
headers grow with their content instead of overflowing.
"""

import math
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

INK = "1F2933"
INK_SOFT = "52606D"
ACCENT = "E8833A"
ACCENT_BG = "FCEFE3"
TEAL = "2F7A8C"
WHITE = "FFFFFF"
TINT = "F4F6F8"
LINE = "D7DDE3"
GREY = "9AA5B1"

LAT, EA = "Calibri", "Yu Gothic"
FIG = Path("reports/figures")

W, H = Inches(13.333), Inches(7.5)
M = Inches(0.62)
CW = W - 2 * M
MSG_SIZE = 22
FOOT_Y = Inches(6.6)


# --- text measurement --------------------------------------------------------
def units(text: str) -> float:
    """Approximate width of a string in 'full-width character' units."""
    return sum(0.55 if ord(c) < 0x2E80 else 1.0 for c in text)


def n_lines(text: str, avail: Emu, size: float) -> int:
    if not text:
        return 0
    return max(1, math.ceil(units(text) * size / 72 * 1.04 / (avail / 914400)))


def block_h(text: str, avail: Emu, size: float, spacing: float = 1.3) -> Emu:
    return Inches(n_lines(text, avail, size) * size * spacing / 72)


def check(cond, msg):
    if not cond:
        raise ValueError(f"layout: {msg}")


# --- primitives --------------------------------------------------------------
def rgb(c):
    return RGBColor.from_string(c)


def style(run, size, *, bold=False, color=INK, latin=LAT, ea=EA):
    f = run.font
    f.color.rgb = rgb(color)
    f.size = Pt(size)
    f.bold = bold
    f.name = latin
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = parse_xml(f'<{tag} xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"/>')
            rPr.append(el)
        el.set("typeface", ea)


def textbox(slide, x, y, w, h, *, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def para(tf, text, *, size=13.5, bold=False, color=INK, first=False, space_after=6,
         align=PP_ALIGN.LEFT, line=1.3):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(0)
    p.space_after = Pt(space_after)
    pPr = p._pPr if p._pPr is not None else p._p.get_or_add_pPr()
    pPr.set("eaLnBrk", "1")
    pPr.set("hangingPunct", "1")
    p.line_spacing = line
    r = p.add_run()
    r.text = text
    style(r, size, bold=bold, color=color)
    return p


def lines(slide, x, y, w, items, *, size=13.5, color=INK, spacing=1.3, gap=5):
    """Draw a stack of paragraphs, returning the y just below them."""
    norm = [(t, color, False) if isinstance(t, str) else (t + (False,) * (3 - len(t))) for t in items]
    h = sum(block_h(t[0], w, size, spacing) + Inches(gap / 72) for t in norm)
    tf = textbox(slide, x, y, w, h)
    for i, (txt, c, bold) in enumerate(norm):
        para(tf, txt, size=size, color=c, bold=bold, first=(i == 0), space_after=gap, line=spacing)
    return y + h


def rect(slide, x, y, w, h, fill, *, shape=MSO_SHAPE.RECTANGLE, radius=None):
    s = slide.shapes.add_shape(shape, int(x), int(y), int(w), int(h))
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    s.line.fill.background()
    s.shadow.inherit = False
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.text_frame.word_wrap = True
    return s


def centred_text(shape, text, *, size, bold=False, color=WHITE, align=PP_ALIGN.CENTER, pad_l=0):
    tf = shape.text_frame
    tf.margin_left = Inches(pad_l)
    tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    para(tf, text, size=size, bold=bold, color=color, first=True, align=align, space_after=0)


def blank(prs, *, dark=False):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    if dark:
        rect(s, 0, 0, W, H, INK)
    return s


def chip(slide, x, y, text, *, fill=ACCENT, color=WHITE):
    w = Inches(0.34 + units(text) * 11.5 / 72)
    c = rect(slide, x, y, w, Inches(0.3), fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.35)
    centred_text(c, text, size=11.5, bold=True, color=color)
    return c


def message(slide, kicker, text, *, sub=None, kicker_fill=ACCENT):
    """Section chip + the one-sentence takeaway; returns the y where the body starts."""
    chip(slide, M, Inches(0.42), kicker, fill=kicker_fill)
    y = Inches(0.86)
    h = block_h(text, CW, MSG_SIZE, 1.2)
    tf = textbox(slide, M, y, CW, h)
    para(tf, text, size=MSG_SIZE, bold=True, first=True, space_after=0, line=1.2)
    y += h + Inches(0.14)
    if sub:
        hs = block_h(sub, CW, 13, 1.3)
        tf = textbox(slide, M, y, CW, hs)
        para(tf, sub, size=13, color=INK_SOFT, first=True, space_after=0)
        y += hs + Inches(0.1)
    return y + Inches(0.22)


def card_height(title, body, w, *, num=False, title_size=15, body_size=12.5):
    pad = Inches(0.24)
    inner = w - 2 * pad
    h = pad * 2 + block_h(title, inner, title_size, 1.2) + Inches(0.1)
    h += sum(block_h(t, inner, body_size, 1.35) + Inches(5 / 72) for t in body)
    if num:
        h += Inches(0.5)
    return h


def card(slide, x, y, w, h, *, title, body, num=None, fill=TINT, title_color=INK,
         accent=ACCENT, title_size=15, body_size=12.5):
    need = card_height(title, body, w, num=num is not None, title_size=title_size, body_size=body_size)
    check(need <= h + Emu(4000), f"card {title!r} needs {need / 914400:.2f}in > {h / 914400:.2f}in")
    rect(slide, x, y, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    pad = Inches(0.24)
    inner = w - 2 * pad
    cy = y + pad
    if num is not None:
        d = rect(slide, x + pad, cy, Inches(0.36), Inches(0.36), accent, shape=MSO_SHAPE.OVAL)
        centred_text(d, str(num), size=13, bold=True)
        cy += Inches(0.5)
    th = block_h(title, inner, title_size, 1.2)
    tf = textbox(slide, x + pad, cy, inner, th)
    para(tf, title, size=title_size, bold=True, color=title_color, first=True, space_after=0, line=1.2)
    cy += th + Inches(0.1)
    lines(slide, x + pad, cy, inner, body, size=body_size, color=INK_SOFT, spacing=1.35)


def table(slide, x, y, w, rows, *, col_w, aligns=None, size=12.5, row_h=Inches(0.36),
          highlight_rows=()):
    cw = [int(w * c) for c in col_w]
    aligns = aligns or ["l"] + ["c"] * (len(col_w) - 1)
    cy = y
    for ri, row in enumerate(rows):
        head = ri == 0
        hl = ri in highlight_rows
        if hl:
            rect(slide, x - Inches(0.08), cy, w + Inches(0.16), row_h, ACCENT_BG,
                 shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.14)
        cx = x
        for ci, cell in enumerate(row):
            avail = cw[ci] - Inches(0.08)
            need = block_h(cell, avail, size, 1.2)
            check(need <= row_h, f"cell {cell!r} needs {need / 914400:.2f}in > row {row_h / 914400:.2f}in")
            tf = textbox(slide, cx + Inches(0.04), cy, avail, row_h, anchor=MSO_ANCHOR.MIDDLE)
            para(tf, cell, size=size, bold=head or (hl and ci == 0),
                 color=INK_SOFT if head else (ACCENT if hl else INK), first=True,
                 align=PP_ALIGN.LEFT if aligns[ci] == "l" else PP_ALIGN.CENTER,
                 space_after=0, line=1.2)
            cx += cw[ci]
        cy += row_h
        if head:
            rect(slide, x, cy - Inches(0.03), w, Emu(9525), LINE)
    return cy


def picture(slide, name, y, *, max_w=CW, max_h=Inches(4.2)):
    iw, ih = Image.open(FIG / name).size
    scale = min(max_w / iw, max_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    slide.shapes.add_picture(str(FIG / name), int((W - w) / 2), int(y), w, h)
    return y + h


def foot(slide, text):
    """Footnote, placed below whatever is already on the slide."""
    bottom = max((sh.top + sh.height for sh in slide.shapes if sh.height < H - Inches(1)), default=0)
    y = max(FOOT_Y, bottom + Inches(0.16))
    h = block_h(text, CW, 10.5, 1.25)
    check(y + h <= H - Inches(0.36), f"footnote does not fit ({(y + h) / 914400:.2f}in)")
    tf = textbox(slide, M, y, CW, h)
    para(tf, text, size=10.5, color=INK_SOFT, first=True, space_after=0, line=1.25)


def audit(prs):
    """Fail the build if any shape leaves the slide or crosses the footer line."""
    for i, sl in enumerate(prs.slides, start=1):
        for sh in sl.shapes:
            if sh.width == W and sh.height == H:
                continue  # full-bleed background
            check(sh.top >= 0 and sh.left >= 0, f"slide {i}: shape above/left of the slide")
            check(sh.top + sh.height <= H - Inches(0.36),
                  f"slide {i}: shape reaches the footer zone ({(sh.top + sh.height) / 914400:.2f}in)")
            check(sh.left + sh.width <= W - Inches(0.12), f"slide {i}: shape overflows the right edge")


def page_numbers(prs):
    for i, s in enumerate(prs.slides, start=1):
        if i == 1:
            continue
        tf = textbox(s, W - M - Inches(0.6), H - Inches(0.42), Inches(0.6), Inches(0.24))
        para(tf, str(i), size=10, color=GREY, first=True, align=PP_ALIGN.RIGHT, space_after=0)


# =============================================================================
prs = Presentation()
prs.slide_width, prs.slide_height = W, H

# --- 1. title ----------------------------------------------------------------
s = blank(prs, dark=True)
tf = textbox(s, M, Inches(2.4), Inches(11.4), Inches(1.9))
para(tf, "変圧器オイル温度予測 PoC", size=42, bold=True, color=WHITE, first=True, space_after=12, line=1.15)
para(tf, "予測に基づく高温リスクの事前警告は、保全判断にどこまで使えるか", size=19, color="CBD2D9", space_after=0)
rect(s, M, Inches(4.6), Inches(0.9), Emu(28575), ACCENT)
tf = textbox(s, M, Inches(5.05), Inches(9.5), Inches(0.9))
para(tf, "技術検証報告 / ETT (Electricity Transformer Temperature) データセット",
     size=13.5, color=GREY, first=True, space_after=5)
para(tf, "2026 年 9 月", size=13.5, color=GREY, space_after=0)

# --- 2. executive summary ----------------------------------------------------
s = blank(prs)
y = message(s, "結論", "日内変動の大きい設備では、6〜12 時間先の高温リスクを実用水準で事前警告できる",
            sub="ただし価値が出る条件は設備特性に依存し、24 時間先は現行手法と差がつかない")
cards = [
    ("効果は設備で大きく異なる",
     ["ETTh2 では 6〜12 時間先の誤差を現状比 56〜59% 削減。",
      "変動の緩やかな ETTh1 では 17〜21% にとどまる。"]),
    ("6 時間前の警告が実用水準",
     ["45°C 超過を検知率 78% / 適合率 85% で事前警告(現状相当は 42% / 43%)。",
      "+5°C 急上昇も 79% を検知。"]),
    ("24 時間先は予測できない",
     ["どのモデルも現状相当と同等。",
      "外気温など外部データなしには 1 日先の温度は決まらない。"]),
    ("負荷の寄与は限定的",
     ["負荷ラグの有無で精度はほぼ不変。",
      "未来の負荷を与えても改善せず、温度履歴と日内周期が予測を支える。"]),
]
cw = (CW - Inches(0.3) * 3) / 4
ch = max(card_height(t, b, cw, num=True, title_size=14) for t, b in cards)
for i, (t, b) in enumerate(cards):
    card(s, M + i * (cw + Inches(0.3)), y, cw, ch, title=t, body=b, num=i + 1, title_size=14)
ry = lines(s, M, y + ch + Inches(0.42), CW, [
    ("提言:まず ETTh2 型(日内変動が大きく高温域に達する)設備を対象に、6 時間先予測を既存の閾値監視に併設する形で試験導入することを推奨する。", INK, True),
], size=15, gap=8)
lines(s, M, ry + Inches(0.12), CW,
      ["ETTh1 型の設備では、予測モデルの導入よりもデータ品質の是正(欠測・前方補完の解消)を優先した方が費用対効果が高い。"],
      size=13.5, color=INK_SOFT)
foot(s, "評価期間: 2017 年 7 月〜2018 年 6 月の 12 ヶ月。各月について、その月より前の全データで再学習するローリング検証。")

# --- 3. use case / framing ---------------------------------------------------
s = blank(prs)
y = message(s, "検証設計", "「何時間先の温度を、どの判断に使うか」を定義してから評価指標を設計した",
            sub="精度指標だけでは「運用に価値が出るか」に答えられないため、業務指標を併設した", kicker_fill=TEAL)
lw = Inches(6.4)
rx = M + lw + Inches(0.5)
rw = CW - lw - Inches(0.5)
lines(s, M, y, lw, ["ユースケース定義"], size=15, gap=0)
rows = [
    ("予測ホライズン", "運用上の問い", "想定アクション"),
    ("1〜3 時間先", "いま負荷を上げてよいか", "負荷配分の調整"),
    ("6〜12 時間先", "今日中に危険域に入るか", "冷却強化・申し送り"),
    ("24 時間先", "明日の巡視をどう組むか", "点検の優先順位付け"),
]
ty = table(s, M, y + Inches(0.42), lw, rows, col_w=(0.28, 0.37, 0.35),
           aligns=["l", "l", "l"], size=12.5, row_h=Inches(0.5))

lines(s, rx, y, rw, ["評価指標の二段構え"], size=15, gap=0)
c1 = ("① 精度指標", ["MAE / RMSE をホライズン別に測定し、ベースライン比の改善率で示す。"])

c2 = ("② 業務指標", ["高温リスクを N 時間前に警告できたかを検知率・適合率で測定。",
                 "閾値超過型と急上昇型の 2 種類を定義。"])
h1 = card_height(*c1, rw)
h2 = card_height(*c2, rw)
card(s, rx, y + Inches(0.42), rw, h1, title=c1[0], body=c1[1])
c2y = y + Inches(0.42) + h1 + Inches(0.2)
card(s, rx, c2y, rw, h2, title=c2[0], body=c2[1], fill=ACCENT_BG, title_color=ACCENT)

base = ("比較対象:現行運用が置いている仮定",
        ["persistence(「N 時間後も現在と同じ温度」)をベースラインとする。これは閾値監視が暗黙に置いている仮定であり、本 PoC はこれを上回れるかで価値を判断した。"])
card(s, M, ty + Inches(0.42), lw, card_height(*base, lw), title=base[0], body=base[1])

# --- 4. EDA: two different transformers --------------------------------------
s = blank(prs)
y = message(s, "EDA", "2 台の変圧器は挙動が大きく異なり、一律のモデル設計は成り立たない", kicker_fill=TEAL)
ey = picture(s, "slide_ot_overview.png", y, max_h=Inches(3.5))
lines(s, M, ey + Inches(0.28), CW, [
    "ETTh1: 季節性に加えて 2 年で温度レベルが大きく低下(負荷も半減)。学習期間と運用期間で温度帯が変わる。",
    "ETTh2: 年周期が安定し、毎夏 50°C 超に到達。高温リスク管理の対象になるのはこちらの設備。",
])
foot(s, "細線: 1 時間値、太線: 月平均。")

# --- 5. EDA: predictability --------------------------------------------------
s = blank(prs)
y = message(s, "EDA", "ETTh2 は日内周期で予測余地が大きく、ETTh1 の短期変動はほぼランダムウォーク",
            sub="この差がそのままモデルの改善幅の差として現れる", kicker_fill=TEAL)
ey = picture(s, "slide_predictability.png", y, max_h=Inches(3.15))
lines(s, M, ey + Inches(0.28), CW, [
    "左: 1 時間差分の自己相関。ETTh2 は 24 時間周期の構造が残る一方、ETTh1 はほぼ相関を持たない。",
    "右: 日内振幅は ETTh2 が ±5°C、ETTh1 は ±1.4°C。曜日効果はどちらも検出されなかった。",
])
foot(s, "原系列の自己相関は両設備とも lag1 で 0.99 超 — 「現在値の持続」が極めて強いベースラインになることを示す。")

# --- 6. EDA: load is not a leading indicator ---------------------------------
s = blank(prs)
y = message(s, "EDA", "負荷と温度の相関は弱く、「負荷上昇の k 時間後に昇温」という遅れ構造は見られない",
            sub="温度変動の主因は観測されていない外部要因(外気温・日射等)にあると考えられる", kicker_fill=TEAL)
ey = picture(s, "slide_load_xcorr.png", y, max_h=Inches(3.05))
lines(s, M, ey + Inches(0.28), CW, [
    ("レベル相関は最大でも 0.49(ETTh2 の MULL)。ラグを変えても相関はほぼ平坦で、先行指標としては機能しない。", INK),
    ("→ 負荷ラグは特徴量に含めるが主役とは見なさず、「有無を比較して寄与を検証する」方針とした。", INK, True),
])
foot(s, "1 時間差分どうしの相関は全変数・全ラグで 0.15 未満。")

# --- 7. EDA: data quality ----------------------------------------------------
s = blank(prs)
y = message(s, "EDA", "機械的に生じた欠測・前方補完が含まれ、これは現行の閾値監視の誤報源でもある", kicker_fill=TEAL)
ey = picture(s, "slide_artifacts.png", y, max_h=Inches(2.95))
items = [
    ("毎月 31 日の前方補完", ["01:00〜23:00 の全列が 00:00 の値と同一。両設備で 322 行(全体の 1.8%)。"]),
    ("オイル温度の欠測", ["OT = 0.00 が ETTh1 で 111 行。負荷は正常値のため、センサ側の欠測と判断。"]),
    ("設備停止区間", ["2016 年 12 月に全負荷ゼロが 58 時間。通常運転とは異なる状態。"]),
]
cw = (CW - Inches(0.32) * 2) / 3
ch = max(card_height(t, b, cw) for t, b in items)
for i, (t, b) in enumerate(items):
    card(s, M + i * (cw + Inches(0.32)), ey + Inches(0.3), cw, ch, title=t, body=b, title_size=14)
foot(s, "本 PoC では該当行を自動検出して学習・評価から除外した(flag_artifacts)。実運用では品質フラグを監視系に組み込むことを推奨する。")

# --- 8. approach / design decisions ------------------------------------------
s = blank(prs)
y = message(s, "アプローチ", "EDA の示唆を設計判断に落とし、実運用で成立する条件を満たすモデルを構築した")
rows = [
    ("EDA で分かったこと", "設計上の判断", "狙い"),
    ("持続性が極めて強く、ETTh1 は温度帯が推移する", "温度そのものではなく「現在からの変化量」を予測", "学習範囲外への外挿破綻を回避"),
    ("未来の負荷は予測時点で未知", "特徴量は温度・負荷の過去値と時刻特徴のみ", "実運用で成立する条件を担保(リーク回避)"),
    ("ETTh2 は日内周期が強い", "時刻の sin/cos と 24 時間前後のラグを投入", "周期構造を明示的に学習"),
    ("負荷との相関が弱い", "負荷ラグ 有り / 無し / 未来値あり の 3 条件を比較", "負荷データ収集の必要性を定量判断"),
    ("固定テスト期間では高温期が偏る", "12 ヶ月の月次ローリング検証", "全季節での性能を確認し運用形態を再現"),
]
ty = table(s, M, y, CW, rows, col_w=(0.35, 0.37, 0.28), aligns=["l", "l", "l"],
           size=12.5, row_h=Inches(0.68))
lines(s, M, ty + Inches(0.38), CW, [
    ("モデルは LightGBM を主軸に選定。特徴量重要度で予測根拠を説明でき、学習が数十秒で済むため月次再学習の運用に乗せやすい。", INK),
    ("比較対象として persistence(現状相当)と seasonal naive(前日同時刻)を全ホライズンで併走させた。", INK_SOFT),
])

# --- 9. validation design ----------------------------------------------------
s = blank(prs)
y = message(s, "アプローチ", "「毎月再学習して翌月を予測する」運用を再現するローリング検証で評価した")
bx, bw, bh, gap = M + Inches(0.05), Inches(11.9), Inches(0.42), Inches(0.14)
folds = [("2016-07 〜 2017-06 のデータで学習", 0.72, "2017-07 を予測"),
         ("2016-07 〜 2017-07 のデータで学習", 0.76, "2017-08 を予測"),
         ("…", 0.80, "…"),
         ("2016-07 〜 2018-05 のデータで学習", 0.84, "2018-06 を予測")]
for i, (ltxt, frac, rtxt) in enumerate(folds):
    ty = y + Inches(0.1) + i * (bh + gap)
    lw_ = int(bw * frac) - int(gap / 2)
    r1 = rect(s, bx, ty, lw_, bh, TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.18)
    centred_text(r1, ltxt, size=12, color=INK_SOFT, align=PP_ALIGN.LEFT, pad_l=0.14)
    r2 = rect(s, bx + lw_ + int(gap / 2), ty, bw - lw_ - int(gap / 2), bh, ACCENT,
              shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.18)
    centred_text(r2, rtxt, size=12, bold=True)
ty = y + Inches(0.1) + 4 * (bh + gap)
lines(s, bx, ty, bw, ["計 12 分割 × 5 ホライズン × 3 モデル条件 = 180 回の学習を両設備で実施"],
      size=12, color=INK_SOFT)
cards2 = [
    ("学習・評価から除外する行", ["品質フラグに該当する時刻 t、および t+h がフラグ該当となる行(予測対象が信用できないため)。"]),
    ("リークの有無を自動テスト化", ["「時刻 t 以降のデータを改変しても t の特徴量が一切変化しない」ことをテストで機械的に検証。"]),
]
cw = (CW - Inches(0.32)) / 2
ch = max(card_height(t, b, cw) for t, b in cards2)
for i, (t, b) in enumerate(cards2):
    card(s, M + i * (cw + Inches(0.32)), ty + Inches(0.5), cw, ch, title=t, body=b)
foot(s, "実装は src/ettpoc/backtest.py、テストは tests/ に 25 件(実装より先に記述)。")

# --- 10. results: accuracy ---------------------------------------------------
s = blank(prs)
y = message(s, "結果", "ETTh2 は 12 時間先まで誤差を 56〜59% 削減、ETTh1 は 17〜21% にとどまる",
            sub="24 時間先はどちらの設備でも現状相当と差がつかない")
ey = picture(s, "slide_mae_by_horizon.png", y, max_h=Inches(3.0))
rows = [
    ("MAE [°C]", "1h", "3h", "6h", "12h", "24h"),
    ("ETTh1  現状相当", "0.50", "0.91", "1.32", "1.73", "1.74"),
    ("ETTh1  LightGBM", "0.46", "0.75", "1.04", "1.37", "1.73"),
    ("ETTh2  現状相当", "0.91", "2.52", "4.39", "6.02", "3.12"),
    ("ETTh2  LightGBM", "0.38", "1.03", "1.82", "2.65", "3.06"),
]
table(s, M + Inches(1.7), ey + Inches(0.3), Inches(8.6), rows,
      col_w=(0.34, 0.132, 0.132, 0.132, 0.132, 0.132), size=12.5, row_h=Inches(0.3),
      highlight_rows=(2, 4))
foot(s, "24 時間先で現状相当の誤差が 12 時間先より小さいのは、日内周期により 24 時間前の値が再び近づくため。")

# --- 11. results: business metrics -------------------------------------------
s = blank(prs)
y = message(s, "結果", "6 時間前の高温警告は検知率 78% / 適合率 85%。急上昇は現状では検知できない",
            sub="「現状相当」は現在温度がそのまま続くと仮定した判断 — 上昇の予兆は原理的に捉えられない")
ey = picture(s, "slide_event_recall.png", y, max_h=Inches(2.95))
cards3 = [
    ("45°C 超過の事前警告(6 時間前)", ["検知率 42% → 78%、適合率 43% → 85%。誤報は 282 → 67 件に減少。"]),
    ("+5°C 急上昇の事前警告(6 時間前)", ["現状相当では検知率 0%(構造上不可能)。本手法では 1,519 件中 79% を事前検知、適合率 75%。"]),
]
cw = (CW - Inches(0.32)) / 2
ch = max(card_height(t, b, cw) for t, b in cards3)
for i, (t, b) in enumerate(cards3):
    card(s, M + i * (cw + Inches(0.32)), ey + Inches(0.26), cw, ch, title=t, body=b,
         fill=ACCENT_BG, title_color=ACCENT, title_size=14)
foot(s, "閾値 45°C・上昇幅 5°C は評価期間の上位 5〜7% に相当する仮置きの値。実運用では設備の許容温度と保全規程に基づく再定義が必要。")

# --- 12. results: example week -----------------------------------------------
s = blank(prs)
y = message(s, "結果", "現状相当は昇温に 6 時間遅れて追随するが、本手法はピークの時刻と高さを捉える")
ey = picture(s, "slide_example_week.png", y + Inches(0.25), max_h=Inches(3.3))
lines(s, M, ey + Inches(0.4), CW, [
    "ETTh2、2017 年 7 月の 1 週間。毎時、その時点までの観測のみを使って 6 時間先を予測した結果を並べたもの。",
    "現状相当(灰)は波形が 6 時間ずれ、45°C を超えてから警報が出る。本手法(橙)は超過前に警告できる。",
])

# --- 13. results: what drives it ---------------------------------------------
s = blank(prs)
y = message(s, "結果", "予測を支えているのは温度自身の履歴と日内周期であり、負荷変数の寄与は限定的",
            sub="負荷ラグを除いた条件、未来の負荷を与えた条件のいずれも精度はほぼ変わらなかった")
ey = picture(s, "slide_importance.png", y, max_h=Inches(2.9))
rows = [
    ("6 時間先 MAE [°C]", "ETTh1", "ETTh2", "解釈"),
    ("温度履歴のみ(負荷なし)", "1.056", "1.772", "負荷なしでもほぼ同等"),
    ("+ 負荷の過去値(採用)", "1.041", "1.822", "改善は誤差の範囲"),
    ("+ 未来の負荷(参考・運用不可)", "1.016", "1.804", "未来の負荷が分かっても改善しない"),
]
table(s, M + Inches(0.9), ey + Inches(0.3), Inches(10.1), rows,
      col_w=(0.33, 0.13, 0.13, 0.41), aligns=["l", "c", "c", "l"], size=12.5, row_h=Inches(0.32))
foot(s, "示唆: 本ユースケースのために負荷データ連携を新規構築する投資対効果は低い。外気温など別のデータ取得を優先すべき。")

# --- 14. where it pays off ---------------------------------------------------
s = blank(prs)
y = message(s, "考察", "予測が価値を生むのは「変動が大きく、閾値に到達しうる設備」に限られる")
cards4 = [
    ("価値が出る条件(ETTh2 型)",
     ["・日内変動が大きい(振幅 5°C 以上)",
      "・運用温度が許容閾値に接近している",
      "・数時間の予告で打てる手がある(冷却強化・負荷調整)",
      "→ 6 時間先予測で検知率 42% → 78%、誤報も 1/4 に減少"]),
    ("価値が出にくい条件(ETTh1 型)",
     ["・短期変動がランダムウォークに近い",
      "・温度が閾値から十分離れて推移している",
      "・変動幅そのものが小さい(日内 ±1.4°C)",
      "→ 改善は 17〜21% にとどまり、高温イベント自体が発生しない"]),
]
cw = (CW - Inches(0.36)) / 2
ch = max(card_height(t, b, cw) for t, b in cards4)
card(s, M, y, cw, ch, title=cards4[0][0], body=cards4[0][1], fill=ACCENT_BG, title_color=ACCENT)
card(s, M + cw + Inches(0.36), y, cw, ch, title=cards4[1][0], body=cards4[1][1])
ry = lines(s, M, y + ch + Inches(0.45), CW, [("導入判断の進め方", INK, True)], size=15, gap=8)
sy = lines(s, M, ry + Inches(0.06), CW, [
    "全設備に一律導入するのではなく、過去データから下記の指標を算出して対象設備をスクリーニングし、価値が見込める設備から段階的に展開することを推奨する。",
], size=13.5, color=INK_SOFT)
screen = [
    ("① 日内変動幅", ["直近 30 日の日内振幅(日最高 − 日最低)の中央値"]),
    ("② 閾値までの余裕", ["温度の 95 パーセンタイルと許容温度の差"]),
    ("③ イベント発生頻度", ["過去 1 年の閾値超過・急上昇イベントの件数"]),
]
cw3 = (CW - Inches(0.32) * 2) / 3
ch3 = max(card_height(t, b, cw3, title_size=14) for t, b in screen)
for i, (t, b) in enumerate(screen):
    card(s, M + i * (cw3 + Inches(0.32)), sy + Inches(0.2), cw3, ch3, title=t, body=b, title_size=14)

# --- 15. limitations & next steps --------------------------------------------
s = blank(prs)
y = message(s, "限界と今後", "現時点の限界は明確であり、それぞれに次の一手を特定できている", kicker_fill=TEAL)
rows = [
    ("現時点の限界", "原因の見立て", "次の一手"),
    ("24 時間先は改善しない", "外気温・日射など未観測の外部要因が支配的", "気象予報データを特徴量に追加し効果を検証"),
    ("点予測のみで警告が決め打ち", "予測の不確実性を表現していない", "分位点回帰で超過確率を出力し、誤報と見逃しのコストで閾値を最適化"),
    ("閾値・イベント定義が仮置き", "設備の許容温度や保全規程が未共有", "実際の警報記録・点検記録と突合して定義を確定"),
    ("検証対象が 2 設備のみ", "設備間の外挿性が未確認", "複数設備の横断学習と、適用可否のスクリーニング基準を策定"),
    ("静的なバックテストのみ", "運用時のデータドリフトを扱っていない", "ドリフト検知と再学習トリガを含む運用設計"),
]
table(s, M, y, CW, rows, col_w=(0.26, 0.33, 0.41), aligns=["l", "l", "l"], size=12.5,
      row_h=Inches(0.74))
foot(s, "15 分粒度データ(ETTm)と深層学習モデルは本 PoC では未検証。ただし本結果からは、短ホライズンでの大幅な改善は見込みにくい。")

# --- 16. closing -------------------------------------------------------------
s = blank(prs, dark=True)
chip(s, M, Inches(0.95), "次のステップ")
tf = textbox(s, M, Inches(1.5), Inches(11.5), Inches(1.0))
para(tf, "本 PoC の結論と、次フェーズへの提案", size=32, bold=True, color=WHITE, first=True, space_after=0, line=1.2)
steps = [
    ("1", "対象設備のスクリーニング", "日内変動幅と閾値までの余裕から、予測が効く設備を全設備データで抽出する(約 2 週間)"),
    ("2", "気象データ統合の効果検証", "外気温・日射を加え、24 時間先の改善余地を定量化する(約 3 週間)"),
    ("3", "警告ルールの運用設計", "誤報・見逃しのコストを現場とすり合わせ、確率予測に基づく警告ルールを設計する(約 3 週間)"),
]
sy = Inches(3.0)
for num, title, body in steps:
    d = rect(s, M, sy, Inches(0.46), Inches(0.46), ACCENT, shape=MSO_SHAPE.OVAL)
    centred_text(d, num, size=15, bold=True)
    tf = textbox(s, M + Inches(0.74), sy - Inches(0.02), Inches(11.2), Inches(0.85))
    para(tf, title, size=17, bold=True, color=WHITE, first=True, space_after=4, line=1.2)
    para(tf, body, size=13.5, color=GREY, space_after=0, line=1.3)
    sy += Inches(1.15)
tf = textbox(s, M, Inches(6.6), Inches(11.5), Inches(0.35))
para(tf, "実装コード・検証ノートブック一式は GitHub リポジトリを参照", size=12.5, color=GREY,
     first=True, space_after=0)

# --- 17. appendix ------------------------------------------------------------
s = blank(prs)
y = message(s, "Appendix", "ETTh2 の改善は高温期(5〜9 月)を含む通年で維持されている", kicker_fill=INK_SOFT)
ey = picture(s, "slide_mae_by_month.png", y + Inches(0.2), max_h=Inches(3.2))
lines(s, M, ey + Inches(0.4), CW, [
    "6 時間先 MAE の月別推移。ETTh2 では夏季に現状相当の誤差が跳ね上がるのに対し、本手法は通年で安定している。",
])
foot(s, "データ出典: Zhou et al., Informer, AAAI 2021 / github.com/zhouhaoyi/ETDataset(追加データの使用なし)")

audit(prs)
page_numbers(prs)
Path("reports").mkdir(exist_ok=True)
prs.save("reports/poc_report.pptx")
print(f"wrote reports/poc_report.pptx ({len(prs.slides._sldIdLst)} slides)")
