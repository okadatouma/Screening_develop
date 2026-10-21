"""図の共通スタイル。日本語フォント、色、キャプションの置き方。"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

INK   = "#141412"
SUB   = "#4a4946"
LINE  = "#c9c7c0"
ENJI  = "#b03a2e"      # 強調（ハミルトニアン等）に限定
BLUE  = "#2a78d6"
GREEN = "#0f7b6c"
PURPLE= "#8a4ab0"
ORANGE= "#eb6834"
PAPER = "#ffffff"


def use():
    ja = False
    for p in fm.findSystemFonts():
        if "NotoSansCJK" in p:
            fm.fontManager.addfont(p); ja = True
    plt.rcParams.update({
        "font.family": (["Noto Sans CJK JP", "DejaVu Sans"] if ja else ["DejaVu Sans"]),
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "figure.facecolor": PAPER, "axes.facecolor": PAPER,
        "axes.edgecolor": INK, "axes.linewidth": 0.9,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.direction": "out", "ytick.direction": "out",
        "font.size": 10.5, "axes.titlesize": 11.5,
        "legend.frameon": False,
        "savefig.facecolor": PAPER,
    })
    return ja


_NO_HEAD = "、。，．）」』】〕〉》"          # 行頭に来てはいけない
_NO_TAIL = "（「『【〔〈《"                    # 行末に来てはいけない


def wrap(text, cols):
    """日本語まじりの文を折り返す。全角を 2 桁として数え、どこでも折れる。

    textwrap は空白でしか折れないので、日本語だと行が揃わない。
    cols は半角換算の桁数。
    """
    def w(ch):
        return 2 if ord(ch) > 0x2E80 else 1

    # $...$ の数式は途中で折らない（折ると mathtext が壊れて生の TeX が出る）
    toks, i = [], 0
    while i < len(text):
        if text[i] == "$":
            j = text.find("$", i + 1)
            if j < 0: j = len(text) - 1
            toks.append(text[i:j + 1]); i = j + 1
        else:
            m = re.match(r"[0-9]+(?:[.,][0-9]+)*%?", text[i:])   # 数字は途中で折らない
            if m:
                toks.append(m.group(0)); i += m.end()
            else:
                toks.append(text[i]); i += 1
    out, line, cur = [], "", 0
    for t in toks:
        if t == "\n":
            out.append(line); line, cur = "", 0; continue
        tw = sum(w(c) for c in t)
        if cur + tw > cols and line and t[0] not in _NO_HEAD and line[-1] not in _NO_TAIL:
            out.append(line); line, cur = "", 0
        line += t; cur += tw
    if line: out.append(line)
    return "\n".join(out)


def caption(fig, title, body, y=0.012, x=0.035, size=9.4):
    """図の下のキャプション。見出しを先に置き、その真下から本文を下へ流す。

    数式を含む行は行送りが読めないので、見出しの位置を本文から推定すると重なる。
    見出しを基準にして本文をぶら下げれば、推定がずれても重ならない。
    """
    n = body.count("\n") + 1
    hpt = fig.get_figheight() * 72.0
    lh = size * 1.95 / hpt
    ytop = y + n * lh + 0.014
    fig.text(x, ytop, title, fontsize=size + 1.0, weight="bold",
             color=INK, va="top", ha="left")
    fig.text(x, ytop - (size + 1.0) * 2.0 / hpt, body, fontsize=size,
             color=SUB, va="top", ha="left", linespacing=1.62)
    return ytop
