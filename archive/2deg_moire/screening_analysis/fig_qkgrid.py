# -*- coding: utf-8 -*-
"""同じヘッドの q·k 行列を、相を縦に並べて比べる。

    python screening_analysis/fig_qkgrid.py --runs 液体=DIR,境界=DIR,結晶=DIR --out F.png

bigN_maps.py の _grid.png は 1 つの run の 16 ヘッドを並べる図なので、
相の違いを見るには 3 枚を見比べる必要がある。ここでは列 = ヘッド（窓の小さい順）、
行 = 相 に組み替えて、列を上から下に読めば「その窓のヘッドが相転移で
どう変わるか」が 1 目でわかるようにする。色の上限は列ごとに 3 相で共通。
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)


def _font():
    import matplotlib
    from matplotlib import font_manager as fm
    for f in ("Noto Sans CJK JP", "IPAexGothic", "Noto Sans JP"):
        if any(f == x.name for x in fm.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = f
            matplotlib.rcParams["mathtext.fontset"] = "cm"
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


def _spin_gap(M, nup):
    """同スピン対と異スピン対で s_ij の平均がどれだけ違うか（対角は除く）。"""
    NE = M.shape[0]; off = ~np.eye(NE, dtype=bool)
    same = np.zeros((NE, NE), bool)
    same[:nup, :nup] = True; same[nup:, nup:] = True
    a = M[same & off].mean(); b = M[(~same) & off].mean()
    return float(a - b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="label=folder をカンマ区切り")
    ap.add_argument("--out", default="fig_qkgrid.png")
    ap.add_argument("--nconf", type=int, default=48)
    ap.add_argument("--ncol", type=int, default=8)
    ap.add_argument("--what", default="W", choices=["W", "S"],
                    help="W = alpha*m（窓込み）, S = q·k のコサイン類似度そのもの")
    a = ap.parse_args()

    from attn_probe import probe
    from head_contrib import cached
    items = []
    for part in a.runs.split(","):
        lab, folder = part.rsplit("=", 1)
        print("=== %s  %s" % (lab, folder))
        r = probe(folder, nconf=a.nconf)
        cfg = json.load(open(os.path.join(folder, "config.json")))
        rs = cfg["system"]["make_local_energy_kwargs"]["potential_kwargs"][
            "interaction_energy_scale"]
        z = cached(folder, nconf=48)
        qp = (z["Cp"] ** 2).sum(0); qp = 100 * qp / qp.sum()
        items.append(dict(lab=lab, rs=rs, r=r, qp=qp))

    r0 = items[0]["r"]
    NE, NH, dref = r0["NE"], r0["NH"], r0["dref"]
    nup = int((np.asarray(r0["spins"][0]) > 0).sum())
    w = r0["w"][0]          # probe は絶対長で返す（殻なら中心 c_h）
    SHELL = bool(r0.get("shell", False))
    LSYM = "c" if SHELL else "w"          # ヘッドを代表する長さの記号
    LNAME = "殻の中心" if SHELL else "窓"
    order = np.argsort(w)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    ja = _font(); T = lambda x, y: x if ja else y            # noqa: E731
    CM = LinearSegmentedColormap.from_list(
        "b", ["#fcfcfb", "#dce8f7", "#93b8ea", "#2a78d6", "#12406f"]); CM.set_bad("#9a9a94")
    ORANGE = "#eb6834"

    NS = len(items)
    NC = a.ncol
    NBLK = int(np.ceil(NH / NC))
    CELL, HEAD, FOOT, LEFT = 1.62, 1.05, 1.30, 1.05
    GAP = 0.62                      # ブロック間の余白 [inch]（列見出しを入れる）
    BLKH = CELL * NS * 1.18         # 1 ブロックの高さ（パネル内の小見出しぶん込み）
    Wfig = CELL * NC + LEFT + 0.25
    Hfig = BLKH * NBLK + GAP * NBLK + HEAD + FOOT
    fig = plt.figure(figsize=(Wfig, Hfig)); fig.patch.set_facecolor("#fcfcfb")
    ax = np.empty((NBLK * NS, NC), dtype=object)
    for b in range(NBLK):
        top_in = Hfig - HEAD - b * (BLKH + GAP) - GAP
        gs = fig.add_gridspec(NS, NC, wspace=.06, hspace=.34,
                              left=LEFT / Wfig, right=1 - 0.14 / Wfig,
                              top=top_in / Hfig, bottom=(top_in - BLKH) / Hfig)
        for i in range(NS):
            for j in range(NC):
                ax[b * NS + i, j] = fig.add_subplot(gs[i, j])
                ax[b * NS + i, j].axis("off")

    for k, h in enumerate(order):
        blk, col = k // NC, k % NC
        # 列（= 1 つのヘッド）の中では 3 相で色の上限をそろえる
        Ms = []
        for it in items:
            M = it["r"]["W"].mean(0)[0, h] if a.what == "W" else it["r"]["S"].mean(0)[0, h]
            Ms.append(M)
        off = ~np.eye(NE, dtype=bool)
        # 対角は他より 1 桁大きく、そこに色域を取られると非対角の模様が見えない。
        # 上限は非対角の 99.5 パーセンタイルから取り、対角は飽和させる。
        vmax = max(np.percentile(M[off], 99.5) for M in Ms)
        vmin = min(0.0, min(M[off].min() for M in Ms)) if a.what == "S" else 0.0
        for si, (it, M) in enumerate(zip(items, Ms)):
            A = ax[blk * NS + si, col]; A.axis("on")
            P = (it["r"]["W"] > 0).mean(0)[0, h]
            A.imshow(np.ma.masked_where(P < 0.005, M), cmap=CM, vmin=vmin,
                     vmax=max(vmax, 1e-9), interpolation="nearest")
            A.axhline(nup - .5, color=ORANGE, lw=.55)
            A.axvline(nup - .5, color=ORANGE, lw=.55)
            A.set_xticks([]); A.set_yticks([])
            for sp in A.spines.values():
                sp.set_linewidth(.6); sp.set_color("#b9b7b1")
            if a.what == "S":
                # s_ij は符号つきの余弦。対角比は意味を持たないので出さない。
                off = ~np.eye(NE, dtype=bool)
                A.set_title(T("同/異スピン差 %+.3f　寄与 %.1f%%", "spin gap %+.3f q %.1f%%")
                            % (_spin_gap(M, nup), it["qp"][h]),
                            fontsize=6.9, loc="left", pad=1.8, color="#2b2b2b")
            else:
                tot = M.sum()
                dia = (100 * M[np.arange(NE), np.arange(NE)].sum() / tot
                       if tot > 1e-8 else np.nan)
                A.set_title(T("自 %s　寄与 %.1f%%", "self %s q %.1f%%")
                            % ("--" if not np.isfinite(dia) else "%.0f%%" % dia,
                               it["qp"][h]),
                            fontsize=6.9, loc="left", pad=1.8, color="#2b2b2b")
            if col == 0:
                A.set_ylabel("%s\n$r_s$ = %g" % (it["lab"], it["rs"]),
                             fontsize=9.6, labelpad=8)
        # 列の見出し（そのブロックの一番上のパネルの上）
        A0 = ax[blk * NS, col]
        pos = A0.get_position()
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.30 * GAP / Hfig,
                 "h%d　$%s$ = %.3f" % (h, LSYM, w[h]), fontsize=9.0, ha="center",
                 va="bottom", color="#12406f", weight="bold")

    fig.text(0.004, 0.995, T("遮蔽アテンション（screening・Screenformer）",
                             "screening attention"),
             fontsize=13.5, weight="bold", color="#0f7b6c", va="top")
    fig.text(0.004, 1 - 0.62 / Hfig,
             T((("内容チャネル  $s_{ij}=\\bar{q}_i\\cdot\\bar{k}_j=\\cos\\theta_{ij}$"
                 "（窓を掛ける前）　")
                if a.what == "S" else
                ("距離込み関連度マップ  $W_{ij}=\\alpha(s_{ij})\\,m_h(d_{ij})$　"))
               + "—　列 = ヘッド（%sの小さい順）, 行 = 相。列を上から下に読む。"
                 "色の上限は列ごとに 3 相で共通" % LNAME,
               "relevance map, columns = heads, rows = phases"),
             fontsize=10.4, color="#0b0b0b", va="top")
    fig.text(0.004, (FOOT - 0.22) / Hfig,
             T("行 = query 電子 $i$、列 = key 電子 $j$。橙線 = スピン境界。配置平均。\n"
               + ("$s_{ij}$ は $q$ と $k$ を単位長にしたあとの内積で、距離も窓も掛かっていない。\n"
                  "同/異スピン差 = 同スピン対と異スピン対での $s_{ij}$ の平均の差（対角を除く）。0 なら内容チャネルはスピンを見ていない。\n"
                  if a.what == "S" else
                  (("灰 = 殻の外（ほぼ常に $W=0$）。\n" if SHELL else "灰 = 窓の外（ほぼ常に $W=0$）。\n")
                   + "自 % = 重み全体に占める対角 $W_{ii}$ の割合。大 = その電子自身の情報を主に使う（局在・結晶側）、"
                     "小 = 他電子との相関を積極的に使う（液体側）。\n"))
               + "寄与 % = $\\|c_h^{\\rm pair}\\|^2/\\sum_{h'}\\|c_{h'}^{\\rm pair}\\|^2$。"
               "そのヘッドが運ぶ 2 体成分の割合。",
               "rows = query, cols = key"),
             fontsize=8.8, color="#2b2b2b", va="top", linespacing=1.9)
    fig.savefig(a.out, dpi=150, facecolor=fig.get_facecolor())
    print("  wrote", a.out)


if __name__ == "__main__":
    main()
