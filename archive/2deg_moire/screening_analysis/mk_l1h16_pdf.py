# -*- coding: utf-8 -*-
"""窓を固定した 1 層 × 16 ヘッドの結果を、r_s ごとに並べて 1 冊にする。

    python screening_analysis/mk_l1h16_pdf.py [--out FILE] [--suffix ""]
"""
import argparse, glob, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
for p in fm.findSystemFonts():
    if "NotoSansCJK" in p:
        fm.fontManager.addfont(p)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.unicode_minus"] = False

W, H = 16.54, 11.69
M = 0.040
INK, SUB, AI, ENJI, RULE = "#141412", "#4a4946", "#155f7a", "#a5262f", "#d6d3cc"
R = "results/2deg-Coulomb/CustomScreenformer"
FOOT = "クーロン 2DEG — 遮蔽アテンション 1 層 × 16 ヘッド, 窓は固定 / 2026-08-31"
def set_foot(lo, hi):
    global FOOT
    FOOT = ("クーロン 2DEG — 遮蔽アテンション 1 層 × 16 ヘッド, 窓は固定 "
            "[%.2f, %.2f] / 2026-08-31" % (lo, hi))
LADDER = np.geomspace(0.45, 3.00, 16)


def squeeze_white(im, maxrun=40):
    """図の中の「真っ白な帯」を詰める。bigN_maps の図は本文用に下余白を
    大きく取ってあり、そのままだと紙面の 3 割が空白になるため。"""
    a = im[..., :3] if im.ndim == 3 else im[..., None]
    # bigN_maps の地色は #fcfcfb（青成分 0.984）なので 0.985 だと 1 行も通らない。
    row_white = (a.min(axis=(1, 2)) > (0.95 if a.max() <= 1.0 else 242))
    keep = np.ones(len(row_white), bool)
    i = 0
    while i < len(row_white):
        if row_white[i]:
            j = i
            while j < len(row_white) and row_white[j]:
                j += 1
            if j - i > maxrun:
                keep[i + maxrun // 2: j - maxrun // 2] = False
            i = j
        else:
            i += 1
    return im[keep]


def crop_panels(im, gap=26):
    """行列パネルの帯だけを切り出す。bigN_maps の図は下に長い解説文を持つが、
    本 PDF は「測り方」のページでそれを済ませているので、ここでは絵だけ大きく見せる。
    青い（カラーマップの）画素がある行を探し、そこから上下に、
    真っ白な行が gap 本続くところまで広げる（パネル上のラベルを拾うため）。"""
    if im.ndim != 3 or im.shape[2] < 3:
        return im
    a = im[..., :3].astype(float)
    if a.max() > 1.5:
        a = a / 255.0
    # 行あたりの「青い」画素数で判定する。見出しの青緑や本文中の数式も
    # 色は付くが、画素数が桁で違うので幅の 2% を閾値にすれば拾わない。
    nb = ((a[..., 2] - a[..., 0]) > 0.05).sum(axis=1)
    blue = nb > 0.02 * a.shape[1]
    if not blue.any():
        return im
    white = a.min(axis=(1, 2)) > 0.95      # 地色 #fcfcfb を白とみなす
    lo, hi = int(np.argmax(blue)), len(blue) - 1 - int(np.argmax(blue[::-1]))
    i, run = lo, 0
    while i > 0 and run < gap:
        i -= 1
        run = run + 1 if white[i] else 0
    top = i
    i, run = hi, 0
    while i < len(white) - 1 and run < gap:
        i += 1
        run = run + 1 if white[i] else 0
    bot = i
    return im[max(0, top - 4): min(len(white), bot + 4)]


def page(pdf, kicker, title, eqs=None, table=None, image=None, note=None,
         eqsize=11.4, tsize=10.2, squeeze=True, crop=False, tw_scale=1.0,
         after_table=None, after_note=None, reserve=0.0):
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    y = 1 - M
    ax.text(M, y, kicker, fontsize=9, color=AI, va="top"); y -= 0.019
    ax.text(M, y, title, fontsize=17, color=INK, va="top", weight="bold"); y -= 0.036
    ax.plot([M, 1 - M], [y, y], color=RULE, lw=1.0); y -= 0.022
    if eqs:
        elh = eqsize * 2.95 / (H * 72); xe = M + 0.132
        y -= elh * 0.25
        for lab, mth in eqs:
            ax.text(M, y, lab, fontsize=eqsize - 2.6, color=AI, va="top")
            ax.text(xe, y, mth, fontsize=eqsize, color=INK, va="top")
            y -= elh * (1.34 if any(k in mth for k in ("\\sum", "\\frac", "\\sqrt")) else 1.0)
        y -= elh * 0.55
    if table:
        lh = tsize * 2.55 / (H * 72)
        hdr, rows = table[0], table[1:]
        ncol = len(hdr); tw = min(1 - 2 * M, (0.108 * ncol + 0.10) * tw_scale)
        xs = [M + tw * j / ncol for j in range(ncol)]
        for j, t in enumerate(hdr):
            ax.text(xs[j], y, t, fontsize=tsize, color=INK, va="top", weight="bold")
        y -= lh * 1.4
        ax.plot([M, M + tw], [y + lh * 0.45, y + lh * 0.45], color=RULE, lw=0.9)
        for r in rows:
            c = ENJI if str(r[0]).startswith("結晶") else SUB
            for j, t in enumerate(r):
                ax.text(xs[j], y, str(t), fontsize=tsize, color=c, va="top")
            y -= lh
        y -= lh * 0.7
    if note:
        nh = 10.0 * 2.5 / (H * 72)
        for ln in note:
            ax.text(M, y, ln, fontsize=10.0, color=SUB, va="top")
            y -= nh
        y -= nh * 0.5
    if image:
        im = mpimg.imread(image)
        if crop:
            im = crop_panels(im)
        elif squeeze:
            im = squeeze_white(im)
        ih, iw = im.shape[0], im.shape[1]
        s = min((1 - 2 * M) * W / iw, (y - 0.045 - reserve) * H / ih)
        dw, dh = iw * s / W, ih * s / H
        axi = fig.add_axes([(1 - dw) / 2, y - dh, dw, dh]); axi.imshow(im); axi.axis("off")
        y = y - dh - 0.028
    if after_table:
        lh = tsize * 2.55 / (H * 72)
        hdr, rows = after_table[0], after_table[1:]
        ncol = len(hdr); tw = min(1 - 2 * M, (0.108 * ncol + 0.10) * tw_scale)
        xs = [M + tw * j / ncol for j in range(ncol)]
        for j, t in enumerate(hdr):
            ax.text(xs[j], y, t, fontsize=tsize, color=INK, va="top", weight="bold")
        y -= lh * 1.4
        ax.plot([M, M + tw], [y + lh * 0.45, y + lh * 0.45], color=RULE, lw=0.9)
        for r in rows:
            for j, t in enumerate(r):
                ax.text(xs[j], y, str(t), fontsize=tsize, color=SUB, va="top")
            y -= lh
        y -= lh * 0.6
    if after_note:
        for ln in after_note:
            ax.text(M, y, ln, fontsize=10.0, color=SUB, va="top")
            y -= 10.0 * 2.5 / (H * 72)
    ax.text(M, 0.018, FOOT, fontsize=8, color=SUB, va="bottom")
    pdf.savefig(fig); plt.close(fig)


def _lat(cfg):
    m = re.search(r"unserializable object: \[\[(.*?)\]\]", json.dumps(cfg))
    return np.array([float(x) for x in re.findall(
        r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(1))][:4]).reshape(2, 2)


def measure(tag, rs):
    d = glob.glob(os.path.join(R, "el13_12_rs*_tri_l2h16_%s" % tag))[0]
    cfg = json.load(open(d + "/config.json")); L = _lat(cfg)
    pk0 = cfg["system"]["make_local_energy_kwargs"]["potential_kwargs"]
    o0 = cfg["network"]["CustomScreenformer"]
    dref0 = 0.0
    ne = sum(cfg["system"]["electrons"])
    st = np.genfromtxt(d + "/train_stats.csv", delimiter=",", names=True)
    step = int(st["step"][-1]); E = float(np.median(st["energy"][-300:])) / ne / rs ** 2
    ck = sorted(glob.glob(d + "/qmcjax_ckpt_*.npz"))[-1]
    X = np.asarray(np.load(ck, allow_pickle=True)["data"].item()
                   ["positions"]).reshape(-1, ne, 2)
    fr = (X[:, :, None, :] - X[:, None, :, :]) @ np.linalg.inv(L).T
    dv = (fr - np.round(fr)) @ L.T; de = np.linalg.norm(dv, axis=-1)
    G = L.T @ L; a = 1 - np.cos(2 * np.pi * fr); b = np.sin(2 * np.pi * fr)
    dp = (1 / (2 * np.pi)) * np.sqrt(np.einsum('...m,mn,...n->...', a, G, a)
                                     + np.einsum('...m,mn,...n->...', b, G, b))
    for i in range(ne):
        de[:, i, i] = 1e9; dp[:, i, i] = 1e9
    p6, nn = [], []
    for bi in range(min(len(X), 400)):
        for i in range(ne):
            k = np.argsort(de[bi, i])[:6]
            p6.append(abs(np.mean(np.exp(6j * np.arctan2(dv[bi, i, k, 1], dv[bi, i, k, 0])))))
            nn.append(de[bi, i, k[0]])
    nn = np.array(nn)
    iu = np.triu_indices(ne, 1)
    v = np.where(dp > 1e8, np.nan, dp)[:, iu[0], iu[1]].ravel()
    from head_contrib import cached
    z = cached(d, nconf=48)
    wv = z["w"][0] * 3.2152
    qp = (z["Cp"] ** 2).sum(0); qp = 100 * qp / qp.sum()
    qt = (z["C"] ** 2).sum(0);  qt = 100 * qt / qt.sum()
    o = np.argsort(wv); wv, qp, qt = wv[o], qp[o], qt[o]
    _f, _sp = o0["init_softmask_w_factor"], o0["init_softmask_w_spread"]
    _dr = float(np.sqrt(np.abs(np.linalg.det(L))) * 0 + 3.2152)
    from periodicwave import CustomScreenformer as _CS
    import jax.numpy as _jnp
    _dr = float(_CS._max_periodic_distance(_jnp.asarray(L)))
    return dict(tag=tag, rs=rs, folder=os.path.basename(d), dir=d, step=step, ne=ne,
                U=float(pk0.get("moire_potential_strength", 0.0)),
                phi=float(pk0.get("moire_potential_phi", 0.0)),
                NL=o0["num_layers"], NH=o0["num_heads"], dref=_dr,
                ladder=np.geomspace(_f / _sp * _dr, _f * _sp * _dr, o0["num_heads"]),
                batch=cfg.get("batch_size", 0),
                E=E, heg=-1.1061 / rs + 0.814 / rs ** 1.5,
                p6=float(np.mean(p6)), fl=float(nn.std() / nn.mean()),
                dnn=float(np.nanmean(np.nanmin(np.where(dp > 1e8, np.nan, dp), axis=2))),
                q=[float(np.nanquantile(v, p)) for p in (1e-4, 1e-3, 1e-2, 5e-2)],
                w=wv, qp=qp, qt=qt,
                wq=float((qp * wv).sum() / qp.sum()),
                w12=float(np.interp(50.0, np.cumsum(qp), wv)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "l1h16_frozen_window.pdf"))
    ap.add_argument("--setup-only", action="store_true",
                    help="セットアップのまとめ 1 ページだけ出す")
    ap.add_argument("--tags", default="fl05,fb30,fc60",
                    help="液体,境界,結晶 のタグ")
    ap.add_argument("--qwin", default=os.path.join(HERE, "..", "out", "fig_qwin.png"))
    ap.add_argument("--dens", default=os.path.join(HERE, "..", "out",
                                                   "fig_coulomb_l1h16.png"))
    ap.add_argument("--wmax", default="", help="梯子の上限を比べた図（あれば入れる）")
    a = ap.parse_args()

    tg = a.tags.split(",")
    RUNS = [("液体", tg[0], 5), ("境界", tg[1], 30), ("結晶", tg[2], 60)]
    D = {t: measure(t, rs) for _, t, rs in RUNS}
    set_foot(D[tg[0]]["w"][0], D[tg[0]]["w"][-1])
    pdf = PdfPages(a.out)

    # ---- 1. まとめ ---------------------------------------------------------
    ml, mb, mc = D[tg[0]], D[tg[1]], D[tg[2]]
    tab = [["", "$r_s$", "step", "$E/N$ [Ha*]", "2D HEG の表式", "差",
            "$\\psi_6$", "ゆらぎ/平均", r"$\langle d_{\rm NN}\rangle$",
            r"$\langle w\rangle_q$"]]
    for lab, t, rs in RUNS:
        m = D[t]
        tab.append([lab, "%g" % rs, "%d" % m["step"], "%.6f" % m["E"], "%.6f" % m["heg"],
                    "%.2f%%" % (100 * abs(m["E"] - m["heg"]) / abs(m["heg"])),
                    "%.3f" % m["p6"], "%.3f" % m["fl"], "%.3f" % m["dnn"],
                    "%.3f" % m["wq"]])
    dead = np.where(mc["qp"] < 1e-9)[0]
    w_dead = mc["w"][dead[-1]] if len(dead) else 0.0
    q_liq = ml["qp"][dead] if len(dead) else np.array([np.nan])
    # 梯子が d_ref を超えている場合、その段は全ての対を含むので長さを選んでいない
    _over = mc["w"] > mc["dref"]
    _ovq = [float(D[t]["qp"][_over].sum()) for _, t, _r in RUNS]
    if len(dead):
        _c1 = ["結論 1 — 結晶では $w_h \\leq %.2f$ のヘッド（%d 本）の 2 体寄与 $q_h$ が $10^{-9}$ を下回る。"
               % (w_dead, len(dead)),
               "　　結晶では 1 万対に 1 対しか %.2f より近づかず、$m_h(d)$ は $d \\geq w_h$ で厳密に 0 だから、"
               % mc["q"][0],
               "　　これらのヘッドには相手が実質 1 個も入らない。液体は同じ窓で "
               "$10^{%d}$〜$10^{%d}$ %%、境界はその中間。"
               % (np.floor(np.log10(q_liq.min())), np.ceil(np.log10(q_liq.max())))]
    elif _over.sum():
        _c1 = ["結論 1 — この梯子は上の %d 段が $w_h > d_{\\rm ref} = %.2f$ にあり、"
               "そこはどの対も必ず窓に入る。" % (int(_over.sum()), mc["dref"]),
               "　　つまりその段は距離を選んでおらず、長さの目盛りとしては働かない。"
               "にもかかわらず 2 体寄与の %.0f%%（液体）〜 %.0f%%（結晶）がそこに乗る。"
               % (_ovq[0], _ovq[2]),
               "　　実際に長さを分解しているのは残りの %.0f%%（結晶）だけである。" % (100 - _ovq[2])]
    else:
        _c1 = ["結論 1 — どのヘッドも 2 体寄与が $10^{-9}$ を上回っており、"
               "梯子の全段が働いている。"]
    page(pdf, "まとめ", "窓を固定した目盛りで、液体 → 境界 → 結晶を読む",
         eqs=[("系", r"$N = 25$（↑13 ↓12）, 三角スーパーセル, "
                    r"$H = -\frac{1}{2}\sum_i \nabla_i^2 + r_s \sum_{i<j} r_{ij}^{-1}$"
                    r"　（モアレなし, $U = 0$）"),
              ("ネットワーク", r"遮蔽アテンション 1 層 × 16 ヘッド, attn 16, value 16, "
                            r"行列式 4, batch 1024, $\eta_0 = 10$, $t_0 = 10^{5}$"),
              ("窓", r"$w_h = d_{\rm ref}\,{\rm softplus}(\omega_h)$ を"
                    "学習させず固定（$\\omega_h$ の勾配を切った）。"
                    r"$d_{\rm ref} = 3.2152$"),
              ("目盛り", "$w_h = " + ",\\ ".join("%.3f" % v for v in ml["w"]) + "$"),
              ("秩序変数", r"$\psi_6 = \left|\langle e^{6i\theta}\rangle\right|$"
                        r"（各電子の最近接 6 個）, "
                        r"ゆらぎ/平均 $= {\rm sd}(d_{\rm NN})/\langle d_{\rm NN}\rangle$"
                        r"（2 次元の融解の目安 0.1 前後）"),
              ("読み出す長さ", r"$\langle w\rangle_q = \sum_h q_h w_h$,　"
                          r"$q_h = \|c_h^{\rm pair}\|^2 \,/\, \sum_{h'}\|c_{h'}^{\rm pair}\|^2$")],
         table=tab, tw_scale=1.06,
         note=["3 つの run で窓が 1 桁も動いていないことをチェックポイントで確認済み。",
               "したがって以降の図はすべて同一の横軸（同じ 16 段の目盛り）の上で比べられる。",
               ""] + _c1 + ["",
               "結論 2 — 狭いヘッドは消えるのではなく、2 体から 1 体へ役割を変える。",
               "　　結晶の h0（$w = %.3f$）は出力への寄与が %.1f%% あるのに、2 体成分は %.2f%% しかない"
               "（液体では %.2f%%）。差は自分自身の情報である。"
               % (mc["w"][0], mc["qt"][0], mc["qp"][0], ml["qp"][0]),
               "",
               "結論 3 — 読み出した長さ $\\langle w \\rangle_q$ は $%.3f \\to %.3f \\to %.3f$、"
               % (ml["wq"], mb["wq"], mc["wq"]),
               "　　実測の最近接距離 $%.3f \\to %.3f \\to %.3f$ と同じ向きに動く。"
               % (ml["dnn"], mb["dnn"], mc["dnn"])])

    # ---- 2. 測り方 ---------------------------------------------------------
    qt = [["", "$r_s$", "下位 0.01%", "下位 0.1%", "下位 1%", "下位 5%",
           r"$\langle d_{\rm NN}\rangle$"]]
    for lab, t, rs in RUNS:
        m = D[t]
        qt.append([lab, "%g" % rs] + ["%.3f" % x for x in m["q"]] + ["%.3f" % m["dnn"]])
    page(pdf, "測り方", "窓は距離のしきい値、寄与率はそのヘッドが足すベクトルの長さ",
         eqs=[("重み", r"$W^{(h)}_{ij} = \alpha(s_{ij})\, m_h(d_{ij})$,　"
                     r"$s_{ij} = \bar q_i \cdot \bar k_j$"),
              ("内容チャネル", r"$\alpha(s) = \left[\max\left(1 - (1-s)/r,\ 0\right)\right]^2$"),
              ("距離チャネル（窓）", r"$m_h(d) = \frac{1}{2}\left[1 + "
                              r"\cos\left(\pi \min(d/w_h,\,1)\right)\right]$,　"
                              r"$d \geq w_h$ で厳密に $0$"),
              ("出力の加法分解", r"${\rm out}_i = e^{s_O} \sum_h "
                            r"\left[{\rm gate} \odot {\rm TanhNorm}({\rm attn})\right]_{i,h}"
                            r"\, W_O^{(h)}$　… 第 $h$ 項が $c_h(i)$"),
              ("2 体成分", r"$c_h^{\rm pair} = c_h - c_h^{\rm self}$,　"
                        r"$c_h^{\rm self}$ は $W^{(h)}_{ij} \to W^{(h)}_{ii}\delta_{ij}$ とした量"),
              ("距離の物差し", r"$d_{ij}$ は周期ノルム。セル内での最大値が $d_{\rm ref} = 3.2152$")],
         table=qt, tw_scale=0.92,
         note=["電子どうしの距離の下位分位点（周期ノルム, $r_s a_B^{*}$）。1024 配置 × 300 対から。",
               "結晶では全体の 0.01% でも 0.793 までしか近づかない。窓がそれより狭いヘッドは、",
               "$m_h(d)$ が $d \\geq w_h$ で厳密に 0 なので、相手が 1 個も入らない。これが下の図の主役。"])

    # ---- 3. 結果 -----------------------------------------------------------
    if os.path.exists(a.qwin):
        page(pdf, "結果 ①", "相が変わると、寄与率が目盛りの上を移動する",
             image=a.qwin, squeeze=False)
    if a.wmax and os.path.exists(a.wmax):
        page(pdf, "結果 ②", "窓の上限を $d_{\\rm ref}$ より上へ伸ばすと、読み出しは鈍る",
             image=a.wmax, squeeze=False)
    if os.path.exists(a.dens):
        page(pdf, "結果 ③", "同じ 3 つの波動関数の実空間：一様 → 三角格子",
             image=a.dens, squeeze=False)

    # ---- 4. r_s ごと -------------------------------------------------------
    for lab, t, rs in RUNS:
        m = D[t]
        f = m["dir"] + "/" + m["folder"]
        rows = [["ヘッド", "$w_h$", r"$q_h$ [%]（2 体）", "全体 [%]"]]
        for i in (0, 4, 8, 11, 13, 14, 15):
            rows.append(["h%d" % i, "%.3f" % m["w"][i],
                         ("%.2e" % m["qp"][i]) if m["qp"][i] < 0.01 else "%.2f" % m["qp"][i],
                         "%.1f" % m["qt"][i]])
        page(pdf, "%s　$r_s$ = %g" % (lab, rs), "$q \\cdot k$ グリッド　—　窓の小さい順",
             image=f + "_grid.png", crop=True, tw_scale=0.66, reserve=0.375,
             note=["step %d,　$E/N = %.6f$ Ha*（表式 %.6f）,　$\\psi_6 = %.3f$,　"
                   "ゆらぎ/平均 $= %.3f$" % (m["step"], m["E"], m["heg"], m["p6"], m["fl"]),
                   "灰色 $=$ 窓の外（$W = 0$）。対角の濃さ $=$ そのヘッドが"
                   "自分自身の情報をどれだけ使っているか。"],
             after_table=rows,
             after_note=["「全体」は 1 体成分を含む $\\|c_h\\|^2$ の割合、"
                         "「2 体」は $\\|c_h^{\\rm pair}\\|^2$ の割合。",
                         "この 2 つが離れているヘッドは、出力には効いているが"
                         "相関は運んでいない（＝ 1 体の特徴変換になっている）。"])
        page(pdf, "%s　$r_s$ = %g" % (lab, rs),
             "ヘッドごとの重みの届き方と、窓 $m_h(d)$", image=f + "_radial.png",
             crop=True, reserve=0.16,
             after_note=[
                 "① の縦線は三角格子の殻位置 $a$, $\\sqrt{3}a$, $2a$（$a = \\sqrt{2\\pi/\\sqrt{3}} = 1.905$）。",
                 "② の各線は 1 つのヘッドの平均重み $\\langle \\alpha\\, m \\rangle$ を距離の関数として見たもの。",
                 "　窓 $w_h$ の手前で必ず 0 に落ちる（$m_h(d)$ が $d \\geq w_h$ で厳密に 0 だから）。",
                 "　この落ち際の位置が 16 本とも固定されているので、相を変えても線の「届く先」は動かない。",
                 "　動くのは線の高さ ＝ そのヘッドが実際にどれだけ重みを拾えたか、である。"])

    # ---- 5. 記録 -----------------------------------------------------------
    rec = [["", "$r_s$", "タグ", "step", "フォルダ"]]
    for lab, t, rs in RUNS:
        rec.append([lab, "%g" % rs, t, "%d" % D[t]["step"], D[t]["folder"]])
    _w1 = (["弱いところ 1 — step 数が揃っていない（%d / %d / %d）。学習量だけでも寄与率は"
            % (ml["step"], mb["step"], mc["step"]),
            "　少数のヘッドに集中するので、相の効果と学習量の効果が厳密には分離できていない。",
            "　チェックポイントは 2 分おきに全部残してあるので、同じ step で揃えた版を作れる。"]
           if len({ml["step"], mb["step"], mc["step"]}) > 1 else
           ["弱いところ 1 — 3 つとも同じ step（%d）で止めてあるが、読み出しの収束は相によって違う。"
            % ml["step"],
            "　液体は 2000 step ほどで落ち着くのに対し、結晶は最後まで上位ヘッドへの集中が進み続けている。",
            "　結晶側だけ長く回した版で確かめる必要がある。"])
    if _over.sum():
        _w2 = ["弱いところ 2 — 梯子の上 %d 段が $w_h > d_{\\rm ref} = %.2f$ にあり、"
               "そこは全ての対を含むので長さを選べない。"
               % (int(_over.sum()), mc["dref"]),
               "　2 体寄与の %.0f〜%.0f%% がその段に乗っており、相のちがいを読む分解能を"
               "そのぶん失っている（前の梯子との比較は「結果 ②」のページ）。" % (min(_ovq), max(_ovq)),
               "　上限は $d_{\\rm ref}$ で止めるのが正しい。$w$ をもっと伸ばしたいなら、"
               "$N$ を増やして $d_{\\rm ref} = 0.3376\\,a_{ee}\\sqrt{N}$ を大きくするしかない。"]
    else:
        _w2 = ["弱いところ 2 — 目盛りの下半分は 3 相とも $q_h$ が小さい。",
               "　次に振るなら効いている範囲を細かく刻む方が分解能が上がる。"]
    page(pdf, "記録", "何を回したか、どこが弱いか", table=rec, tw_scale=1.25,
         eqs=[("環境変数", "PW_FREEZE_W=1,  PW_W_LO=%.4f,  PW_W_HI=%.4f,  "
                       "PW_LAYERS=%d,  PW_HEADS=%d,  PW_BATCH=%d"
                       % (ml["w"][0], ml["w"][-1], ml["NL"], ml["NH"], ml["batch"])),
              ("実行", "ws (manatoCPU),  --gres=gpu:2 を 1 ジョブで取り, "
                     r"$r_s{=}60$ を 1 枚専有, $r_s{=}30$ と $r_s{=}5$ でもう 1 枚を共有"),
              ("梯子の検証", "チェックポイントから復元した $w$ と指定値の差は最大 "
                        r"$2.2\times10^{-5}$（3 run とも）")],
         note=_w1 + [""] + _w2 + ["",
             "窓を学習させなかった理由 — $r_s{=}60$ で 10,000 step 回しても $\\omega_h$ は"
             "ほとんど動かない（h4〜h7 で 0.02〜0.3%）。",
             "　下端は $w <$ 最短の電子間距離 になった瞬間 $m \\equiv 0$ となり 1 体チャネルに落ちて"
             "戻れない暴走、",
             "　上端は $\\partial m/\\partial w \\propto (\\pi/2)\\sin(\\pi d/w)\\,d/w^2$ が"
             "$1/w^2$ で潰れるうえ、エネルギーが $w$ に対してほぼ平坦。",
         ])

    pdf.close()
    print("->", a.out)


if __name__ == "__main__":
    main()
