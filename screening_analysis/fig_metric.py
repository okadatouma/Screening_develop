# -*- coding: utf-8 -*-
"""セル・周期ノルム・d_ref を、実際の計算セットアップの上で説明する図。

    python screening_analysis/fig_metric.py --folder <結晶の run> --out fig_metric.png
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from fig_qwin import _font, _lattice, _pnorm                        # noqa

AI, ENJI, MIDORI, HAI, SUMI = "#12406f", "#a5262f", "#0f7b6c", "#8a8880", "#141412"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out", default="fig_metric.png")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, FancyArrow
    ja = _font()
    T = lambda x, y: x if ja else y                                  # noqa: E731

    cfg = json.load(open(os.path.join(a.folder, "config.json")))
    L = _lattice(cfg)                       # 列が L1, L2
    ne = sum(cfg["system"]["electrons"])
    rs = cfg["system"]["make_local_energy_kwargs"]["potential_kwargs"][
        "interaction_energy_scale"]
    from periodicwave.CustomScreenformer import _max_periodic_distance as DR
    dref = float(DR(L))
    a_ee = float(np.sqrt(2 * np.pi / np.sqrt(3)))

    ck = sorted(glob.glob(os.path.join(a.folder, "qmcjax_ckpt_*.npz")))
    P = np.load(ck[-1], allow_pickle=True)["data"].item()["positions"]
    P = np.asarray(P).reshape(-1, ne, 2)
    R = P[0]                                                   # 1 配置
    fr0 = np.mod(np.linalg.solve(L, R.T).T, 1.0)               # セル内へ折り返す
    R = fr0 @ L.T

    HFIG = 13.1
    fig = plt.figure(figsize=(14.6, HFIG))
    _y = lambda inch: 1.0 - inch / HFIG                       # noqa: E731
    gs = fig.add_gridspec(2, 3, width_ratios=[1.12, 1.02, 1.0],
                          height_ratios=[1.0, 0.98], wspace=0.30, hspace=0.46,
                          left=0.045, right=0.955, top=_y(1.10), bottom=_y(4.75))
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[0, 2]); ax5 = fig.add_subplot(gs[1, 2])
    gs2 = fig.add_gridspec(1, 3, width_ratios=[1.10, 1.00, 1.02], wspace=0.30,
                           left=0.055, right=0.955, top=_y(7.95), bottom=_y(10.45))
    ax6 = fig.add_subplot(gs2[0, 0]); ax7 = fig.add_subplot(gs2[0, 1])
    ax8 = fig.add_subplot(gs2[0, 2])

    # ---- ① スーパーセルと周期像 -------------------------------------------
    corners = np.array([[0, 0], L[:, 0], L[:, 0] + L[:, 1], L[:, 1]])
    for m in (-1, 0, 1):
        for n in (-1, 0, 1):
            sh = m * L[:, 0] + n * L[:, 1]
            mid = (m == 0 and n == 0)
            ax1.add_patch(Polygon(corners + sh, closed=True, fill=mid,
                                  fc="#f2f4f7" if mid else "none",
                                  ec=SUMI if mid else HAI,
                                  lw=1.9 if mid else 0.8,
                                  ls="-" if mid else (0, (3, 3)),
                                  alpha=1.0 if mid else .65, zorder=1))
            ax1.plot(R[:, 0] + sh[0], R[:, 1] + sh[1], "o", ms=5.2 if mid else 3.0,
                     color=AI if mid else HAI, alpha=1.0 if mid else .40,
                     mec="white", mew=.7 if mid else 0, zorder=3 if mid else 2)
    for v, lab, col in ((L[:, 0], r"$\mathbf{L}_1$", ENJI),
                        (L[:, 1], r"$\mathbf{L}_2$", ENJI)):
        ax1.annotate("", xy=(v[0], v[1]), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", lw=2.2, color=col,
                                     shrinkA=0, shrinkB=0), zorder=5)
        ax1.text(v[0] * 0.56 + (0.7 if v[1] > 1 else 0.0),
                 v[1] * 0.56 + (0.0 if v[1] > 1 else -0.95), lab, color=col,
                 fontsize=13, ha="center", va="center", zorder=6,
                 bbox=dict(fc="white", ec="none", alpha=.85, pad=1.5))
    ax1.set_aspect("equal"); ax1.axis("off")
    lim = 1.35
    ax1.set_xlim((-lim) * abs(L[0, 0]) + L[0, 1] * 0.6, lim * abs(L[0, 0]))
    ax1.set_ylim(-lim * L[1, 1] * 0.85, lim * L[1, 1] * 1.05)
    ax1.set_title(T("① スーパーセル ＝ 実際に計算している箱",
                    "① the supercell"), fontsize=11.6, pad=6)
    ax1.text(0.5, -0.055,
             T("濃い箱の中の %d 個だけを解き、外は同じ箱の繰り返しとして扱う（周期境界条件）。\n"
               "箱の面積は $\\pi N$、電子の間隔は $a_{ee}=\\sqrt{2\\pi/\\sqrt{3}}=%.3f$"
               % (ne, a_ee),
               "periodic boundary conditions"),
             transform=ax1.transAxes, fontsize=9.4, ha="center", va="top",
             color="#2b2b2b", linespacing=1.8)

    # ---- ② 1 次元で見た「距離」---------------------------------------------
    Lx = float(np.linalg.norm(L[:, 0]))
    x = np.linspace(0, Lx, 400)
    euc = np.minimum(x, Lx - x)
    sine = (Lx / np.pi) * np.sin(np.pi * x / Lx)
    ax2.plot(x, euc, "-", color=HAI, lw=2.2,
             label=T("最小イメージ（Euclid）", "min-image"))
    ax2.plot(x, sine, "-", color=AI, lw=2.4, label=T("周期ノルム", "periodic norm"))
    ax2.plot([Lx / 2], [Lx / 2], "o", ms=9, mfc="none", mec=ENJI, mew=2.2, zorder=5)
    ax2.annotate(T("ここで折れる\n→ $\\nabla^2\\log\\psi$ が壊れる", "kink"),
                 xy=(Lx / 2, Lx / 2), xytext=(Lx * 0.10, Lx * 0.455),
                 fontsize=8.8, color=ENJI, ha="left", va="center", linespacing=1.7,
                 arrowprops=dict(arrowstyle="-", lw=.9, color=ENJI,
                                 shrinkA=2, shrinkB=6))
    ax2.set_xlim(0, Lx); ax2.set_ylim(0, Lx / 2 * 1.22)
    ax2.set_xticks([0, Lx / 2, Lx])
    ax2.set_xticklabels(["0", r"$|\mathbf{L}_1|/2$", r"$|\mathbf{L}_1|$"])
    ax2.set_xlabel(T("2 電子の隔たり（$\\mathbf{L}_1$ に沿って歩いた量）", "separation"),
                   fontsize=10.0)
    ax2.set_ylabel(T("2 電子の距離 $d$", "distance $d$"), fontsize=10.0)
    ax2.grid(alpha=.22, lw=.7); ax2.legend(fontsize=8.8, frameon=False, loc="lower center")
    ax2.set_title(T("② なぜ正弦の物差しを使うか", "② why the sine metric"),
                  fontsize=11.0, pad=6)
    # 正弦ノルムの正体：箱を輪に丸めたときの「弦」。最小イメージは輪に沿った「弧」。
    axi = ax2.inset_axes([0.615, 0.545, 0.36, 0.42], facecolor="white")
    th = np.linspace(0, 2 * np.pi, 300)
    axi.plot(np.cos(th), np.sin(th), "-", color=HAI, lw=1.1, alpha=.8)
    t0, t1 = np.pi / 2, np.pi / 2 + 2.1
    ta = np.linspace(t0, t1, 120)
    axi.plot(np.cos(ta), np.sin(ta), "-", color=HAI, lw=3.6, solid_capstyle="round")
    axi.plot([np.cos(t0), np.cos(t1)], [np.sin(t0), np.sin(t1)], "-", color=AI, lw=2.6)
    axi.plot([np.cos(t0), np.cos(t1)], [np.sin(t0), np.sin(t1)], "o", ms=5.5,
             color=SUMI, zorder=5)
    axi.text(1.30, 0.52, T("弧 ＝ 最小イメージ", "arc"), color="#6f6d68", fontsize=7.6,
             ha="left", va="center")
    axi.text(1.30, -0.32, T("弦 ＝ 周期ノルム", "chord"), color=AI, fontsize=7.6,
             ha="left", va="center")
    axi.text(-1.15, 1.42, T("箱を輪に丸めると", "roll the box"), color="#52514e",
             fontsize=7.4, ha="left", va="center")
    axi.set_xlim(-1.25, 4.10); axi.set_ylim(-1.35, 1.62)
    axi.set_aspect("equal"); axi.axis("off"); axi.patch.set_alpha(.92)

    # ---- ③ 定義式 -----------------------------------------------------------
    ax3.axis("off")
    ax3.text(0.0, 1.0,
             T("③ 周期ノルムの式（$d$ 次元）", "③ periodic norm"), fontsize=10.6,
             color=MIDORI, va="top", weight="bold", transform=ax3.transAxes)
    ax3.text(0.0, 0.875,
             r"$d^{\rm per} = \dfrac{1}{2\pi}\sqrt{\sum_{mn} G_{mn}"
             r"\left(a_m a_n + b_m b_n\right)}$",
             fontsize=13.5, va="top", transform=ax3.transAxes)
    ax3.text(0.0, 0.585,
             r"$a_m = 1-\cos 2\pi f_m,\quad b_m = \sin 2\pi f_m,"
             r"\quad G = \mathbf{L}^{T}\mathbf{L}$",
             fontsize=11.2, va="top", transform=ax3.transAxes)
    ax3.text(0.0, 0.435,
             T("$\\mathbf{L}$ は①の 2 本の矢印を並べた行列（列が $\\mathbf{L}_1,\\ "
               "\\mathbf{L}_2$）。\n"
               "$f_m$ はその 2 本で測った、2 電子の隔たりの分数座標。\n"
               "$a_m,\\ b_m$ は箱を輪に丸めたときの弦の成分（②の挿図）。\n"
               "$d^{\\rm per}$ は弧ではなく弦の長さ。だから折れ点がない。",
               "fractional coordinates; chord, not arc"),
             fontsize=8.9, va="top", transform=ax3.transAxes, color="#2b2b2b",
             linespacing=1.62)

    # ---- ④ d_ref はセルのどこで出るか --------------------------------------
    # 片方の電子をセルの中央に置いて、そこから見た d^per を塗る（f を -1/2..1/2 で取る）
    ng = 300
    u = np.linspace(-0.5, 0.5, ng, endpoint=False)
    F1, F2 = np.meshgrid(u, u, indexing="ij")
    FR = np.stack([F1, F2], axis=-1)
    D = _pnorm(FR, L)
    XY = FR @ L.T
    pc = ax4.pcolormesh(XY[..., 0], XY[..., 1], D, cmap="magma", shading="nearest",
                        vmin=0, vmax=dref)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    ax4.plot([XY[i, j, 0]], [XY[i, j, 1]], "*", ms=19, color=SUMI,
             mec="white", mew=1.3, zorder=5)
    ax4.plot([0], [0], "o", ms=8.5, color="white", mec=SUMI, mew=1.4, zorder=5)
    ax4.annotate(T("片方の電子", "one electron"), (0, 0), textcoords="offset points",
                 xytext=(0, -14), fontsize=8.8, color="white", va="top", ha="center")
    ax4.annotate(T("いちばん遠い点\n$d_{\\rm ref}$ = %.3f" % dref, "$d_{ref}$"),
                 (XY[i, j, 0], XY[i, j, 1]), textcoords="offset points",
                 xytext=(15, 0), fontsize=9.0, color=SUMI, ha="left", va="center",
                 linespacing=1.6,
                 bbox=dict(fc="white", ec="none", alpha=.82, pad=1.8))
    _c = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) @ L.T
    ax4.add_patch(Polygon(_c, closed=True, fill=False, ec="white", lw=1.5))
    ax4.set_aspect("equal"); ax4.axis("off")
    _mx, _my = np.abs(_c[:, 0]).max(), np.abs(_c[:, 1]).max()
    ax4.set_xlim(-_mx * 1.07, _mx * 1.07); ax4.set_ylim(-_my * 1.12, _my * 1.14)
    cb = fig.colorbar(pc, ax=ax4, fraction=0.042, pad=0.02)
    cb.set_label(T("2 電子の距離 $d^{\\rm per}$", "$d^{per}$"), fontsize=9.0)
    cb.ax.tick_params(labelsize=8.2)
    ax4.set_title(T("④ $d_{\\rm ref}$ ＝ この物差しの上限",
                    "④ $d_{ref}$"), fontsize=11.0, pad=6)

    # ---- ⑤ その軸の上に窓が乗る --------------------------------------------
    Q = P[:400]
    fr = (Q[:, :, None, :] - Q[:, None, :, :]) @ np.linalg.inv(L).T
    dp = _pnorm(fr, L); iu = np.triu_indices(ne, 1)
    dd = dp[:, iu[0], iu[1]].ravel()
    xs = np.linspace(0, 5.0, 400)
    h, e = np.histogram(dd, bins=70, range=(0, dref), density=True)
    ax5.fill_between(0.5 * (e[1:] + e[:-1]), 0, h / h.max(), color=HAI, alpha=.30,
                     lw=0, zorder=0, label=T("電子どうしの距離の分布", "pairs"))
    for w, c, xl in ((2.0, "#4c1d80", 0.30), (2.8, "#1f7a6c", 0.46),
                     (3.9, "#c9a227", 0.62), (5.0, "#c96a12", 0.78)):
        ax5.plot(xs, 0.5 * (1 + np.cos(np.pi * np.clip(xs / w, 0, 1))), lw=2.0,
                 color=c, zorder=3)
        ax5.text(w * xl, 0.5 * (1 + np.cos(np.pi * xl)) + 0.035,
                 "$w_h$=%.1f" % w, color=c, fontsize=8.6, ha="center", zorder=4,
                 bbox=dict(fc="white", ec="none", alpha=.75, pad=1.0))
    ax5.axvspan(dref, 5.05, color=ENJI, alpha=.085, zorder=1)
    ax5.axvline(dref, color=ENJI, lw=1.3, ls=(0, (4, 2.5)), zorder=2)
    ax5.text(dref * 1.03, 0.79, T(" $d_{\\rm ref}$ より右は\n 全ての対が入る",
                                  " all pairs inside"),
             color=ENJI, fontsize=8.6, va="top", ha="left", linespacing=1.6, zorder=6,
             bbox=dict(fc="white", ec="none", alpha=.75, pad=1.6))
    ax5.set_xlim(0, 5.05); ax5.set_ylim(0, 1.20)
    ax5.set_xlabel(T("電子どうしの距離 $d$（周期ノルム, $r_s a_B^{*}$）", "$d$"),
                   fontsize=10.0)
    ax5.set_ylabel(T("窓 $m_h(d)$", "$m_h(d)$"), fontsize=10.0)
    ax5.grid(alpha=.22, lw=.7)
    ax5.legend(fontsize=8.6, frameon=False, loc="lower left")
    ax5.set_title(T("⑤ ヘッドの窓はこの軸の上に置く", "⑤ windows on that axis"),
                  fontsize=11.0, pad=6)
    ax5.text(0.015, 0.985,
             T("横軸 ＝ 距離（配置ごとに変わる）\n曲線 ＝ 窓（固定）",
               "x = distance, curves = windows"),
             transform=ax5.transAxes, fontsize=8.4, color="#2b2b2b", va="top",
             ha="left", linespacing=1.7, zorder=6,
             bbox=dict(fc="white", ec="none", alpha=.80, pad=2.0))

    # ---- 表題と説明 --------------------------------------------------------
    fig.text(0.006, 0.985, T("セル・周期ノルム・$d_{\\rm ref}$",
                             "cell, periodic norm, $d_{ref}$"),
             fontsize=13.4, weight="bold", color=MIDORI, va="top")
    fig.text(0.006, 0.930,
             T("クーロン 2DEG, $N$ = %d（↑%d ↓%d）, 三角スーパーセル, $r_s$ = %g の配置を使用"
               % (ne, cfg["system"]["electrons"][0], cfg["system"]["electrons"][1], rs),
               "Coulomb 2DEG"),
             fontsize=10.4, color="#0b0b0b", va="top")
    lines = [
        "セル ＝ 有限個の電子を入れて解くための箱。外側は同じ箱の繰り返しなので、"
        "電子は箱の端から出ると反対側から入ってくる（①）。したがって「2 電子の距離」は"
        "箱をまたいだ測り方が要る。",
        "素直な最小イメージは箱の半分のところで折れてしまい、そこで 2 階微分が発散する。"
        "局所エネルギーには $\\nabla^2 \\log \\psi$ が入るので、これは使えない（②）。"
        "代わりに滑らかで周期的な正弦の物差しを使う（③）。",
        "その物差しでいちばん遠い点の値が $d_{\\rm ref}$ = %.4f。菱形セルでは"
        "$f=(2/3,\\,1/3)$ の深い穴で出る（④の★）。どの 2 電子も $d \\leq d_{\\rm ref}$ なので、"
        "これが距離軸の天井になる。" % dref,
        "距離 $d$ と窓 $m_h$ は別のもの。$d$ は電子の位置から決まる入力で、配置ごとに変わる。"
        "窓 $m_h(d) = \\frac{1}{2}\\left[1+\\cos\\left(\\pi\\min(d/w_h,1)\\right)\\right]$ は"
        "その $d$ を $0$〜$1$ の重みに変える関数で、ヘッドごとに 1 本、今回は学習させず固定してある（⑤）。",
        "　　両者は $W^{(h)}_{ij} = \\alpha(s_{ij})\\, m_h(d_{ij})$ でつながる"
        "（$\\alpha$ は内容チャネル）。$w_h > d_{\\rm ref}$ の窓はどの対でも $m_h > 0$ になってしまい、"
        "距離で選ぶ働きを失う ＝ 赤い帯。",
    ] if ja else ["see figure"]
    fig.text(0.008, _y(5.35), "\n".join(lines), fontsize=9.4, color="#2b2b2b",
             va="top", linespacing=2.0)

    # ================= 下段：窓を広くするための手法 =========================
    _section_B(fig, ax6, ax7, ax8, dd, dref, a_ee, ja, T, _y)

    fig.savefig(a.out, dpi=155)
    print("  wrote %s   d_ref=%.4f  a_ee=%.4f" % (a.out, dref, a_ee))


def _er(M):
    """窓の族が実質いくつの独立な形を持つか（対ごとにヘッド平均を引いて SVD）。"""
    X = M - M.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False) ** 2
    return float((s.sum() ** 2) / (s ** 2).sum())


def _mono(w, b, x):
    """立ち上がり幅 b で切る入れ子窓。b = w が今使っている生の余弦窓。"""
    t = np.clip((x - w + b) / b, 0.0, 1.0)
    return 0.5 * (1 + np.cos(np.pi * t))


def _shell(c, sg, x):
    t = np.clip((x - c) / sg, -1.0, 1.0)
    return 0.5 * (1 + np.cos(np.pi * t))


def _section_B(fig, ax6, ax7, ax8, dd, dref, a_ee, ja, T, _y):
    """下段：窓を広くする（＝ヘッドの分解能を上げる）ための 2 つの手法。"""
    import matplotlib.pyplot as plt
    H = 16
    ds = dd[::7]
    xs = np.linspace(0, 5.0, 500)
    h, e = np.histogram(dd, bins=70, range=(0, dref), density=True)
    ctr = 0.5 * (e[1:] + e[:-1])

    # ---- ⑥ シェル窓 --------------------------------------------------------
    cc = np.linspace(1.0, dref, H)
    sg = np.full(H, (dref - 1.0) / (H - 1))
    ax6.fill_between(ctr, 0, h / h.max(), color=HAI, alpha=.30, lw=0, zorder=0,
                     label=T("電子どうしの距離の分布", "pairs"))
    for i, (c, g) in enumerate(zip(cc, sg)):
        ax6.plot(xs, _shell(c, g, xs), lw=1.5, color=plt.cm.viridis(i / (H - 1)),
                 zorder=3)
    ax6.axvspan(dref, 5.05, color=ENJI, alpha=.085, zorder=1)
    ax6.axvline(dref, color=ENJI, lw=1.3, ls=(0, (4, 2.5)), zorder=2)
    _c, _g = cc[-3], sg[-3]
    ax6.annotate("", xy=(_c - _g, 0.05), xytext=(_c, 0.05),
                 arrowprops=dict(arrowstyle="<|-", lw=1.6, color=ENJI), zorder=6)
    ax6.text(_c - _g - 0.08, 0.13,
             T("選別するのは内側の端 $c_h-\sigma_h$\n（ここより内は厳密に 0）",
               "inner edge selects"),
             color=ENJI, fontsize=8.4, ha="right", va="bottom", linespacing=1.65,
             zorder=6, bbox=dict(fc="white", ec="none", alpha=.85, pad=1.6))
    ax6.set_xlim(0, 5.05); ax6.set_ylim(0, 1.22)
    ax6.set_xlabel(T("電子どうしの距離 $d$（周期ノルム, $r_s a_B^{*}$）", "$d$"), fontsize=10.0)
    ax6.set_ylabel(T("窓 $m_h(d)$", "$m_h(d)$"), fontsize=10.0)
    ax6.grid(alpha=.22, lw=.7); ax6.legend(fontsize=8.6, frameon=False, loc="upper right")
    ax6.set_title(T("⑥ 手法 1：シェル窓（帯域通過）にする", "⑥ shell windows"),
                  fontsize=11.0, pad=6)

    # ---- ⑦ 窓を鋭くする ----------------------------------------------------
    w0 = 2.8
    ax7.fill_between(ctr, 0, h / h.max(), color=HAI, alpha=.30, lw=0, zorder=0)
    for b, col, lab in ((w0, "#c96a12", T("$b=w_h$（いまの窓）", "now")),
                        (0.8, "#1f7a6c", "$b=0.8$"), (0.2, AI, "$b=0.2$")):
        ax7.plot(xs, _mono(w0, b, xs), lw=2.1, color=col, zorder=3, label=lab)
    ax7.axvline(w0, color=SUMI, lw=1.0, ls=":", zorder=2)
    ax7.text(w0 * 1.02, 1.19, "$w_h$", fontsize=9.2, color=SUMI, ha="left", va="top")
    ax7.set_xlim(0, 5.05); ax7.set_ylim(0, 1.22)
    ax7.set_xlabel(T("電子どうしの距離 $d$", "$d$"), fontsize=10.0)
    ax7.set_ylabel(T("窓 $m_h(d)$", "$m_h(d)$"), fontsize=10.0)
    ax7.grid(alpha=.22, lw=.7); ax7.legend(fontsize=8.8, frameon=False, loc="upper right")
    ax7.set_title(T("⑦ 手法 2：立ち上がりを鋭くする", "⑦ sharpen the roll-off"),
                  fontsize=11.0, pad=6)

    # ---- ⑧ 効果 ------------------------------------------------------------
    bs = np.geomspace(3.0, 0.02, 18)
    W = np.quantile(ds, np.linspace(1.0 / H, 1.0, H))
    q = np.quantile(ds, np.linspace(0.02, 0.98, H))
    rm = [_er(np.array([_mono(w, b, ds) for w in W])) for b in bs]
    rs_ = [_er(np.array([_shell(c, b, ds) for c in q])) for b in bs]
    cv = np.pi ** 2 / (2 * bs ** 2)
    ax8.plot(cv, rs_, "-s", color=AI, lw=2.2, ms=5.0, mfc="white",
             label=T("シェル窓（⑥）", "shell"))
    ax8.plot(cv, rm, "-o", color=ENJI, lw=2.2, ms=5.0,
             label=T("入れ子の窓（いまの形）", "nested"))
    ax8.axhline(2.5, color=ENJI, ls=(0, (5, 3)), lw=1.4, alpha=.85)
    ax8.text(cv[0] * 1.15, 2.62, T("入れ子の上限 2.5", "cap 2.5"), color=ENJI,
             fontsize=8.8, va="bottom")
    ax8.plot([cv[0]], [rm[0]], "*", ms=16, color=ENJI, mec="white", mew=1.2, zorder=6)
    ax8.annotate(T("いまの窓 %.1f" % rm[0], "now"), (cv[0], rm[0]),
                 textcoords="offset points", xytext=(13, 9), fontsize=8.8, color=ENJI)
    ax8.set_xscale("log"); ax8.set_yscale("log"); ax8.minorticks_off()
    ax8.set_xticks([1, 10, 100, 1000, 10000])
    ax8.set_xticklabels(["1", "10", "100", "1000", "$10^4$"], fontsize=9.0)
    ax8.set_yticks([1, 2, 2.5, 5, 10]); ax8.set_yticklabels(["1", "2", "2.5", "5", "10"])
    ax8.set_xlabel(T("窓の鋭さ  $\\max|m_h''| = \\pi^2/2b^2$（右ほど鋭い）", "sharpness"),
                   fontsize=10.0)
    ax8.set_ylabel(T("実質いくつの形を持つか", "effective rank"), fontsize=10.0)
    ax8.grid(alpha=.22, lw=.7, which="both")
    ax8.legend(fontsize=8.8, frameon=False, loc="upper left")
    ax8.set_title(T("⑧ 効果（16 ヘッドでの有効ランク）", "⑧ effect"), fontsize=11.0, pad=6)

    # ---- 見出しと説明 ------------------------------------------------------
    fig.add_artist(plt.Line2D([0.008, 0.992], [_y(6.85)] * 2, color=HAI, lw=1.0))
    fig.text(0.008, _y(7.10), T("窓を広くするための手法", "how to widen the windows"),
             fontsize=13.0, weight="bold", color=MIDORI, va="top")
    fig.text(0.008, _y(7.44),
             T("⑤ の窓は入れ子なので、16 本あっても実質 1 通りの形しか持たない。"
               "$w_h$ の上限を下げても、それは直らない。", ""),
             fontsize=10.2, color="#0b0b0b", va="top")

    i2 = int(np.argmin(np.abs(bs - 0.2)))
    ln = [
        "⑥ シェル窓　$m_h(d) = \\frac{1}{2}\\left[1+\\cos\\left(\\pi\\,{\\rm clip}"
        "\\left((d-c_h)/\\sigma_h,\\ -1,\\ 1\\right)\\right)\\right]$"
        "　… 中心 $c_h$、半値幅 $\\sigma_h$ の帯。$d \\leq c_h-\\sigma_h$ で厳密に 0 になる。",
        "　　選別するのは外側の端ではなく内側の端なので、帯が $d_{\\rm ref}$ をはみ出しても情報を失わない。"
        "制約が $w_h < d_{\\rm ref}$ から $c_h-\\sigma_h <$ 対の最大距離 に移り、赤い帯の問題が消える。"
        "$c_h = 0$ とすれば ⑤ の窓に厳密に戻る（後方互換）。",
        "⑦ 立ち上がり幅 $b$ を狭めて鋭くする。物差しを尖らせるのと数学的に同じ操作で"
        "（窓は $m_h(d) = \\varphi(g(d)/v_h)$ の形なので、物差し $g$ を歪めても断面 $\\varphi$ を変えても"
        "同じ族になる）、こちらのほうが $d$ を使う他の場所に副作用が出ない。",
        "⑧ 鋭くすれば入れ子でも上がるが（%.1f → %.1f）、2.5 で頭打ちになる。"
        "入れ子（$\\{d<w_h\\}$ という劣位集合の列）である限り相関行列の固有値が $1/k^2$ に落ち着き、"
        "その参加率が厳密に $90/36 = 2.5$ だからで、ヘッドを増やしても変わらない。" % (rm[0], rm[-1]),
        "　　シェル窓は劣位集合ではなく重なった帯なので上限を越える。同じ鋭さ（$b = 0.2$）で比べて"
        "入れ子 %.1f に対しシェル %.1f。ただし鋭くするほど $\\max|m_h''| = \\pi^2/2b^2$ が大きくなるので、\n"
        "　　$\\nabla^2\\log\\psi$ の分散が悪化しないかは $\\sigma$ を振って確かめる必要がある（未検証）。"
        % (rm[i2], rs_[i2]),
    ] if ja else ["shell windows and sharpening"]
    fig.text(0.008, _y(11.25), "\n".join(ln), fontsize=9.4, color="#2b2b2b",
             va="top", linespacing=2.0)
    print("  有効ランク: 入れ子 %.2f→%.2f  シェル %.2f→%.2f"
          % (rm[0], rm[-1], rs_[0], rs_[-1]))


if __name__ == "__main__":
    main()
