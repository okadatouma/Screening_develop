# -*- coding: utf-8 -*-
"""λ-jellium の 12 本の図：(1) 対密度 g(x,y) と S(G)、(2) q·k 行列、(3) 擬スピン差のまとめ。

    python fig_lj_maps.py <CustomScreenformer フォルダ> <出力フォルダ> <2DEG 結晶> <2DEG 液体> [ラベル]
"""
import os, sys, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import style as S
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from lj_maps import load, first_shell, structure_factor, pair_density, qk, head_lengths, LAMS

A_NN = np.sqrt(2 * np.pi / np.sqrt(3))       # 三角格子の最近接距離（長さの単位 r_s a_B）
WIN = [("jn", "入れ子窓（16 ヘッド）"), ("jg", "ガウスの輪（8 ヘッド）")]
CM_G = LinearSegmentedColormap.from_list("g", ["#ffffff", "#dce8f7", "#93b8ea", "#2a78d6", "#12406f", "#0b1f38"])
CM_S = LinearSegmentedColormap.from_list("s", ["#12406f", "#5e8bc0", "#f4f4f2", "#f0a27a", "#b0461a"])


def folder(base, w, lam):
    tag = f"{w}{int(round(lam * 100)):02d}"
    return glob.glob(os.path.join(base, f"N36_rs15_lam*_{tag}"))[0]


def fig_density(base, out, ref_c, ref_l, label, width=9.6, name="lj_pair_density", header=True):
    R = 4.2
    fig, axs = plt.subplots(2, 7, figsize=(width, width * 3.5 / 9.6 + (0.15 if header else -0.12)))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80 if header else 0.83, bottom=0.04, wspace=0.06, hspace=0.18)
    for r, (w, wname) in enumerate(WIN):
        for c, lam in enumerate(LAMS):
            cfg, z, pos, sp, L, ne, ck = load(folder(base, w, lam))
            g, e = pair_density(pos, L, R=R)
            sg = structure_factor(pos, L, first_shell(L, ne))
            ax = axs[r, c]
            ax.imshow(g, origin="lower", extent=[-R, R, -R, R], cmap=CM_G, vmin=0, vmax=2.4)
            ax.text(-R + 0.25, R - 0.3, "$S(G)$=%.2f" % sg, fontsize=6.6, va="top", color=S.INK,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
            if r == 0:
                ax.set_title("$\\lambda$ = %.2f" % lam, fontsize=8.5, pad=3)
        if header:
            axs[r, 0].set_ylabel(wname, fontsize=8.0, labelpad=4)
        else:   # 報告用は行の高さが足りないので 2 行に分ける
            axs[r, 0].set_ylabel(wname.replace("（", "\n（"), fontsize=7.4, labelpad=3, linespacing=1.15)
    for r, (f, nm) in enumerate([(ref_c, "結晶 $r_s$=60"), (ref_l, "液体 $r_s$=5")]):
        cfg, z, pos, sp, L, ne, ck = load(f)
        g, e = pair_density(pos, L, R=R)
        sg = structure_factor(pos, L, first_shell(L, ne))
        ax = axs[r, 6]
        ax.imshow(g, origin="lower", extent=[-R, R, -R, R], cmap=CM_G, vmin=0, vmax=2.4)
        ax.text(-R + 0.25, R - 0.3, "$S(G)$=%.2f" % sg, fontsize=6.6, va="top", color=S.INK,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
        ax.text(0, -R + 0.25, nm, fontsize=6.8, ha="center", va="bottom", color=S.INK,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
    axs[0, 6].set_title("参考：2 次元電子", fontsize=8.5, pad=3)
    for ax in axs.flat:
        ax.add_patch(Circle((0, 0), A_NN, fc="none", ec=S.ORANGE, lw=0.6, ls=(0, (2, 2))))
        ax.plot([0], [0], "+", color=S.ORANGE, ms=4, mew=0.8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_linewidth(0.5); sp_.set_color("#b9b7b1")
    if header:
        fig.text(0.07, 0.975, "基準の電子（+）から見た他の電子の密度 $g(x,y)$（平均密度で割ったもの）" + label,
                 fontsize=9.2, va="top")
    cax = fig.add_axes([0.80, 0.905 if header else 0.905, 0.17, 0.02 if header else 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CM_G, norm=plt.Normalize(0, 2.4)), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1, 2]); cb.ax.tick_params(labelsize=6.5, length=2)
    cb.ax.xaxis.set_ticks_position("top")
    fig.text(0.795, 0.915 if header else 0.916, "$g$", fontsize=8, ha="right", va="center")
    fig.savefig(os.path.join(out, name + ".png"), dpi=200)
    fig.savefig(os.path.join(out, name + ".pdf"))
    plt.close(fig)


def sorted_matrix(Sw, spw, posw, L):
    """1 配置の s^h_ij を、擬スピン +1 → −1、その中は分数座標の行優先で並べ替える。"""
    s = (posw @ np.linalg.inv(L).T) % 1.0
    key = np.lexsort((s[:, 0], np.floor(6 * s[:, 1]), -spw))
    return Sw[:, key][:, :, key], int((spw > 0).sum())


def spin_gap(Sall, sp):
    """配置平均：同じ擬スピンの対と違う擬スピンの対での s^h_ij の平均の差（対角を除く）。[ヘッド]"""
    nw, H, ne, _ = Sall.shape
    same = sp[:, :, None] == sp[:, None, :]
    off = ~np.eye(ne, dtype=bool)[None]
    out = []
    for h in range(H):
        M = Sall[:, h]
        out.append(M[same & off].mean() - M[(~same) & off].mean())
    return np.array(out)


def fig_qk(base, out, w, wname, label, nconf=64, walker=0, heads=None, name=None, cell=1.02, header=True):
    rows = []
    for lam in LAMS:
        cfg, z, pos, sp, L, ne, ck = load(folder(base, w, lam))
        params = z["params"].item()
        Sall = qk(params, pos[:nconf], sp[:nconf], L)
        lens, lname = head_lengths(params, cfg, L)
        M, nup = sorted_matrix(Sall[walker], sp[walker], pos[walker], L)
        rows.append(dict(lam=lam, M=M, nup=nup, gap=spin_gap(Sall, sp[:nconf]), lens=lens))
    order = np.argsort(rows[0]["lens"])
    if heads is not None:
        order = order[heads]
    H = len(order)
    ncol = 8; nblk = int(np.ceil(H / ncol))
    HT = 1.3 if header else 0.45
    fig = plt.figure(figsize=(0.95 + cell * ncol, HT + nblk * (len(rows) * cell + 0.42)))
    Hf = fig.get_figheight(); Wf = fig.get_figwidth()
    for b in range(nblk):
        top = Hf - (HT - 0.1) - b * (len(rows) * cell + 0.42)
        gs = fig.add_gridspec(len(rows), ncol, left=0.82 / Wf, right=1 - 0.1 / Wf,
                              top=top / Hf, bottom=(top - len(rows) * cell + 0.1) / Hf, wspace=0.08, hspace=0.32)
        for k in range(ncol):
            hk = b * ncol + k
            if hk >= H: continue
            h = order[hk]
            for i, rw in enumerate(rows):
                ax = fig.add_subplot(gs[i, k])
                ax.imshow(rw["M"][h], cmap=CM_S, vmin=-1, vmax=1, interpolation="nearest")
                ax.axhline(rw["nup"] - 0.5, color=S.INK, lw=0.5); ax.axvline(rw["nup"] - 0.5, color=S.INK, lw=0.5)
                ax.set_xticks([]); ax.set_yticks([])
                for sp_ in ax.spines.values():
                    sp_.set_linewidth(0.4); sp_.set_color("#b9b7b1")
                ax.set_title("%+.2f" % rw["gap"][h], fontsize=6.0, pad=1.2, loc="left", color=S.SUB)
                if i == 0:
                    ax.text(0.5, 1.33, "h%d  %.2f" % (h, rw["lens"][h]), transform=ax.transAxes,
                            ha="center", va="bottom", fontsize=6.8, color="#12406f", weight="bold")
                if k == 0:
                    ax.set_ylabel("$\\lambda$=%.2f" % rw["lam"], fontsize=7.4, labelpad=3)
    if header:
      fig.text(0.82 / Wf, 1 - 0.12 / Hf, "q·k 行列 $s^h_{ij}=\\bar q_i\\cdot\\bar k_j$：" + wname + label,
             fontsize=9.0, va="top")
    if header:
      fig.text(0.82 / Wf, 1 - 0.36 / Hf,
             "列：ヘッド（青の数字は" + ("窓の幅 $w_h$" if w == "jn" else "輪の中心") + "）。行：$\\lambda$。"
             "各マスは 1 つの配置の 36×36 行列（行 $i$、列 $j$）で、黒線の左上が擬スピン +1 の電子、"
             "その中は位置の順。\nマスの左上の数字は擬スピン差（同じ擬スピンの対と違う擬スピンの対での $s^h_{ij}$ の平均の差、64 配置で平均）。"
             "色は $-1$（青）から $+1$（橙）。", fontsize=6.8, va="top", color=S.SUB, linespacing=1.6)
    fig.savefig(os.path.join(out, (name or f"lj_qk_{w}") + ".png"), dpi=200)
    plt.close(fig)
    return rows


def fig_gap(rows_by_w, out, label, header=True, name="lj_qk_gap"):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6 if header else 2.15), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.78 if header else 0.9, bottom=0.18 if header else 0.2, wspace=0.08)
    for ax, (w, wname) in zip(axs, WIN):
        rows = rows_by_w[w]
        lens = rows[0]["lens"]; order = np.argsort(lens)
        G = np.array([r["gap"] for r in rows])            # [lam, head]
        cmap = plt.get_cmap("viridis")
        for k, h in enumerate(order):
            ax.plot(LAMS, G[:, h], "o-", ms=2.5, lw=0.9, color=cmap(k / max(len(order) - 1, 1)))
        ax.plot(LAMS, G.mean(1), "-", color=S.INK, lw=2.0, label="ヘッドの平均")
        ax.axhline(0, color=S.LINE, lw=0.6, zorder=0)
        ax.set_title(wname, fontsize=8.6, pad=3); ax.set_xlabel("$\\lambda$")
        ax.legend(loc="upper left", fontsize=7, handlelength=1.4)
    axs[0].set_ylabel("擬スピン差")
    if header:
        fig.text(0.08, 0.985, "q·k の擬スピン差（同じ擬スピンの対 − 違う擬スピンの対）と $\\lambda$" + label,
                 fontsize=8.6, va="top")
        fig.text(0.08, 0.915, "細線：各ヘッド（色は窓の幅・輪の中心の順で、紫が内側、黄が外側）。太線：ヘッドの平均。",
                 fontsize=7.2, va="top", color=S.SUB)
    fig.savefig(os.path.join(out, name + ".png"), dpi=200)
    fig.savefig(os.path.join(out, name + ".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    base, out, ref_c, ref_l = sys.argv[1:5]
    label = sys.argv[5] if len(sys.argv) > 5 else ""
    os.makedirs(out, exist_ok=True)
    S.use()
    plt.rcParams.update({"font.size": 8})
    fig_density(base, out, ref_c, ref_l, label)
    print("density done")
    rows_by_w = {}
    for w, wname in WIN:
        rows_by_w[w] = fig_qk(base, out, w, wname, label)
        print("qk", w, "done")
    fig_gap(rows_by_w, out, label)
