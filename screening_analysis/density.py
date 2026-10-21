"""walker から「電子が実際にどう並んでいるか」を見る。

    python screening_analysis/density.py

出すもの:
  ① 一体密度 ρ(r)            — PBC の有限系では対称性が破れないので平坦なはず（確認用）
  ② 対相関 g(r)              — 液体/結晶の違いはここに出る。三角格子の殻位置を併記
  ③ 二体密度 g(x,y)          — 基準電子から見た周囲の電子分布（六方構造が見える）
  ④ 構造因子 S(k)            — 結晶なら逆格子点に Bragg ピーク
  ⑤ 窓の中の電子数 n_h        — 各ヘッドが「何個の電子を見ているか」。w を個数に翻訳する

コード単位では 1 電子あたりの面積が π なので密度は ρ = 1/π。
半径 R の円内の他電子数（連続近似）は ρπR² = R²。
"""
import glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)


def load(folder):
    cfg = json.load(open(os.path.join(folder, "config.json")))
    m = re.search(r"unserializable object: \[\[(.*?)\]\]", json.dumps(cfg))
    L = np.array([float(x) for x in re.findall(
        r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(1))][:4]).reshape(2, 2)
    ne = sum(cfg["system"]["electrons"])
    rs = float(cfg["system"]["make_local_energy_kwargs"]["potential_kwargs"]
               ["interaction_energy_scale"])
    ck = sorted(glob.glob(os.path.join(folder, "qmcjax_ckpt_*.npz")))[-1]
    pos = np.asarray(np.load(ck, allow_pickle=True)["data"].item()["positions"][0])
    return pos.reshape(-1, ne, 2), L, ne, rs


def rel(R, L):
    """最小像の相対ベクトル (B,N,N,2) と距離 (B,N,N)。"""
    dr = R[:, :, None, :] - R[:, None, :, :]
    f = dr @ np.linalg.inv(L).T
    dr = (f - np.round(f)) @ L.T
    return dr, np.linalg.norm(dr, axis=-1)


def gofr(d, ne, area, nb=60, rmax=None):
    """rmax は内接円半径まで。それ以上は最小像規約で角度の一部しか数えられず g(r) が落ちる。"""
    iu = np.triu_indices(ne, 1)
    x = d[:, iu[0], iu[1]].ravel()
    rmax = rmax or x.max()
    x = x[x <= rmax]
    h, e = np.histogram(x, bins=nb, range=(0, rmax))
    r = .5 * (e[1:] + e[:-1])
    shell = np.pi * (e[1:] ** 2 - e[:-1] ** 2)
    return r, h / (len(d) * ne * (ne / area) * shell / 2) / 1.0


def sofk(R, L, ne, nmax=7):
    """許される k = m b1 + n b2 上の S(k)。b はセルの逆格子。"""
    B = 2 * np.pi * np.linalg.inv(L).T                      # 行が b1, b2
    ks, S = [], []
    for m in range(-nmax, nmax + 1):
        for n in range(-nmax, nmax + 1):
            if m == 0 and n == 0:
                continue
            k = m * B[:, 0] + n * B[:, 1]
            ph = np.exp(1j * (R @ k))                       # (B,N)
            S.append(float((np.abs(ph.sum(1)) ** 2).mean() / ne))
            ks.append(k)
    return np.array(ks), np.array(S)


def window_counts(folder, nconf=48):
    """各ヘッドの窓に入る電子数。hard = d<w の個数、soft = Σ m(d) の窓重み付き個数。"""
    from attn_probe import probe
    a = probe(folder, nconf=nconf)
    NE, off = a["NE"], ~np.eye(a["NE"], dtype=bool)
    d = a["d"][:, off].reshape(a["d"].shape[0], NE, NE - 1)   # 周期ノルム
    w = a["w"]                                               # (L,H) 絶対長
    hard = np.array([[(d < w[l, h]).sum(-1).mean() for h in range(a["NH"])]
                     for l in range(a["NL"])])
    soft = np.array([[(0.5 * (1 + np.cos(np.pi * np.clip(d / w[l, h], 0, 1)))).sum(-1).mean()
                      for h in range(a["NH"])] for l in range(a["NL"])])
    return hard, soft, w, a["dref"], a


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.font_manager as fm
    for p in fm.findSystemFonts():
        if "NotoSansCJK" in p: fm.fontManager.addfont(p)
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED, BLUE, ORANGE, GREEN, RED = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#008300", "#a5262f"
    ROOT = os.path.dirname(HERE)
    D9 = os.path.join(ROOT, "results/2deg-Coulomb/CustomScreenformer/el5_4_rs%d_tri_det")
    D49 = os.path.join(ROOT, "gpu/results/2deg-Coulomb/CustomScreenformer/el25_24_rs5_tri_l2h16")
    A = np.sqrt(2 * np.pi / np.sqrt(3))                      # 第1近接 1.9046

    fig = plt.figure(figsize=(16.8, 9.4)); fig.patch.set_facecolor("#fcfcfb")
    gs = fig.add_gridspec(2, 4, hspace=.40, wspace=.34, left=.048, right=.988, top=.895, bottom=.115,
                          width_ratios=[1.02, 1.0, 1.0, 1.16])

    # ---------- セットアップ（図の顔として最初に置く）
    axs = fig.add_subplot(gs[:, 3]); axs.axis("off")
    axs.add_patch(plt.Rectangle((0, 0), 1, 1, transform=axs.transAxes, facecolor="#f2f1ec",
                                edgecolor="#d8d4ca", lw=1.0))
    SETUP = [
        (r"系", r"2 次元電子ガス（2DEG）"),
        (r"", r"一様な正電荷背景 + 裸の Coulomb $1/r$"),
        (r"", r"周期境界、Ewald 和（遮蔽項は入れていない）"),
        (r"スーパーセル", r"三角格子と整合する菱形。面積 $A_{\rm sc} = \pi N$"),
        (r"", r"→ 密度 $\rho=1/\pi$、第 1 近接 $a=1.9046$"),
        (r"単位", r"長さ = $r_s a_B$、エネルギー = $\mathrm{Ha}/r_s^2$"),
        (r"", r"この単位では $a$ は $r_s$ に依らず一定"),
        (r"電子数", r"$N=9$ (↑5↓4) / 25 (↑13↓12) / 49 (↑25↓24)"),
        (r"", r"正準集団。$N$ は固定"),
        (r"波動関数", r"CustomScreenformer（遮蔽注意）"),
        (r"", r"対照 = CustomPsiformer（softmax 注意）"),
        (r"", r"複素出力、行列式 4 個、Jastrow なし"),
        (r"", r"エンベロープなし（PBC feature layer）"),
        (r"構成", r"$N$=9: 4 層×4 ヘッド、batch 64、400 step"),
        (r"", r"$N$=25,49: 2 層×16 ヘッド、batch 512、1000 step"),
        (r"", r"attn_dim = value_dim = 16、mlp_dim = 64"),
        (r"初期値", r"$r_{\rm trim}$=1.9、窓は $0.8\,d_{\rm ref}$ を中心に幾何級数"),
        (r"最適化", r"KFAC、lr 0.1、MCMC 10 step、burn-in 100"),
        (r"局所E", r"folx forward Laplacian（$T$ と $V$ を分離可）"),
        (r"種", r"$N$=9 は PW_SEED=42 固定 → ヘッド番号が"),
        (r"", r"条件間で対応する"),
        (r"参照", r"Wigner 転移 $r_s=31(1)$（Drummond–Needs）"),
        (r"", r"Madelung $-V_1/N=1.106103$ を 7 桁再現"),
    ]
    axs.text(.06, .975, "セットアップ", transform=axs.transAxes, fontsize=12.5, va="top",
             ha="left", weight="bold", color="#0b0b0b")
    y = .925
    for k, v in SETUP:
        if k:
            axs.text(.045, y, k, transform=axs.transAxes, fontsize=8.9, va="top", ha="left",
                     color="#a5262f", weight="bold")
        axs.text(.255, y, v, transform=axs.transAxes, fontsize=8.6, va="top", ha="left",
                 color="#2b2b28")
        y -= .0395

    # ---------- ① 一体密度の確認 + ② g(r)
    ax = fig.add_subplot(gs[0, 0])
    def _rin(L):
        return 0.5 * min(abs(np.cross(L[:, 0], L[:, 1])) / np.linalg.norm(L[:, i]) for i in (0, 1))
    for rs, c in [(1, BLUE), (5, "#6a5acd"), (20, "#c07a2a"), (40, ORANGE)]:
        R, L, ne, _ = load(D9 % rs); _, d = rel(R, L)
        r, g = gofr(d, ne, abs(np.linalg.det(L)), nb=34, rmax=_rin(L))
        ax.plot(r, g, "-", color=c, lw=1.9, label="$r_s=%d$" % rs)
    R, L, ne, _ = load(D49); _, d49 = rel(R, L)
    r, g = gofr(d49, ne, abs(np.linalg.det(L)), nb=60, rmax=_rin(L))
    ax.plot(r, g, "-", color=RED, lw=2.4, label="$N=49$, $r_s=5$")
    for k, lab in [(1, "$a$"), (3, "$\\sqrt{3}a$"), (4, "$2a$")]:
        ax.axvline(A * np.sqrt(k), color=GREEN, lw=1.0, ls=":")
        ax.annotate(lab, (A * np.sqrt(k), 1.72), color=GREEN, fontsize=9, ha="center")
    ax.axhline(1, color=MUTED, lw=.7)
    ax.set_xlabel("$r$ [コード単位 = $r_s a_B$]（各系の内接円半径まで）"); ax.set_ylabel("$g(r)$")
    ax.set_title("① 対相関 $g(r)$ — 結晶側でピークが立つ", fontsize=11, loc="left")
    ax.legend(fontsize=8.5); ax.grid(alpha=.25); ax.set_ylim(0, 1.85)

    # ---------- ③ 二体密度 g(x,y)
    for k, (tag, folder, rs) in enumerate([("$N=49$, $r_s=5$（液体）", D49, 5),
                                           ("$N=9$, $r_s=40$（結晶側）", D9 % 40, 40)]):
        ax = fig.add_subplot(gs[0, 1 + k])
        R, L, ne, _ = load(folder); dv, dd = rel(R, L)
        off = ~np.eye(ne, dtype=bool)
        xy = dv[:, off].reshape(-1, 2)
        lim = 0.5 * min(np.linalg.norm(L[:, 0]), np.linalg.norm(L[:, 1]))
        H, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=61,
                                   range=[[-lim, lim], [-lim, lim]])
        cell = (xe[1] - xe[0]) * (ye[1] - ye[0])
        H = H / (len(R) * ne * (ne / abs(np.linalg.det(L))) * cell)
        rin = 0.5 * min(abs(np.cross(L[:, 0], L[:, 1])) / np.linalg.norm(L[:, i]) for i in (0, 1))
        XX, YY = np.meshgrid(.5 * (xe[1:] + xe[:-1]), .5 * (ye[1:] + ye[:-1]), indexing="ij")
        H = np.ma.masked_where(XX ** 2 + YY ** 2 > rin ** 2, H)
        lim = rin
        cmp_ = plt.get_cmap("magma").copy(); cmp_.set_bad("#fcfcfb")
        im = ax.imshow(H.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
                       cmap=cmp_, vmin=0, vmax=np.percentile(H.compressed(), 99))
        ax.set_xlim(-lim * 1.04, lim * 1.04); ax.set_ylim(-lim * 1.04, lim * 1.04)
        ax.set_aspect("equal")
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(A * np.cos(th), A * np.sin(th), color="#7ef9c8", lw=1.1, ls="--")
        ax.set_title("② %s\n基準電子から見た周囲の電子密度 $g(x,y)$" % tag, fontsize=10.5, loc="left")
        ax.set_xlabel("$x$"); ax.set_ylabel("$y$" if k == 0 else "")
        fig.colorbar(im, ax=ax, fraction=.046, pad=.03, label="$g(x,y)$")

    # ---------- ④ S(k)
    ax = fig.add_subplot(gs[1, 0])
    for folder, tag, c in [(D9 % 1, "$N=9$, $r_s=1$", BLUE), (D9 % 40, "$N=9$, $r_s=40$", ORANGE),
                           (D49, "$N=49$, $r_s=5$", RED)]:
        R, L, ne, _ = load(folder)
        ks, S = sofk(R, L, ne)
        kk = np.linalg.norm(ks, axis=1)
        o = np.argsort(kk)
        ax.plot(kk[o], S[o], "o", ms=3.4, color=c, alpha=.7, label=tag)
    G1 = 4 * np.pi / (np.sqrt(3) * A)
    ax.axvline(G1, color=GREEN, lw=1.2, ls=":")
    ax.annotate(" 第1逆格子点 $|G_1|$=%.2f" % G1, (G1, 3.4), color=GREEN, fontsize=9)
    ax.set_xlabel("$|k|$"); ax.set_ylabel("$S(k)$")
    ax.set_title("③ 構造因子 $S(k)$ — Bragg ピークの立ち方", fontsize=11, loc="left")
    ax.legend(fontsize=8.5); ax.grid(alpha=.25); ax.set_xlim(0, 8); ax.set_yscale("log")

    # ---------- ⑤ 窓の中の電子数
    ax = fig.add_subplot(gs[1, 1:3])
    hard, soft, w, dref, a49 = window_counts(D49)
    C = np.load(os.path.join(D49, "head_contrib.npz"))["C"]
    ww = w.mean(0)
    ax.plot(ww, hard.mean(0), "o-", color=BLUE, lw=2, ms=7, label="窓の中の電子数（$d<w$ を数える）")
    ax.plot(ww, soft.mean(0), "s--", color=RED, lw=2, ms=6, label="窓で重み付けした電子数 $\\sum_j m(d_{ij})$")
    rr = np.linspace(0, ww.max() * 1.05, 100)
    ax.plot(rr, rr ** 2, "-", color=MUTED, lw=1.2, label="一様密度の予想 $\\rho\\pi R^2=R^2$")
    for k, n, lab in [(1, 6, "第1殻 6 個"), (3, 12, "＋第2殻 12 個"), (4, 18, "＋第3殻 18 個")]:
        ax.axvline(A * np.sqrt(k), color=GREEN, lw=1.0, ls=":")
        ax.annotate(lab, (A * np.sqrt(k), 46), color=GREEN, fontsize=8.5, rotation=90,
                    ha="right", va="top")
    wstar = float((C * w).sum() / C.sum())
    ax.axvline(wstar, color="#a5262f", lw=2.2)
    o = np.argsort(ww)
    nsoft = np.interp(wstar, ww[o], soft.mean(0)[o])
    ax.annotate("寄与で重み付けした窓 $w^*$ = %.2f\n→ 窓重み付きで %.1f 個" % (wstar, nsoft),
                (wstar, 26), color="#a5262f", fontsize=9.5, ha="right",
                xytext=(wstar - .15, 26))
    ax.annotate("青が 48 で頭打ちなのは幾何ではなく\n周期ノルムが $d_{\\rm ref}$=%.2f で飽和するため" % dref,
                (7.0, 41), color=BLUE, fontsize=8.6, ha="center")
    ax.set_xlabel("ヘッドの窓幅 $w$ [コード単位]"); ax.set_ylabel("窓の中の他電子の数")
    ax.set_title("④ 各ヘッドは何個の電子を見ているか — $w$ を「個数」に翻訳する（$N=49$, $r_s=5$）",
                 fontsize=11, loc="left")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=.25); ax.set_ylim(0, 50)

    fig.suptitle("電子はどう並んでいて、各ヘッドはそのうち何個を見ているか", fontsize=13.5,
                 x=.006, ha="left", y=.975)
    fig.text(.006, .022,
             "コード単位（長さ = $r_s a_B$）では 1 電子あたりの面積が $\\pi$ なので密度は $\\rho=1/\\pi$、"
             "半径 $R$ の円内の他電子数は $\\rho\\pi R^2=R^2$（灰線）。三角格子の殻は $a$, $\\sqrt{3}a$, $2a$ に 6, 12, 18 個（累積）。\n"
             "一体密度 $\\rho(\\mathbf{r})$ は PBC の有限系では並進対称性が破れないので厳密に平坦 — だから「結晶かどうか」は $g(r)$・$g(x,y)$・$S(k)$ で見る。"
             "$S(G_1)$/背景 = 0.96, 1.30, 1.58, 1.51, 1.75, 2.70（$r_s$=1,5,20,30,35,40）。真の結晶なら $S(G)\\approx N$ なので、"
             "400 step のこれらは秩序が芽生えた液体。", fontsize=9, color=MUTED)
    fig.savefig(os.path.join(HERE, "fig_density.png"), dpi=150, facecolor=fig.get_facecolor())
    print("saved fig_density.png")

    print("\n  ヘッドごとの窓と中の電子数（N=49, r_s=5, 層平均）")
    print("   H    w      w/d_ref   窓内の個数   窓重み付き   寄与%")
    sq = 100 * C ** 2 / (C ** 2).sum()
    for h in np.argsort(ww):
        print("   %-3d %6.3f %8.3f %11.1f %12.1f %8.1f%%"
              % (h, ww[h], ww[h] / dref, hard.mean(0)[h], soft.mean(0)[h], sq[:, h].sum()))
    print("\n  寄与で重み付けした窓 w* = %.3f → 窓内 %.1f 個 / 窓重み付き %.1f 個"
          % (wstar, np.interp(wstar, ww, hard.mean(0)), np.interp(wstar, ww, soft.mean(0))))
