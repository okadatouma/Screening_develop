# -*- coding: utf-8 -*-
"""窓を固定したときの「波動関数への寄与率 vs 窓の長さ」を 3 相で重ねる。

    python screening_analysis/fig_qwin.py --runs liquid=DIR,boundary=DIR,crystal=DIR \
        --out fig_qwin.png

窓 w_h を学習させず 16 段の目盛りとして固定してあるので、3 つの run で横軸が完全に共通になる。
したがって曲線の形の違いは、そのまま「波動関数がどの長さを使っているか」の違いになる。
"""
import argparse, glob, json, os, re, sys
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


def _lattice(cfg):
    m = re.search(r"unserializable object: \[\[(.*?)\]\]", json.dumps(cfg))
    return np.array([float(x) for x in re.findall(
        r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(1))][:4]).reshape(2, 2)


def _pnorm(fr, L):
    M = L.T @ L
    a = 1.0 - np.cos(2 * np.pi * fr); b = np.sin(2 * np.pi * fr)
    return (1 / (2 * np.pi)) * np.sqrt(
        np.einsum('...m,mn,...n->...', a, M, a) + np.einsum('...m,mn,...n->...', b, M, b))


def load(folder, nconf=48):
    """1 つの run から (w[H], q_pair[H], q_tot[H], d_nn, d_min, rs, E) を取り出す。"""
    from head_contrib import cached
    z = cached(folder, nconf=nconf)
    C, Cp, w = z["C"], z["Cp"], z["w"]                 # (L,H), (L,H), (L,H) w は w/d_ref
    cfg = json.load(open(os.path.join(folder, "config.json")))
    L = _lattice(cfg)
    from periodicwave import CustomScreenformer as CS
    import jax.numpy as jnp
    dref = float(CS._max_periodic_distance(jnp.asarray(L)))
    o = cfg["network"]["CustomScreenformer"]
    NL = o["num_layers"]
    # 窓を固定した 1 層構成が前提。多層なら層をまとめる（寄与は層について加法）。
    if NL > 1 and not np.allclose(w[0], w[1], rtol=1e-3):
        print("  警告: 層ごとに窓が違う。層 0 の窓で描く。")
    wv = w[0] * dref
    qp = (Cp ** 2).sum(0); qp = 100 * qp / qp.sum()
    qt = (C ** 2).sum(0);  qt = 100 * qt / qt.sum()
    order = np.argsort(wv)
    pk = cfg["system"]["make_local_energy_kwargs"]["potential_kwargs"]
    rs = pk["interaction_energy_scale"]
    U = float(pk.get("moire_potential_strength", 0.0))

    # 実際の電子配置から最近接距離（周期ノルム、= ネットワークが窓に入れる距離）
    d_nn = d_min = np.nan
    ck = sorted(glob.glob(os.path.join(folder, "qmcjax_ckpt_*.npz")))
    try:
        dat = np.load(ck[-1], allow_pickle=True)["data"].item()
        P = np.asarray(dat["positions"])
        ne = sum(cfg["system"]["electrons"])
        P = P.reshape(-1, ne, 2)[:400]
        fr = (P[:, :, None, :] - P[:, None, :, :]) @ np.linalg.inv(L).T
        dp = _pnorm(fr, L)
        iu = np.triu_indices(ne, 1)
        d_min = float(np.quantile(dp[:, iu[0], iu[1]], 0.01))
        dd = dp.copy()
        for i in range(ne):
            dd[:, i, i] = 1e9
        d_nn = float(dd.min(axis=2).mean())
    except Exception as e:
        print("  (positions skipped: %s)" % e)

    E = np.nan
    try:
        t = np.genfromtxt(os.path.join(folder, "train_stats.csv"), delimiter=",",
                          names=True)
        E = float(np.median(np.asarray(t[t.dtype.names[1]])[-200:]))
    except Exception:
        pass
    return dict(w=wv[order], qp=qp[order], qt=qt[order], dref=dref,
                shell=bool(z.get("shell", False)),
                d_nn=d_nn, d_min=d_min, rs=rs, U=U, E=E, ne=sum(cfg["system"]["electrons"]),
                NH=len(wv), NL=NL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="label=folder をカンマ区切り（左から順に描く）")
    ap.add_argument("--out", default="fig_qwin.png")
    ap.add_argument("--nconf", type=int, default=48)
    ap.add_argument("--panel2", default="auto", choices=["auto", "log", "ratio", "cum"])
    ap.add_argument("--title", default="相が変わると、波動関数が使う長さが変わる")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ja = _font()
    T = lambda x, y: x if ja else y            # noqa: E731

    items = []
    for part in a.runs.split(","):
        lab, folder = part.rsplit("=", 1)   # ラベル側に = があってもよいよう後ろから割る
        print("=== %s  %s" % (lab, folder))
        items.append((lab, load(folder, a.nconf)))

    SHELL = bool(items[0][1].get("shell", False))
    LSYM = "c_h" if SHELL else "w_h"        # ヘッドを代表する長さの記号
    LHALF = "c_{1/2}" if SHELL else "w_{1/2}"
    LNAME = "殻の中心" if SHELL else "窓"
    LAVG = "\\langle c \\rangle_q" if SHELL else "\\langle w \\rangle_q"
    COL = ["#2a78d6", "#c98a12", "#c0392b", "#0f7b6c"]
    MRK = ["o", "s", "^", "D"]
    fig = plt.figure(figsize=(13.6, 6.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.05, 1.0, 1.02], wspace=0.36,
                          left=0.058, right=0.945, top=0.815, bottom=0.275)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # --- 左: 寄与率 vs 窓 -------------------------------------------------
    for i, (lab, d) in enumerate(items):
        ax.plot(d["w"], d["qp"], "-", marker=MRK[i], ms=6.0, lw=2.0,
                color=COL[i], label=lab, zorder=3)
    w0_ = items[0][1]["w"]
    inside = all(w0_[0] <= d["d_nn"] <= w0_[-1] for _, d in items if np.isfinite(d["d_nn"]))
    if inside:
        for i, (lab, d) in enumerate(items):
            if np.isfinite(d["d_nn"]):
                ax.axvline(d["d_nn"], color=COL[i], ls=(0, (5, 3)), lw=1.25, alpha=.75, zorder=1)
    # 窓が周期セル全体を覆ってしまう領域（w > d_ref）。ここの段は「距離で選ぶ」
    # 働きを失い、どれも全対を含むので、長さの読み出しには使えない。
    _dref = items[0][1]["dref"]
    _cover = w0_[-1] > _dref * 1.001
    if _cover:
        ax.axvspan(_dref, w0_[-1] * 1.05, color="#b0392b", alpha=.065, zorder=0)
        ax.axvline(_dref, color="#a5262f", lw=1.3, ls=(0, (4, 2.5)), alpha=.9, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(w0_[0] / 1.05, w0_[-1] * 1.05)
    w0 = items[0][1]["w"]
    ax.set_xticks(w0)
    _sk = 1 if len(w0) <= 10 else 2          # 目盛りが詰まるときは 1 つおきに数字を出す
    ax.set_xticklabels(["%.2f" % v if i % _sk == 0 else "" for i, v in enumerate(w0)],
                       fontsize=7.6, rotation=90)
    ax.minorticks_off()
    ax.set_xlabel(T("固定した%s $%s$（周期ノルム, $r_s a_B^{*}$）— %d 段の目盛り"
                    % (LNAME, LSYM, len(w0)), "fixed $%s$" % LSYM),
                  fontsize=10.6, labelpad=6)
    ax.set_ylabel(T("波動関数への寄与率  $q_h$ [%]", "contribution share [%]"), fontsize=10.6)
    ax.grid(alpha=.22, lw=.7)
    ax.legend(fontsize=10.0, frameon=False, loc="upper left")
    ax.set_title(T("① どの距離のヘッドが効いているか", "① contribution vs length"),
                 fontsize=11.4, pad=8)
    ytop = max(d["qp"].max() for _, d in items) * 1.20
    ax.set_ylim(0, ytop)
    if _cover:
        ax.text(_dref * 0.985, ytop * 0.035,
                T("$d_{\\rm ref}$ = %.2f " % _dref, "$d_{ref}$ = %.2f " % _dref),
                color="#a5262f", fontsize=8.8, va="bottom", ha="right",
                rotation=90, zorder=4,
                bbox=dict(fc="white", ec="none", alpha=.85, pad=1.0))
        ax.text(0.985, 0.035,
                T("この帯では窓がセル全体を覆う\n（全ての対が入り、長さを選べない）",
                  "window covers the whole cell here"),
                transform=ax.transAxes, color="#a5262f", fontsize=8.8,
                va="bottom", ha="right", linespacing=1.6, zorder=4,
                bbox=dict(fc="white", ec="none", alpha=.80, pad=2.0))
    if inside:
        for i, (lab, d) in enumerate(items):
            if np.isfinite(d["d_nn"]):
                ax.text(d["d_nn"], ytop * 0.035,
                        T("最近接 %.2f " % d["d_nn"], "NN %.2f " % d["d_nn"]),
                        color=COL[i], fontsize=8.4, va="bottom", ha="right",
                        rotation=90, zorder=4,
                        bbox=dict(fc="white", ec="none", alpha=.80, pad=1.0))
    else:
        ax.text(0.985, 0.965, T("実測の最近接距離（最小の%sより内）\n" % LNAME,
                                "NN distance (off scale)\n")
                + "  ".join("%s %.2f" % (lab, d["d_nn"]) for lab, d in items),
                transform=ax.transAxes, fontsize=8.6, color="#52514e",
                va="top", ha="right", linespacing=1.6)

    # --- 中: 目盛りが広いときは対数、狭いときは液体を基準にした比 ---------
    span = float(items[0][1]["w"][-1] / items[0][1]["w"][0])
    mode = a.panel2 if a.panel2 != "auto" else ("log" if span > 3.0 else "cum")
    if mode == "log":
        for i, (lab, d) in enumerate(items):
            ax2.plot(d["w"], np.maximum(d["qp"], 1e-7), "-", marker=MRK[i], ms=4.6,
                     lw=1.9, color=COL[i], label=lab)
        for i, (lab, d) in enumerate(items):
            if np.isfinite(d["d_nn"]):
                ax2.axvline(d["d_nn"], color=COL[i], ls=(0, (5, 3)), lw=1.15, alpha=.7, zorder=1)
        ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.minorticks_off()
        ax2.set_xticks([0.5, 1.0, 2.0, 3.0]); ax2.set_xticklabels(["0.5", "1.0", "2.0", "3.0"])
        ax2.set_ylabel(T("寄与率  $q_h$ [%]（対数）", "$q_h$ [%] (log)"), fontsize=10.4)
        ax2.set_title(T("② 同じものを対数で見る", "② same, log scale"), fontsize=11.4, pad=8)
    elif mode == "cum":
        for i, (lab, d) in enumerate(items):
            ax2.plot(d["w"], np.cumsum(d["qp"]), "-", marker=MRK[i], ms=4.6, lw=1.9,
                     color=COL[i], label=lab)
        ax2.axhline(50, color="#52514e", lw=.9, ls=":")
        for i, (lab, d) in enumerate(items):
            cs = np.cumsum(d["qp"])
            if cs[0] < 50 < cs[-1]:
                ax2.plot([float(np.interp(50.0, cs, d["w"]))], [50], marker="v", ms=8,
                         color=COL[i], zorder=5)
        if _cover:
            ax2.axvspan(_dref, items[0][1]["w"][-1] * 1.05, color="#b0392b",
                        alpha=.065, zorder=0)
            ax2.axvline(_dref, color="#a5262f", lw=1.2, ls=(0, (4, 2.5)), alpha=.9)
        ax2.set_xscale("log"); ax2.minorticks_off()
        ax2.set_xlim(items[0][1]["w"][0] / 1.05, items[0][1]["w"][-1] * 1.05)
        _tk = [t for t in (2.0, 2.5, 3.0, 4.0, 5.0) if
               items[0][1]["w"][0] <= t <= items[0][1]["w"][-1]]
        ax2.set_xticks(_tk); ax2.set_xticklabels(["%.1f" % t for t in _tk])
        ax2.set_ylim(0, 103)
        ax2.set_ylabel(T("累積寄与率 [%]", "cumulative [%]"), fontsize=10.4)
        ax2.set_title(T("② 累積（▽ が半分に達する%s $%s$）" % (LNAME, LHALF),
                        "② cumulative"),
                      fontsize=11.4, pad=8)
    else:
        base = items[0][1]["qp"]
        ok = base > 1e-6
        ax2.axhline(1.0, color="#52514e", lw=.9, ls=":")
        for i, (lab, d) in enumerate(items):
            ax2.plot(d["w"][ok], d["qp"][ok] / base[ok], "-", marker=MRK[i], ms=4.6,
                     lw=1.9, color=COL[i], label=lab)
        ax2.set_yscale("log"); ax2.minorticks_off()
        ax2.set_yticks([0.25, 0.5, 1.0, 2.0, 4.0])
        ax2.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
        ax2.set_ylabel(T("%s を 1 としたときの比" % items[0][0], "ratio to %s" % items[0][0]),
                       fontsize=10.4)
        ax2.set_title(T("② どの距離で入れ替わるか", "② where the ranking flips"),
                      fontsize=11.4, pad=8)
        # 比が 1 を跨ぐ窓 w*（最後の相 / 最初の相）
        last = items[-1][1]["qp"]
        m2 = ok & (last > 1e-6)
        if m2.sum() > 2:
            lr = np.log(last[m2] / base[m2]); ww = items[0][1]["w"][m2]
            if lr.min() < 0 < lr.max():
                wstar = float(np.interp(0.0, lr, ww))
                ax2.axvline(wstar, color="#a5262f", lw=1.2, ls=(0, (4, 3)), alpha=.85)
                ax2.text(wstar, ax2.get_ylim()[1], T(" $w^{*}$ = %.2f" % wstar,
                                                     " w*=%.2f" % wstar),
                         color="#a5262f", fontsize=9.4, va="top", ha="left")
    ax2.set_xlabel(T("%s $%s$" % (LNAME, LSYM), "$%s$" % LSYM), fontsize=10.4)
    ax2.grid(alpha=.22, lw=.7, which="both")
    ax2.legend(fontsize=9.0, frameon=False, loc="best")

    # --- 右: 1 個の読み出し値 --------------------------------------------
    xs = np.arange(len(items))
    wq, whalf = [], []
    for lab, d in items:
        wq.append(float((d["qp"] * d["w"]).sum() / d["qp"].sum()))
        cs = np.cumsum(d["qp"])
        whalf.append(float(np.interp(50.0, cs, d["w"])))
    ax3.plot(xs, wq, "-o", color="#12406f", lw=2.2, ms=8,
             label=T("$%s=\\sum_h q_h %s$" % (LAVG, LSYM), "$%s$" % LAVG))
    ax3.plot(xs, whalf, "--s", color="#eb6834", lw=1.8, ms=7,
             label=T("$%s$（累積 50%%）" % LHALF, "$%s$" % LHALF))
    for i, (lab, d) in enumerate(items):
        if np.isfinite(d["d_nn"]):
            ax3.plot([i], [d["d_nn"]], marker="_", ms=22, mew=2.4, color="#52514e",
                     label=T("最近接距離", "NN distance") if i == 0 else None)
    ax3.set_xticks(xs)
    _byU = len({d["rs"] for _, d in items}) == 1 and len({d["U"] for _, d in items}) > 1
    _key = (lambda d: "$U$=%g" % d["U"]) if _byU else (lambda d: "$r_s$=%g" % d["rs"])
    ax3.set_xticklabels(["%s\n%s" % (lab.split("$r_s$")[0].strip(), _key(d))
                         for lab, d in items], fontsize=9.6)
    ax3.set_ylabel(T("長さ（$r_s a_B^{*}$）", "length"), fontsize=10.4)
    ax3.grid(alpha=.22, lw=.7)
    ax3.legend(fontsize=8.8, frameon=True, framealpha=.88, edgecolor="none",
               loc="upper left", borderpad=.35)
    _yl = ax3.get_ylim(); ax3.set_ylim(_yl[0] - 0.05 * (_yl[1] - _yl[0]),
                                       _yl[1] + 0.34 * (_yl[1] - _yl[0]))
    ax3.set_title(T("③ 長さと、使うヘッドの数", "③ length and spread"),
                  fontsize=11.4, pad=8)
    ax3b = ax3.twinx()
    part = [float(1.0 / ((d["qp"] / 100.0) ** 2).sum()) for _, d in items]
    ax3b.plot(xs, part, ":D", color="#0f7b6c", lw=1.8, ms=6.5, mfc="white")
    ax3b.set_ylabel(T("使うヘッドの数 $1/\\sum_h q_h^2$", "participation ratio"),
                    fontsize=9.6, color="#0f7b6c")
    ax3b.tick_params(axis="y", colors="#0f7b6c", labelsize=9)
    _y3 = ax3b.get_ylim()
    ax3b.set_ylim(_y3[0] - 0.12 * (_y3[1] - _y3[0]), _y3[1] + 0.55 * (_y3[1] - _y3[0]))
    for i, v in enumerate(part):
        ax3b.text(xs[i], v, "  %.1f" % v, color="#0f7b6c", fontsize=8.8, va="bottom")

    _bU = len({d["rs"] for _, d in items}) == 1 and len({d["U"] for _, d in items}) > 1
    d0 = items[0][1]
    fig.text(0.006, 0.985,
             T("遮蔽アテンション（%sを固定した目盛り）" % LNAME, "screening attention, frozen windows"),
             fontsize=13.2, weight="bold", color="#0f7b6c", va="top")
    fig.text(0.006, 0.930, a.title + "　—　" +
             T(("モアレ三角ポテンシャル, $r_s$=%g, " % d0["rs"] if _bU else "クーロンのみ, ")
               + "$N=%d$, %d 層 × %d ヘッド, %s は学習させず固定"
               % (d0["ne"], d0["NL"], d0["NH"], "帯域窓（殻）" if SHELL else "窓"),
               "Coulomb only, N=%d, %dx%d, windows frozen" % (d0["ne"], d0["NL"], d0["NH"])),
             fontsize=10.6, color="#0b0b0b", va="top")

    lines = ([
        ("%s $%s$ は学習させず %d 段の目盛りとして固定した。3 つの相で横軸が完全に共通なので、"
         "曲線の違いはそのまま波動関数の違いになる。" % (LNAME, LSYM, d0["NH"])),
        ("　　 殻は隣どうしが重ならず $\\sum_h m_h = 1$ なので、$q_h$ はそのまま"
         "「距離 $c_h$ にどれだけ相関があるか」— 動径分解として読める。" if SHELL else None),
        (("①  液体では近い殻のヘッドにも仕事があるが、結晶では電子が離れて座るため、"
          "近い殻に相手が 1 個も入らない。寄与は遠い殻へ移る。" if SHELL else
          "①  液体では窓の狭いヘッドにも仕事があるが、結晶では電子が離れて座るため狭い窓に相手が入らず、寄与が広い窓へ移る。")
         + ("破線は各相で実測した最近接距離。" if inside else "ヘッドごとのばらつきがあるので、滑らかな量は ② の累積で見る。")
         + ("\n　　 赤い帯は $%s>d_{\\rm ref}$ の範囲。" % LSYM + "$d_{\\rm ref}$ は周期ノルムがセル内で取り得る最大値なので、"
            "そこを超えた窓はどれも全ての対を含み、長さの区別ができない。" if _cover else "")),
        ("②  ▽ は累積が半分に達する%s $%s$。相が硬くなるほど右へ動く。" % (LNAME, LHALF)
         if mode == "cum" else
         "②  下半分のヘッドは線形では 0 に見えるが、対数で見ると液体と結晶で 2〜3 桁ちがう。窓が最近接距離より狭いヘッドは、結晶では相手が 1 個も入らない。" if mode == "log" else "②  液体を 1 としたときの比。1 を上下に跨ぐ窓が、相転移でヘッドの順位が入れ替わる位置。"),
        ("③  $%s = \\sum_h q_h %s$ と $%s$（左軸, 灰色の横棒は実測の最近接距離）、"
         "および $1/\\sum_h q_h^2$（右軸）＝ 相関を何本のヘッドに分けて持っているか。"
         % (LAVG, LSYM, LHALF)),
    ] if ja else [
        "Windows are frozen as a 16-step ruler, so the three phases share one x axis.",
    ])
    lines = [x for x in lines if x]
    # 説明文が 4 行を超えたら、その行数ぶんだけ図の下端を上げて場所を作る
    _nl = sum(1 + x.count("\n") for x in lines)
    _sh = 0.0395 * max(_nl - 4, 0)
    if _sh > 0:
        gs.update(bottom=0.275 + _sh)
    fig.text(0.008, 0.185 + _sh, "\n".join(lines),
             fontsize=9.5, color="#2b2b2b", va="top", linespacing=2.05)
    fig.savefig(a.out, dpi=155)
    print("  wrote %s" % a.out)
    for (lab, d), a1, a2 in zip(items, wq, whalf):
        print("  %-9s rs=%-4g U=%-5g  <w>_q=%.3f  w_1/2=%.3f  d_nn=%.3f  E=%.4f"
              % (lab, d["rs"], d["U"], a1, a2, d["d_nn"], d["E"]))
    print("  参加ヘッド数 1/Σq²: " + "  ".join(
        "%s %.2f" % (lab, 1.0 / ((d["qp"] / 100.0) ** 2).sum()) for lab, d in items))
    if len(items) > 1:
        p0 = items[0][1]["qp"] / 100.0
        for lab, d in items[1:]:
            print("  配分の移動（全変動距離）%s → %s : %.3f"
                  % (items[0][0], lab, 0.5 * np.abs(d["qp"] / 100.0 - p0).sum()))
        print("  参考: 分布まるごと 1 ヘッドぶんずらすと %.3f"
              % (0.5 * np.abs(np.roll(p0, 1) - p0).sum()))
    # 目盛りが d_ref を超えている場合、その段は全対を含むので長さを選んでいない。
    _dr = items[0][1]["dref"]
    if items[0][1]["w"][-1] > _dr * 1.001:
        print("  セル全体を覆う段（w > d_ref = %.4f）: %d / %d 段"
              % (_dr, int((items[0][1]["w"] > _dr).sum()), items[0][1]["NH"]))
        for lab, d in items:
            ov = d["w"] > _dr
            print("    %-9s その段が持つ寄与 %5.1f%%  （残り %5.1f%% だけが長さを分解している）"
                  % (lab, d["qp"][ov].sum(), d["qp"][~ov].sum()))


if __name__ == "__main__":
    main()
