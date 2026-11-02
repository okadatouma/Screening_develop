# -*- coding: utf-8 -*-
"""結果の解釈に使う 2 つの対照図。

    python screening_analysis/fig_interp.py --runs 液体=DIR,境界=DIR,結晶=DIR --out F.png

左：秩序変数を、完全三角格子にガウス変位を入れた模型で較正する。
右：ヘッド別の寄与率を、幾何だけで決まる帰無モデルと比べる。
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import style as ST                                                   # noqa

AEE = np.sqrt(2 * np.pi / np.sqrt(3))          # 1.904626（r_s a_B* 単位）


def _lat(n=5):
    a1 = AEE * np.array([1.0, 0.0]); a2 = AEE * np.array([-0.5, np.sqrt(3) / 2])
    L = np.column_stack([n * a1, n * a2])      # 列 = 格子ベクトル
    ij = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 2)
    return ij @ np.stack([a1, a2]), L


def order(R, L):
    """psi6、最近接距離の平均、そのゆらぎ。完全三角格子で (1, a_ee, 0) になる。"""
    dr = R[:, None, :] - R[None, :, :]
    f = dr @ np.linalg.inv(L).T
    dv = (f - np.round(f)) @ L.T
    dd = np.linalg.norm(dv, axis=-1); np.fill_diagonal(dd, 1e9)
    p6 = []
    for i in range(len(R)):
        k = np.argsort(dd[i])[:6]
        th = np.arctan2(dv[i, k, 1], dv[i, k, 0])
        p6.append(abs(np.mean(np.exp(6j * th))))
    nn = dd.min(axis=1)
    return float(np.mean(p6)), float(nn.mean()), float(nn.std() / nn.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="fig_interp.png")
    ap.add_argument("--nconf", type=int, default=64)
    ap.add_argument("--nsamp", type=int, default=384)
    a = ap.parse_args()
    ST.use()
    import matplotlib.pyplot as plt
    from attn_probe import probe
    import head_contrib as HC
    import moire_analysis as MA, sampler as SM

    items = []
    for part in a.runs.split(","):
        lab, f = part.rsplit("=", 1); f = f.rstrip("/")
        pr = probe(f, nconf=a.nconf)
        C, Cd, Cp, X, Rn, w, sm, mt = HC.compute(f, nconf=a.nconf)
        NE = pr["NE"]; off = ~np.eye(NE, dtype=bool)
        d = pr["d"][:, off].ravel(); wh = pr["w"][0]
        P = np.array([np.mean(0.5 * (1 + np.cos(np.pi * np.clip(d / v, 0, 1)))) for v in wh])
        q = (Cp[0] ** 2); q = 100 * q / q.sum()
        dd = MA.load(f); dd = dict(dd)
        dd["R"] = np.asarray(SM.cached(f, a.nsamp)).reshape(-1, int(dd["ne"]), 2)
        Lr = np.asarray(dd["L"])
        p6, dnn, fl = np.mean([order(R, Lr) for R in dd["R"]], axis=0)
        items.append(dict(lab=lab, w=wh, q=q, qn=100 * P / P.sum(), p6=p6, dnn=dnn,
                          fl=fl, rs=float(dd["rs"])))

    # --- ガウス変位模型の較正曲線 ------------------------------------------
    # 完全三角格子に独立なガウス変位 u を入れ、3 つの量を u の関数として作る。
    # 実測値をこの曲線で逆に解くと、量ごとに u の推定値が出る。3 つが揃えば
    # 「格子 + 変位」という描像がその相で成り立っている、という確認になる。
    R0, L0 = _lat()
    us = np.linspace(0.0, 0.36, 37)
    rng = np.random.default_rng(0)
    cal = np.array([np.mean([order(R0 + rng.normal(0, u * AEE, R0.shape), L0)
                             for _ in range(160)], axis=0) for u in us])   # (U,3)

    def invert(col, val, dec):
        """較正曲線を逆に解いて u/a_ee を返す。dec=True なら単調減少の量。"""
        y = cal[:, col]
        if dec:
            return float(np.interp(-val, -y, us))
        return float(np.interp(val, y, us))

    QOBS = [("$\\psi_6$", 0, True), ("$\\langle d_{nn}\\rangle$", 1, True),
            ("$\\sigma_{nn}/\\langle d_{nn}\\rangle$", 2, False)]
    for it in items:
        it["u"] = [invert(c, it[["p6", "dnn", "fl"][c]], dec) for _, c, dec in QOBS]

    fig = plt.figure(figsize=(13.0, 5.9))
    fig.text(0.035, 0.962, "結果の解釈に必要な 2 つの対照",
             fontsize=15.0, weight="bold", color=ST.GREEN, va="top")
    fig.text(0.035, 0.915,
             "左：相の乱れを 1 つの変位 $u$ で測る　　右：ヘッドの配分を幾何だけの帰無モデルと比べる",
             fontsize=10.8, color=ST.SUB, va="top")
    COL = [ST.BLUE, "#c98a12", "#c0392b"]; MRK = ["o", "s", "^"]

    # --- 左 -----------------------------------------------------------------
    ax = fig.add_axes([0.055, 0.335, 0.365, 0.480])
    OM = ["o", "D", "s"]
    for k, (nm, _, _) in enumerate(QOBS):
        ax.plot([i + (k - 1) * 0.17 for i in range(len(items))],
                [it["u"][k] for it in items], OM[k], ms=9, mfc="white", mew=1.7,
                color="#12406f", ls="none", label="%s から" % nm)
    for i, it in enumerate(items):
        u = it["u"]
        # psi6 は乱れが大きいところで頭打ちになるので、平均でなく中央値を採る
        um = float(np.median(u))
        ax.plot([i - 0.24, i + 0.24], [um] * 2, "-", color=COL[i], lw=2.4,
                alpha=.55, zorder=0)
        ax.text(i + 0.27, um, "%.2f" % um, ha="left", va="center", fontsize=10.6,
                color=COL[i], weight="bold")
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([it["lab"] for it in items], fontsize=10.2)
    ax.set_xlim(-0.5, len(items) - 0.5)
    ax.set_ylabel("格子からの変位  $u/a_{ee}$")
    ax.set_ylim(0, max(max(it["u"]) for it in items) * 1.20)
    ax.legend(fontsize=9.0, loc="lower center", ncol=3, columnspacing=1.0,
              handletextpad=0.35, borderaxespad=0.5)
    ax.grid(alpha=.22, lw=.7, axis="y")
    ax.set_title("① 乱れの大きさ（3 つの量から別々に読む）", pad=8)

    # --- 右 -----------------------------------------------------------------
    ax2 = fig.add_axes([0.545, 0.335, 0.415, 0.480])
    for i, it in enumerate(items):
        ax2.plot(it["w"], it["q"], "-", marker=MRK[i], ms=5.6, lw=1.9, color=COL[i],
                 label=it["lab"], zorder=3)
        ax2.plot(it["w"], it["qn"], ":", lw=1.5, color=COL[i], alpha=.85, zorder=2)
    ax2.set_xlabel("固定した窓 $w_h$  （周期ノルム, $r_s a_B^*$）", labelpad=3)
    ax2.set_ylabel("2 体の寄与率  $q_h$ [%]")
    ax2.grid(alpha=.22, lw=.7)
    ax2.legend(fontsize=9.4, loc="upper left")
    ax2.set_title("② 実測（実線）と帰無モデル（点線）", pad=8)

    ST.caption(
        fig, "読み方",
        "①  完全三角格子に独立なガウス変位 $u$ を入れた模型で、3 つの量を較正した"
        "（$u=0$ で $\\psi_6 = 1$, $\\langle d_{nn}\\rangle = a_{ee}$, ゆらぎ $=0$ を確認済み）。\n"
        "     実測値を較正曲線で逆に解くと、量ごとに $u/a_{ee}$ が出る。3 つが揃えば「格子 + 変位」で"
        "その相を記述してよい。横棒と数字は中央値\n"
        "     （$\\psi_6$ は乱れが大きいと頭打ちになるので、液体では上振れする）。\n"
        "②  点線は「そのヘッドが何対を見ているか」だけで決まる配分 $q_h^{\\rm null} \\propto \\langle m_h(d)\\rangle_{\\rm pairs}$。"
        "対の分布は相とともに変わるので、点線も動く。\n"
        "     実線と点線の差が、幾何では説明できない — 波動関数が長さを選び直したぶん。",
        y=0.030, size=9.3)

    fig.savefig(a.out, dpi=170, facecolor=fig.get_facecolor())
    print("  wrote", a.out)
    for it in items:
        print("  %-12s psi6=%.4f dnn=%.4f ゆらぎ=%.4f u/a=%s  <w>_q=%.4f  <w>_null=%.4f"
              % (it["lab"], it["p6"], it["dnn"], it["fl"],
                 " ".join("%.3f" % v for v in it["u"]),
                 (it["q"] * it["w"]).sum() / 100, (it["qn"] * it["w"]).sum() / 100))


if __name__ == "__main__":
    main()
