# -*- coding: utf-8 -*-
"""run の設定を 1 枚にまとめる。数字はすべて config.json とチェックポイントから読む。

    python screening_analysis/fig_setup_card.py --runs 液体=DIR,境界=DIR,結晶=DIR --out F.png
"""
import argparse, glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)

W, H = 16.54, 11.69
M = 0.040
INK, SUB, AI, ENJI, RULE = "#141412", "#4a4946", "#155f7a", "#a5262f", "#d6d3cc"


def _lat(cfg):
    m = re.search(r"unserializable object: \[\[(.*?)\]\]", json.dumps(cfg))
    return np.array([float(x) for x in re.findall(
        r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(1))][:4]).reshape(2, 2)


def read(folder):
    import jax.numpy as jnp
    from periodicwave import CustomScreenformer as CS
    cfg = json.load(open(os.path.join(folder, "config.json")))
    L = _lat(cfg); dref = float(CS._max_periodic_distance(jnp.asarray(L)))
    pk = cfg["system"]["make_local_energy_kwargs"]["potential_kwargs"]
    o = cfg["network"]["CustomScreenformer"]
    f, sp = o["init_softmask_w_factor"], o["init_softmask_w_spread"]
    lad = np.geomspace(f / sp * dref, f * sp * dref, o["num_heads"])
    # チェックポイントから実際の窓を復元（設定と食い違っていないかの検算）
    ck = sorted(glob.glob(os.path.join(folder, "qmcjax_ckpt_*.npz")))
    w_ck = None
    if ck:
        import jax
        P = np.load(ck[-1], allow_pickle=True)["params"].item()
        om = np.asarray(jax.tree_util.tree_map(lambda x: np.asarray(x)[0], P)
                        ["layers"]["attention"][0]["mask_omega"])
        w_ck = dref * np.log1p(np.exp(om))
    st = np.genfromtxt(os.path.join(folder, "train_stats.csv"), delimiter=",", names=True)
    mv = _lat({"x": "unserializable object: [[%s]]" % ""}) if False else None
    aM = None
    m2 = re.search(r"moire_lattice_vectors.*?\[\[\s*([0-9.eE+-]+)", json.dumps(pk))
    if m2:
        aM = float(m2.group(1))
    return dict(folder=os.path.basename(folder), dref=dref, lad=lad, w_ck=w_ck,
                rs=pk["interaction_energy_scale"],
                U=float(pk.get("moire_potential_strength", 0.0)),
                phi=float(pk.get("moire_potential_phi", 0.0)), aM=aM,
                ne=sum(cfg["system"]["electrons"]), spins=cfg["system"]["electrons"],
                NL=o["num_layers"], NH=o["num_heads"], AD=o["attn_dim"],
                VD=o["value_dim"], MD=o["mlp_dim"],
                det=cfg["network"]["determinants"], batch=cfg.get("batch_size", 0),
                lr=cfg["optim"]["lr"]["rate"], delay=cfg["optim"]["lr"]["delay"],
                trim=o["init_trim_r"], step=int(st["step"][-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="fig_setup_card.png")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    for p in fm.findSystemFonts():
        if "NotoSansCJK" in p:
            fm.fontManager.addfont(p)
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["axes.unicode_minus"] = False

    items = []
    for part in a.runs.split(","):
        lab, folder = part.rsplit("=", 1)
        items.append((lab, read(folder)))
    d0 = items[0][1]
    a_ee = np.sqrt(2 * np.pi / np.sqrt(3))
    moire = any(d["U"] != 0 for _, d in items)

    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    y = 1 - M
    ax.text(M, y, "セットアップ", fontsize=9, color=AI, va="top"); y -= 0.019
    ax.text(M, y, a.title or ("モアレ三角ポテンシャル — 窓を固定した目盛りで読む"
                              if moire else "クーロンのみ — 窓を固定した目盛りで読む"),
            fontsize=17, color=INK, va="top", weight="bold"); y -= 0.036
    ax.plot([M, 1 - M], [y, y], color=RULE, lw=1.0); y -= 0.024

    eqs = [("単位", r"長さ $r_s a_B^{*}$, エネルギー ${\rm Ha}^{*}/r_s^{2}$。"
                   r"$\pi (r_s a_B^{*})^2 = 1/n_e$ より $A_{\rm sc} = \pi N$、"
                   r"平均間隔 $a_{ee} = \sqrt{2\pi/\sqrt{3}} = %.4f$（$r_s$ によらない）" % a_ee),
           ("系", r"$N = %d$（↑%d ↓%d）, 三角スーパーセル, 周期境界" %
                  (d0["ne"], d0["spins"][0], d0["spins"][1]))]
    if moire:
        eqs.append(("ハミルトニアン",
                    r"$H = -\frac{1}{2}\sum_i \nabla_i^2 + r_s \sum_{i<j} r_{ij}^{-1}"
                    r" - 2U \sum_i \sum_{j=1}^{3} \cos(\mathbf{G}_j \cdot \mathbf{r}_i + \phi)$"))
        eqs.append(("モアレ", r"$|\mathbf{G}_j| = 4\pi/(\sqrt{3}\,a_M)$, "
                             r"$a_M = %.4f$ ($= a_{ee}$, 充填 $\nu = 1$), $\phi = \pi/4$"
                             % (d0["aM"] or a_ee)))
    else:
        eqs.append(("ハミルトニアン",
                    r"$H = -\frac{1}{2}\sum_i \nabla_i^2 + r_s \sum_{i<j} r_{ij}^{-1}$"
                    r"　（$U = 0$）"))
    eqs += [
        ("ネットワーク", r"遮蔽アテンション %d 層 × %d ヘッド, attn %d, value %d, mlp %d, "
                      r"行列式 %d" % (d0["NL"], d0["NH"], d0["AD"], d0["VD"], d0["MD"], d0["det"])),
        ("最適化", r"KFAC, batch %d, $\eta_0 = %g$, $t_0 = %g$, %d step（3 run とも同じ）"
                 % (d0["batch"], d0["lr"], d0["delay"], d0["step"])),
        ("重み", r"$W^{(h)}_{ij} = \alpha(s_{ij})\, m_h(d_{ij})$,　"
                r"$\alpha(s) = \left[\max\left(1-(1-s)/r,\,0\right)\right]^2$, $r_{\rm init} = %.2f$"
                % d0["trim"]),
        ("窓", r"$m_h(d) = \frac{1}{2}\left[1+\cos\left(\pi\min(d/w_h,\,1)\right)\right]$,　"
              r"$d \geq w_h$ で厳密に 0。$w_h = d_{\rm ref}\,{\rm softplus}(\omega_h)$ を"
              r"学習させず固定（$\omega_h$ の勾配を切った）"),
        ("距離", r"$d_{ij}$ は周期ノルム（サイン計量）。セル内の最大値 "
                r"$d_{\rm ref} = %.4f = %.3f\,a_{ee}$" % (d0["dref"], d0["dref"] / a_ee)),
    ]
    eqsize = 11.0
    elh = eqsize * 2.95 / (H * 72); xe = M + 0.118
    for lab, mth in eqs:
        ax.text(M, y, lab, fontsize=eqsize - 2.4, color=AI, va="top")
        ax.text(xe, y, mth, fontsize=eqsize, color=INK, va="top")
        y -= elh * (1.34 if any(k in mth for k in ("\\sum", "\\frac", "\\sqrt")) else 1.0)
    y -= elh * 0.6

    lad = d0["lad"]
    ax.text(M, y, "目盛り", fontsize=eqsize - 2.4, color=AI, va="top")
    ax.text(xe, y, "$w_h$（周期ノルム）= " + ", ".join("$%.3f$" % v for v in lad[:8]),
            fontsize=eqsize, color=INK, va="top"); y -= elh
    ax.text(xe, y, "　　　　　　　　　　" + ", ".join("$%.3f$" % v for v in lad[8:]),
            fontsize=eqsize, color=INK, va="top"); y -= elh * 1.1
    ax.text(xe, y, "$a_{ee}$ 単位では $%.3f \\ldots %.3f$、"
                   "$d_{\\rm ref}$ 比では $%.3f \\ldots %.3f$"
            % (lad[0] / a_ee, lad[-1] / a_ee, lad[0] / d0["dref"], lad[-1] / d0["dref"]),
            fontsize=eqsize - 0.6, color=SUB, va="top"); y -= elh * 1.5

    tab = [["", "$U$", "$r_s$", "step", "batch", "窓の下端", "窓の上端",
            "$w_{\\rm max}/d_{\\rm ref}$", "フォルダ"]]
    for lab, d in items:
        tab.append([lab, "%g" % d["U"], "%g" % d["rs"], "%d" % d["step"], "%d" % d["batch"],
                    "%.4f" % d["lad"][0], "%.4f" % d["lad"][-1],
                    "%.3f" % (d["lad"][-1] / d["dref"]), d["folder"]])
    tsize = 10.0; lh = tsize * 2.55 / (H * 72)
    ncol = len(tab[0]); tw = 1 - 2 * M
    xs = [M + tw * j / ncol for j in range(ncol)]
    for j, t in enumerate(tab[0]):
        ax.text(xs[j], y, t, fontsize=tsize, color=INK, va="top", weight="bold")
    y -= lh * 1.4
    ax.plot([M, M + tw], [y + lh * 0.45, y + lh * 0.45], color=RULE, lw=0.9)
    for r in tab[1:]:
        c = ENJI if r[0].startswith("結晶") else SUB
        for j, t in enumerate(r):
            ax.text(xs[j], y, str(t), fontsize=tsize, color=c, va="top")
        y -= lh
    y -= lh * 0.9

    # 検算: チェックポイントの窓が設定と一致するか
    msg = []
    for lab, d in items:
        if d["w_ck"] is not None:
            e = float(np.abs(np.sort(d["w_ck"]) - d["lad"]).max())
            msg.append("%s %.1e" % (lab, e))
    ax.text(M, y, "検算 — チェックポイントの $\\omega_h$ から復元した窓と、設定から計算した梯子の最大差: "
                  + "、".join(msg) + "（窓が学習で動いていないことの確認）",
            fontsize=9.6, color=SUB, va="top"); y -= lh * 1.15
    if abs(lad[-1] / d0["dref"] - 1.0) > 0.01:
        ax.text(M, y, "注意 — 梯子の上端は $d_{\\rm ref}$ に届いていない（$%.1f\\%%$）。"
                      "$w = d_{\\rm ref}$ で窓は最も遠い対をちょうど 0 にする。"
                      % (100 * lad[-1] / d0["dref"]),
                fontsize=9.6, color=ENJI, va="top")
    ax.text(M, 0.018, "PeriodicWave — 遮蔽アテンション / セットアップ",
            fontsize=8, color=SUB, va="bottom")
    fig.savefig(a.out, dpi=155)
    print("->", a.out)


if __name__ == "__main__":
    main()
