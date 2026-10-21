"""結合強度 |V|/T を軸にして、遮蔽注意の内部量がどう変わるかを見る。

    # まず r_s を振って学習（GPU なら iterations を増やすこと）
    python screening_analysis/coupling_scan.py --train 1 2 5 10 20 25 30 35 40 --iters 400
    # 学習済みフォルダを解析して図にする
    python screening_analysis/coupling_scan.py --analyze results/2deg-Coulomb/CustomScreenformer/el5_4_rs*_tri_ctl

なぜ |V|/T を軸にするか。r_s は「入力パラメータ」だが、|V|/T は波動関数を通した
実際の結合強度で、無次元。古典 2D プラズマの Gamma に対応する量なので、
注意機構の内部量（遮蔽長 w、スコア s の分散）がこれに応答するなら、
それは「注意が相関の強さを読んでいる」ことの直接の証拠になる。

単位の注意。configs/run_2deg_screen*.py は長さを r_s a_B にとるので、
supercell の大きさは r_s に依らず一定（面積 = pi*N）。d_ref も一定。
つまり幾何は固定で、変わるのは結合定数だけ。w/d_ref を r_s 間で直接比べてよい。

V は背景電荷を引いた後の値（Hartree 項は相殺済み）なので、|V|/T は
「交換相関 + Madelung」対「運動」の比。素の Hartree 比ではない。

Wigner 転移は 2DEG で r_s = 31(1)（Drummond & Needs, PRL 102, 126402 (2009)）。
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import jax

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

RS_WIGNER = 31.0


# --------------------------------------------------------------- 学習
def train(rs_list, iters, nspins="5_4", shape="tri", config=None, nice=10):
    """r_s を振って同一設定で学習する。既に結果があるものは飛ばす。"""
    cfg = config or os.path.join(REPO, "periodicwave", "configs", "run_2deg_screen_ctl.py")
    if not os.path.exists(cfg):
        cfg = os.path.join(REPO, "periodicwave", "configs", "run_2deg_screen.py")
    up, dn = nspins.split("_")
    env = dict(os.environ, PYTHONPATH=REPO)
    for rs in rs_list:
        tag = "ctl" if cfg.endswith("_ctl.py") else "scan"
        fold = os.path.join(REPO, "results", "2deg-Coulomb", "CustomScreenformer",
                            f"el{up}_{dn}_rs{_fmt(rs)}_{shape}_{tag}")
        if glob.glob(os.path.join(fold, "qmcjax_ckpt_*.npz")):
            print(f"  skip r_s = {rs} (already done)")
            continue
        print(f"  train r_s = {rs} ...", flush=True)
        cmd = [sys.executable, "-u", cfg, nspins, str(rs), "CustomScreenformer", shape, str(iters)]
        if nice and hasattr(os, "nice"):
            cmd = ["nice", "-n", str(nice)] + cmd
        with open(os.path.join(REPO, f"scan_rs{_fmt(rs)}.log"), "w") as fh:
            subprocess.run(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False)


def _fmt(rs):
    """configs 側の folder 名規則に合わせる（整数なら小数点を落とす）。"""
    return str(int(rs)) if float(rs) == int(rs) else str(rs)


# --------------------------------------------------------------- 解析
def analyze(folders, blocks=25, mcmc_steps=10, cache=True):
    """各フォルダの結果を <folder>/coupling_row.npy にキャッシュする。

    JAX は 1 プロセス内でフォルダごとに新しいカーネルを積み上げるので、
    9 個も回すとメモリを使い切って 'Cannot allocate memory' で落ちる。
    キャッシュがあれば再実行で飛ばせるので、シェル側で 1 フォルダ 1 プロセスに
    分けて回すのが確実（scan_all.sh 相当）。jax.clear_caches() も一応入れてある。
    """
    from tv_split import build, measure, _err            # noqa: E402
    from attn_probe import probe                         # noqa: E402

    rows = []
    for f in sorted(folders, key=_rs_of):
        cf = os.path.join(f, "coupling_row.npy")
        if cache and os.path.exists(cf):
            rows.append(np.load(cf, allow_pickle=True).item())
            print("  cached %-24s |V|/T=%6.2f" % (rows[-1]["name"], rows[-1]["ratio"]), flush=True)
            continue
        b = build(f)
        T, V = measure(b, blocks=blocks, mcmc_steps=mcmc_steps)
        a = probe(f)
        NE, off = a["NE"], ~np.eye(a["NE"], dtype=bool)
        S = a["S"]                                        # (B,L,H,N,N)
        rows.append(dict(
            name=os.path.basename(f), r_s=b["r_s"], ne=b["ne"],
            T=T.mean(), V=V.mean(), eT=_err(T), eV=_err(V),
            ratio=abs(V.mean()) / T.mean(),
            w=a["w"] / a["dref"],                          # (L,H) 遮蔽長 / 箱
            trim=a["r"],                                   # (L,H)
            s_sd=S[:, :, :, off].std(axis=(0, 3)),         # (L,H) スコアの散らばり
            s_mean=S[:, :, :, off].mean(axis=(0, 3)),
            d_mean=float(a["d"][:, off].mean()),
            d_min=float(np.percentile(a["d"][:, off], 1)),
        ))
        print("  %-24s r_s=%-5g |V|/T=%6.2f  w/d_ref(H0)=%.3f  sd(s)=%.3f"
              % (rows[-1]["name"], rows[-1]["r_s"], rows[-1]["ratio"],
                 rows[-1]["w"].mean(0)[0], rows[-1]["s_sd"].mean()), flush=True)
        if cache:
            np.save(cf, rows[-1], allow_pickle=True)
        jax.clear_caches()
    return rows


def _rs_of(f):
    import re
    m = re.search(r"_rs([0-9.]+)_", os.path.basename(f))
    return float(m.group(1)) if m else 0.0


# --------------------------------------------------------------- 図
def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    cjk = False
    for p in fm.findSystemFonts():
        if "NotoSansCJK" in p or "NotoSerifCJK" in p:
            fm.fontManager.addfont(p); cjk = True
    if cjk:
        plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    L = (lambda ja, en: ja if cjk else en)

    rs = np.array([r["r_s"] for r in rows])
    ra = np.array([r["ratio"] for r in rows])
    T = np.array([r["T"] for r in rows])
    V = np.array([r["V"] for r in rows])
    W = np.stack([r["w"].mean(0) for r in rows])          # (R,H)
    SD = np.stack([r["s_sd"].mean(0) for r in rows])      # (R,H)
    NH = W.shape[1]

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.4))
    fig.patch.set_facecolor("#fcfcfb")

    a = ax[0, 0]
    a.loglog(rs, ra, "o-", color="#eb6834", lw=2, ms=7)
    a.loglog(rs, rs, "--", color="#9a9a95", lw=1, label=r"$|V|/T = r_s$")
    if len(rs) > 2:
        p = np.polyfit(np.log(rs), np.log(ra), 1)
        a.loglog(rs, np.exp(np.polyval(p, np.log(rs))), ":", color="#2a78d6", lw=1.6,
                 label=r"fit: $r_s^{%.2f}$" % p[0])
    a.axvline(RS_WIGNER, color="#008300", lw=1, ls="-.")
    a.text(RS_WIGNER, a.get_ylim()[0] * 1.4, " Wigner", color="#008300", fontsize=9)
    a.set_xlabel(r"$r_s$"); a.set_ylabel(r"$|V|/T$")
    a.set_title(L("結合強度は r_s に比例しない", "coupling is sublinear in $r_s$"), fontsize=11)
    a.legend(fontsize=9); a.grid(alpha=.25, which="both")

    a = ax[0, 1]
    a.loglog(rs, T, "o-", label=r"$\langle T\rangle$", color="#2a78d6")
    a.loglog(rs, -V, "s-", label=r"$|\langle V\rangle|$", color="#eb6834")
    a.axvline(RS_WIGNER, color="#008300", lw=1, ls="-.")
    a.set_xlabel(r"$r_s$"); a.set_ylabel(L("エネルギー（コード単位）", "energy (code units)"))
    a.set_title(L("T も r_s とともに増える（局在の代償）",
                  "$T$ grows too (cost of localisation)"), fontsize=11)
    a.legend(fontsize=9); a.grid(alpha=.25, which="both")

    a = ax[1, 0]
    for h in range(NH):
        a.semilogx(ra, W[:, h], "o-", label=f"head {h}", lw=1.8)
    a.set_xlabel(r"$|V|/T$"); a.set_ylabel(r"$w\,/\,d_{\rm ref}$")
    a.set_title(L("各ヘッドが選んだ遮蔽長", "screening length chosen per head"), fontsize=11)
    a.legend(fontsize=8, ncol=2); a.grid(alpha=.25, which="both")

    a = ax[1, 1]
    for h in range(NH):
        a.semilogx(ra, SD[:, h], "o-", label=f"head {h}", lw=1.8)
    a.semilogx(ra, SD.mean(1), "k--", lw=2, label=L("平均", "mean"))
    a.set_xlabel(r"$|V|/T$"); a.set_ylabel(r"${\rm sd}(s)$,  $s=\bar q\cdot\bar k$")
    a.set_title(L("スコアの散らばり = 対をどれだけ区別しているか",
                  "score spread = how much pairs are distinguished"), fontsize=11)
    a.legend(fontsize=8, ncol=2); a.grid(alpha=.25, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    print("  wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="*", type=float, default=None,
                    help="学習する r_s のリスト")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--nspins", default="5_4")
    ap.add_argument("--analyze", nargs="*", default=None, help="解析する結果フォルダ")
    ap.add_argument("--blocks", type=int, default=25)
    ap.add_argument("--mcmc", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "coupling_scan"))
    a = ap.parse_args()

    if a.train:
        train(a.train, a.iters, nspins=a.nspins)
    if a.analyze:
        rows = analyze(a.analyze, blocks=a.blocks, mcmc_steps=a.mcmc)
        np.save(a.out + ".npy", rows, allow_pickle=True)
        plot(rows, a.out + ".png")
        print("\n  %-24s %7s %9s %9s %8s %9s %8s"
              % ("run", "r_s", "<T>", "<V>", "|V|/T", "w/d_ref H0", "sd(s)"))
        for r in rows:
            print("  %-24s %7g %9.3f %9.3f %8.2f %9.3f %8.3f"
                  % (r["name"], r["r_s"], r["T"], r["V"], r["ratio"],
                     r["w"].mean(0)[0], r["s_sd"].mean()))
