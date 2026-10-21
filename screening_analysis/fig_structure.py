# -*- coding: utf-8 -*-
"""構造の検討に使う 2 パネル：どの行列が学習したか／カスプの有無。

    python screening_analysis/fig_structure.py <run_folder> --out F.png
"""
import argparse, glob, os, sys
import numpy as np, jax, jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import style as ST                                                   # noqa


def pr(M):
    s = np.linalg.svd(M, compute_uv=False) ** 2
    return s.sum() ** 2 / (s ** 2).sum()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("folder")
    ap.add_argument("--out", default="fig_structure.png"); a = ap.parse_args()
    ST.use()
    import matplotlib.pyplot as plt
    z = np.load(sorted(glob.glob(os.path.join(a.folder, "qmcjax_ckpt_*.npz")))[-1], allow_pickle=True)
    P = jax.tree_util.tree_map(lambda x: np.asarray(x)[0], z["params"].item())
    ap_ = P["layers"]["attention"][0]
    mats = [("埋め込み", P["layers"]["embed"]),
            ("$W_Q$", ap_["q_w"]), ("$W_K$", ap_["k_w"]), ("$W_V$", ap_["v_w"]),
            ("ゲート $W_G$", ap_["g_w"]), ("$W_O$", ap_["attn_output"]),
            ("MLP 1 層目", P["layers"]["mlp"][0][0]["w"]), ("MLP 2 層目", P["layers"]["mlp"][0][1]["w"]),
            ("軌道への射影", P["orbital"][0]["w"])]
    rng = np.random.default_rng(0)
    rat = []
    for nm, M in mats:
        M = np.asarray(M)
        prr = np.mean([pr(rng.normal(0, 1 / np.sqrt(M.shape[0]), M.shape)) for _ in range(8)])
        rat.append(pr(M) / prr)

    # --- カスプの走査 -----------------------------------------------------
    import sampler as SM
    from energy_check import build_signed
    net, params, le, data, cfg = build_signed(a.folder)
    NE = sum(cfg["system"]["electrons"])
    rs_ = float(cfg["system"]["make_local_energy_kwargs"]["potential_kwargs"]["interaction_energy_scale"])
    pos = np.asarray(data["positions"][0])[0].reshape(NE, 2); sp = np.asarray(data["spins"][0])[0]
    at = jnp.asarray(np.asarray(data["atoms"][0])[0]); ch = jnp.asarray(np.asarray(data["charges"][0])[0])
    def logpsi(x): return float(np.asarray(net.apply(params, jnp.asarray(x.ravel()), jnp.asarray(sp), at, ch)[1]))
    up = np.where(sp > 0)[0]; dn = np.where(sp < 0)[0]
    rr = np.linspace(-0.25, 0.25, 51)
    def scan(i, j):
        d = pos[j] - pos[i]; d = d / np.linalg.norm(d)
        out = []
        for r in rr:
            x2 = pos.copy(); x2[j] = pos[i] + r * d; out.append(logpsi(x2))
        return np.array(out)
    # 配置平均で背景の傾きを消す: 何組かの対で平均
    rng2 = np.random.default_rng(1)
    opp = np.mean([scan(up[k], dn[k]) for k in range(4)], axis=0)
    par = np.mean([scan(up[k], up[k + 1]) for k in range(4)], axis=0)

    fig = plt.figure(figsize=(13.0, 5.4))
    fig.text(0.035, 0.962, "学習したのはどこか、カスプはあるか", fontsize=15.0, weight="bold",
             color=ST.GREEN, va="top")
    fig.text(0.035, 0.915, "左：各行列の特異値スペクトルを、同じ形の乱数行列と比べる　　右：2 電子を近づけたときの $\\log|\\psi|$",
             fontsize=10.8, color=ST.SUB, va="top")

    ax = fig.add_axes([0.085, 0.40, 0.375, 0.42])
    cols = [ST.BLUE if r > 0.8 else ST.ENJI for r in rat]
    ax.barh(range(len(mats)), rat, color=cols, height=0.62)
    ax.set_yticks(range(len(mats))); ax.set_yticklabels([m[0] for m in mats], fontsize=9.8)
    ax.invert_yaxis()
    ax.axvline(1.0, color=ST.LINE, lw=1.0, ls="--")
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("特異値の参加率 ／ 乱数行列の参加率", labelpad=4)
    ax.set_title("① どの行列が学習したか", pad=8)
    ax.text(1.02, 0.5, "乱数と\n区別がつかない", fontsize=8.8, color=ST.BLUE, va="center",
            transform=ax.get_yaxis_transform())
    for i, r in enumerate(rat):
        if r < 0.8:
            ax.text(r + 0.02, i, "%.2f" % r, va="center", fontsize=9, color=ST.ENJI)

    ax2 = fig.add_axes([0.56, 0.40, 0.41, 0.42])
    ax2.plot(rr, opp - opp[len(rr) // 2], "-", color=ST.BLUE, lw=2.0, label="異スピン対 ↑↓")
    ax2.plot(rr, par - par.max(), "-", color=ST.ENJI, lw=2.0, label="同スピン対 ↑↑")
    # 期待されるカスプ: log psi ≈ r_s |r| （異スピン, 2 次元）
    _m = np.abs(rr) <= 0.06          # カスプ条件は合体点での傾きなので、近傍だけ描く
    ax2.plot(rr[_m], rs_ * np.abs(rr[_m]), ":", color=ST.BLUE, lw=1.8,
             label="カスプ条件（合体点での傾き $\\pm r_s$）")
    ax2.set_xlabel("対の距離 $r$（$r_s a_B^*$ 単位、符号は向き）", labelpad=4)
    ax2.set_ylabel("$\\log|\\psi|$（$r=0$ を基準）")
    ax2.set_ylim(-12, 5)
    ax2.legend(fontsize=9.2, loc="lower left")
    ax2.grid(alpha=.22, lw=.7)
    ax2.set_title("② 2 電子の合体点まわり（$r_s = %g$）" % rs_, pad=8)

    ST.caption(
        fig, "読み方",
        "①  乱数行列に近い参加率（青）は、その行列がスペクトルの上で初期化から見分けられないことを意味する。"
        "$W_Q, W_K, W_V, W_G$ と MLP 1 層目がそれにあたり、\n     学習が見えるのは $W_O$・MLP 2 層目・軌道への射影（赤）。"
        "つまり注意の中身は「乱数の特徴 → 学習した線形読み出し」になっている。\n"
        "②  異スピン対は $r = 0$ を滑らかに通り、点線の折れ（2 次元のカスプ条件 $\\psi \\simeq 1 + r_s r$）がない。"
        "同スピン対は行列式の節で $-\\infty$ に落ちる。4 組の対で平均し、他電子による背景の傾きを消してある。",
        y=0.030, size=9.3)
    fig.savefig(a.out, dpi=170, facecolor=fig.get_facecolor())
    print("  wrote", a.out)
    for (nm, _), r in zip(mats, rat): print("   %-14s %.3f" % (nm, r))


if __name__ == "__main__":
    main()
