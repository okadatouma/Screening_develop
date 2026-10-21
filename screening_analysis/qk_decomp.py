# -*- coding: utf-8 -*-
"""内容チャネルは 5x5 の双 1 次形式に潰れる。その中身を分解する。

入力は電子あたり 5 個 f = [sin b1.r, sin b2.r, cos b1.r, cos b2.r, sigma] なので

    s_ij  ∝  f_i A_h f_j^T ,      A_h = (W_E W_Q^(h)) (W_E W_K^(h))^T   （5x5）

内容チャネルが表現できるのはこの 25 成分だけ。位置 x 位置の 16 成分を、
同じ高調波の 2x2 ブロックごとに

    B = a I + b J + c sz + d sx        J = [[0,-1],[1,0]]

と分けると、I と J は theta_i - theta_j にしか依らない（並進不変）、
sz と sx は theta_i + theta_j（重心、並進を破る）。異なる高調波の交差項は全部
並進を破る。残りがスピン x 位置とスピン x スピン。

    python screening_analysis/qk_decomp.py [--nseed 8]
"""
import argparse, glob, json, os, re, sys
import numpy as np, jax, jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from periodicwave import CustomScreenformer as CS, networks, envelopes
from periodicwave.pbc import feature_layer as FL

B0 = "/home/claude/run/results/2deg-Coulomb/CustomScreenformer/el13_12_"
RUNS = [("液体 rs=5", "rs5_tri_l2h16_sl05"), ("境界 rs=30", "rs30_tri_l2h16_sb30"),
        ("結晶 rs=60", "rs60_tri_l2h16_sc60")]
CAT = ["並進不変の対", "重心（並進を破る）", "交差高調波", "スピン×位置", "スピン×スピン"]


def _lattice(cfg):
    m = re.search(r"unserializable object: \[\[(.*?)\]\]", json.dumps(cfg))
    return np.array([float(x) for x in re.findall(
        r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(1))][:4]).reshape(2, 2)


def decomp(A, sd):
    """A: (5,5)。sd: 特徴の標準偏差 (5,)。分散への寄与を 5 分類に分ける。"""
    W = np.outer(sd, sd)                       # 各成分が s の分散に効く重み
    M = A * W                                  # 重み込みの成分
    ti = cm = 0.0
    for m in range(2):                         # 高調波 m の (sin_m, cos_m) ブロック
        i, j = m, m + 2
        b = M[np.ix_([i, j], [i, j])]
        a_ = (b[0, 0] + b[1, 1]) / 2; bb = (b[1, 0] - b[0, 1]) / 2
        c_ = (b[0, 0] - b[1, 1]) / 2; d_ = (b[0, 1] + b[1, 0]) / 2
        ti += 2 * (a_ ** 2 + bb ** 2); cm += 2 * (c_ ** 2 + d_ ** 2)
    pos = M[:4, :4]
    cross = (pos ** 2).sum() - ti - cm         # 異なる高調波どうし
    sp = (M[4, :4] ** 2).sum() + (M[:4, 4] ** 2).sum()
    ss = M[4, 4] ** 2
    v = np.array([ti, cm, cross, sp, ss])
    return v / v.sum()


def A_of(P, NH, AD):
    """A_h = (W_E W_Q)(W_E W_K)^T をヘッドごとに。"""
    E = np.asarray(P["layers"]["embed"])                       # (5, dE)
    ap = P["layers"]["attention"][0]
    Q = np.asarray(ap["q_w"]).reshape(-1, NH, AD)              # (dE, H, dA)
    K = np.asarray(ap["k_w"]).reshape(-1, NH, AD)
    EQ = np.einsum("fe,ehd->fhd", E, Q); EK = np.einsum("fe,ehd->fhd", E, K)
    return np.einsum("ahd,bhd->hab", EQ, EK)                   # (H, 5, 5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nseed", type=int, default=8)
    ap.add_argument("--nconf", type=int, default=24); a = ap.parse_args()
    folder = B0 + RUNS[1][1]
    cfg = json.load(open(os.path.join(folder, "config.json")))
    o = cfg["network"]["CustomScreenformer"]
    L = jnp.asarray(_lattice(cfg)); nspins = tuple(cfg["system"]["electrons"])
    NH, AD = o["num_heads"], o["attn_dim"]
    fl = FL.make_pbc_feature_layer(natoms=1, nspins=nspins, ndim=2, lattice=L,
                                   include_r_ae=False)
    cif = networks.make_periodic_construct_input_features(L)

    # 特徴の標準偏差（実配置から）
    z = np.load(sorted(glob.glob(os.path.join(folder, "qmcjax_ckpt_*.npz")))[-1],
                allow_pickle=True)
    d = z["data"].item()
    pos = jnp.asarray(np.asarray(d["positions"][0])[:a.nconf])
    spn = jnp.asarray(np.asarray(d["spins"][0])[:a.nconf])
    atm = jnp.asarray(np.asarray(d["atoms"][0])[:a.nconf])
    def feat(p, s, at):
        ae, ee, r_ae, r_ee = cif(p, at, ndim=2)
        f, _ = fl.apply(ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee)
        return jnp.concatenate((f, s[..., None]), axis=-1)
    F = np.asarray(jax.vmap(feat)(pos, spn, atm)).reshape(-1, 5)
    sd = F.std(0)
    print("特徴の標準偏差 [sin b1, sin b2, cos b1, cos b2, spin] =", np.round(sd, 3))
    print()
    hdr = "%-14s " % "" + " ".join("%12s" % c for c in CAT)
    print(hdr)

    # 学習済み
    for nm, f in RUNS:
        cfg2 = json.load(open(os.path.join(B0 + f, "config.json")))
        zz = np.load(sorted(glob.glob(os.path.join(B0 + f, "qmcjax_ckpt_*.npz")))[-1],
                     allow_pickle=True)
        P = jax.tree_util.tree_map(lambda x: np.asarray(x)[0], zz["params"].item())
        V = np.stack([decomp(A, sd) for A in A_of(P, NH, AD)]).mean(0)
        print("%-14s " % nm + " ".join("%11.1f%%" % (100 * v) for v in V))

    # 初期化（学習なし）
    net = CS.make_fermi_net(
        nspins, jnp.zeros(1), ndim=2, determinants=cfg["network"]["determinants"],
        envelope=envelopes.make_null_envelope(), feature_layer=fl, jastrow="NONE",
        jastrow_kwargs={}, bias_orbitals=False, complex_output=False, pbc_lattice=L,
        num_layers=o["num_layers"], num_heads=NH, attn_dim=AD,
        value_dim=o["value_dim"], mlp_dim=o["mlp_dim"],
        num_perceptrons_per_layer=o["num_perceptrons_per_layer"],
        use_layer_norm=o["use_layer_norm"], mlp_activation_fct=o["mlp_activation_fct"],
        init_trim_r=o["init_trim_r"],
        init_softmask_w_factor=o["init_softmask_w_factor"],
        init_softmask_w_spread=o["init_softmask_w_spread"])
    Vs = []
    for s in range(a.nseed):
        P = net.init(jax.random.PRNGKey(1234 + s))
        P = jax.tree_util.tree_map(np.asarray, P)
        Vs.append(np.stack([decomp(A, sd) for A in A_of(P, NH, AD)]).mean(0))
    Vs = np.stack(Vs)
    print("%-14s " % ("初期化 %d 種 平均" % a.nseed)
          + " ".join("%11.1f%%" % (100 * v) for v in Vs.mean(0)))
    print("%-14s " % "　（標準偏差）"
          + " ".join("%11.1f%%" % (100 * v) for v in Vs.std(0)))
    print("%-14s " % "次元の数え上げ"
          + " ".join("%11.1f%%" % v for v in [16, 16, 32, 32, 4]))


if __name__ == "__main__":
    main()
