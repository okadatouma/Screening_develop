# -*- coding: utf-8 -*-
"""内容チャネルのスピン差は初期化の残りか、学習したものか。

初期化直後のネットワークを乱数種だけ変えて何本も作り、ヘッドごとの
「同スピン対 − 異スピン対」の s_ij の平均差を測る。

    python screening_analysis/seed_spin.py [--nseed 12]

学習済み run は deterministic = False（種は time 由来）なので、3 相はそれぞれ
独立な初期化。したがって「相どうしでヘッドごとの符号が揃う」なら、
それは種ではなく窓（＝アーキテクチャ）か学習が決めていることになる。
"""
import argparse, glob, json, os, re, sys
import numpy as np, jax, jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from periodicwave import CustomScreenformer as CS, networks, envelopes
from periodicwave.pbc import feature_layer as FL

B = "/home/claude/run/results/2deg-Coulomb/CustomScreenformer/el13_12_"
RUNS = [("液体", "rs5_tri_l2h16_sl05"), ("境界", "rs30_tri_l2h16_sb30"),
        ("結晶", "rs60_tri_l2h16_sc60")]


def _lattice(cfg):
    m = re.search(r"unserializable object: \[\[(.*?)\]\]", json.dumps(cfg))
    return np.array([float(x) for x in re.findall(
        r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(1))][:4]).reshape(2, 2)


def spin_gap(S, nup):
    """[head, i, j] の s_ij から、ヘッドごとの (同スピン − 異スピン) の平均差。"""
    NE = S.shape[-1]; off = ~np.eye(NE, dtype=bool)
    same = np.zeros((NE, NE), bool); same[:nup, :nup] = True; same[nup:, nup:] = True
    return np.array([S[h][same & off].mean() - S[h][(~same) & off].mean()
                     for h in range(S.shape[0])])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nseed", type=int, default=12)
    ap.add_argument("--nconf", type=int, default=24); a = ap.parse_args()

    folder = B + RUNS[1][1]
    cfg = json.load(open(os.path.join(folder, "config.json")))
    o = cfg["network"]["CustomScreenformer"]
    L = jnp.asarray(_lattice(cfg)); nspins = tuple(cfg["system"]["electrons"])
    NE, nup = sum(nspins), nspins[0]
    NH, AD = o["num_heads"], o["attn_dim"]
    ck = sorted(glob.glob(os.path.join(folder, "qmcjax_ckpt_*.npz")))[-1]
    z = np.load(ck, allow_pickle=True); data = z["data"].item()
    pos = np.asarray(data["positions"][0])[:a.nconf]
    sp = np.asarray(data["spins"][0])[:a.nconf]
    at_ = np.asarray(data["atoms"][0])[:a.nconf]

    fl = FL.make_pbc_feature_layer(natoms=1, nspins=nspins, ndim=2, lattice=L,
                                   include_r_ae=False)
    cif = networks.make_periodic_construct_input_features(L)
    net = CS.make_fermi_net(
        nspins, jnp.zeros(1), ndim=2, determinants=cfg["network"]["determinants"],
        envelope=envelopes.make_null_envelope(), feature_layer=fl,
        jastrow="NONE", jastrow_kwargs={}, bias_orbitals=False, complex_output=False,
        pbc_lattice=L, num_layers=o["num_layers"], num_heads=NH,
        attn_dim=AD, value_dim=o["value_dim"], mlp_dim=o["mlp_dim"],
        num_perceptrons_per_layer=o["num_perceptrons_per_layer"],
        use_layer_norm=o["use_layer_norm"], mlp_activation_fct=o["mlp_activation_fct"],
        init_trim_r=o["init_trim_r"],
        init_softmask_w_factor=o["init_softmask_w_factor"],
        init_softmask_w_spread=o["init_softmask_w_spread"])

    def s_of(P, pos1, sp1, at1):
        ae, ee, r_ae, r_ee = cif(pos1, at1, ndim=2)
        f, _ = fl.apply(ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee)
        f = jnp.concatenate((f, sp1[..., None]), axis=-1)
        x = jnp.dot(f, P["layers"]["embed"])
        ap0 = P["layers"]["attention"][0]
        q = jnp.dot(x, ap0["q_w"]).reshape(NE, NH, AD)
        k = jnp.dot(x, ap0["k_w"]).reshape(NE, NH, AD)
        return jnp.einsum("ihd,jhd->hij", CS._unit_length(q), CS._unit_length(k))

    batched = jax.jit(jax.vmap(s_of, in_axes=(None, 0, 0, 0)))
    dref = float(CS._max_periodic_distance(np.asarray(_lattice(cfg))))
    G = []
    for s in range(a.nseed):
        P = net.init(jax.random.PRNGKey(1234 + s))
        S = np.asarray(batched(P, jnp.asarray(pos), jnp.asarray(sp), jnp.asarray(at_))).mean(0)
        G.append(spin_gap(S, nup))
        w0 = dref * np.asarray(jax.nn.softplus(P["layers"]["attention"][0]["mask_omega"]))
    G = np.stack(G)                                    # [seed, head]

    print("初期化直後（学習なし）、種を %d 本" % a.nseed)
    print("  ヘッドごとのスピン差: 標準偏差 %.4f（種・ヘッドをまとめて）  最大 |差| %.3f"
          % (G.std(), np.abs(G).max()))
    print("  種をまたいだヘッド平均の標準偏差 %.4f  → 窓で決まる成分" % G.mean(0).std())
    cc = [np.corrcoef(G[i], G[j])[0, 1] for i in range(a.nseed) for j in range(i + 1, a.nseed)]
    print("  種どうしの相関（ヘッド番号で対応づけ）: 平均 %.3f  標準偏差 %.3f  範囲 %.2f..%.2f"
          % (np.mean(cc), np.std(cc), min(cc), max(cc)))
    print("  参考: n = %d の帰無分布の標準偏差 = %.3f" % (NH, 1 / np.sqrt(NH - 3)))
    np.save(os.path.join(HERE, "seed_spin_init.npy"), G)


if __name__ == "__main__":
    main()
