"""通常の softmax 注意（CustomPsiformer）から q·k と注意重みを取り出す。

    python screening_analysis/psi_probe.py results/2deg-Coulomb/CustomPsiformer/el13_12_rs5_tri_psi_l2h16

attn_probe.py（遮蔽注意用）と同じ形の辞書を返すので、bigN_maps.py がそのまま使える。
違いは 2 点だけ:
  - 重み W は softmax(q·k/√d)。行の総和が 1 に規格化される（遮蔽は規格化しない）
  - 遮蔽窓 w も Trim 幅 r も存在しないので、w は NaN を入れて「窓なし」を明示する

本体は jax.nn.dot_product_attention（Flash attention）を使っていて重みが露出しないため、
ここでは同じ q, k から softmax を明示的に計算し直している。1/√attn_dim のスケールは
dot_product_attention の既定と揃えてある。
"""
import glob, json, os, re, sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from periodicwave import networks, CustomPsiformer as CP          # noqa: E402
from periodicwave.pbc import feature_layer as FL                  # noqa: E402


def lattice_from_config(cfg):
    m = re.search(r"unserializable object: \[\[(.*?)\]\]", json.dumps(cfg))
    v = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(1))]
    return np.array(v[:4]).reshape(2, 2)


def probe(folder, checkpoint=None, nconf=None):
    ck = checkpoint or sorted(glob.glob(os.path.join(folder, "qmcjax_ckpt_*.npz")))[-1]
    z = np.load(ck, allow_pickle=True)
    data = z["data"].item()
    P = jax.tree_util.tree_map(lambda a: np.asarray(a)[0], z["params"].item())
    cfg = json.load(open(os.path.join(folder, "config.json")))
    o = cfg["network"]["CustomPsiformer"]
    L = lattice_from_config(cfg)
    NH, AD, VD, MD, NL = o["num_heads"], o["attn_dim"], o["value_dim"], o["mlp_dim"], o["num_layers"]
    NE = sum(cfg["system"]["electrons"])

    fl = FL.make_pbc_feature_layer(natoms=1, nspins=tuple(cfg["system"]["electrons"]), ndim=2,
                                   lattice=jnp.asarray(L), include_r_ae=False)
    cif = networks.make_periodic_construct_input_features(jnp.asarray(L))
    _, attn = CP.make_multi_head_attention(NH, MD, AD, VD, NE)
    _, ln = CP.make_layer_norm()
    _, mlp = CP.make_mlp(o["num_perceptrons_per_layer"], o["mlp_activation_fct"])
    use_ln = o["use_layer_norm"]

    def forward(pos, spins, atoms):
        ae, ee, r_ae, r_ee = cif(pos, atoms, ndim=2)
        f, _ = fl.apply(ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee)
        f = jnp.concatenate((f, spins[..., None]), axis=-1)
        x = jnp.dot(f, P["layers"]["embed"])
        d_ee = r_ee[..., 0]
        S, W = [], []
        for l in range(NL):
            ap = P["layers"]["attention"][l]
            q = jnp.dot(x, ap["q_w"]).reshape(NE, NH, AD)
            k = jnp.dot(x, ap["k_w"]).reshape(NE, NH, AD)
            s = jnp.einsum("ihd,jhd->hij", q, k) / jnp.sqrt(float(AD))
            S.append(s)
            W.append(jax.nn.softmax(s, axis=-1))
            x = x + attn(ap, x)
            if use_ln:
                x = ln(P["layers"]["ln"][l][0], x)
            x = x + mlp(P["layers"]["mlp"][l], x)
            if use_ln:
                x = ln(P["layers"]["ln"][l][1], x)
        return jnp.stack(S), jnp.stack(W), d_ee

    pos = np.asarray(data["positions"][0])
    sp_ = np.asarray(data["spins"][0])
    at_ = np.asarray(data["atoms"][0])
    if nconf is not None and pos.shape[0] > nconf:
        i = np.linspace(0, pos.shape[0] - 1, nconf).astype(int)
        pos, sp_, at_ = pos[i], sp_[i], at_[i]
    S, W, d = jax.vmap(forward)(jnp.asarray(pos), jnp.asarray(sp_), jnp.asarray(at_))

    # 遮蔽注意と同じ辞書の形。窓は存在しないので NaN。d_ref は幾何から計算。
    from periodicwave import CustomScreenformer as CS
    dref = CS._max_periodic_distance(jnp.asarray(L))
    return dict(S=np.asarray(S), W=np.asarray(W), d=np.asarray(d),
                w=np.full((NL, NH), np.nan), r=np.full((NL, NH), np.nan), dref=float(dref),
                spins=sp_, positions=pos, lattice=np.asarray(L),
                NE=NE, NL=NL, NH=NH, checkpoint=os.path.basename(ck), folder=folder,
                softmax=True)


if __name__ == "__main__":
    a = probe(sys.argv[1], nconf=int(sys.argv[2]) if len(sys.argv) > 2 else 128)
    off = ~np.eye(a["NE"], dtype=bool)
    print("  %s  N=%d  L=%d H=%d  配置 %d" % (os.path.basename(a["folder"]), a["NE"],
                                              a["NL"], a["NH"], a["S"].shape[0]))
    print("  自己重み [%] 層平均:",
          np.round(100 * a["W"][:, :, :, np.arange(a["NE"]), np.arange(a["NE"])].mean(axis=(0, 3)).mean(0), 1))
    print("  非対角の重み平均    :", np.round(a["W"][:, :, :, off].mean(axis=(0, 3)).mean(0), 4))
