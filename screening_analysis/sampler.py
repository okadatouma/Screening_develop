"""学習済みの波動関数から配置を大量に取り直す。

    python screening_analysis/sampler.py <folder> [--nconf 4096] [--nwalk 128]

チェックポイントに残っているのは walker 64 本ぶんだけで、密度図には足りない。
ここでは学習と同じ Metropolis（全電子移動・10 step/回）を回して独立な配置を貯める。
"""
import argparse, glob, json, os, re, sys
import numpy as np, jax, jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from periodicwave import networks, envelopes, mcmc as MC          # noqa
from periodicwave import CustomScreenformer as CS                 # noqa
from periodicwave import CustomPsiformer as CP                    # noqa
from periodicwave.pbc import feature_layer as FL                  # noqa


def _mats(cfg):
    out = []
    for blk in re.findall(r"unserializable object: \[\[(.*?)\]\]", json.dumps(cfg)):
        v = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", blk)][:4]
        out.append(np.array(v).reshape(2, 2))
    return out



def _net_kwargs(cfg, ntype, params):
    """config.json のネットワーク引数を、checkpoint に合わせて直したもの。

    帯域窓（殻）の中心 init_softmask_centres は numpy 配列なので config.json には
    "unserializable object: [...]" という文字列でしか残らない。そこからは復元
    できないので、checkpoint の 'mask_centre' / 'mask_omega' から作り直す。
    checkpoint に無ければ単調窓なので None にする。
    """
    kw = dict(cfg["network"][ntype])
    if "init_softmask_centres" not in kw and "init_softmask_rings" not in kw:
        return kw
    if "init_softmask_centres" in kw:
        kw["init_softmask_centres"] = None
    if "init_softmask_rings" in kw:
        kw["init_softmask_rings"] = None
    try:
        a = params["layers"]["attention"][0]
    except (KeyError, TypeError, IndexError):
        return kw
    if "mask_centre" in a:
        c = np.asarray(a["mask_centre"], dtype=np.float64)
        sg = np.log1p(np.exp(np.asarray(a["mask_omega"], dtype=np.float64)))
        kw["init_softmask_centres"] = (c, sg)
    if "mask_rings" in a:
        # 全体窓の輪切り：(境界 b_h/d_ref, 遷移の半幅 delta/d_ref, 指数 p)
        b = np.asarray(a["mask_rings"], dtype=np.float64)
        dl = float(np.log1p(np.exp(np.asarray(a["mask_omega"], dtype=np.float64)))[0])
        kw["init_softmask_rings"] = (b, dl, float(np.asarray(a["mask_envp"])))
        if "mask_gauss" in a:                  # ガウス核の輪切り（dl は標準偏差 sigma / d_ref）
            kw["init_softmask_rings"] = kw["init_softmask_rings"] + ("gauss",)
    return kw


def build(folder, checkpoint=None):
    """(batch_network, params, data0, cfg, L) を返す。"""
    ck = checkpoint or sorted(glob.glob(os.path.join(folder, "qmcjax_ckpt_*.npz")))[-1]
    z = np.load(ck, allow_pickle=True)
    data = z["data"].item()
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(np.asarray(x)[0]), z["params"].item())
    cfg = json.load(open(os.path.join(folder, "config.json")))
    L = _mats(cfg)[0]
    nspins = tuple(cfg["system"]["electrons"])
    ntype = cfg["network"]["network_type"]

    atoms = jnp.asarray(np.asarray(data["atoms"][0])[0])
    charges = jnp.asarray(np.asarray(data["charges"][0])[0])
    fl = FL.make_pbc_feature_layer(natoms=charges.shape[0], nspins=nspins, ndim=2,
                                   lattice=jnp.asarray(L), include_r_ae=False)
    mk = CP.make_fermi_net if ntype == "CustomPsiformer" else CS.make_fermi_net
    net = mk(nspins, charges, ndim=2,
             determinants=cfg["network"]["determinants"],
             envelope=envelopes.make_null_envelope(), feature_layer=fl,
             jastrow=cfg["network"].get("jastrow", "NONE"),
             jastrow_kwargs=cfg["network"]["jastrow_kwargs"],
             bias_orbitals=cfg["network"]["bias_orbitals"],
             complex_output=cfg["network"].get("complex", False),
             pbc_lattice=jnp.asarray(L), **_net_kwargs(cfg, ntype, params))
    logabs = lambda p, x, s, a, c: net.apply(p, x, s, a, c)[1]
    bnet = jax.vmap(logabs, in_axes=(None, 0, 0, 0, 0), out_axes=0)
    d0 = networks.FermiNetData(
        positions=jnp.asarray(np.asarray(data["positions"][0])),
        spins=jnp.asarray(np.asarray(data["spins"][0])),
        atoms=jnp.asarray(np.asarray(data["atoms"][0])),
        charges=jnp.asarray(np.asarray(data["charges"][0])))
    return bnet, params, d0, cfg, L


def make_hop(bnet, ne, L):
    """電子 1 個をセル内の一様乱数点へ飛ばす Metropolis 移動。

    全電子ガウス移動は幅が 0.1 程度しかないので、電子が谷から谷へ移れない。
    そのままだと「どの谷が埋まっているか」がチェックポイントの初期条件のまま
    凍りついてしまい、密度や占有の測定が信用できなくなる。飛び移りを混ぜて
    その自由度をきちんとサンプルする。提案は左右対称なので詳細釣り合いは保たれる。
    """
    Lj = jnp.asarray(L)

    def hop(params, pos, spins, atoms, charges, key, lp, nhop):
        nw = pos.shape[0]
        def body(c, _):
            pos, lp, key, acc = c
            key, k1, k2, k3 = jax.random.split(key, 4)
            i = jax.random.randint(k1, (nw,), 0, ne)
            f = jax.random.uniform(k2, (nw, 2))
            r = f @ Lj.T
            p2 = pos.reshape(nw, ne, 2).at[jnp.arange(nw), i].set(r).reshape(nw, -1)
            lp2 = 2.0 * bnet(params, p2, spins, atoms, charges)
            ok = jnp.log(jax.random.uniform(k3, (nw,))) < (lp2 - lp)
            return (jnp.where(ok[:, None], p2, pos), jnp.where(ok, lp2, lp),
                    key, acc + ok.mean()), None
        (pos, lp, key, acc), _ = jax.lax.scan(body, (pos, lp, key, 0.0), None, length=nhop)
        return pos, lp, key, acc / nhop
    return hop


def sample(folder, nconf=4096, nwalk=128, decorr=4, burn=30, width=0.2,
           seed=0, checkpoint=None, verbose=True, nhop=4, params_override=None):
    """(R, pmove) を返す。R.shape = (nconf, ne, 2)。

    params_override を渡すと、その params の |ψ|² からサンプルする（アブレーション用）。
    """
    bnet, params, d0, cfg, L = build(folder, checkpoint)
    if params_override is not None:
        params = params_override
    ne = sum(cfg["system"]["electrons"])
    nb = int(d0.positions.shape[0])
    rep = int(np.ceil(nwalk / nb))
    key = jax.random.PRNGKey(seed)

    def tile(a):
        a = jnp.concatenate([a] * rep, axis=0)[:nwalk]
        return a
    key, sk = jax.random.split(key)
    pos = tile(d0.positions) + 0.05 * jax.random.normal(sk, tile(d0.positions).shape)
    data = networks.FermiNetData(positions=pos, spins=tile(d0.spins),
                                 atoms=tile(d0.atoms), charges=tile(d0.charges))
    step = jax.jit(MC.make_mcmc_step(bnet, nwalk, steps=10, ndim=2, blocks=1))
    hop = jax.jit(make_hop(bnet, ne, L), static_argnums=(7,)) if nhop else None

    def sweep(data, key, w):
        key, sk = jax.random.split(key)
        data, pm = step(params, data, sk, w)
        ha = 0.0
        if hop is not None:
            key, sk = jax.random.split(key)
            lp = 2.0 * bnet(params, data.positions, data.spins, data.atoms, data.charges)
            pos, lp, _, ha = hop(params, data.positions, data.spins, data.atoms,
                                 data.charges, sk, lp, nhop)
            data = networks.FermiNetData(**(dict(data) | {"positions": pos}))
        return data, pm, float(ha)

    w = jnp.asarray(width)
    for i in range(burn):                                   # burn-in + width 調整
        data, pm, ha = sweep(data, key, w)
        key, _ = jax.random.split(key)
        if (i + 1) % 2 == 0:
            w = w * (1.15 if pm > 0.55 else (1 / 1.15 if pm < 0.45 else 1.0))
    out, pms, has = [], [], []
    while sum(x.shape[0] for x in out) < nconf:
        for _ in range(decorr):
            key, _ = jax.random.split(key)
            data, pm, ha = sweep(data, key, w)
            has.append(ha)
        out.append(np.asarray(data.positions)); pms.append(float(pm))
        if verbose:
            print(f"\r  sampling {sum(x.shape[0] for x in out):6d}/{nconf}"
                  f"  pmove={np.mean(pms):.3f} 飛移り={np.mean(has):.3f} "
                  f"width={float(w):.3f}", end="", flush=True)
    if verbose: print()
    R = np.concatenate(out, 0)[:nconf].reshape(-1, ne, 2)
    return R, float(np.mean(pms))


def cached(folder, nconf=4096, nwalk=128, **kw):
    f = os.path.join(folder, f"samples_{nconf}.npy")
    if os.path.exists(f): return np.load(f)
    R, _ = sample(folder, nconf, nwalk, **kw)
    np.save(f, R); return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("--nconf", type=int, default=4096)
    ap.add_argument("--nwalk", type=int, default=128)
    a = ap.parse_args()
    R = cached(a.folder, a.nconf, a.nwalk)
    print("R", R.shape, "-> ", os.path.join(a.folder, f"samples_{a.nconf}.npy"))
