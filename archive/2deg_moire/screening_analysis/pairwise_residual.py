"""波動関数のドリフト場が「2体で書けるか」を直接測る。

    python screening_analysis/pairwise_residual.py results/2deg-Coulomb/CustomScreenformer/el5_4_rs*_tri_det

## 何を測るか

ラフリン状態は ln ψ = Σ_{i<j} m ln(z_i - z_j) + 一体項。つまり **ln ψ が対の和**で、
勾配（ドリフト場）は

    ∇_i ln ψ = Σ_{j≠i} f(r_ij) r̂_ij            ← 対の力の重ね合わせ

の形しか取れない。一様系なので一体項は無い。そこでネットワークの実際の
∇_i ln ψ を上の形に最小二乗で当てはめ、**説明できた分散の割合 R²** を測る。

    R² = 1        完全に2体（ラフリン型で書ける）
    1 - R²        2体では原理的に書けない残り = 3体以上の寄与

f は動径ガウス基底で展開する（線形最小二乗）。スピン依存も許す版（f_same, f_opp）と
許さない版の両方を出す。ラフリンは単一の指数 m なのでスピン非依存版が本来の比較対象、
スピン依存版は Slater–Jastrow 相当まで許した場合の上限になる。

## なぜこれが「見るべき量」なのか

- ゲージや規格化に依らない。q·k のような内部表現量と違い、波動関数そのものの性質
- 単位に依らない（分散比）。r_s 間で直接比べられる
- 人手の2体波動関数（ラフリン、Jastrow）との距離を直接与える

行列式の寄与も「2体で書けない分」に入る。ラフリンは反対称性を対の因子だけで作るので、
その扱いで正しい。
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from periodicwave import networks, envelopes, mcmc, CustomScreenformer, CustomPsiformer  # noqa
from periodicwave.pbc import feature_layer as FL                                          # noqa


def _lattice(cfg):
    m = re.search(r"unserializable object: \[\[(.*?)\]\]", json.dumps(cfg))
    v = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(1))]
    return np.array(v[:4]).reshape(2, 2)


def _min_image(dr, L, Linv):
    """最小像規約。格子ベクトルは L の列なので分数座標は dr @ Linv.T（転置を落とすと壊れる）。"""
    f = dr @ Linv.T
    f = f - jnp.round(f)
    return f @ L.T


def load(folder):
    ck = sorted(glob.glob(os.path.join(folder, "qmcjax_ckpt_*.npz")))[-1]
    z = np.load(ck, allow_pickle=True)
    data = z["data"].item()
    params = jax.tree_util.tree_map(lambda a: jnp.asarray(a)[0], z["params"].item())
    cfg = json.load(open(os.path.join(folder, "config.json")))
    L = jnp.asarray(_lattice(cfg))
    nspins = tuple(cfg["system"]["electrons"])
    ndim = cfg["system"]["ndim"]
    charges = jnp.zeros(1)
    ntype = cfg["network"]["network_type"]
    fl = FL.make_pbc_feature_layer(natoms=1, nspins=nspins, ndim=ndim, lattice=L, include_r_ae=False)
    mod = CustomScreenformer if ntype == "CustomScreenformer" else CustomPsiformer
    net = mod.make_fermi_net(nspins, charges, ndim=ndim,
                             determinants=cfg["network"]["determinants"],
                             envelope=envelopes.make_null_envelope(), feature_layer=fl,
                             jastrow=cfg["network"]["jastrow"],
                             jastrow_kwargs=cfg["network"]["jastrow_kwargs"],
                             bias_orbitals=cfg["network"]["bias_orbitals"],
                             complex_output=cfg["network"]["complex"], pbc_lattice=L,
                             **cfg["network"][ntype])
    logabs = lambda p, x, s, a, c: net.apply(p, x, s, a, c)[1]                    # noqa: E731
    return dict(params=params, data=data, logabs=logabs, L=np.asarray(L), ndim=ndim,
                ne=sum(nspins), nup=nspins[0],
                r_s=float(cfg["system"]["make_local_energy_kwargs"]
                          ["potential_kwargs"]["interaction_energy_scale"]),
                name=os.path.basename(folder))


def sample(b, blocks=12, mcmc_steps=15, width=0.2, seed=0):
    d = b["data"]
    pos = jnp.asarray(d["positions"][0])
    spins, atoms, chg = (jnp.asarray(d[k][0]) for k in ("spins", "atoms", "charges"))
    bn = jax.vmap(b["logabs"], in_axes=(None, 0, 0, 0, 0))
    step = mcmc.make_mcmc_step(bn, pos.shape[0], steps=mcmc_steps, ndim=b["ndim"])
    grad = jax.jit(jax.vmap(jax.grad(b["logabs"], argnums=1), in_axes=(None, 0, 0, 0, 0)))
    key = jax.random.PRNGKey(seed)
    P, G = [], []
    for _ in range(blocks):
        key, sub = jax.random.split(key)
        nd, _ = step(b["params"], networks.FermiNetData(positions=pos, spins=spins,
                                                        atoms=atoms, charges=chg), sub, width)
        pos = nd.positions
        P.append(np.asarray(pos)); G.append(np.asarray(grad(b["params"], pos, spins, atoms, chg)))
    return np.concatenate(P), np.concatenate(G), np.asarray(spins[0])


def fit(pos, G, sp, L, ne, ndim=2, nbasis=16, spin_dep=True):
    """∇_i lnψ ≈ Σ_{j≠i} f(r_ij) r̂_ij を線形最小二乗で当てる。"""
    Linv = np.linalg.inv(L)
    B = pos.shape[0]
    R = pos.reshape(B, ne, ndim)
    dr = R[:, :, None, :] - R[:, None, :, :]                        # (B,i,j,d)
    f = dr @ Linv.T
    dr = (f - np.round(f)) @ L.T                                    # 最小像
    r = np.linalg.norm(dr, axis=-1)                                 # (B,i,j)
    np.fill_diagonal(r[0], 1.0)
    eye = np.eye(ne, dtype=bool)
    r = np.where(eye[None], 1.0, r)
    hat = dr / np.maximum(r, 1e-9)[..., None]

    rmax = float(np.percentile(r[:, ~eye], 99.5))
    cen = np.linspace(r[:, ~eye].min() * 0.9, rmax, nbasis)
    wid = (cen[1] - cen[0]) * 1.1
    # 基底: 各中心のガウス。f(r) = Σ_a c_a exp(-(r-cen_a)^2/2wid^2)
    phi = np.exp(-0.5 * ((r[..., None] - cen) / wid) ** 2)          # (B,i,j,a)
    phi = np.where(eye[None, :, :, None], 0.0, phi)

    same = (sp[:, None] * sp[None, :] > 0) & ~eye
    groups = [np.ones_like(same, bool) & ~eye] if not spin_dep else [same, (~same) & ~eye]
    # 設計行列: X[b,i,d, (g,a)] = Σ_j 1_g(i,j) phi[b,i,j,a] hat[b,i,j,d]
    cols = []
    for g in groups:
        cols.append(np.einsum("bija,bijd,ij->bida", phi, hat, g.astype(float)))
    X = np.concatenate(cols, axis=-1).reshape(-1, nbasis * len(groups))
    y = G.reshape(B, ne, ndim).reshape(-1)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ c
    r2 = 1.0 - res.var() / y.var()
    return r2, c, cen, res.var(), y.var()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--mcmc", type=int, default=15)
    ap.add_argument("--nbasis", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    for f in a.folders:
        b = load(f)
        pos, G, sp = sample(b, blocks=a.blocks, mcmc_steps=a.mcmc)
        r2s, *_ = fit(pos, G, sp, b["L"], b["ne"], nbasis=a.nbasis, spin_dep=False)
        r2d, c, cen, rv, yv = fit(pos, G, sp, b["L"], b["ne"], nbasis=a.nbasis, spin_dep=True)
        rows.append(dict(name=b["name"], r_s=b["r_s"], r2_plain=r2s, r2_spin=r2d,
                         resid=rv, total=yv, coef=c, cen=cen))
        print("  %-24s r_s=%-5g  R2(2体)=%.4f  R2(2体+スピン)=%.4f  → 3体以上 %.1f%%"
              % (b["name"], b["r_s"], r2s, r2d, 100 * (1 - r2d)), flush=True)
        jax.clear_caches()
    if a.out:
        np.save(a.out, rows, allow_pickle=True)
