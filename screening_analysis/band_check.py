# -*- coding: utf-8 -*-
"""帯窓（輪ごとに 1 -> 0）の検算。GPU に投げる前にここで全部確かめる。

    python screening_analysis/band_check.py
"""
import os, sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(HERE)); sys.path.insert(0,HERE)
import jax, jax.numpy as jnp
from mini_vmc import build, AEE
from periodicwave import networks, CustomScreenformer as CS

CKPT = ("results/2deg-Coulomb/CustomScreenformer/el13_12_rs30_tri_l2h16_sb30/"
        "qmcjax_ckpt_007999.npz")


def window(d, A, D):
    t = np.clip((d[None] - A[:, None]) / D, -1.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * t))


def main():
    n = 5
    a1 = AEE * np.array([1., 0.]); a2 = AEE * np.array([-0.5, np.sqrt(3) / 2])
    L = np.column_stack([n * a1, n * a2])
    dref = float(CS._max_periodic_distance(jnp.asarray(L)))
    print("d_ref = %.4f" % dref)

    # ---- 1. 窓そのもの -------------------------------------------------
    for H in (8, 16):
        D = dref / max(H - 1, 1); A = D * np.arange(H)
        d = np.linspace(0, dref, 40001); M = window(d, A, D)
        core = d <= A[-1]
        print("H=%2d  Delta=%.4f  a_h = %s ... %s   (d_ref = %.4f)"
              % (H, D, "%.3f" % A[0], "%.3f" % A[-1], dref))
        print("      [a_0, a_{H-1}] = [0, d_ref] で sum_h m_h : min %.10f  max %.10f"
              % (M.sum(0)[core].min(), M.sum(0)[core].max()))
        sup = [(d[M[h] > 1e-9].min(), d[M[h] > 1e-9].max()) for h in (0, H - 1)]
        print("      台: h0 [%.3f, %.3f]  h%d [%.3f, %.3f]（外は厳密に 0）"
              % (sup[0][0], sup[0][1], H - 1, sup[1][0], sup[1][1]))
        print("      自分の輪 [a_h, a_{h+1}] で単調減少: %s   m_h(a_h)=%.6f  m_h(a_{h+1})=%.2e"
              % (bool(np.all(np.diff(M[3][(d >= A[3]) & (d <= A[3] + D)]) <= 1e-12)),
                 M[3][np.argmin(abs(d - A[3]))], M[3][np.argmin(abs(d - A[3] - D))]))
        print("      max|m''| = %.1f  （曲率 pi^2/(2 Delta^2)）" % (np.pi ** 2 / (2 * D ** 2)))

    # ---- 2. 学習済み run の配置での局所エネルギーのばらつき ------------
    z = np.load(os.path.join(os.path.dirname(HERE), CKPT), allow_pickle=True)
    pos = np.asarray(z["data"].item()["positions"]).reshape(-1, 50)
    rng = np.random.default_rng(0)
    x = jnp.asarray(pos[rng.choice(len(pos), 48, replace=False)])
    B = x.shape[0]
    print("\n配置 %d 点（学習済み run rs=30 の walker から）" % B)
    store = {}
    for mode in ("base", "fixv", "band8", "band16"):
        net, le, _, _, NE, nspins = build(mode, rs=30.0)
        p = net.init(jax.random.PRNGKey(0))
        sp = jnp.asarray(np.tile(np.array([1.] * nspins[0] + [-1.] * nspins[1]), (B, 1)))
        dat = networks.FermiNetData(positions=x, spins=sp,
                                    atoms=jnp.zeros((B, 1, 2)), charges=jnp.zeros((B, 1)))
        E = np.real(np.asarray(jax.vmap(le, in_axes=(None, 0, 0))(
            p, jax.random.split(jax.random.PRNGKey(1), B), dat)[0]))
        npar = sum(np.size(v) for v in jax.tree_util.tree_leaves(p))
        store[mode] = E
        mad = float(np.median(np.abs(E - np.median(E))))
        print("  %-7s  標準偏差 %8.1f   中央絶対偏差 %7.1f   パラメータ %6d"
              % (mode, E.std(), mad, npar))

    np.savez(os.path.join(HERE, "init_spread.npz"), **store)

    # ---- 3. 反対称性 ---------------------------------------------------
    net, le, _, _, NE, nspins = build("band8", rs=30.0)
    p = net.init(jax.random.PRNGKey(0))
    x1 = np.asarray(x[:1]).reshape(1, NE, 2).copy()
    x2 = x1.copy(); x2[0, [0, 1]] = x2[0, [1, 0]]          # 同スピン交換
    s1 = jnp.asarray(np.array([[1.] * nspins[0] + [-1.] * nspins[1]]))
    o1 = net.apply(p, jnp.asarray(x1.reshape(1, -1))[0], s1[0], jnp.zeros((1, 2)), jnp.zeros(1))
    o2 = net.apply(p, jnp.asarray(x2.reshape(1, -1))[0], s1[0], jnp.zeros((1, 2)), jnp.zeros(1))
    dl = abs(float(o1[1]) - float(o2[1]))
    print("\n同スピン交換: 符号 %+d -> %+d,  log|psi| の差 %.1e（|log|psi|| = %.1f, 相対 %.1e）"
          % (np.sign(o1[0]), np.sign(o2[0]), dl, abs(float(o1[1])), dl / abs(float(o1[1]))))
    print("W_V:", "生成されない" if not any('v_w' in str(k)
          for k, _ in jax.tree_util.tree_flatten_with_path(p)[0]) else "あり")


if __name__ == "__main__":
    main()
