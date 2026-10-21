# -*- coding: utf-8 -*-
"""複素の波動関数で、位相の運動エネルギーと Im E_L がどれだけあるかを測る。

    python screening_analysis/imag_check.py <結果フォルダ> [--nconf 128]

psi = exp(u + i phi) と書くと、運動エネルギーの局所値は
    E_kin = -1/2 [lap u + |grad u|^2 - |grad phi|^2] - i/2 [lap phi + 2 grad u . grad phi]
で、期待値は <E_kin> = 1/2 <|grad u|^2> + 1/2 <|grad phi|^2>（虚部の平均は 0）。
pbc.hamiltonian.local_energy の folx・複素の枝は Re E_L だけを返すので、勾配
2 Re<(E_L - E) O*> から <Im E_L Im O> が抜け、位相だけを動かす方向の勾配が 0 になる。
エネルギーの推定値そのもの（Re E_L の平均）は正しい。ここでは
  P = 1/2 <|grad phi|^2>（位相の運動エネルギー。実数の基底状態なら 0 まで下げられる）
  Im E_L の大きさ
を学習済みのチェックポイントで測る。
"""
import argparse, os, sys
import numpy as np, jax, jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import energy_check as EC                                  # noqa
from periodicwave import networks                          # noqa


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("folder")
    ap.add_argument("--nconf", type=int, default=128)
    a = ap.parse_args()
    net, params, le, data, cfg = EC.build_signed(a.folder)
    f = net.apply

    def one(x, s, at, ch):
        ph = lambda z: f(params, z, s, at, ch)[0]
        la = lambda z: f(params, z, s, at, ch)[1]
        gu, hu = jax.grad(la)(x), jax.hessian(la)(x)
        gp, hp = jax.grad(ph)(x), jax.hessian(ph)(x)
        lap_u, lap_p = jnp.trace(hu), jnp.trace(hp)
        re = -0.5 * (lap_u + gu @ gu - gp @ gp)
        im = -0.5 * (lap_p + 2 * gu @ gp)
        return jnp.stack([re, im, 0.5 * gp @ gp, 0.5 * gu @ gu])

    g = jax.jit(jax.vmap(one))
    pos = np.asarray(data["positions"][0]); sp = np.asarray(data["spins"][0])
    at = np.asarray(data["atoms"][0]); ch = np.asarray(data["charges"][0])
    n = min(a.nconf, pos.shape[0])
    out = []
    for i in range(0, n, 16):
        sl = slice(i, min(i + 16, n))
        out.append(np.asarray(g(jnp.asarray(pos[sl]), jnp.asarray(sp[sl]), jnp.asarray(at[sl]), jnp.asarray(ch[sl]))))
    X = np.concatenate(out)
    m = X.mean(0); se = X.std(0, ddof=1) / np.sqrt(len(X))
    print(f"{os.path.basename(os.path.normpath(a.folder))}: walkers {n}")
    print(f"  <Re E_kin>          {m[0]:10.4f} ± {se[0]:.4f}")
    print(f"  1/2<|grad u|^2>     {m[3]:10.4f} ± {se[3]:.4f}")
    print(f"  P = 1/2<|grad phi|^2> {m[2]:8.4f} ± {se[2]:.4f}   (P / <Re E_kin> = {m[2] / m[0]:.3f})")
    print(f"  Im E_kin: 平均 {m[1]:+.4f} ± {se[1]:.4f}, 標準偏差 {X[:, 1].std():.3f}   (Re E_kin の標準偏差 {X[:, 0].std():.3f})")


if __name__ == "__main__":
    main()
