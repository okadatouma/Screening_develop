# -*- coding: utf-8 -*-
"""value を電子平均に潰したとき、波動関数とエネルギーがどれだけ変わるか。

    python screening_analysis/value_ablation.py <run_folder> [--nconf 256]

value が「定数」なら、潰しても log|ψ| は配置によらない定数だけ動き、エネルギーは変わらない。
"""
import argparse, os, sys
import numpy as np, jax, jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from periodicwave import networks                        # noqa
import sampler as SM                                     # noqa
from energy_check import build_signed                    # noqa


def with_vmode(params, mode):
    p2 = jax.tree_util.tree_map(lambda x: x, params)
    for ap in p2["layers"]["attention"]:
        NH = ap["trim_rho"].shape[0]
        ap["_v_mode"] = jnp.asarray(np.full(NH, float(mode), dtype=np.float32))
    return p2


def main():
    a = argparse.ArgumentParser(); a.add_argument("folder")
    a.add_argument("--nconf", type=int, default=256); a.add_argument("--nwalk", type=int, default=64)
    a = a.parse_args()
    net, params, le, data, cfg = build_signed(a.folder)
    R, _ = SM.sample(a.folder, a.nconf, a.nwalk, verbose=False)
    ne = sum(cfg["system"]["electrons"]); n = len(R)
    sp = jnp.asarray(np.tile(np.asarray(data["spins"][0])[0], (n, 1)))
    at = jnp.asarray(np.tile(np.asarray(data["atoms"][0])[0], (n, 1, 1)))
    ch = jnp.asarray(np.tile(np.asarray(data["charges"][0])[0], (n, 1)))
    x = jnp.asarray(R.reshape(n, -1))
    dat = networks.FermiNetData(positions=x, spins=sp, atoms=at, charges=ch)
    key = jax.random.split(jax.random.PRNGKey(0), n)
    fE = jax.jit(jax.vmap(le, in_axes=(None, 0, 0)))
    fL = jax.jit(jax.vmap(lambda p, x_, s_, a_, c_: net.apply(p, x_, s_, a_, c_)[1],
                          in_axes=(None, 0, 0, 0, 0)))

    def energy(p, R_):
        n_ = len(R_)
        x_ = jnp.asarray(R_.reshape(n_, -1))
        dat_ = networks.FermiNetData(positions=x_, spins=sp[:n_], atoms=at[:n_], charges=ch[:n_])
        key_ = jax.random.split(jax.random.PRNGKey(1), n_)
        out = fE(p, key_, dat_)
        E = np.real(np.asarray(out[0] if isinstance(out, tuple) else out))
        lo, hi = np.percentile(E, [1, 99]); Ec = E[(E > lo) & (E < hi)]
        return Ec.mean(), Ec.std(), Ec.std() / np.sqrt(len(Ec))

    pa = with_vmode(params, 1.0)
    # 潰した波動関数は自分の |ψ|² からサンプルし直す（でないと変分量にならない）
    Ra, _ = SM.sample(a.folder, a.nconf, a.nwalk, verbose=False, params_override=pa)
    E0, s0, e0 = energy(params, R)
    E1, s1, e1 = energy(pa, Ra)
    la0 = np.asarray(fL(params, x, sp, at, ch)); la1 = np.asarray(fL(pa, x, sp, at, ch))
    d = la1 - la0
    print("  配置数 %d" % n)
    print("  元のモデル            : E = %.4f ± %.4f   局所エネルギーの標準偏差 %.3f" % (E0, e0, s0))
    print("  value を電子平均に潰す: E = %.4f ± %.4f   局所エネルギーの標準偏差 %.3f" % (E1, e1, s1))
    print("  ΔE = %+.4f  （1 電子あたり %+.5f, 相対 %+.3f%%）" % (E1 - E0, (E1 - E0) / ne, 100 * (E1 - E0) / abs(E0)))
    print("  Δlog|ψ| の配置間の標準偏差 %.3f   ← value が定数なら 0" % d.std())


if __name__ == "__main__":
    main()
