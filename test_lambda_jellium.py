# -*- coding: utf-8 -*-
"""λ-jellium の局所エネルギーと擬スピンのモンテカルロの検算（CPU、倍精度）。

    JAX_PLATFORMS=cpu PYTHONPATH=. python test_lambda_jellium.py

1. 1 粒子の固有状態：下のバンドのスピノル平面波で E_L = q^2/2、上のバンドで q^2/2 + Delta(1 + lam^2 q^2)
2. 自由フェルミオン（スピノル平面波の Slater 行列式、N = 3）で E_L が一定 = 軌道エネルギーの和
3. 乱数の CustomScreenformer（N = 3、複素）で E_L を差分の力ずく計算と比べる（folx / default の両方）
4. lam = 0 で従来の pbc.hamiltonian.local_energy と一致（全員 s = +1、Coulomb あり）
5. モンテカルロ：1 粒子で P(s = -1) = lam^2 q^2 / (1 + lam^2 q^2)、2 粒子で擬スピン配置の分布
"""
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from periodicwave import networks, mcmc, CustomScreenformer as CS, envelopes
from periodicwave.pbc import lambda_jellium as LJ, lattices, feature_layer as FL
from periodicwave.pbc import hamiltonian as PH

A_WC = np.sqrt(2 * np.pi / np.sqrt(3))       # 電子 1 個あたり面積 pi の三角格子の格子定数
ATOMS = jnp.zeros((1, 2)); CHARGES = jnp.zeros(1)
FAILS = []


def check(name, ok, detail=""):
    print(("  ok    " if ok else "  FAIL  ") + name + ("   " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


def cell(n):
    lat, _ = lattices._triangular_lattice_vecs_periodic_potential(A_WC, n)
    return jnp.asarray(lat)


def recip(lat):
    return 2 * np.pi * np.linalg.inv(np.asarray(lat))     # 行 i が b_i


def data_of(x, s):
    return networks.FermiNetData(positions=jnp.asarray(x), spins=jnp.asarray(s, jnp.float64),
                                 atoms=ATOMS, charges=CHARGES)


def spinor(q, lam, band):
    qp = q[0] + 1j * q[1]; qm = q[0] - 1j * q[1]
    return (1.0 + 0j, lam * qp) if band == 'lower' else (-lam * qm, 1.0 + 0j)


def make_det_wf(ks, lam, bands):
    """det[ u_k(s_i) exp(i k.r_i) ]。ks: (N,2)、bands: 各軌道の 'lower' / 'upper'。"""
    ks = jnp.asarray(ks)
    U = jnp.array([spinor(np.asarray(k), lam, b) for k, b in zip(ks, bands)])   # (N, 2)

    def f(params, x, spins, atoms, charges):
        r = x.reshape(-1, 2)
        up = (spins > 0)[:, None]
        amp = jnp.where(up, U[None, :, 0], U[None, :, 1])          # (i, k)
        M = amp * jnp.exp(1j * r @ ks.T)
        sign, logdet = jnp.linalg.slogdet(M)
        return jnp.angle(sign), logdet
    return f


def orbital_energy(k, lam, delta, band):
    q2 = float(np.dot(k, k))
    return 0.5 * q2 + (delta * (1 + lam ** 2 * q2) if band == 'upper' else 0.0)


def e_l_fn(f, lat, lam, delta, method, coulomb=False, r_s=1.0):
    return LJ.local_energy(
        f, CHARGES, (0, 0), complex_output=True, lattice=lat, lam=lam, delta=delta,
        potential_type='Coulomb' if coulomb else 'none',
        potential_kwargs={'interaction_energy_scale': r_s},
        kinetic_kwargs={'laplacian_method': method})


def test_single_particle():
    print("[1] 1 粒子のスピノル平面波")
    lat = cell(1); B = recip(lat)
    lam, delta = 0.72, 12.0
    rng = np.random.default_rng(0)
    for m in [(1, 0), (1, 1), (2, -1), (-1, 3)]:
        q = m[0] * B[0] + m[1] * B[1]
        for band in ['lower', 'upper']:
            f = make_det_wf([q], lam, [band])
            want = orbital_energy(q, lam, delta, band)
            for method in ['folx', 'default']:
                el = jax.jit(e_l_fn(f, lat, lam, delta, method))
                got = []
                for s in (+1.0, -1.0):
                    x = rng.normal(size=2)
                    got.append(complex(el(None, None, data_of(x, [s]))[0]))
                err = max(abs(g - want) for g in got)
                check(f"q=m{m} {band:5s} {method:7s} E_L={got[0].real:.6f}{got[0].imag:+.1e}i (want {want:.6f})",
                      err < 1e-8 * max(1, abs(want)), f"err {err:.1e}")


def test_free_fermions():
    print("[2] 自由フェルミオン（スピノル平面波の行列式、N = 3）")
    lat = cell(9); B = recip(lat)
    lam, delta = 0.6, 7.0
    ks = [B[0] + B[1], B[0], B[1]]
    rng = np.random.default_rng(1)
    for bands in (['lower'] * 3, ['lower', 'upper', 'lower']):
        f = make_det_wf(ks, lam, bands)
        want = sum(orbital_energy(k, lam, delta, b) for k, b in zip(ks, bands))
        for method in ['folx', 'default']:
            el = jax.jit(e_l_fn(f, lat, lam, delta, method))
            vals = []
            for _ in range(6):
                x = rng.normal(size=6) * 2
                s = rng.choice([-1.0, 1.0], size=3)
                vals.append(complex(el(None, None, data_of(x, s))[0]))
            err = max(abs(v - want) for v in vals)
            check(f"bands={bands} {method:7s} E_L = {vals[0].real:.6f} (want {want:.6f})",
                  err < 1e-7 * max(1, abs(want)), f"max err over 6 configs {err:.1e}")


def build_net(lat, n, heads=2, layers=1, det=2, seed=3):
    feat = FL.make_pbc_feature_layer(natoms=1, nspins=(n, 0), ndim=2, lattice=lat, include_r_ae=False)
    net = CS.make_fermi_net(
        (n, 0), CHARGES, ndim=2, determinants=det, envelope=envelopes.make_null_envelope(),
        feature_layer=feat, jastrow='NONE', jastrow_kwargs={'ndim': 2}, complex_output=True,
        bias_orbitals=False, num_layers=layers, num_heads=heads, attn_dim=8, value_dim=8, mlp_dim=16,
        num_perceptrons_per_layer=1, use_layer_norm=True, mlp_activation_fct='GELU', pbc_lattice=lat,
        init_softmask_w_factor=0.7, init_softmask_w_spread=1.5)
    params = net.init(jax.random.PRNGKey(seed))
    return net, params


def brute_force_e_l(f, params, x, s, lam, delta, h=2e-3):
    """[H Psi]/Psi を中心差分で（4 次精度）。"""
    x = np.asarray(x, np.float64); s = np.asarray(s, np.float64)
    n = s.shape[0]

    def logpsi(xx, ss):
        ph, la = f(params, jnp.asarray(xx), jnp.asarray(ss), ATOMS, CHARGES)
        return complex(la) + 1j * complex(ph)

    L0 = logpsi(x, s)
    rel = lambda xx, ss: np.exp(logpsi(xx, ss) - L0)            # Psi(xx, ss) / Psi(x, s)
    e = 0j
    for i in range(n):
        c = 0.5 + delta * lam ** 2 * (s[i] > 0)
        lap = 0j
        for k in (2 * i, 2 * i + 1):
            d = np.zeros_like(x); d[k] = h
            lap += (-rel(x + 2 * d, s) + 16 * rel(x + d, s) - 30 + 16 * rel(x - d, s) - rel(x - 2 * d, s)) / (12 * h * h)
        e += -c * lap + delta * (s[i] < 0)
        si = s.copy(); si[i] = -si[i]
        der = []
        for k in (2 * i, 2 * i + 1):
            d = np.zeros_like(x); d[k] = h
            der.append((-rel(x + 2 * d, si) + 8 * rel(x + d, si) - 8 * rel(x - d, si) + rel(x - 2 * d, si)) / (12 * h))
        e += 1j * delta * lam * (der[0] - 1j * s[i] * der[1])
    return e


def test_network_vs_brute_force():
    print("[3] 乱数の CustomScreenformer（N = 3）で差分の力ずく計算と比較")
    lat = cell(3)
    net, params = build_net(lat, 3)
    lam, delta = 0.72, 12.0
    rng = np.random.default_rng(2)
    els = {m: jax.jit(e_l_fn(net.apply, lat, lam, delta, m)) for m in ['folx', 'default']}
    for trial in range(4):
        x = rng.normal(size=6)
        s = rng.choice([-1.0, 1.0], size=3)
        ref = brute_force_e_l(net.apply, params, x, s, lam, delta)
        for m, el in els.items():
            got = complex(el(params, None, data_of(x, s))[0])
            err = abs(got - ref)
            check(f"trial {trial} s={s.astype(int).tolist()} {m:7s} E_L = {got:.5f}  (差分 {ref:.5f})",
                  err < 2e-5 * max(1.0, abs(ref)), f"|diff| {err:.1e}")


def test_lambda_zero_matches_existing():
    print("[4] lam = 0 で従来の pbc.hamiltonian.local_energy と一致（Coulomb あり）")
    lat = cell(3)
    net, params = build_net(lat, 3, seed=5)
    r_s = 10.0
    old = PH.local_energy(net.apply, CHARGES, (3, 0), complex_output=True, lattice=lat,
                          potential_type='Coulomb', potential_kwargs={'interaction_energy_scale': r_s},
                          kinetic_kwargs={'laplacian_method': 'default'})
    new = LJ.local_energy(net.apply, CHARGES, (3, 0), complex_output=True, lattice=lat, lam=0.0,
                          potential_type='Coulomb', potential_kwargs={'interaction_energy_scale': r_s},
                          kinetic_kwargs={'laplacian_method': 'folx'})
    rng = np.random.default_rng(4)
    for trial in range(3):
        x = rng.normal(size=6)
        d = data_of(x, [1.0, 1.0, 1.0])
        a = complex(jax.jit(old)(params, None, d)[0]); b = complex(jax.jit(new)(params, None, d)[0])
        check(f"trial {trial}: 従来 {a:.6f}  新 {b:.6f}", abs(a - b) < 1e-8 * max(1, abs(a)), f"|diff| {abs(a-b):.1e}")
    # 擬スピンが混ざっていても lam = 0 なら 従来の運動エネルギー + Delta n_-
    s = [1.0, -1.0, -1.0]; d = data_of(rng.normal(size=6), s)
    a = complex(jax.jit(old)(params, None, d)[0]) + r_s * 2; b = complex(jax.jit(new)(params, None, d)[0])
    check(f"s = {s}: 従来 + 2 Delta = {a:.6f}  新 {b:.6f}", abs(a - b) < 1e-8 * max(1, abs(a)))


def test_mcmc_single():
    print("[5a] モンテカルロ、1 粒子：P(s=-1) = lam^2 q^2 / (1 + lam^2 q^2)")
    lat = cell(1); B = recip(lat)
    lam = 0.5
    for m in [(1, 0), (1, 1)]:
        q = m[0] * B[0] + m[1] * B[1]
        f = make_det_wf([q], lam, ['lower'])
        bn = jax.vmap(lambda p, x, s, a, c: f(p, x, s, a, c)[1], in_axes=(None, 0, 0, 0, 0))
        nw = 8192
        step = jax.jit(mcmc.make_mcmc_step(bn, nw, steps=20, ndim=2, spin_flips=1))
        d = networks.FermiNetData(positions=jnp.zeros((nw, 2)), spins=jnp.ones((nw, 1)),
                                  atoms=jnp.zeros((nw, 1, 2)), charges=jnp.zeros((nw, 1)))
        key = jax.random.PRNGKey(0)
        for _ in range(3):
            key, sk = jax.random.split(key)
            d, pm = step(None, d, sk, 0.3)
        frac = []
        for _ in range(10):
            key, sk = jax.random.split(key)
            d, pm = step(None, d, sk, 0.3)
            frac.append(float(jnp.mean(d.spins < 0)))
        q2 = float(np.dot(q, q)); want = lam ** 2 * q2 / (1 + lam ** 2 * q2)
        got = float(np.mean(frac)); err = np.std(frac) / np.sqrt(len(frac)) + 0.5 / np.sqrt(nw * 10)
        check(f"q=m{m}: P(s=-1) = {got:.4f} (want {want:.4f}), pmove {float(pm[0]):.2f} pflip {float(pm[1]):.2f}",
              abs(got - want) < 4 * err, f"4 sigma = {4*err:.4f}")


def test_mcmc_two():
    print("[5b] モンテカルロ、2 粒子の行列式：擬スピン配置 (s1, s2) の分布")
    lat = cell(4); B = recip(lat)
    lam = 0.8
    ks = [B[0], B[0] + B[1]]
    f = make_det_wf(ks, lam, ['lower', 'lower'])
    bn = jax.vmap(lambda p, x, s, a, c: f(p, x, s, a, c)[1], in_axes=(None, 0, 0, 0, 0))
    nw = 8192
    step = jax.jit(mcmc.make_mcmc_step(bn, nw, steps=10, ndim=2, spin_flips=2))
    key = jax.random.PRNGKey(1)
    d = networks.FermiNetData(positions=jax.random.normal(key, (nw, 4)) * 2, spins=jnp.ones((nw, 2)),
                              atoms=jnp.zeros((nw, 1, 2)), charges=jnp.zeros((nw, 1)))
    for _ in range(20):
        key, sk = jax.random.split(key)
        d, pm = step(None, d, sk, 0.5)
    counts = np.zeros(4)
    for _ in range(20):
        key, sk = jax.random.split(key)
        d, pm = step(None, d, sk, 0.5)
        idx = np.asarray((d.spins[:, 0] < 0) * 2 + (d.spins[:, 1] < 0), int)
        counts += np.bincount(idx, minlength=4)
    got = counts / counts.sum()
    # 平面波は直交するので int |Psi(R,S)|^2 dR ∝ sum_P prod_j |u_{k_P(j)}(s_j)|^2
    U = np.array([[abs(v) ** 2 for v in spinor(np.asarray(k), lam, 'lower')] for k in ks])   # (k, s)
    want = np.zeros(4)
    for a, s1 in enumerate([0, 1]):
        for b, s2 in enumerate([0, 1]):
            want[2 * a + b] = U[0, s1] * U[1, s2] + U[1, s1] * U[0, s2]
    want /= want.sum()
    err = np.sqrt(want * (1 - want) / (nw * 20)) * 3        # 自己相関の分を 3 倍で見込む
    check("P(++,+-,-+,--) = " + " ".join(f"{v:.4f}" for v in got) + "  want " + " ".join(f"{v:.4f}" for v in want),
          np.all(np.abs(got - want) < 4 * err + 1e-3), f"pmove {float(pm[0]):.2f} pflip {float(pm[1]):.2f}")


if __name__ == "__main__":
    test_single_particle()
    test_free_fermions()
    test_network_vs_brute_force()
    test_lambda_zero_matches_existing()
    test_mcmc_single()
    test_mcmc_two()
    print("\nALL PASSED" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
