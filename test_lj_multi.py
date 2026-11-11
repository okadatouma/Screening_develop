# -*- coding: utf-8 -*-
"""λ-jellium の多点学習（HF なし、1 本のネットワークで多くの (r_s, lam)、periodicwave/lj_multi.py）の検算（CPU、倍精度）。

    JAX_PLATFORMS=cpu PYTHONPATH=. python test_lj_multi.py

1. 多点の網（入力 12 成分 + (r_s, lam)、HF なしの軌道、ウォーカーごとのカスプ）の E_L を、差分の力ずく計算と比べる
   （folx・default、ウォーカーごとに違う (r_s, lam)、Coulomb なし）
2. ウォーカーごとの E_L（per_walker）が、同じ (r_s, lam) を定数で渡した 1 点の E_L と一致する（Coulomb あり）
3. 点ごとに中心を取る勾配（loss の num_groups）が、点ごとの 1 点の勾配の平均と一致する（E_L の切り詰めを含む）
4. 事前学習の的（相互作用のない下のバンドの軌道 e^{ik·r} u_k(s; lam)）の行列式は、Coulomb なしの H の固有状態
   （E_L = sum_k k^2/2、どの配置・擬スピンでも同じ）
5. 事前学習で、網の軌道が的に近づく（loss が下がり、行列式の比 psi_net/psi_0 のばらつきが小さくなる）
"""
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import kfac_jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from periodicwave import networks, CustomScreenformer as CS, envelopes, lj_multi
from periodicwave import loss as qmc_loss
from periodicwave.pbc import lambda_jellium as LJ, lattices, feature_layer as FL

A_WC = np.sqrt(2 * np.pi / np.sqrt(3))
ATOMS = jnp.zeros((1, 2))
FAILS = []


def check(name, ok, detail=""):
    print(("  ok    " if ok else "  FAIL  ") + name + ("   " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


def cell(n):
    lat, crys = lattices._triangular_lattice_vecs_periodic_potential(A_WC, n)
    return jnp.asarray(lat), CS.crystal_harmonics(lat, crys)


def build_multi_net(lat, harm, n, center, scale, heads=2, det=2, seed=3):
    feat = FL.make_pbc_feature_layer(natoms=1, nspins=(n, 0), ndim=2, lattice=lat, include_r_ae=False)
    net = CS.make_fermi_net(
        (n, 0), jnp.zeros(1), ndim=2, determinants=det, envelope=envelopes.make_null_envelope(),
        feature_layer=feat, jastrow='NONE', jastrow_kwargs={'ndim': 2}, complex_output=True,
        bias_orbitals=False, num_layers=1, num_heads=heads, attn_dim=8, value_dim=8, mlp_dim=16,
        num_perceptrons_per_layer=1, use_layer_norm=True, mlp_activation_fct='GELU', pbc_lattice=lat,
        init_softmask_w_factor=0.7, init_softmask_w_spread=1.5,
        input_mode='lj12', input_harmonics=harm, input_init='random', trim_fixed=True,
        input_cond=True, cond_center=center, cond_scale=scale,
        lj_net_only=True, lj_per_walker=True, lj_delta_mult=1.0, lj_net_scale=1.0, lj_jastrow=True)
    return net, net.init(jax.random.PRNGKey(seed))


def data_of(x, s, ch):
    return networks.FermiNetData(positions=jnp.asarray(x), spins=jnp.asarray(s, jnp.float64),
                                 atoms=ATOMS, charges=jnp.asarray(ch, jnp.float64))


def e_l_multi(f, lat, method, coulomb):
    return LJ.local_energy(f, jnp.zeros(1), (0, 0), complex_output=True, lattice=lat,
                           potential_type='Coulomb' if coulomb else 'none',
                           kinetic_kwargs={'laplacian_method': method}, per_walker=True, delta_mult=1.0)


def e_l_single(f, lat, r_s, lam, method, coulomb):
    return LJ.local_energy(f, jnp.zeros(1), (0, 0), complex_output=True, lattice=lat, lam=lam, delta=r_s,
                           potential_type='Coulomb' if coulomb else 'none',
                           potential_kwargs={'interaction_energy_scale': r_s},
                           kinetic_kwargs={'laplacian_method': method})


def brute_force_e_l(f, params, x, s, ch, h=2e-3):
    """[H Psi]/Psi（Coulomb なし）を中心差分で（4 次精度）。r_s, lam はウォーカーの charges から、Delta = r_s。"""
    x = np.asarray(x, np.float64); s = np.asarray(s, np.float64)
    r_s, lam = float(ch[0]), float(ch[1]); delta = r_s
    n = s.shape[0]

    def logpsi(xx, ss):
        ph, la = f(params, jnp.asarray(xx), jnp.asarray(ss), ATOMS, jnp.asarray(ch))
        return complex(la) + 1j * complex(ph)

    L0 = logpsi(x, s)
    rel = lambda xx, ss: np.exp(logpsi(xx, ss) - L0)
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


def test_brute_force():
    print("[1] 多点の網（N = 3）の E_L を差分の力ずく計算と比べる（ウォーカーごとに違う (r_s, lam)）")
    lat, harm = cell(3)
    net, params = build_multi_net(lat, harm, 3, center=(15.0, 0.55), scale=(5.0, 0.25))
    nin = params['layers']['embed'].shape[0]
    check(f"入力の次元 {nin} = 4 + 2 x {len(harm)} + 2 + 2（(r_s, lam)）、HF の項なし",
          nin == 4 + 2 * len(harm) + 2 + 2)
    rng = np.random.default_rng(5)
    els = {m: jax.jit(e_l_multi(net.apply, lat, m, coulomb=False)) for m in ['folx', 'default']}
    for trial, ch in enumerate([(12.0, 0.72, 0.0), (21.0, 0.35, 1.0), (24.0, 0.70, 2.0)]):
        x = rng.normal(size=6)
        s = rng.choice([-1.0, 1.0], size=3)
        ref = brute_force_e_l(net.apply, params, x, s, ch)
        for m, el in els.items():
            got = complex(el(params, None, data_of(x, s, ch))[0])
            err = abs(got - ref)
            check(f"(r_s, lam) = {ch[:2]} s={s.astype(int).tolist()} {m:7s} E_L = {got:.4f}  (差分 {ref:.4f})",
                  err < 2e-5 * max(1.0, abs(ref)), f"|diff| {err:.1e}")
    # (r_s, lam) を変えると網の値が変わる（入力とカスプに入っている）
    x = jnp.asarray(rng.normal(size=6)); s = jnp.asarray([1.0, -1.0, 1.0])
    a = net.apply(params, x, s, ATOMS, jnp.asarray([12.0, 0.72, 0.0]))[1]
    b = net.apply(params, x, s, ATOMS, jnp.asarray([21.0, 0.35, 1.0]))[1]
    check(f"(r_s, lam) で log|psi| が変わる（{float(a):.4f} / {float(b):.4f}）", abs(float(a - b)) > 1e-6)


def test_per_walker_equals_single():
    print("[2] ウォーカーごとの E_L ＝ 同じ (r_s, lam) を定数で渡した 1 点の E_L（Coulomb あり、N = 9）")
    lat, harm = cell(9)
    net, params = build_multi_net(lat, harm, 9, center=(18.0, 0.55), scale=(8.0, 0.25))
    rng = np.random.default_rng(9)
    for method in ['folx', 'default']:
        multi = jax.jit(e_l_multi(net.apply, lat, method, coulomb=True))
        for ch in [(12.0, 0.72, 0.0), (21.0, 0.35, 1.0)]:
            single = jax.jit(e_l_single(net.apply, lat, ch[0], ch[1], method, coulomb=True))
            worst = 0.0
            for _ in range(2):
                x = rng.uniform(-4, 4, size=18)
                s = rng.choice([-1.0, 1.0], size=9)
                d = data_of(x, s, ch)
                e1, c1 = multi(params, None, d)
                e2, c2 = single(params, None, d)
                worst = max(worst, abs(complex(e1 - e2)) / max(1.0, abs(complex(e2))), float(jnp.max(jnp.abs(c1 - c2))))
            check(f"{method:7s} (r_s, lam) = {ch[:2]}：相対差の最大 {worst:.1e}", worst < 1e-12)


def test_group_gradient():
    print("[3] 点ごとに中心を取る勾配 ＝ 点ごとの 1 点の勾配の平均（切り詰めつき、3 点 x 4 ウォーカー、N = 3）")
    lat, harm = cell(3)
    net, params = build_multi_net(lat, harm, 3, center=(15.0, 0.55), scale=(5.0, 0.25))
    pts = [(11.0, 0.74), (19.0, 0.35), (24.0, 0.70)]
    per = 4
    rng = np.random.default_rng(21)
    x = jnp.asarray(rng.normal(size=(len(pts) * per, 6)))
    s = jnp.asarray(rng.choice([-1.0, 1.0], size=(len(pts) * per, 3)))
    ch = jnp.asarray(lj_multi.walker_params(np.asarray(pts), len(pts) * per))
    atoms = jnp.zeros((len(pts) * per, 1, 2))

    def log_network(*a):
        ph, la = net.apply(*a)
        return la + 1j * ph

    key = jax.random.PRNGKey(0)
    for clip in (5.0, 0.5):           # 0.5 なら実際に切り詰めが効く
        multi = qmc_loss.make_loss(log_network, e_l_multi(net.apply, lat, 'folx', coulomb=True),
                                   clip_local_energy=clip, clip_from_median=False, center_at_clipped_energy=True,
                                   complex_output=True, num_groups=len(pts))
        (_, aux), g_multi = jax.value_and_grad(multi, has_aux=True)(
            params, key, networks.FermiNetData(positions=x, spins=s, atoms=atoms, charges=ch))
        grads = []
        for p, (r_s, lam) in enumerate(pts):
            sl = slice(p * per, (p + 1) * per)
            single = qmc_loss.make_loss(log_network, e_l_single(net.apply, lat, r_s, lam, 'folx', coulomb=True),
                                        clip_local_energy=clip, clip_from_median=False,
                                        center_at_clipped_energy=True, complex_output=True)
            grads.append(jax.value_and_grad(single, has_aux=True)(
                params, key, networks.FermiNetData(positions=x[sl], spins=s[sl], atoms=atoms[sl], charges=ch[sl]))[1])
        g_avg = jax.tree_util.tree_map(lambda *g: sum(g) / len(g), *grads)
        num = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(jax.tree_util.tree_leaves(g_multi),
                                                                 jax.tree_util.tree_leaves(g_avg)))
        den = max(float(jnp.max(jnp.abs(b))) for b in jax.tree_util.tree_leaves(g_avg))
        nclip = int(jnp.sum(jnp.abs(aux.clipped_energy - aux.local_energy) > 1e-12))
        # 1 点の loss は Σ c O* と (平均) Σ O を別々に足してから引くので、大きな数の打ち消しで相対 1e-9 ほどの丸めが出る
        # （式が違えば相対 O(1) の差になる）。
        check(f"切り詰め {clip}（切り詰めたウォーカー {nclip} 本）：差の最大 {num:.1e}（勾配の大きさ {den:.2e}）",
              num < 1e-8 * max(1.0, den))


def test_free_band_eigenstate():
    print("[4] 事前学習の的（相互作用のない下のバンド、N = 9 の第 1 ブリルアン域）は Coulomb なしの H の固有状態")
    lat, _ = cell(9)
    kpts = lj_multi.free_band_kpoints(3)
    want = float(0.5 * np.sum(kpts ** 2))

    def f(params, x, spins, atoms, charges):
        M = lj_multi.free_band_orbitals(x, spins, charges[1], kpts)
        sign, logdet = jnp.linalg.slogdet(M)
        return jnp.angle(sign), logdet

    rng = np.random.default_rng(3)
    for method in ['folx', 'default']:
        el = jax.jit(e_l_multi(f, lat, method, coulomb=False))
        for ch in [(12.0, 0.72, 0.0), (21.0, 0.35, 1.0)]:
            vals = []
            for _ in range(4):
                x = rng.uniform(-4, 4, size=18)
                s = rng.choice([-1.0, 1.0], size=9)
                vals.append(complex(el(None, None, data_of(x, s, ch))[0]))
            err = max(abs(v - want) for v in vals)
            check(f"{method:7s} (r_s, lam) = {ch[:2]}：E_L = {vals[0].real:.6f}{vals[0].imag:+.1e}i（sum k^2/2 = {want:.6f}）",
                  err < 1e-8 * max(1.0, want), f"最大の差 {err:.1e}")


def test_pretrain():
    print("[5] 事前学習（N = 9、2 点 x 4 ウォーカー、Adam 1500 step）で網の軌道が的に近づく")
    lat, harm = cell(9)
    net, params = build_multi_net(lat, harm, 9, center=(18.0, 0.55), scale=(8.0, 0.25), det=2, seed=4)
    pts = np.array([[12.0, 0.72], [21.0, 0.35]])
    nd = jax.local_device_count()
    per_dev = 2 * 4
    ch = kfac_jax.utils.replicate_all_local_devices(jnp.asarray(lj_multi.walker_params(pts, per_dev)))
    data = networks.FermiNetData(
        positions=kfac_jax.utils.broadcast_all_local_devices(jnp.zeros((nd, per_dev, 18))),
        spins=kfac_jax.utils.broadcast_all_local_devices(jnp.ones((nd, per_dev, 9))),
        atoms=kfac_jax.utils.replicate_all_local_devices(jnp.zeros((per_dev, 1, 2))), charges=ch)
    kpts = lj_multi.free_band_kpoints(3)
    rng = np.random.default_rng(8)
    xs = rng.uniform(-4, 4, size=(16, 18)); ss = rng.choice([-1.0, 1.0], size=(16, 9))

    def spread(p):
        """行列式の比 psi_net/psi_0 の |.| のばらつき（中央値で割った標準偏差）。一致すれば 0。"""
        out = []
        for c in ([12.0, 0.72, 0.0], [21.0, 0.35, 1.0]):
            r = []
            for x, s in zip(xs, ss):
                phi = net.orbitals(p, jnp.asarray(x), jnp.asarray(s), ATOMS, jnp.asarray(c), with_jastrow=False)[0]
                tot = jnp.sum(jnp.stack([jnp.linalg.det(m) for m in phi]))
                r.append(complex(tot / jnp.linalg.det(lj_multi.free_band_orbitals(jnp.asarray(x), jnp.asarray(s), c[1], kpts))))
            r = np.asarray(r)
            out.append(np.std(np.abs(r)) / np.median(np.abs(r)))
        return max(out)

    p0 = jax.tree_util.tree_map(lambda a: a[0], kfac_jax.utils.replicate_all_local_devices(params))
    before = spread(p0)
    p_rep = kfac_jax.utils.replicate_all_local_devices(params)
    p_rep = lj_multi.pretrain(p_rep, data, net, lat, 3, steps=1500, lr=3e-3, key=jax.random.PRNGKey(1), log_every=500)
    after = spread(jax.tree_util.tree_map(lambda a: a[0], p_rep))
    check(f"|psi_net/psi_0| のばらつき {before:.2f} → {after:.3f}", after < 0.1 * before)


if __name__ == "__main__":
    test_brute_force()
    test_per_walker_equals_single()
    test_group_gradient()
    test_free_band_eigenstate()
    test_pretrain()
    print()
    print("ALL PASSED" if not FAILS else f"FAILED: {len(FAILS)}")
    for f in FAILS:
        print("  - " + f)
    sys.exit(1 if FAILS else 0)
