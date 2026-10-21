# [SCREEN] Quick correctness checks for CustomScreenformer vs CustomPsiformer.
#
# Checks (both networks, 2DEG setup as in run_2deg.py / run_2deg_screen.py):
#   1. forward pass is finite
#   2. antisymmetry under exchange of two same-spin electrons
#      (log|psi| unchanged, phase flips by pi)
#   3. periodicity: translating all electrons by a supercell lattice vector
#      leaves the wavefunction unchanged
#   4. gradient wrt positions is finite
#   5. folx forward Laplacian works (this is what the local energy uses)
#
# Run:  python test_screenformer.py

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "float32")

from periodicwave.pbc import lattices
from periodicwave.pbc import feature_layer as pbc_feature_layer
from periodicwave import envelopes
from periodicwave import CustomPsiformer
from periodicwave import CustomScreenformer

nspins = (2, 0)
ne = sum(nspins)
ndim = 2

supercell_a = np.sqrt(2 * np.pi / np.sqrt(3) * ne)
lattice, _ = lattices._triangular_lattice_vecs_periodic_potential(supercell_a, 1)
lattice = jnp.asarray(lattice)

feature_layer = pbc_feature_layer.make_pbc_feature_layer(
    natoms=1, nspins=nspins, ndim=ndim, lattice=lattice, include_r_ae=False)
envelope = envelopes.make_null_envelope()

atoms = jnp.array([[0.0, 0.0]])
charges = jnp.array([0.0])
spins = jnp.concatenate((jnp.ones(nspins[0]), -jnp.ones(nspins[1])))

common = dict(
    ndim=ndim, determinants=4, envelope=envelope, feature_layer=feature_layer,
    jastrow="NONE", jastrow_kwargs={'ndim': ndim}, complex_output=True,
    bias_orbitals=False, num_layers=4, num_heads=4, attn_dim=16, value_dim=16,
    mlp_dim=64, num_perceptrons_per_layer=2, use_layer_norm=True,
    mlp_activation_fct="GELU", pbc_lattice=lattice)

nets = {
    'CustomPsiformer': CustomPsiformer.make_fermi_net(nspins, charges, **common),
    'CustomScreenformer': CustomScreenformer.make_fermi_net(
        nspins, charges, **common,
        use_softmask=True, use_gate=True, use_tanh_norm=True,
        init_trim_r=1.0, init_softmask_w_factor=0.8,
        init_softmask_w_spread=2.5, use_output_scale=True),
}

d_ref = CustomScreenformer._max_periodic_distance(lattice)
print(f"lattice a = {supercell_a:.4f}, max periodic distance d_ref = {d_ref:.4f}")

key = jax.random.PRNGKey(23)
pos = jax.random.uniform(key, (ne * ndim,)) * supercell_a

for name, net in nets.items():
    print(f"\n=== {name} ===")
    params = net.init(jax.random.PRNGKey(7))

    if name == 'CustomScreenformer':
        att0 = params['layers']['attention'][0]
        r0 = 2.0 * jax.nn.sigmoid(att0['trim_rho'])
        w0 = d_ref * jax.nn.softplus(att0['mask_omega'])
        print(f"  init trim r per head      : {np.asarray(r0)}")
        print(f"  init screening w per head : {np.asarray(w0)}  (d_ref={d_ref:.3f})")

    # NOTE: for complex_output=True the first return value is the phase ANGLE
    # theta of psi (see network_blocks.logdet_matmul), not exp(i theta).
    theta, logabs = net.apply(params, pos, spins, atoms, charges)
    print(f"  psi: theta={float(theta):+.6f} rad, log|psi|={float(logabs):+.6f}",
          "finite OK" if np.isfinite(float(logabs)) else "NOT FINITE")

    def angle_dist(a, b):
        """distance between angles mod 2*pi"""
        d = (float(a) - float(b)) % (2 * np.pi)
        return min(d, 2 * np.pi - d)

    # --- antisymmetry (swap the two same-spin electrons) ---
    pos_swap = jnp.concatenate([pos[ndim:2*ndim], pos[0:ndim]])
    theta_s, logabs_s = net.apply(params, pos_swap, spins, atoms, charges)
    dlog = abs(float(logabs) - float(logabs_s))
    dth = angle_dist(theta_s, float(theta) + np.pi)
    print(f"  swap: dlog|psi|={dlog:.2e} (want 0), dtheta from pi={dth:.2e} (want 0)")
    assert dlog < 1e-4 and dth < 1e-4, "antisymmetry violated"

    # --- periodicity (translate everyone by lattice vector = column of A) ---
    shift = jnp.tile(lattice[:, 0], ne)
    theta_t, logabs_t = net.apply(params, pos + shift, spins, atoms, charges)
    dlog = abs(float(logabs) - float(logabs_t))
    dth = angle_dist(theta_t, theta)
    print(f"  lattice shift: dlog|psi|={dlog:.2e}, dtheta={dth:.2e} (want 0)")
    assert dlog < 1e-3 and dth < 1e-3, "periodicity violated"

    # --- gradient ---
    g = jax.grad(lambda p: net.apply(params, p, spins, atoms, charges)[1])(pos)
    print(f"  grad log|psi|: finite={bool(np.all(np.isfinite(np.asarray(g))))}, |g|={float(jnp.linalg.norm(g)):.4f}")

    # --- folx forward Laplacian (as used by the kinetic energy) ---
    from folx import forward_laplacian
    f = lambda p: net.apply(params, p, spins, atoms, charges)[1]
    lap_res = forward_laplacian(f)(pos)
    lap = float(lap_res.laplacian)
    print(f"  folx laplacian log|psi| = {lap:+.6f}, finite={np.isfinite(lap)}")
    assert np.isfinite(lap), "laplacian not finite"

print("\nAll checks passed.")
