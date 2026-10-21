# -*- coding: utf-8 -*-
"""軌道と行列式の作り方の検算（doc/orbitals_det.pdf の数値）。N = 25（13 上 + 12 下）、run 1929 と同じ網の形。

1. 各段の形（特徴 5、埋め込み 64、16 ヘッド x 16、軌道の出力 25 x 4 x 2 = 200、行列 4 枚の 25 x 25 複素）とパラメータ数
2. 電子 p と q を入れ替える（位置とスピンを一緒に）と、4 枚の行列の行 p と q が入れ替わり、psi の符号が変わる
3. アテンションの枝を 0 にすると（e^{s_O} -> 0）、行 i は電子 i だけの関数（他の電子を動かしても変わらない）
4. ヤコビの公式 d ln det M / d M_ij = (M^{-1})_ji
5. 同じ位置に同じスピンの電子が 2 個あると行が一致して psi = 0
6. 行列の形：全電子の h_i を縦に並べた Z（25 x 64）と、W_orb の列を並べ直した複素の W^k（64 x 25）で M^k = Z W^k

    JAX_PLATFORMS=cpu PYTHONPATH=. python screening_analysis/check_orbitals_det.py
"""
import os, sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from periodicwave import CustomScreenformer as CS, envelopes, network_blocks, networks
from periodicwave.pbc import lattices, feature_layer as FL

A_WC = np.sqrt(2 * np.pi / np.sqrt(3))
# run 1929 と同じ格子（config.json の pbc_lattice。列が格子ベクトル、2 本のなす角 120 度）
lat = jnp.asarray([[9.52312807, -4.76156403], [0.0, 8.24727083]])
N, NUP = 25, 13
dref = CS._max_periodic_distance(np.asarray(lat))
lo, hi = A_WC, dref
feat = FL.make_pbc_feature_layer(natoms=1, nspins=(N, 0), ndim=2, lattice=lat, include_r_ae=False)
net = CS.make_fermi_net(
    (NUP, N - NUP), jnp.zeros(1), ndim=2, determinants=4, envelope=envelopes.make_null_envelope(),
    feature_layer=feat, jastrow='NONE', jastrow_kwargs={'ndim': 2}, complex_output=True, bias_orbitals=False,
    num_layers=1, num_heads=16, attn_dim=16, value_dim=16, mlp_dim=64, num_perceptrons_per_layer=2,
    use_layer_norm=True, mlp_activation_fct='GELU', pbc_lattice=lat,
    init_softmask_w_factor=float(np.sqrt(lo * hi) / dref), init_softmask_w_spread=float(np.sqrt(hi / lo)))
params = net.init(jax.random.PRNGKey(0))
atoms = jnp.zeros((1, 2)); charges = jnp.zeros(1)
spins = jnp.array([1.0] * NUP + [-1.0] * (N - NUP))
rng = np.random.default_rng(0)
pos = jnp.asarray(rng.uniform(-4, 4, size=2 * N))

print(f"d_ref = {dref:.4f}, 入れ子窓の幅 w_h = {lo:.3f} .. {hi:.3f}")
print("1. 形とパラメータ数")
L = params['layers']
print("   埋め込み W_E", L['embed'].shape, "  （特徴 4 + スピン 1 = 5 -> 64）")
at = L['attention'][0]
for k in ('q_w', 'k_w', 'v_w', 'g_w', 'attn_output'):
    print(f"   {k:12s}", at[k].shape)
print("   MLP", [p['w'].shape for p in L['mlp'][0]])
print("   軌道の線形層 W_orb", params['orbital'][0]['w'].shape, " bias" if 'b' in params['orbital'][0] else " （バイアスなし）")
cnt = lambda t: sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(t))
print(f"   パラメータ数：全体 {cnt(params)}、うち埋め込み {cnt(L['embed'])}、アテンション {cnt(L['attention'])}、"
      f"LayerNorm {cnt(L['ln'])}、MLP {cnt(L['mlp'])}、軌道 {cnt(params['orbital'])}")
M = net.orbitals(params, pos, spins, atoms, charges)[0]
print("   行列 M の形", M.shape, M.dtype)
ph, la = net.apply(params, pos, spins, atoms, charges)
psi = jnp.exp(la + 1j * ph)
psi_direct = sum(jnp.linalg.det(M[k]) for k in range(4))
print(f"   psi（logdet_matmul）= {complex(psi):.6e}  /  sum_k det M^k = {complex(psi_direct):.6e}")

print("2. 電子 p, q の入れ替え（同じスピンどうし、違うスピンどうし）")
for p, q in ((0, 5), (2, 20)):
    perm = np.arange(N); perm[[p, q]] = perm[[q, p]]
    pos2 = pos.reshape(N, 2)[perm].reshape(-1); sp2 = spins[perm]
    M2 = net.orbitals(params, pos2, sp2, atoms, charges)[0]
    rows_swapped = float(jnp.abs(M2 - M[:, perm, :]).max())
    ph2, la2 = net.apply(params, pos2, sp2, atoms, charges)
    print(f"   ({p},{q}) スピン {int(spins[p]):+d},{int(spins[q]):+d}: 行が入れ替わっただけ（ずれ {rows_swapped:.1e}）、"
          f"psi'/psi = {complex(jnp.exp(la2 - la + 1j * (ph2 - ph))):.6f}")

print("3. アテンションの枝を切る（e^{s_O} = 0）")
p0 = jax.tree_util.tree_map(lambda x: x, params)
p0['layers']['attention'][0]['out_scale'] = jnp.asarray(-1e9)
Ma = net.orbitals(p0, pos, spins, atoms, charges)[0]
pos3 = pos.at[2 * 7:2 * 7 + 2].add(jnp.array([0.37, -0.21]))        # 電子 7 を動かす
Mb = net.orbitals(p0, pos3, spins, atoms, charges)[0]
others = [i for i in range(N) if i != 7]
print(f"   電子 7 を動かしても、他の行の変化 {float(jnp.abs(Mb[:, others] - Ma[:, others]).max()):.1e}、行 7 の変化 "
      f"{float(jnp.abs(Mb[:, 7] - Ma[:, 7]).max()):.2e}")
Mc = net.orbitals(params, pos3, spins, atoms, charges)[0]
print(f"   （枝があると他の行も変わる：{float(jnp.abs(Mc[:, others] - M[:, others]).max()):.2e}）")

print("4. ヤコビの公式")
Mk = M[0]
g = jax.grad(lambda A: jnp.log(jnp.abs(jnp.linalg.det(A))) , holomorphic=False)
# 複素の行列：実部を少し動かして数値微分
e = 1e-6; i, j = 3, 8
dM = jnp.zeros_like(Mk).at[i, j].set(e)
num = (jnp.log(jnp.linalg.det(Mk + dM)) - jnp.log(jnp.linalg.det(Mk - dM))) / (2 * e)
print(f"   d ln det M / d M_{{{i}{j}}}：差分 {complex(num):.6f}、(M^-1)_{{{j}{i}}} = {complex(jnp.linalg.inv(Mk)[j, i]):.6f}")

print("5. 同じ位置・同じスピンの電子が 2 個")
pos5 = pos.at[2 * 4:2 * 4 + 2].set(pos[2 * 1:2 * 1 + 2])               # 電子 4 を電子 1 の位置へ（どちらも上スピン）
M5 = net.orbitals(params, pos5, spins, atoms, charges)[0]
print(f"   行 1 と行 4 の差 {float(jnp.abs(M5[:, 1] - M5[:, 4]).max()):.1e}、|det M^k| = "
      + ", ".join(f"{float(abs(jnp.linalg.det(M5[k]))):.1e}" for k in range(4)) + f"（もとは {float(abs(jnp.linalg.det(M[0]))):.1e}）")

print("6. 行列の形 M^k = Z W^k")
ae, ee, r_ae, r_ee = networks.make_periodic_construct_input_features(lat)(pos, atoms, ndim=2)
_, lay_apply = CS.make_screenformer_layers((NUP, N - NUP), 1, net.options)
Hm = lay_apply(params['layers'], ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, spins=spins, charges=charges)   # 25 x 64
W = params['orbital'][0]['w']                                                                         # 64 x 200
err = 0.0
for k in range(4):
    Wk = W[:, [2 * (25 * k + j) for j in range(25)]] + 1j * W[:, [2 * (25 * k + j) + 1 for j in range(25)]]
    err = max(err, float(jnp.abs(M[k] - Hm @ Wk).max()))
print(f"   Z {Hm.shape}、W_orb {W.shape}、max|M^k - Z W^k| = {err:.1e}")
