"""N=9 の三角 Wigner 結晶を「解析的に」評価する — Madelung + 調和フォノン零点振動。

リポジトリ本体の Ewald（make_2DCoulomb_potential）をそのまま使い、
  E_harm(r_s) = r_s·V1 + (1/2)·sqrt(r_s)·Σ'√λ_i
を出す。V1 = 結晶配置での静電エネルギー（scale=1）、λ_i は Hessian の固有値
（並進ゼロモード 2 個を除く）。コードのハミルトニアンと同一の関数を微分するので
単位の取り違えが起きない。これを NN の実測エネルギーと同じコード単位で直接比べる。
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, "/home/claude/run")
from periodicwave import networks                      # noqa: E402
from periodicwave.pbc import hamiltonian as HAM        # noqa: E402

jax.config.update("jax_enable_x64", True)

L = jnp.array([[5.71387684, -2.85693842],
               [0.0,         4.9483625]])
N = 9
pot = HAM.make_2DCoulomb_potential(L, jnp.array([0.0]), jnp.zeros(1), 5, 1.0)  # scale = 1
atoms = jnp.zeros((1, 2))


def V1(flat):
    """診断用（値のみ正しい。対角の 0 距離が autodiff で NaN を生むので微分には使わない）"""
    ae, ee, _, _ = networks.construct_input_features(flat, atoms, ndim=2)
    return jnp.real(pot(ae, ee))


def Vsafe(flat):
    """微分安全版。対角の変位をダミーの非ゼロベクトルに置き換える。
    pot は対角の ewald 値を明示的に 0 に潰すので、値も勾配も対角に依存しない。"""
    R = flat.reshape(N, 2)
    ee = R[:, None, :] - R[None, :, :]
    ee = ee.at[jnp.arange(N), jnp.arange(N)].set(jnp.array([1.0, 0.7]))
    return jnp.real(pot(jnp.zeros((N, 1, 2)), ee))


# 3x3 三角部分格子（格子ベクトルは L の列）
a1, a2 = L[:, 0] / 3.0, L[:, 1] / 3.0
pos = jnp.array([m * a1 + n * a2 for m in range(3) for n in range(3)]).reshape(-1)

v1 = float(V1(pos))
v1s = float(Vsafe(pos))
assert abs(v1 - v1s) < 1e-9 * abs(v1), (v1, v1s)
g = np.asarray(jax.grad(Vsafe)(pos))
H = np.asarray(jax.hessian(Vsafe)(pos))
lam = np.linalg.eigvalsh(H)
nzero = int((np.abs(lam) < 1e-8 * np.abs(lam).max()).sum())
S = float(np.sqrt(lam[lam > 1e-8 * lam.max()]).sum())

print("V1(静電, scale=1)      = %.6f   （粒子あたり %.6f）" % (v1, v1 / N))
print("|grad| max             = %.2e   （0 なら格子は停留点）" % np.abs(g).max())
print("Hessian 固有値: 最小 %.3e / ゼロモード %d 個 / Σ'√λ = %.6f" % (lam.min(), nzero, S))
print("Madelung 係数 −V1/N     = %.6f   （B–M 熱力学極限: 1.106103 Ry = 0.553051 Ha）" % (-v1 / N))
print("零点係数 Σ'√λ/(2N)     = %.6f" % (S / (2 * N)))
print()

# NN 実測（det 系列, ewmean, コード単位の全エネルギー）
nn = {1: -1.1485, 5: -33.264, 20: -166.058, 30: -256.924, 35: -304.997, 40: -350.705}

print("  r_s     E_harm(コード)   E_NN(det実測)    差[%]     E_harm/N/r_s^2")
for rs in [1, 5, 10, 20, 25, 30, 31, 35, 40, 60, 100]:
    eh = rs * v1 + 0.5 * np.sqrt(rs) * S
    row = "  %-6g  %12.3f " % (rs, eh)
    if rs in nn:
        row += "  %12.3f   %+6.1f%%" % (nn[rs], 100 * (nn[rs] - eh) / abs(eh))
    else:
        row += "  %12s   %7s" % ("-", "-")
    row += "   %10.6f" % (eh / N / rs ** 2)
    print(row)

# 交点: E_harm と NN 点列の交差（コード単位のまま補間）
rs_pts = np.array(sorted(nn))
e_nn = np.array([nn[r] for r in rs_pts])
rr = np.linspace(1, 40, 2000)
e_h = rr * v1 + 0.5 * np.sqrt(rr) * S
e_n = np.interp(rr, rs_pts, e_nn)
d = e_h - e_n
sign = np.sign(d)
cross = rr[np.where(np.diff(sign) != 0)[0]]
print("\nE_harm が NN 実測を下回り始める r_s =", np.round(cross, 2) if len(cross) else "なし（全域で上/下）")
print("（NN は 400 step の変分上界＝真値より高い。調和は大 r_s で漸近的に正確）")
