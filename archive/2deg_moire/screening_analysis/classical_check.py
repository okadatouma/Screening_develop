"""古典極限での見積もり: nu=2/3 でハニカム配置がほかの「3 つに 2 つ」配置より
本当に得か、そしてその差が r_s とともにどう効くかを確かめる。

    python screening_analysis/classical_check.py <run_folder>
"""
import json, os, sys
import numpy as np, jax, jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import moire_analysis as MA
from periodicwave.pbc import hamiltonian as HAM


def patterns(d):
    """スーパーセル内の谷を 3 つの部分格子に分け、いくつかの配置を返す。"""
    L, MO = d["L"], d["MO"]
    site, lab = [], []
    seen = set()
    for m in range(-40, 41):
        for n in range(-40, 41):
            r = m * MO[:, 0] + n * MO[:, 1]
            f = np.mod(np.linalg.solve(L, r), 1.0)
            f = np.where((f < 1e-7) | (f > 1 - 1e-7), 0.0, f)
            k = (round(float(f[0]), 5), round(float(f[1]), 5))
            if k in seen: continue
            seen.add(k); site.append(f @ L.T); lab.append(((m + n) % 3, (m - n) % 3, m % 3))
    site = np.array(site); lab = np.array(lab)
    out = {}
    out["ハニカム (√3×√3)"] = site[np.isin(lab[:, 0], (0, 1))]
    out["縞 (m-n)"] = site[np.isin(lab[:, 1], (0, 1))]
    out["縞 (m)"] = site[np.isin(lab[:, 2], (0, 1))]
    rng = np.random.default_rng(0)
    for t in range(3):
        idx = rng.permutation(len(site))[:len(out["ハニカム (√3×√3)"])]
        out["でたらめ %d" % (t + 1)] = site[idx]
    return out


def main(folder):
    d = MA.load(folder)
    L, MO, ne, rs, U, phi = d["L"], d["MO"], d["ne"], d["rs"], d["U"], d["phi"]
    pot = HAM.make_2DCoulomb_potential(jnp.asarray(L), ne, 1, interaction_energy_scale=rs)
    mo = HAM.make_triangular_potential(jnp.asarray(MO), jnp.array([0.0]),
                                       jnp.zeros(1), U, phi)
    print("  %s   r_s = %g,  U = %g,  a_M = %.4f" %
          (os.path.basename(folder), rs, U, np.linalg.norm(MO[:, 0])))
    print("  %-20s %12s %12s %12s" % ("配置", "Coulomb", "モアレ", "合計"))
    base = None
    for name, R in patterns(d).items():
        if len(R) != ne:
            print("  %-20s (谷の数が合わない: %d)" % (name, len(R))); continue
        ae = jnp.asarray(R)[:, None, :]
        ee = jnp.asarray(R)[:, None, :] - jnp.asarray(R)[None, :, :]
        Vc = float(pot(ae, ee)); Vm = float(mo(ae))
        tot = Vc + Vm
        if base is None: base = tot
        print("  %-20s %12.4f %12.4f %12.4f   (ハニカムとの差 %+8.4f)"
              % (name, Vc, Vm, tot, tot - base))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
