"""walker から相関長の候補を測る — 窓幅 w* と直接比べるため。

    python screening_analysis/corr_length.py <folder> [<folder> ...]

出す量（すべてコード長 = r_s a_B 単位）:
  r_hole  : g(r) = 0.5 を最初に横切る距離（相関ホールの半径）
  r_peak  : g(r) の第 1 ピーク位置
  a_latt  : 三角格子の第 1 近接距離 √(2π/√3) = 1.9046（r_s に依らない）
  λ_TF    : Thomas–Fermi 遮蔽長 1/(2 r_s)（コード単位。r_s とともに縮む）

距離は「最小像 Euclid」と「周期ノルム（窓 m(d) が実際に使う物差し）」の両方で測る。
"""
import glob, json, os, re, sys
import numpy as np, jax, jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from periodicwave import networks                                  # noqa


def lattice(cfg):
    m = re.search(r"unserializable object: \[\[(.*?)\]\]", json.dumps(cfg))
    return np.array([float(x) for x in re.findall(
        r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(1))][:4]).reshape(2, 2)


def gofr(d, ne, area, nb=70, rmax=None):
    d = d[d > 0]
    rmax = rmax or d.max()
    h, e = np.histogram(d, bins=nb, range=(0, rmax))
    r = 0.5 * (e[1:] + e[:-1])
    shell = np.pi * (e[1:] ** 2 - e[:-1] ** 2)
    rho = ne / area
    npair = len(d) / (ne * (ne - 1))            # 配置数
    return r, h / (npair * ne * rho * shell)


def cross(r, g, lvl=0.5):
    i = np.argmax(g > lvl)
    if i == 0:
        return np.nan
    return float(np.interp(lvl, [g[i - 1], g[i]], [r[i - 1], r[i]]))


def run(folder):
    cfg = json.load(open(os.path.join(folder, "config.json")))
    L = lattice(cfg); ne = sum(cfg["system"]["electrons"])
    rs = float(cfg["system"]["make_local_energy_kwargs"]["potential_kwargs"]
               ["interaction_energy_scale"])
    ck = sorted(glob.glob(os.path.join(folder, "qmcjax_ckpt_*.npz")))[-1]
    pos = np.asarray(np.load(ck, allow_pickle=True)["data"].item()["positions"][0])
    R = pos.reshape(-1, ne, 2)
    area = abs(np.linalg.det(L))

    dr = R[:, :, None, :] - R[:, None, :, :]
    f = dr @ np.linalg.inv(L).T
    de = np.linalg.norm((f - np.round(f)) @ L.T, axis=-1)           # 最小像 Euclid

    cif = networks.make_periodic_construct_input_features(jnp.asarray(L))
    dp = np.asarray(jax.vmap(lambda p: cif(p, jnp.zeros((1, 2)), ndim=2)[3][..., 0])(
        jnp.asarray(pos)))                                          # 周期ノルム

    out = dict(rs=rs, ne=ne, a_latt=float(np.linalg.norm(L[:, 0]) / np.sqrt(ne)),
               tf=1.0 / (2 * rs))
    for tag, d in [("eu", de), ("pn", dp)]:
        iu = np.triu_indices(ne, 1)
        x = d[:, iu[0], iu[1]].ravel()
        r, g = gofr(x, ne, area, rmax=d.max())
        out["r_hole_" + tag] = cross(r, g, 0.5)
        out["r_peak_" + tag] = float(r[np.argmax(g)])
        out["gmax_" + tag] = float(g.max())
    return out


if __name__ == "__main__":
    rows = [run(f) for f in sys.argv[1:]]
    print("\n  %6s %6s | %9s %9s %8s | %9s %9s | %8s %8s"
          % ("r_s", "N", "r_hole(eu)", "r_peak(eu)", "g_max",
             "r_hole(pn)", "r_peak(pn)", "a_latt", "λ_TF"))
    print("  " + "-" * 92)
    for r in rows:
        print("  %6g %6d | %9.3f %9.3f %8.2f | %9.3f %9.3f | %8.3f %8.4f"
              % (r["rs"], r["ne"], r["r_hole_eu"], r["r_peak_eu"], r["gmax_eu"],
                 r["r_hole_pn"], r["r_peak_pn"], r["a_latt"], r["tf"]))
