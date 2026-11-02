# -*- coding: utf-8 -*-
"""なめらかな窓でも液体と結晶の違いが見えるか —— 窓だけで集めた対の分布（幾何の読み出し）で比べる。

入れ子窓の run（ジョブ 1929、N = 25、U = 0）の学習後のウォーカー（各 1024 本）から周期ノルムの対距離を作り、
輪切りの窓 m_h(d)（全体の窓 g を含む）で重みを付けて足したヘッドごとの割合

    p_h = < sum_{i<j} m_h(d_ij) > / sum_h < sum_{i<j} m_h(d_ij) >

を、いまの輪切り（tau = 0.5, 1）とガウス核（sigma = 0.5 Delta, 0.3 Delta）で比べる。
違いの大きさは全変動距離 TV = 1/2 sum_h |p_h^A - p_h^B|、雑音の床は同じ相の 128 本ずつの組どうしの TV の平均。
あわせて、周期 lambda の対分布の揺らぎがどれだけ読み出しに残るか（伝達率）を式で出す：

    T(lambda) = |sinc(pi Delta / lambda)| * |k^(2 pi / lambda)|

    python screening_analysis/check_ring_readout.py
"""
import glob, json, os, re
import numpy as np
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DREF, H, P_ENV = 3.2152, 8, 3.0
DL = DREF / H
RUNS = {"液体 r_s=5": "sl05", "境界 r_s=30": "sb30", "結晶 r_s=60": "sc60"}


def lattice(cfg):
    m = re.search(r"unserializable object: \[\[(.*?)\]\]", json.dumps(cfg))
    return np.array([float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(1))][:4]).reshape(2, 2)


def pair_distances(tag):
    f = sorted(glob.glob(os.path.join(ROOT, f"results/2deg-Coulomb/CustomScreenformer/el13_12_rs*_tri_l2h16_{tag}/qmcjax_ckpt_*.npz")))[-1]
    cfg = json.load(open(os.path.join(os.path.dirname(f), "config.json")))
    L = lattice(cfg); M = L.T @ L; Linv = np.linalg.inv(L)
    x = np.asarray(np.load(f, allow_pickle=True)["data"].item()["positions"])[0]      # (walkers, 2N)
    r = x.reshape(x.shape[0], -1, 2)
    iu = np.triu_indices(r.shape[1], 1)
    ee = r[:, iu[1], :] - r[:, iu[0], :]                                                # (walkers, pairs, 2)
    s = np.einsum('il,wpl->wpi', Linv, ee)
    a, b = 1 - np.cos(2 * np.pi * s), np.sin(2 * np.pi * s)
    return np.sqrt(np.einsum('wpm,mn,wpn->wp', a, M, a) + np.einsum('wpm,mn,wpn->wp', b, M, b)) / (2 * np.pi)


def windows(d, kind, width):
    """(H, ...) の窓。kind = 'sin'（width = tau）/ 'gauss'（width = sigma / Delta）。全体の窓 g を含む。"""
    b = np.arange(H + 1) * DL
    if kind == 'sin':
        dl = width * DL / 2
        th = [0.5 * (1 + np.sin(0.5 * np.pi * np.clip((d - bk) / dl, -1, 1))) for bk in b]
    else:
        sg = width * DL
        th = [ndtr((d - bk) / sg) + ndtr((-d - bk) / sg) for bk in b]
    th[0] = np.ones_like(d); th[-1] = np.zeros_like(d)
    g = 0.5 * (1 + np.cos(np.pi * np.clip(d / DREF, 0, 1) ** P_ENV))
    return np.array([g * (th[h] - th[h + 1]) for h in range(H)])


def transfer(lam, kind, width):
    w = 2 * np.pi / lam
    sinc = abs(np.sinc(DL / lam))                      # np.sinc(x) = sin(pi x)/(pi x)
    if kind == 'sin':
        l = width * DL; z = w * l / 2
        k = (np.pi ** 2 / 4) * np.cos(z) / ((np.pi / 2) ** 2 - z ** 2)
    else:
        k = np.exp(-(w * width * DL) ** 2 / 2)
    return sinc * abs(k)


WINS = [("いま tau=0.5", 'sin', 0.5), ("いま tau=1", 'sin', 1.0),
        ("ガウス sigma=0.5D", 'gauss', 0.5), ("ガウス sigma=0.3D", 'gauss', 0.3)]


def readouts(d, kind, width, groups=8):
    """全体と 8 組それぞれの p_h。"""
    m = windows(d, kind, width)                        # (H, walkers, pairs)
    s = m.sum(-1)                                      # (H, walkers)
    tot = s.mean(1); p = tot / tot.sum()
    sg = np.array_split(np.arange(d.shape[0]), groups)
    pg = np.array([s[:, g].mean(1) / s[:, g].mean(1).sum() for g in sg])
    return p, pg


def tv(p, q):
    return 0.5 * np.abs(p - q).sum(-1)


def main():
    D = {k: pair_distances(t) for k, t in RUNS.items()}
    print(f"Delta = {DL:.4f}, 対の数 {D['液体 r_s=5'].shape[1]}、ウォーカー {D['液体 r_s=5'].shape[0]}")
    print("\n伝達率 T(lambda)：対分布の周期 lambda の揺らぎが読み出しに残る割合")
    for lam in (1.905, 1.0, 0.8):
        print(f"  lambda = {lam:5.3f}: " + "  ".join(f"{n} {transfer(lam, k, w):.3f}" for n, k, w in WINS))
    out = {}
    for n, k, w in WINS:
        R = {ph: readouts(d, k, w) for ph, d in D.items()}
        floor = np.mean([np.mean([tv(pg[i], pg[j]) for i in range(8) for j in range(i + 1, 8)]) for _, pg in R.values()])
        lc = tv(R["液体 r_s=5"][0], R["結晶 r_s=60"][0]); bc = tv(R["境界 r_s=30"][0], R["結晶 r_s=60"][0])
        # 組ごとの TV のばらつき（標準誤差）
        lcg = [tv(R["液体 r_s=5"][1][i], R["結晶 r_s=60"][1][i]) for i in range(8)]
        out[n] = R
        print(f"\n{n}: TV(液体, 結晶) = {lc:.4f} ± {np.std(lcg)/np.sqrt(8):.4f}   TV(境界, 結晶) = {bc:.4f}"
              f"   雑音の床 {floor:.4f}   床の {lc/floor:.0f} 倍 / {bc/floor:.1f} 倍")
        for ph in RUNS:
            print(f"   {ph:10s} p_h = " + " ".join(f"{v:.3f}" for v in R[ph][0]))
        dp = R["結晶 r_s=60"][0] - R["液体 r_s=5"][0]
        print(f"   結晶 - 液体 = " + " ".join(f"{v:+.3f}" for v in dp))
    np.savez(os.path.join(HERE, "out_gauss_rings", "readout.npz"),
             **{f"{n}|{ph}": out[n][ph][0] for n in out for ph in RUNS},
             **{f"err|{n}|{ph}": out[n][ph][1].std(0) / np.sqrt(8) for n in out for ph in RUNS},
             **{f"hist|{ph}": np.histogram(D[ph].ravel(), bins=np.linspace(0, DREF, 65))[0] for ph in RUNS})
    print("\nwrote out_gauss_rings/readout.npz")


if __name__ == "__main__":
    main()
