# -*- coding: utf-8 -*-
"""lj2 の初期値：下のバンドに射影した HF の WC 解と AHC 解を lambda に沿って追い、VMC 用の表を書く。

  python screening_analysis/lj_hf_branch.py <kind> <seed.npy> <lam,lam,...> <C>
    kind = wc / ahc、seed.npy = 出発の係数 c_k(G)（LJHF.solve の c）、C = 守るチャーン数（0 か 1）
各 lambda で、前の lambda の解を種にして解き、C が変わらなければ採用（変われば種のまま書く）。
出力：hf_N36_lam{lam:.2f}_{kind}.txt（lj_orbitals.load_hf_table が読む）、branch_{kind}.json。
2026-10-16 の表（ws_pw_kit/lj2_hf/）は、WC は lam = 0.33 のガウスの種（LJHF.seed_wc(0.5)）、
AHC は lam = 0.72 の乱数の種（LJHF.seed_random、np.random.default_rng(720) の 1 本目）から。gcut = 2.01（G 19 本）。
"""
import numpy as np, sys, time, json
sys.path.insert(0, '.')
from periodicwave.pbc.lj_hf import LJHF

kind, start_file = sys.argv[1], sys.argv[2]
lams = [float(v) for v in sys.argv[3].split(',')]
C_want = int(sys.argv[4])
c_prev = np.load(start_file)
res = []
for lam in lams:
    hf = LJHF(lam, 15.0, gcut=2.01)
    D0 = np.einsum('ka,kb->kab', c_prev, c_prev.conj())
    t = time.time()
    o = hf.solve(D0, maxit=150, mix=0.5, tol=1e-6)
    C = hf.chern(o['c'])
    ok = abs(C - C_want) < 0.1 and o['hist'][-1][1] < 1e-4
    Dp = np.einsum('ka,kb->kab', c_prev, c_prev.conj())
    E_seed = hf.energy(Dp, hf.hartree(Dp), hf.fock(Dp)); C_seed = hf.chern(c_prev)
    use = o['c'] if ok else c_prev
    np.savetxt(f"hf_N36_lam{lam:.2f}_{kind}.txt", hf.orbitals_text(use), fmt='%.12e')
    r = dict(lam=lam, kind=kind, E=float(o['E']), C=float(C), rhoG=hf.density_fourier(o['c']), its=len(o['hist']),
             dD=float(o['hist'][-1][1]), converged_same_C=bool(ok), E_seed=float(E_seed), C_seed=float(C_seed),
             used='scf' if ok else 'seed', gap=float(np.min(o['w'][:, 1] - o['w'][:, 0])), sec=round(time.time() - t))
    print(json.dumps(r), flush=True)
    res.append(r)
    if ok:
        c_prev = o['c']
json.dump(res, open(f"branch_{kind}.json", "w"), indent=1)
