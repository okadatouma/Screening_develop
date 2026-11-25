"""Liquid/solid diagnostics for rs-sweep cases (read-only; never trains).

python -m screening_analysis.liquid_solid_analysis OUTPUT CASE_DIR [CASE_DIR ...]
    [--equilibrium-root DIR] [--samples 1024] [--probe 256]

Per case (CASE_DIR = rs-sweep case folder containing training/):
  1. Pseudospin: polarization, g_same/g_opp(r), 6-NN <s_i s_j>, S_z(q) on first G shell / smallest q.
  2. Window-width trajectory over all checkpoints.
  3. Solid order: global/local psi6, g6(r), Lindemann ratio (Hungarian assignment to the
     commensurate 6x6-type lattice, per-config translation fit), conditional pair density
     and its m=6 angular modulation in shells.
  4. Local-energy tail statistics (only if equilibrium samples carry local energies).
  5. Within-model link: attention weights on probe configs; per electron/head effective
     distance, same-pseudospin weight fraction, effective neighbours, vs local psi6,
     nearest-neighbour distance and local same-spin fraction (Spearman, and partial
     Spearman controlling the nearest-neighbour distance).
Configurations: <equilibrium-root>/<case>/equilibrium-measurements.npz if present, otherwise
fresh Metropolis chains with an auto-tuned proposal width. Not a phase certification.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


def min_image(dx, L, Li):
    f = dx @ Li.T
    f -= np.round(f)
    return f @ L.T


def pair_geometry(x, L, Li):
    d = x[:, None, :] - x[None, :, :]
    d = min_image(d, L, Li)
    r = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(r, np.inf)
    return d, r


def lattice_sites(L, n):
    side = int(round(np.sqrt(n)))
    g = np.array([(i / side, j / side) for i in range(side) for j in range(side)])
    return g @ L.T, np.linalg.norm(L[:, 0]) / side


def lindemann_u2(x, sites, L, Li, iters=6):
    t = np.zeros(2)
    for _ in range(iters):
        cost = np.linalg.norm(min_image(x[:, None, :] - (sites[None] + t), L, Li), axis=-1) ** 2
        r, c = linear_sum_assignment(cost)
        u = min_image(x[r] - (sites[c] + t), L, Li)
        t = t + u.mean(0)
    u = min_image(x[r] - (sites[c] + t), L, Li)
    u -= u.mean(0)
    return float((u ** 2).sum(1).mean())


def shell_vectors(L, n, which='first'):
    G = 2 * np.pi * np.linalg.inv(L).T     # columns: reciprocal of supercell
    side = int(round(np.sqrt(n)))
    m = np.array([(a, b) for a in range(-2 * side, 2 * side + 1) for b in range(-2 * side, 2 * side + 1)
                  if (a, b) != (0, 0)])
    k = m @ G.T
    kn = np.linalg.norm(k, axis=1)
    if which == 'smallest':
        sel = kn < kn.min() * 1.01
    else:  # first Bragg shell of the commensurate crystal = |k| of side * smallest
        target = kn.min() * side
        sel = np.abs(kn - target) < 1e-6 * target + 1e-9
    return k[sel]


def sample_configs(case, n_samples, seed):
    from screening_analysis.checkpoint_model import load_model
    from screening_analysis.physical_loops import make_network_evaluator, sample_wavefunction
    with load_model(case / 'training') as (net, params, data, info):
        L = np.asarray(info['lattice']); n = sum(info['config']['system']['electrons'])
        ev = make_network_evaluator(net, params, data['atoms'][0], data['charges'][0])
        best = None
        for width in (0.02, 0.01, 0.005, 0.0025, 0.00125):
            s = sample_wavefunction(ev, L, n, seed=seed, chains=4, sweeps=16, burn=64,
                                    steps_per_sample=4, width=width, pseudospin=True)
            acc = float(np.mean(s['acceptance']))
            best = width
            if acc > 0.2:
                break
        chains = 8
        sweeps = max(n_samples // chains, 2)
        s = sample_wavefunction(ev, L, n, seed=seed + 1, chains=chains, sweeps=sweeps, burn=1024,
                                steps_per_sample=8, width=best, pseudospin=True)
    pos = np.asarray(s['positions']).reshape(-1, n, 2)
    spins = np.asarray(s['spins']).reshape(-1, n)
    return pos, spins, None, dict(source='fresh MCMC', width=best,
                                  acceptance=float(np.mean(s['acceptance'])),
                                  spin_acceptance=float(np.mean(s['spin_acceptance'])))


def width_trajectory(case):
    rows = []
    for ck in sorted((case / 'training').glob('qmcjax_ckpt_*.npz')):
        with np.load(ck, allow_pickle=True) as z:
            p = z['params'].item(); step = int(z['t'])
        for layer, ap in enumerate(p['layers']['attention']):
            om = np.asarray(ap['mask_omega']).reshape(-1) if 'mask_omega' in ap else None
            if om is None:
                continue
            sp = np.log1p(np.exp(om))
            for h, v in enumerate(sp):
                rows.append(dict(step=step, layer=layer, head=h, softplus_omega=float(v)))
    return rows


def analyze(case, out, eq_root, n_samples, n_probe, seed):
    from screening_analysis.checkpoint_model import load_model
    from screening_analysis.attn_probe import probe
    out.mkdir(parents=True, exist_ok=True)
    eq = eq_root / case.name / 'equilibrium-measurements.npz' if eq_root else None
    with load_model(case / 'training') as (net, params, data, info):
        L = np.asarray(info['lattice']); n = sum(info['config']['system']['electrons'])
        cfg_path, lat = info['config_path'], info['lattice']; ckpt = info['checkpoint']
    if eq is not None and eq.exists():
        z = np.load(eq)
        pos = z['positions'].reshape(-1, n, 2); spins = z['spins'].reshape(-1, n)
        eloc = z['local_energy'] if z['local_energy'].size else None
        src = dict(source=str(eq))
    else:
        pos, spins, eloc, src = sample_configs(case, n_samples, seed)
    Li = np.linalg.inv(L); area = abs(np.linalg.det(L))
    sites, a = lattice_sites(L, n)
    B = len(pos)
    rmax = 0.5 * min(np.linalg.norm(L[:, 0]), np.linalg.norm(L[:, 1])) * np.sqrt(3) / 2
    edges = np.linspace(0, rmax, 49); rc = (edges[1:] + edges[:-1]) / 2; shell = np.pi * np.diff(edges ** 2)
    hs = np.zeros(48); ho = np.zeros(48); ns = no = 0
    g6n = np.zeros(48, complex); g6c = np.zeros(48)
    psi6g = []; nn_ss = []; u2 = []; m = []; d1_all = []
    cond = np.zeros((81, 81)); cext = 2.5 * a
    Gs = shell_vectors(L, n, 'first'); Gq = shell_vectors(L, n, 'smallest')
    szG = []; szq = []; sG = []
    local = []
    for b in range(B):
        x = pos[b]; s = spins[b]
        d, r = pair_geometry(x, L, Li)
        iu = np.triu_indices(n, 1)
        same = (s[:, None] == s[None, :])[iu]
        hs += np.histogram(r[iu][same], edges)[0]; ho += np.histogram(r[iu][~same], edges)[0]
        ns += same.sum(); no += (~same).sum()
        nn = np.argsort(r, axis=1)[:, :6]
        vec = np.take_along_axis(d, nn[:, :, None], axis=1)
        th = np.arctan2(vec[..., 1], vec[..., 0])
        p6 = np.exp(6j * th).mean(1)
        psi6g.append(abs(p6.mean()))
        prod = (p6[:, None] * p6[None, :].conj())[iu]
        idx = np.digitize(r[iu], edges) - 1; ok = (idx >= 0) & (idx < 48)
        np.add.at(g6n, idx[ok], prod[ok]); np.add.at(g6c, idx[ok], 1)
        ssn = s[nn] * s[:, None]; nn_ss.append(ssn.mean())
        m.append(s.mean())
        d1 = np.take_along_axis(r, nn[:, :1], axis=1)[:, 0]; d1_all.append(d1)
        local.append(dict(psi6=np.abs(p6), d1=d1, same_nn=(s[nn] == s[:, None]).mean(1)))
        if b < 512:
            u2.append(lindemann_u2(x, sites, L, Li))
        dd = d[~np.eye(n, dtype=bool)]
        H, _, _ = np.histogram2d(dd[:, 0], dd[:, 1], bins=81, range=[[-cext, cext], [-cext, cext]])
        cond += H
        rho = np.exp(1j * x @ Gs.T).sum(0); sG.append(np.mean(abs(rho) ** 2) / n)
        szG.append(np.mean(abs((s[:, None] * np.exp(1j * x @ Gs.T)).sum(0)) ** 2) / n)
        szq.append(np.mean(abs((s[:, None] * np.exp(1j * x @ Gq.T)).sum(0)) ** 2) / n)
    norm_s = ns / area * shell / B * B; norm_o = no / area * shell / B * B
    g_same = hs / (ns / area * shell) if ns else hs * np.nan
    g_opp = ho / (no / area * shell) if no else ho * np.nan
    g6 = np.where(g6c > 0, (g6n / np.maximum(g6c, 1)).real, np.nan)
    pix = (2 * cext / 81) ** 2
    cond_density = cond / (B * n) / pix / (n / area)          # 1 = uniform
    yy, xx = np.mgrid[0:81, 0:81]; cx = (xx - 40) * 2 * cext / 81; cy = (yy - 40) * 2 * cext / 81
    rr = np.hypot(cx, cy); ang = np.arctan2(cy, cx)
    # Site contrast: density on ideal triangular sites / off-site density within each shell.
    # (A single m=6 annulus spanning sqrt3*a and 2a cancels: those shells are rotated by 30 deg.)
    D = cond_density.T                # histogram2d is [x, y] -> [y, x]
    best = None
    for t in np.linspace(0, np.pi / 3, 121, endpoint=False):
        sc = sum(D[np.hypot(cx - a * np.cos(t + k * np.pi / 3), cy - a * np.sin(t + k * np.pi / 3)) < .25 * a].mean()
                 for k in range(6))
        if best is None or sc > best[0]:
            best = (sc, t)
    t0 = best[1]
    a1 = a * np.array([np.cos(t0), np.sin(t0)]); a2 = a * np.array([np.cos(t0 + np.pi / 3), np.sin(t0 + np.pi / 3)])
    ring = dict(orientation_deg=float(np.degrees(t0)))
    for name, lo, hi in (('1a', .8, 1.2), ('sqrt3a', 1.55, 1.9), ('2a', 1.9, 2.2)):
        st = [i * a1 + j * a2 for i in range(-4, 5) for j in range(-4, 5)]
        st = [v for v in st if lo * a <= np.linalg.norm(v) < hi * a]
        on = np.zeros_like(rr, bool)
        for v in st:
            on |= np.hypot(cx - v[0], cy - v[1]) < .2 * a
        band = (rr >= (lo - .2) * a) & (rr < (hi + .2) * a) & (rr < cext)
        off_ = band & ~on
        ring['site_contrast_' + name] = (float(D[on & (rr < cext)].mean() / D[off_].mean())
                                         if st and (on & (rr < cext)).any() and off_.any() else None)
    lind = float(np.sqrt(np.mean(u2)) / a) if u2 else None
    tail = None
    if eloc is not None and np.size(eloc):
        e = np.real(np.asarray(eloc)); med = np.median(e); mad = np.median(abs(e - med))
        tail = dict(mean=float(e.mean()), std=float(e.std()), median=float(med), mad=float(mad),
                    kurtosis_excess=float(((e - e.mean()) ** 4).mean() / e.var() ** 2 - 3),
                    frac_beyond_10mad=float(np.mean(abs(e - med) > 10 * mad)), max_abs_dev=float(abs(e - med).max()))
    # ---- within-model attention link
    sel = np.linspace(0, B - 1, min(n_probe, B)).astype(int)
    batch = dict(positions=pos[sel].reshape(len(sel), -1), spins=spins[sel],
                 atoms=np.zeros((len(sel), 1, 2)), charges=np.zeros((len(sel), 1)))
    p = probe(case / 'training', checkpoint=ckpt, config_path=cfg_path, lattice=lat,
              evaluation_data=batch, nconf=len(sel), collect_internal=False)
    W = np.asarray(p['W']); dist = np.asarray(p['d'])
    NL, NH = W.shape[1], W.shape[2]
    off = ~np.eye(n, dtype=bool)
    link = []; per_head = []
    for l in range(NL):
        for h in range(NH):
            deff, fsame, neff = [], [], []
            for k, b in enumerate(sel):
                w = np.where(off, W[k, l, h], 0.0); w = np.clip(w, 0, None)
                tot = w.sum(1); tot = np.where(tot > 0, tot, np.nan)
                deff.append((w * dist[k]).sum(1) / tot)
                sm = spins[b][:, None] == spins[b][None, :]
                fsame.append((w * sm).sum(1) / tot)
                q = w / tot[:, None]
                with np.errstate(divide='ignore', invalid='ignore'):
                    H_ = -np.nansum(np.where(q > 0, q * np.log(q), 0), 1)
                neff.append(np.exp(H_))
            deff, fsame, neff = map(np.concatenate, (deff, fsame, neff))
            ps6 = np.concatenate([local[b]['psi6'] for b in sel]); d1 = np.concatenate([local[b]['d1'] for b in sel])
            snn = np.concatenate([local[b]['same_nn'] for b in sel])
            row = dict(layer=l, head=h, mean_deff=float(np.nanmean(deff)), mean_same_fraction=float(np.nanmean(fsame)),
                       mean_neff=float(np.nanmean(neff)))
            ok = np.isfinite(deff) & np.isfinite(fsame)

            def sp(a_, b_):
                return float(spearmanr(a_[ok], b_[ok]).statistic)

            def partial(a_, b_, c_):
                ra, rb, rc_ = (np.argsort(np.argsort(v[ok])).astype(float) for v in (a_, b_, c_))
                res = lambda y: y - np.polyval(np.polyfit(rc_, y, 1), rc_)
                return float(np.corrcoef(res(ra), res(rb))[0, 1])
            row.update(rho_psi6_deff=sp(ps6, deff), rho_psi6_same=sp(ps6, fsame), rho_psi6_neff=sp(ps6, neff),
                       rho_d1_deff=sp(d1, deff), rho_samenn_same=sp(snn, fsame),
                       partial_psi6_deff_given_d1=partial(ps6, deff, d1),
                       partial_psi6_same_given_samenn=partial(ps6, fsame, snn), n_points=int(ok.sum()))
            per_head.append(row)
            # configuration-level: global psi6 vs mean head metrics
            cfg_psi = np.array([psi6g[b] for b in sel])
            cfg_deff = deff.reshape(len(sel), n).mean(1) if deff.size == len(sel) * n else None
            if cfg_deff is not None:
                link.append(dict(layer=l, head=h, rho_cfg_psi6_deff=float(spearmanr(cfg_psi, cfg_deff).statistic),
                                 rho_cfg_psi6_same=float(spearmanr(cfg_psi, fsame.reshape(len(sel), n).mean(1)).statistic)))
    traj = width_trajectory(case)
    summary = dict(case=case.name, checkpoint=ckpt, n=n, a_lattice=float(a), samples=int(B), sampling=src,
                   pseudospin=dict(mean_abs_m=float(np.mean(np.abs(m))), mean_m=float(np.mean(m)),
                                   nn6_ss=float(np.mean(nn_ss)), Sz_firstG=float(np.mean(szG)),
                                   Sz_smallest_q=float(np.mean(szq)), Sz_q0=float(np.mean(np.square(m)) * n)),
                   solid=dict(S_G_over_N=float(np.mean(sG) / n), psi6_global=float(np.mean(psi6g)),
                              psi6_global_std=float(np.std(psi6g)), lindemann=lind, lindemann_configs=len(u2),
                              conditional_density=ring, g6_at_2a=float(np.interp(2 * a, rc, g6)),
                              g6_at_far=float(np.nanmean(g6[rc > 0.8 * rmax]))),
                   local_energy=tail, attention_link=per_head, attention_link_config=link,
                   width_trajectory_last=[r for r in traj if r['step'] == traj[-1]['step']] if traj else [],
                   note='Finite N, single checkpoint; diagnostics, not phase certification.')
    np.savez_compressed(out / 'liquid-solid.npz', r=rc, g_same=g_same, g_opp=g_opp, g6=g6,
                        cond_density=cond_density, cond_extent=cext, a=a)
    (out / 'width-trajectory.json').write_text(json.dumps(traj, indent=1))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('output', type=Path)
    ap.add_argument('cases', type=Path, nargs='+')
    ap.add_argument('--equilibrium-root', type=Path)
    ap.add_argument('--samples', type=int, default=1024)
    ap.add_argument('--probe', type=int, default=256)
    ap.add_argument('--seed', type=int, default=20261125)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    allrows = []
    for case in a.cases:
        case = case.resolve(); out = a.output / case.name
        if (out / 'summary.json').exists():
            allrows.append(json.loads((out / 'summary.json').read_text())); continue
        s = analyze(case, out, a.equilibrium_root, a.samples, a.probe, a.seed)
        allrows.append(s)
        print(json.dumps(dict(case=s['case'], pseudospin=s['pseudospin'], solid=s['solid']), ensure_ascii=False), flush=True)
    (a.output / 'all-summaries.json').write_text(json.dumps(allrows, indent=2))


if __name__ == '__main__':
    main()
