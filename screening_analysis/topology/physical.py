"""Small actual-candidate 1RDM evaluation using the already validated estimator."""
import argparse
import json
from pathlib import Path
import numpy as np
from .workflow import ROOT,provenance,sha,write
from screening_analysis.checkpoint_model import load_model
from screening_analysis import physical_loops as physical
from screening_analysis.topology_cde_run import load_probe
from screening_analysis.report_topology_benchmark import line_svg


def run(output,samples=16,grids=(4,6)):
    import jax.numpy as jnp
    out=Path(output);out.mkdir(parents=True,exist_ok=False);rows=[]
    for lam in ('033','072'):
        identifier=f'N9-lam{lam}-off-seed11';folder=ROOT/'screening_analysis/topology_benchmark_2026-10-26'/identifier
        source=ROOT/'screening_analysis/topology_benchmark_analysis_2026-10-26_final'/identifier
        with load_model(folder,folder/'qmcjax_ckpt_000023.npz') as (net,params,data,info):
            with np.load(source/'fresh_measurement.npz',allow_pickle=False) as z:
                x=z['positions'][:samples].reshape(samples,9,2);s=z['spins'][:samples];atoms=z['atoms'][0];charges=z['charges'][0]
            evaluator=physical.make_network_evaluator(net,params,jnp.asarray(atoms),jnp.asarray(charges),'angle')
            basis=physical.dvr_basis(info['lattice'],(3,3),components=(-1,1))
            probe=load_probe(source/'probe_A.npz');windows=physical.basis_windows(basis,probe['metadata']['window_parameters'],probe['metadata']['settings'],float(probe['dref']),int(probe['NH']))
            for grid in grids:
                points,spins,weights=physical.uniform_quadrature(info['lattice'],(grid,grid),components=(-1,1))
                local=physical.one_body_samples(evaluator,x,s,basis,points,spins,weights,chunk=4)
                measured=physical.density_summary(local,chain_ids=np.arange(samples)%4)
                loops=physical.physical_loop_statistics(measured['G'],basis,window=windows,chain_matrices=measured['independent_chain_means'])
                row=dict(id=identifier,seed=11,N=9,rs=15,lam=float('0.'+lam[1:]),phase='unknown',phase_verified=False,
                    samples=samples,chains=4,sweeps=samples//4,quadrature_grid=grid,quadrature_points=len(points),basis=basis.metadata(),
                    trace_real=float(measured['trace'].real),trace_imag=float(measured['trace'].imag),hermiticity_residual=float(measured['hermiticity_residual']),
                    matrix_sem_frobenius=float(np.linalg.norm(np.hypot(measured['sem_real'],measured['sem_imag']))),
                    plugin_loop_rms=float(np.sqrt(np.mean(loops['imaginary_loop']**2))),
                    independent_loop_mean=float(loops['independent_product'].imag.mean()),
                    independent_loop_rms=float(np.sqrt(np.mean(loops['independent_product'].imag**2))),
                    mean_jackknife_loop_sem=float(np.mean(loops['independent_jackknife_sem_imag'])),
                    window_weighted_loop=loops['window_weighted_mean'],triangles=len(loops['triangles']),
                    source_samples_sha256=sha(source/'fresh_measurement.npz'),checkpoint_sha256=sha(folder/'qmcjax_ckpt_000023.npz'),
                    interpretation='finite-subspace projected density, unhermitized and unrenormalized; few samples; no physical phase/Chern assignment')
                np.savez_compressed(out/f'{identifier}-q{grid}.npz',local_samples=local,G=measured['G'],chain_means=measured['independent_chain_means'],
                    triangles=loops['triangles'],independent_product=loops['independent_product'],plugin_product=loops['product'],window_mass=loops['window_loop_mass'])
                rows.append(row);print(identifier,grid,row['trace_real'],row['independent_loop_rms'],flush=True)
    write(out/'summary.json',dict(results=rows,provenance=provenance(11),status='executed; small unconverged candidate MC and finite quadrature, not a reliable phase diagnostic',
        sampler='reuse first complete time slices of previously saved fresh A chains, NOT training configurations; independent chains not independent seeds'))
    panels=[]
    for identifier in sorted({r['id'] for r in rows}):
        selected=[r for r in rows if r['id']==identifier]
        panels.append((identifier,[('raw projected trace',[r['quadrature_grid'] for r in selected],[r['trace_real'] for r in selected])]))
    line_svg(panels,out/'actual-projected-trace.svg','Actual N9 candidates: quadrature sensitivity, NOT enforced trace=N',xlabel='quadrature side (both spins summed)',ylabel='Re Tr(G_projected)')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=16);a=p.parse_args()
    if a.samples<4 or a.samples>64 or a.samples%4:raise ValueError('samples must be a multiple of 4, in [4,64]')
    run(a.output,a.samples)
