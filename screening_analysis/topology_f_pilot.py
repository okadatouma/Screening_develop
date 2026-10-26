"""Physical validation: exact 1RDM, independently twisted Chern model, real trial MC.

python -m screening_analysis.topology_f_pilot --output NEW_DIRECTORY
This does not assign WC/AHC labels or compute a Screenformer many-body Chern number.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from screening_analysis import physical_loops as physical,berry_validation as berry
from screening_analysis.topology_cde_run import heatmap_svg,load_probe
from screening_analysis.topology_run import json_safe


def exact_density(out,previous):
    """Two continuous-space fermions in a known finite Fourier Slater state."""
    from itertools import product
    probe=load_probe(previous/'synthetic-cos-seed11-ordered/probe.npz')
    # A 5x5 spatial basis supplies multiple contractible triangle lengths; on a
    # 3x3 triangular basis all retained shortest-image triangles have one length.
    basis=physical.dvr_basis(probe['lattice'],shape=(5,5))
    q,qs,qw=physical.uniform_quadrature(basis.lattice,(5,5),shift=(0,0))
    rng=np.random.default_rng(19)
    coefficients=np.linalg.qr(rng.normal(size=(basis.size,2))+1j*rng.normal(size=(basis.size,2)))[0]
    pairs=np.array(list(product(range(len(q)),repeat=2)));x=q[pairs];s=np.ones((len(x),2))
    def evaluate(x,s):
        psi=np.linalg.det(basis(x,s)@coefficients);la=np.full(len(psi),-np.inf)
        np.log(abs(psi),out=la,where=abs(psi)>0)
        return np.angle(psi),la
    _,logabs=evaluate(x,s);weights=np.exp(2*logabs);keep=weights>1e-20
    local=physical.one_body_samples(evaluate,x[keep],s[keep],basis,q,qs,qw)
    measured=physical.density_summary(local,weights[keep]);g=measured['G'];expected=coefficients.conj()@coefficients.T
    error=float(np.max(abs(g-expected)));assert error<2e-13
    loops=physical.physical_loop_statistics(g,basis)
    scale=[];arrays=dict(G=g,reference_G=expected,local_samples=local,weights=weights[keep],
                         loop_triangles=loops['triangles'],imaginary_loop=loops['imaginary_loop'],perimeter=loops['perimeter'])
    for mode in ('cos','wide','smooth7','band','sin','gauss','fixed'):
        p=load_probe(previous/f'synthetic-{mode}-seed11-ordered/probe.npz')
        windows=physical.basis_windows(basis,p['metadata']['window_parameters'],p['metadata']['settings'],float(p['dref']),int(p['NH']))
        result=physical.physical_loop_statistics(g,basis,window=windows)
        arrays[mode+'_basis_window']=windows;arrays[mode+'_loop_mass']=result['window_loop_mass']
        arrays[mode+'_weighted_loop']=result['window_weighted_mean']
        scale.append(dict(window=mode,head_layer_values=result['window_weighted_mean'],
                          head_permutation_invariant_mean=float(np.mean(result['window_weighted_mean']))))
    with (out/'exact-density.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    heatmap_svg([('Exact G real',g.real),('Exact G imaginary',g.imag),('Estimator absolute error',abs(g-expected))],
        out/'exact-density.svg','Fixed physical DVR basis: two-fermion analytic validation',xlabel='orbital b',ylabel='orbital a')
    return dict(case='continuous-space two-fermion finite Fourier Slater; exact weighted quadrature, not MC',
        basis=basis.metadata(),max_error=error,trace_real=float(np.trace(g).real),trace_imag=float(np.trace(g).imag),
        hermiticity_residual=float(measured['hermiticity_residual']),
        ccw_imaginary_loop_mean=float(loops['imaginary_loop'].mean()),
        ccw_imaginary_loop_rms=float(np.sqrt(np.mean(loops['imaginary_loop']**2))),
        window_scale_comparison=scale,
        window_interpretation='same physical G, frozen untrained-network radial Windows evaluated on basis centres; token alpha not transplanted')


def chern_benchmark(out):
    records=[];arrays={};panels=[]
    for mass,expected in ((-1.,-1),(1.,1),(3.,0)):
        # Fock-basis many-body amplitudes provide an independent overlap check.
        for grid in (6,10):
            states,gap=berry.qwz_manybody_grid(grid,mass)
            result=berry.berry_links(states);assert abs(result['chern']-expected)<1e-11
            arrays[f'm{mass}_fock_grid{grid}_curvature']=result['curvature']
            records.append(dict(mass=mass,size=2,grid=grid,method='explicit 70-dimensional fixed Fock basis',
                chern=result['chern'],gap=gap,minimum_overlap=result['minimum_link_overlap'],max_phase=result['max_plaquette_phase']))
        kubo=berry.qwz_kubo_integral(48,mass);assert abs(kubo-expected)<1e-8
        for size in (3,5):
            x=np.array([(i,j) for i in range(size) for j in range(size)])
            basis=SimpleNamespace(size=2*size*size,centres=np.repeat(x,2,axis=0),lattice=np.eye(2)*size)
            energy,v=np.linalg.eigh(berry.qwz_twist_hamiltonian([0.,0.],mass,(size,size)))
            occupied=v[:,:size*size];g=occupied.conj()@occupied.T
            loops=physical.physical_loop_statistics(g,basis)
            for grid in (6,10):
                orbitals,gap=berry.qwz_occupied_grid(grid,mass,(size,size))
                result=berry.slater_berry_links(orbitals);assert abs(result['chern']-expected)<1e-11
                arrays[f'm{mass}_n{size}_grid{grid}_curvature']=result['curvature']
                records.append(dict(mass=mass,size=size,grid=grid,method='exact many-body determinant overlaps in fixed site basis',
                    chern=result['chern'],gap=gap,minimum_overlap=result['minimum_link_overlap'],max_phase=result['max_plaquette_phase'],
                    independent_kubo_size2=kubo,ccw_imaginary_loop_mean=float(loops['imaginary_loop'].mean()),
                    area_weighted_loop_mean=float(np.mean(loops['imaginary_loop']*loops['area'])),
                    max_abs_loop=float(np.max(abs(loops['imaginary_loop']))),loop_count=len(loops['triangles'])))
                if size==3 and grid==10:panels.append((f'QWZ m={mass:g}, C={result["chern"]:.0f}',result['curvature']))
            arrays[f'm{mass}_n{size}_G']=g;arrays[f'm{mass}_n{size}_loop']=loops['imaginary_loop']
    with (out/'chern-benchmark.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    heatmap_svg(panels,out/'chern-benchmark.svg','Actual seam-twist Hamiltonian: independent QWZ many-body benchmark',
                xlabel='theta y grid',ylabel='theta x grid',xvalues=[0,2*np.pi],yvalues=[0,2*np.pi])
    return dict(model='QWZ half-filled noninteracting lattice model, NOT lambda-jellium or Screenformer',
        hamiltonian='sin(kx)sigma_x+sin(ky)sigma_y+(m+cos(kx)+cos(ky))sigma_z',
        twist='Peierls phases ONLY on bonds crossing each torus seam; H(theta+2pi)=H(theta) in fixed site basis',
        tracking='entire gapped occupied subspace; Slater determinant, no individual-band index matching',
        convention='links <psi(theta)|psi(theta+dtheta)>, CCW theta_x/theta_y; lower band C=-1 for m=-1',
        independent_checks='analytic twist derivatives, exact Fourier spectrum, Fock vs determinant overlaps, U(1)/U(Nocc) gauge, Kubo curvature, grid/size refinement',
        records=records)


def checkpoint_density(out,root,previous,sweeps=1024,quadrature_grid=16):
    """Fresh |Psi|² chains at an existing trained N1 free-system checkpoint."""
    import jax
    import jax.numpy as jnp
    from periodicwave import CustomScreenformer as CS
    # Validation-only constructor: the matching saved config/summary is checked.
    from tests.check_g3a_vmc import build,blocking
    source=root/'tests/g3a_validation/screen-complex64'
    config=json.loads((source/'effective-config.json').read_text())['config']
    assert config['system']['electrons']==[1,0] and config['network']['complex']
    assert config['system']['make_local_energy_kwargs']['potential_kwargs']['interaction_energy_scale']==0.
    probe=load_probe(previous/'screen-complex64-000015/probe.npz');settings=probe['metadata']['settings']
    # Dictionary insertion order is not a physical setting contract.
    for key in ('TRIM_MAX','TRIM_POW','WIN_SHAPE','FREEZE_W','CLIP_R','CLIP_BETA'):setattr(CS,key,settings[key])
    jax.config.update('jax_enable_x64',True);jax.config.update('jax_default_matmul_precision',settings['jax_default_matmul_precision'])
    net,_=build('screen',1,True)
    path=source/'run/qmcjax_ckpt_000015.npz';before=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path,allow_pickle=True) as z:
        params=jax.tree_util.tree_map(lambda a:np.asarray(a)[0],z['params'].item())
    evaluate=physical.make_network_evaluator(net,params,jnp.zeros((1,2)),jnp.zeros(1),'angle')
    L=np.asarray(config['system']['pbc_lattice']);basis=physical.dvr_basis(L)
    chain=physical.sample_wavefunction(evaluate,L,1,seed=1701,chains=8,sweeps=sweeps,burn=256,steps_per_sample=4,width=.2)
    T,C=chain['positions'].shape[:2];x=chain['positions'].reshape(T*C,1,2);s=chain['spins'].reshape(T*C,1)
    q,qs,qw=physical.uniform_quadrature(L,(quadrature_grid,quadrature_grid))
    samples=physical.one_body_samples(evaluate,x,s,basis,q,qs,qw,chunk=32)
    ids=np.tile(np.arange(C),T);result=physical.density_summary(samples,chain_ids=ids)
    def reference(grid):
        p,ps,pw=physical.uniform_quadrature(L,(grid,grid));phase,logabs=evaluate(p[:,None,:],np.ones((len(p),1)))
        amplitude=np.exp(logabs-logabs.max()+1j*phase)
        norm=np.sum(pw*abs(amplitude)**2);coeff=np.einsum('q,qa,q->a',pw,basis(p).conj(),amplitude)/np.sqrt(norm)
        return coeff.conj()[:,None]*coeff[None,:]
    ref16=reference(16);ref32=reference(32);ref64=reference(64);ref128=reference(128)
    quad_budget=float(np.max(abs(ref64-ref128)))
    refined_grid=2*quadrature_grid
    qr,sr,wr=physical.uniform_quadrature(L,(refined_grid,refined_grid))
    local_refined=physical.one_body_samples(evaluate,x[:128],s[:128],basis,qr,sr,wr)
    ratio_quad_difference=float(np.max(abs(local_refined-samples[:128])))
    windows=physical.basis_windows(basis,probe['metadata']['window_parameters'],settings,float(probe['dref']),int(probe['NH']))
    loops=physical.physical_loop_statistics(result['G'],basis,windows,result['independent_chain_means'])
    difference=result['G']-ref128
    sem=np.sqrt(result['sem_real']**2+result['sem_imag']**2)
    trace_series=np.trace(samples,axis1=-2,axis2=-1).real.reshape(T,C)
    scalar=samples[:,0,1].real.reshape(T,C)
    positions=chain['positions'];cos_series=np.cos(2*np.pi*positions[:,:,0,0]/4.)
    with (out/'checkpoint-density.npz').open('xb') as f:np.savez_compressed(f,
        positions=positions,spins=chain['spins'],chain_ids=ids,local_G=samples,G=result['G'],reference_G=ref128,
        sem_real=result['sem_real'],sem_imag=result['sem_imag'],chain_mean_G=result['independent_chain_means'],
        triangles=loops['triangles'],loop_plugin=loops['product'],loop_independent=loops['independent_product'],
        loop_independent_jackknife_sem=loops['independent_jackknife_sem_imag'],basis_window=windows)
    heatmap_svg([('MC G real',result['G'].real),('MC G imaginary',result['G'].imag),('reference G real',ref128.real),
                 ('absolute MC error',abs(difference)),('complex chain SEM',sem),('reference G imaginary',ref128.imag)],
        out/'checkpoint-density.svg','Existing trained free-electron trial state: fresh independent MC chains',xlabel='orbital b',ylabel='orbital a')
    assert before==hashlib.sha256(path.read_bytes()).hexdigest()
    maximum_ratio=float(np.max(np.divide(abs(difference),sem+quad_budget+ratio_quad_difference)))
    return dict(checkpoint=str(path),checkpoint_sha256=before,checkpoint_unchanged=True,settings=settings,
        config_sha256=hashlib.sha256((source/'effective-config.json').read_bytes()).hexdigest(),
        validation_constructor='tests.check_g3a_vmc.build screen,N1,complex; known matching saved config; no init/training called',
        basis=basis.metadata(),sampling=dict(seed=1701,chains=C,sweeps=T,burn=256,steps_per_sample=4,width=.2,
            position_acceptance=chain['acceptance']),
        trace_real=float(result['trace'].real),trace_imag=float(result['trace'].imag),reference_projected_trace=float(np.trace(ref128).real),
        hermiticity_residual=float(result['hermiticity_residual']),matrix_error_frobenius=float(np.linalg.norm(difference)),
        chain_sem_frobenius=float(np.linalg.norm(sem)),maximum_matrix_error_over_complex_SEM_plus_quad=maximum_ratio,
        reference_grid16_32_difference=float(np.max(abs(ref16-ref32))),reference_grid32_64_difference=float(np.max(abs(ref32-ref64))),
        reference_grid64_128_difference=quad_budget,
        local_quadrature_refinement=dict(grid=quadrature_grid,refined_grid=refined_grid,checked_samples=128,max_difference=ratio_quad_difference),
        blocking_cos_x=blocking(cos_series),blocking_G01_real=blocking(scalar),blocking_trace_real=blocking(trace_series),
        retained_triangle_count=len(loops['triangles']),loop_plugin_mean_imag=float(loops['product'].imag.mean()),
        loop_independent_mean_imag=float(loops['independent_product'].imag.mean()),
        loop_reference_max_abs_imag=float(np.max(abs(physical.physical_loop_statistics(ref128,basis)['imaginary_loop']))),
        loop_jackknife_sem_max=float(np.max(loops['independent_jackknife_sem_imag'])),
        window_weighted_loop_mean=loops['window_weighted_mean'],
        interpretation='N1 trial wavefunction, not WC/AHC; exact 1RDM is rank one and physical triangular loops have zero imaginary part',
        uncertainty='independent-chain SEM and chain jackknife; finite-chain error estimates, not a convergence proof')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous-results',type=Path,default=Path('screening_analysis/topology_results_2026-10-26_final'))
    parser.add_argument('--skip-mc',action='store_true');parser.add_argument('--sweeps',type=int,default=1024)
    parser.add_argument('--quadrature-grid',type=int,default=16);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('qmcjax_ckpt_*.npz')}
    exact=exact_density(out,args.previous_results);print('exact fixed-basis density checked',flush=True)
    chern=chern_benchmark(out);print('twist Hamiltonian / many-body Chern independently checked',flush=True)
    mc=None if args.skip_mc else checkpoint_density(out,root,args.previous_results,args.sweeps,args.quadrature_grid)
    result=dict(exact_density=exact,independent_chern_benchmark=chern,checkpoint_density=mc,
        original_checkpoint_count=len(before),all_checkpoints_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in before.items()),
        screenformer_chern='UNVERIFIED: no physically twisted lambda-jellium Hamiltonian/state family or independent WC/AHC labels',
        screenformer_rotation_eigenvalue='UNVERIFIED: no validated symmetry state family provided',
        source_sha256={str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest()
                       for p in (__file__,physical.__file__,berry.__file__)})
    (out/'summary.json').write_text(json.dumps(json_safe(result),indent=2,allow_nan=False))


if __name__=='__main__':main()
