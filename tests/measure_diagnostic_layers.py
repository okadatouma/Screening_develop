"""D': 分離した診断の誤差を実測する。基準・本体・既存許容誤差は変更しない。

Usage: JAX_PLATFORMS=cpu .venv-pw/bin/python -u -m tests.measure_diagnostic_layers NEW_DIRECTORY
"""
import argparse
from attr import evolve
import hashlib
import json
from pathlib import Path
import tempfile
import time

from tests.support import require_numerics, make_case
require_numerics()
import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import CustomScreenformer as CS, networks, network_blocks
from periodicwave.pbc import feature_layer, hamiltonian
from tests.diagnostic_layers import diagnostic_namespace, make_diagnostic
from tests.baseline import metadata


def metrics(reference, diagnostic):
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(diagnostic, dtype=np.float64)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    delta = b - a
    scale = float(np.sqrt(np.mean(a*a)))
    rms = float(np.sqrt(np.mean(delta*delta)))
    floor = max(scale * 1e-6, np.finfo(np.float64).tiny)
    mask = np.abs(a) >= floor
    return dict(max_abs=float(np.max(np.abs(delta))), rms=rms, reference_rms=scale,
                reference_max=float(np.max(np.abs(a))), relative_rms=rms/max(scale, 1e-300),
                max_relative_above_floor=float(np.max(np.abs(delta[mask]/a[mask]))) if mask.any() else None,
                relative_floor=floor, excluded_near_zero=int((~mask).sum()), size=int(a.size))


def flatten(tree):
    return np.concatenate([np.asarray(x).reshape(-1) for x in jax.tree_util.tree_leaves(tree)])


def kinetic(g, h):
    """独立な複素恒等式。u=(theta,a=log|psi|)。本体folxの既知問題とは別評価。

    T=-1/2[Delta a+|grad a|²-|grad theta|²+i(Delta theta+2 grad a.grad theta)]。
    同じNumPy float64縮約で両側を評価し、診断由来の微分差の影響を分離する。
    """
    g, h = np.asarray(g, np.float64), np.asarray(h, np.float64)
    lap = np.trace(h, axis1=-2, axis2=-1)
    return -0.5*(lap[1]+np.sum(g[1]**2)-np.sum(g[0]**2)
                 + 1j*(lap[0]+2*np.sum(g[1]*g[0])))


def ready(value):
    jax.block_until_ready(value)
    return value


def benchmark(compiled, args):
    for _ in range(3):ready(compiled(*args))
    times=[]
    for _ in range(9):
        start=time.perf_counter()
        for _ in range(10):ready(compiled(*args))
        times.append((time.perf_counter()-start)/10)
    mem=compiled.memory_analysis()
    return dict(median_ms=1000*float(np.median(times)), samples_ms=[1000*x for x in times],
                memory={k:int(getattr(mem,k)) for k in
                        ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')})


CASES = [(n,ln,2,'cos') for n in (1,2,3) for ln in (False,True)] + [
    (n,True,4,'cos') for n in (1,2,3)] + [
    (2,True,2,mode) for mode in ('smooth7','pow4','sin','gauss','nomask','fixed')]


def run_case(directory, x64, n, ln, ne, mode, width=4, formal_api=False):
    jax.config.update('jax_enable_x64',x64)
    key=f"{'f64' if x64 else 'f32'}-L{n}-LN{int(ln)}-N{ne}-{mode}"
    if width!=4:key+=f'-E{width}'
    print('START',key,flush=True)
    ns=None if formal_api else diagnostic_namespace()
    settings={k:getattr(CS,k) for k in ('TRIM_MAX','TRIM_POW','WIN_SHAPE','FREEZE_W','CLIP_R','CLIP_BETA')}
    # 構築・初回traceに同じ実効設定を使用。Productionグローバルには介入しない。
    # 異なるWindow設定は診断名前空間だけ変更せず、以下の独立factory空間で両側を構築する。
    settings.update(TRIM_POW=4 if mode=='pow4' else 2,WIN_SHAPE='smooth7' if mode=='smooth7' else 'cos')
    original_namespace=dict(CS.__dict__)
    import ast
    original_tree=ast.parse(Path(CS.__file__).read_text())
    names={'make_screening_attention','make_self_attention_block','make_screenformer_layers','make_fermi_net',
           'make_screenformer_diagnostic'}
    original_tree.body=[x for x in original_tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    exec(compile(original_tree,'<unchanged production source: isolated settings>','exec'),original_namespace)
    original_namespace.update(settings)
    if formal_api:
        ns=original_namespace
        ns['source_sha256']=hashlib.sha256(Path(CS.__file__).read_bytes()).hexdigest()
    else:
        # Historical prototype namespace: use only its archived candidate factories.
        ns.pop('make_screenformer_diagnostic',None)
        ns.update(settings)
    with tempfile.TemporaryDirectory() as tmp:
        _,_,data,lattice,opts,_=make_case('screen',tmp)
    nspins=(ne//2,ne//2)
    data.positions=jnp.asarray([.23,.41,1.17,.89,.72,1.61,1.91,1.35][:2*ne])
    data.spins=jnp.asarray([1.]*(ne//2)+[-1.]*(ne//2))
    opts.update(num_layers=n,use_layer_norm=ln,attn_dim=3,value_dim=2,mlp_dim=width,
                num_perceptrons_per_layer=2,use_fixed_value=mode=='fixed',use_softmask=mode!='nomask',
                feature_layer=feature_layer.make_pbc_feature_layer(
                    natoms=1,nspins=nspins,ndim=2,lattice=lattice,include_r_ae=False))
    if mode in ('sin','gauss'):opts['init_softmask_rings']=([0.,.5,1.],.16,1.,mode)
    net=original_namespace['make_fermi_net'](nspins,data.charges,**opts)
    params=net.init(jax.random.PRNGKey(7))
    diagnostic=make_diagnostic(net,nspins,data,ns)
    def production(p,r):return jnp.stack(net.apply(p,r,data.spins,data.atoms,data.charges))
    args=(params,data.positions)
    start=time.perf_counter()
    prod=jax.jit(production).lower(*args).compile()
    prod_compile=time.perf_counter()-start
    before=ready(prod(*args));hlo_before=prod.as_text()
    start=time.perf_counter()
    diag=jax.jit(diagnostic).lower(*args).compile()
    diag_compile=time.perf_counter()-start
    value,layers=ready(diag(*args))
    after=ready(prod(*args))
    np.testing.assert_array_equal(before,after)
    assert hlo_before==prod.as_text()
    # 内部量を捨てた場合は同じ通常グラフに戻ることを別に検証。
    discarded=ready(jax.jit(lambda p,r:diagnostic(p,r)[0])(*args))
    np.testing.assert_array_equal(before,discarded)
    eager,erecords=diagnostic(*args)
    np.testing.assert_array_equal(np.asarray(production(*args)),np.asarray(eager))
    _,norm=network_blocks.make_layer_norm();_,mlp=CS.make_mlp(2,'GELU')
    for i,record in enumerate(erecords):
        np.testing.assert_array_equal(record['attention_residual'],record['input']+record['attention_output'])
        first=norm(params['layers']['ln'][i][0],record['attention_residual']) if ln else record['attention_residual']
        np.testing.assert_array_equal(record['ln1_output'],first)
        np.testing.assert_array_equal(record['mlp_output'],mlp(params['layers']['mlp'][i],first))
        residual=first+record['mlp_output']
        second=norm(params['layers']['ln'][i][1],residual) if ln else residual
        np.testing.assert_array_equal(record['output'],second)
        if i+1<n:np.testing.assert_array_equal(layers[i]['output'],layers[i+1]['input'])
    grad=ready(jax.jit(jax.jacrev(production,argnums=1))(*args))
    dg,_=ready(jax.jit(jax.jacrev(diagnostic,argnums=1,has_aux=True))(*args))
    hess=ready(jax.jit(jax.hessian(production,argnums=1))(*args))
    dh,_=ready(jax.jit(jax.hessian(diagnostic,argnums=1,has_aux=True))(*args))
    pg=ready(jax.jit(jax.jacrev(production,argnums=0))(*args))
    dpg,_=ready(jax.jit(jax.jacrev(diagnostic,argnums=0,has_aux=True))(*args))
    # 指標ごとに別コンパイル。has_aux=Trueの中間量はコンパイル出力に保持する。
    values=dict(phase=(before[0],value[0]),logabs=(before[1],value[1]),
        coordinate_gradient_phase=(grad[0],dg[0]),coordinate_gradient_logabs=(grad[1],dg[1]),
        hessian_phase=(hess[0],dh[0]),hessian_logabs=(hess[1],dh[1]),
        parameter_gradient=(flatten(pg),flatten(dpg)),
        parameter_gradient_phase=(flatten(jax.tree_util.tree_map(lambda x:x[0],pg)),
                                  flatten(jax.tree_util.tree_map(lambda x:x[0],dpg))),
        parameter_gradient_logabs=(flatten(jax.tree_util.tree_map(lambda x:x[1],pg)),
                                   flatten(jax.tree_util.tree_map(lambda x:x[1],dpg))))
    geometry=networks.make_periodic_construct_input_features(lattice)
    ae,ee,ra,re=geometry(data.positions,data.atoms,ndim=2)
    potential=hamiltonian.make_2DCoulomb_potential(
        lattice,data.atoms,data.charges,truncation_limit=2,interaction_energy_scale=2.)
    potential_value=float(potential(ae,ee))
    t0,t1=kinetic(grad,hess),kinetic(dg,dh)
    e0,e1=t0+potential_value,t1+potential_value
    values.update(kinetic_real=(t0.real,t1.real),kinetic_imag=(t0.imag,t1.imag),
                  local_energy_real=(e0.real,e1.real),local_energy_imag=(e0.imag,e1.imag),
                  laplacian_logabs=(np.trace(hess[1]),np.trace(dh[1])),
                  laplacian_phase=(np.trace(hess[0]),np.trace(dh[0])))
    for i,record in enumerate(layers):
        # 途中値はProduction内から直接観測できない。通常ソースのprefixを独立JITで評価。
        for count,label in ((i,'input'),(i+1,'output')):
            _,prefix=original_namespace['make_screenformer_layers'](nspins,1,evolve(net.options,num_layers=count))
            ref=ready(jax.jit(lambda p:prefix(p['layers'],ae=ae,ee=ee,r_ae=ra,r_ee=re,
                           spins=data.spins,charges=data.charges))(params))
            values[f'layer{i}_{label}_vs_separate_prefix']=(ref,record[label])
        # 同じ診断ソースでWeightだけを出力した独立JITとの比較（Production内部値ではない）。
        weights=ready(jax.jit(lambda p,r:diagnostic(p,r)[1][i]['attention']['aggregation_weights'])(*args))
        values[f'layer{i}_weight_vs_separate_weight_jit']=(weights,record['attention']['aggregation_weights'])
    arrays={'positions':np.asarray(data.positions),'spins':np.asarray(data.spins),'lattice':np.asarray(lattice)}
    for i,leaf in enumerate(jax.tree_util.tree_leaves(params)):arrays[f'params_{i}']=np.asarray(leaf)
    for name,(a,b) in values.items():arrays[name+'_production_reference']=np.asarray(a);arrays[name+'_diagnostic']=np.asarray(b)
    for i,record in enumerate(layers):
        for path,leaf in jax.tree_util.tree_flatten_with_path(record)[0]:
            arrays[f'aux_layer{i}_'+''.join(str(p) for p in path)]=np.asarray(leaf)
    np.savez_compressed(directory/(key+'.npz'),**arrays)
    result=dict(case=key,x64=x64,layers=n,layernorm=ln,electrons=ne,mode=mode,seed=7,
                effective_settings=settings,source_sha256=ns['source_sha256'],
                model=dict(nspins=nspins,ndim=2,determinants=2,num_heads=2,attn_dim=3,value_dim=2,
                           mlp_dim=width,num_perceptrons_per_layer=2,activation='GELU',complex_output=True,
                           jastrow='NONE',init_trim_r=1.9,init_softmask_w_factor=.8,
                           init_softmask_w_spread=2.5,init_output_scale=0.,
                           init_softmask_rings=opts.get('init_softmask_rings')),
                archive_sha256=hashlib.sha256((directory/(key+'.npz')).read_bytes()).hexdigest(),
                formal_api=formal_api,candidate_sha256=ns.get('candidate_sha256'),params_sha256=hashlib.sha256(flatten(params).tobytes()).hexdigest(),
                params_dtypes=sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(params)}),
                production_unchanged=True,discarded_aux_exact=True,eager_relations_exact=True,
                metrics={k:metrics(a,b) for k,(a,b) in values.items()},
                performance={'production':benchmark(prod,args),'diagnostic':benchmark(diag,args),
                             'production_lower_compile_seconds':prod_compile,
                             'diagnostic_lower_compile_seconds':diag_compile},
                local_energy_definition='same float64 complex kinetic identity from measured derivatives + same Ewald potential; NOT legacy folx local_energy',
                intermediate_reference='separately compiled source prefixes / output-pruned weight JIT; NOT unobservable Production internal bits')
    if n==2 and ln and ne==2 and mode=='cos':
        (directory/(key+'-production-hlo.txt')).write_text(hlo_before)
        (directory/(key+'-diagnostic-hlo.txt')).write_text(diag.as_text())
    (directory/(key+'.json')).write_text(json.dumps(result,indent=2))
    print('DONE',key,'maxH=',max(result['metrics']['hessian_'+x]['max_abs'] for x in ('phase','logabs')),
          'deltaT=',abs(t1-t0),flush=True)
    jax.clear_caches()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--smoke',action='store_true',help='最初の1・2層だけ')
    parser.add_argument('--wide-electrons',action='store_true',help='4電子・幅8の非特異な追加条件だけ')
    parser.add_argument('--formal-api',action='store_true',help='正式な多層Diagnostic APIを評価')
    args=parser.parse_args()
    if args.directory.exists():parser.error('既存の計測記録は上書きしない')
    args.directory.mkdir(parents=True)
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('periodicwave').rglob('*.py')}
    (args.directory/'source-hashes.json').write_text(json.dumps(before,indent=2))
    environment=metadata()
    environment['runtime']=dict(backend=jax.default_backend(),devices=[str(d) for d in jax.devices()],
                                matmul_precision=str(jax.config.jax_default_matmul_precision),
                                compiler_options='default; no override')
    (args.directory/'environment.json').write_text(json.dumps(environment,indent=2,default=str))
    reports=[]
    cases=[(1,False,2,'cos'),(2,True,2,'cos')] if args.smoke else CASES
    if args.wide_electrons:cases=[(n,True,4,'cos') for n in (1,2,3)]
    for x64 in (False,True):
        for case in cases:reports.append(run_case(args.directory,x64,*case,width=8 if args.wide_electrons else 4,
                                                formal_api=args.formal_api))
    assert before=={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before}
    (args.directory/'summary.json').write_text(json.dumps(dict(production_source_unchanged=True,cases=reports),indent=2))


if __name__=='__main__':main()
