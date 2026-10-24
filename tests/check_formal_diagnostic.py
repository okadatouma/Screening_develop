"""正式な独立診断API。Productionは厳密、診断はD'の固定測定との比較で判定する。"""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.support import require_numerics, make_case, ROOT
require_numerics()
import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import CustomScreenformer as CS, networks, network_blocks, checkpoint
from tests.check_layers import fixture, CASES
from tests.check_window import assert_exact


class FormalDiagnosticChecks(unittest.TestCase):
    def tearDown(self):jax.config.update('jax_enable_x64',True)

    def test_production_old_loop_and_derivatives_remain_exact(self):
        # 変更前B2ループとの厳密比較。診断の許容誤差をここへ持ち込まない。
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for n,ln,mode in CASES:
                with self.subTest(x64=x64,n=n,ln=ln,mode=mode):
                    p,previous,apply,old_apply,settings,inputs,pos=fixture(n,ln,mode)
                    assert_exact(self,p,previous)
                    def loss(fn,pp,r):return jnp.sum(jnp.sin(fn(pp,*inputs(r))))
                    with patch.multiple(CS,**settings):
                        assert_exact(self,jax.jit(apply)(p,*inputs(pos)),jax.jit(old_apply)(p,*inputs(pos)))
                        if mode in ('cos','pow4','fixed','freeze'):
                            for arg in (0,1):
                                assert_exact(self,jax.jit(jax.grad(lambda pp,r:loss(apply,pp,r),argnums=arg))(p,pos),
                                             jax.jit(jax.grad(lambda pp,r:loss(old_apply,pp,r),argnums=arg))(p,pos))
                            assert_exact(self,jax.jit(jax.hessian(lambda r:loss(apply,p,r)))(pos),
                                         jax.jit(jax.hessian(lambda r:loss(old_apply,p,r)))(pos))
        jax.clear_caches()

    def test_shared_intermediates_and_attention_dimensions(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for ln,heads,qdim,vdim,mode in ((False,1,2,5,'cos'),(True,3,5,2,'fixed')):
                with self.subTest(x64=x64,ln=ln,mode=mode),tempfile.TemporaryDirectory() as tmp:
                    _,_,data,_,opts,_=make_case('screen',tmp)
                    opts.update(num_layers=3,num_heads=heads,attn_dim=qdim,value_dim=vdim,use_layer_norm=ln)
                    opts['use_fixed_value']=mode=='fixed'
                    net=CS.make_fermi_net((1,1),data.charges,**opts)
                    p=net.init(jax.random.PRNGKey(7))
                    fn=CS.make_screenformer_diagnostic((1,1),net.options)
                    args=(p,data.positions,data.spins,data.atoms,data.charges)
                    x,records=fn(*args)
                    _,norm=network_blocks.make_layer_norm();_,mlp=CS.make_mlp(1,'GELU')
                    for i,r in enumerate(records):
                        self.assertEqual(r['attention']['q'].shape,(2,heads,qdim))
                        a=r['attention']
                        if mode=='fixed':
                            aggregate=jnp.einsum('hij,ijd->ihd',a['aggregation_weights'],a['pair_features'])
                            np.testing.assert_array_equal(np.diagonal(np.asarray(a['aggregation_weights']),axis1=1,axis2=2),0.)
                        else:aggregate=jnp.einsum('hij,jhd->ihd',a['aggregation_weights'],a['v_effective'])
                        assert_exact(self,aggregate,a['aggregate'])
                        assert_exact(self,r['attention_residual'],r['input']+r['attention_output'])
                        expected=norm(p['layers']['ln'][i][0],r['attention_residual']) if ln else r['attention_residual']
                        assert_exact(self,r['ln1_output'],expected)
                        assert_exact(self,r['mlp_output'],mlp(p['layers']['mlp'][i],expected))
                        expected=expected+r['mlp_output']
                        expected=norm(p['layers']['ln'][i][1],expected) if ln else expected
                        assert_exact(self,r['output'],expected)
                        if i+1<3:assert_exact(self,r['output'],records[i+1]['input'])
                    compiled,aux=jax.jit(fn)(*args)
                    assert_exact(self,compiled,aux[-1]['output'])
                    for a,b in zip(aux,aux[1:]):assert_exact(self,a['output'],b['input'])
        jax.clear_caches()

    def test_rank_status_keeps_attention_defined(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            # D'の4電子・幅4の保存済み診断X。Attention自体は有限で定義されている。
            directory=Path(__file__).parent/'diagnostic_validation/matrix-v1'
            for n in (1,2,3):
                path=directory/f"{'f64' if x64 else 'f32'}-L{n}-LN1-N4-cos.npz"
                with np.load(path) as data:
                    x=jnp.asarray(data[f"aux_layer{n-1}_['output']"])
                    # rankを変えない射影で構造的な節を独立に確認する。
                    orbitals=[x[None,...].astype(jnp.complex128 if x64 else jnp.complex64)]
                    status=jax.jit(CS.screening_diagnostic_status)(x,orbitals=orbitals)
                    self.assertTrue(bool(status['feature_rank_deficient']))
                    self.assertTrue(bool(status['all_determinants_rank_deficient']))
                    self.assertTrue(bool(status['wavefunction_unstable_or_undefined']))
                    self.assertTrue(bool(status['local_energy_unstable_or_undefined']))
                    for i in range(n):
                        prefix=f"aux_layer{i}_['attention']"
                        w=data[prefix+"['aggregation_weights']"]
                        v=data[prefix+"['v_effective']"]
                        aggregate=data[prefix+"['aggregate']"]
                        self.assertTrue(np.isfinite(w).all())
                        reference=np.einsum('hij,jhd->ihd',w.astype(np.float64),v.astype(np.float64))
                        magnitude=np.einsum('hij,jhd->ihd',np.abs(w).astype(np.float64),np.abs(v).astype(np.float64))
                        eps=np.finfo(aggregate.dtype).eps
                        # 通常の内積の前進誤差上界。log微分の特異性とは独立に検算。
                        bound=(4*eps/(1-4*eps))*magnitude+eps*np.abs(reference)
                        self.assertTrue(np.all(np.abs(aggregate-reference)<=bound))
            unchecked=CS.screening_diagnostic_status(jnp.eye(2))
            self.assertIsNone(unchecked['wavefunction_unstable_or_undefined'])
            invalid=CS.screening_diagnostic_status(jnp.eye(2),phase_logabs=jnp.array([0.,-jnp.inf]))
            self.assertTrue(bool(invalid['wavefunction_unstable_or_undefined']))
        jax.clear_caches()

    def test_actual_orbitals_status_for_nodal_and_regular_models(self):
        from periodicwave.pbc import feature_layer
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for width in (4,8):
                with self.subTest(x64=x64,width=width),tempfile.TemporaryDirectory() as tmp:
                    _,_,data,lattice,opts,_=make_case('screen',tmp)
                    data.positions=jnp.array([.23,.41,1.17,.89,.72,1.61,1.91,1.35])
                    data.spins=jnp.array([1.,1.,-1.,-1.])
                    opts.update(num_layers=2,mlp_dim=width,attn_dim=3,value_dim=2,
                        feature_layer=feature_layer.make_pbc_feature_layer(
                            natoms=1,nspins=(2,2),ndim=2,lattice=lattice,include_r_ae=False))
                    net=CS.make_fermi_net((2,2),data.charges,**opts)
                    p=net.init(jax.random.PRNGKey(7))
                    x,_=jax.jit(CS.make_screenformer_diagnostic((2,2),net.options))(
                        p,data.positions,data.spins,data.atoms,data.charges)
                    init,_=CS.make_screenformer_layers((2,2),1,net.options)
                    _,orbital_fn=networks.make_orbitals((2,2),data.charges,net.options,
                                                       (init,lambda *args,**kwargs:x))
                    orbitals=orbital_fn(p,data.positions,data.spins,data.atoms,data.charges)
                    wave=network_blocks.logdet_matmul(orbitals)
                    status=jax.jit(CS.screening_diagnostic_status)(x,orbitals=orbitals,phase_logabs=wave)
                    self.assertEqual(bool(status['all_determinants_rank_deficient']),width==4)
                    self.assertEqual(bool(status['wavefunction_unstable_or_undefined']),width==4)
                    self.assertTrue(bool(status['orbital_check_performed']))
        jax.clear_caches()

    def test_saved_checkpoint_entry(self):
        root=ROOT/'tests/layernorm_validation/before-checkpoints'
        for record in json.loads((root/'evaluations.json').read_text()):
            if record['route'] not in ('screen','lambda','hf'):continue
            jax.config.update('jax_enable_x64',record['x64'])
            path=ROOT/record['path']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),record['sha256'])
            with tempfile.TemporaryDirectory() as tmp:
                net,initial,_,lattice,_,_=make_case(record['route'],tmp)
            _,data,p,_,_=checkpoint.restore(str(path),batch_size=2)
            p=jax.tree_util.tree_map(lambda x:jnp.asarray(x[0]),p)
            data=jax.tree_util.tree_map(lambda x:jnp.asarray(x[0,0]),data)
            assert_exact(self,p,initial)
            nspins=(1,1) if record['route']=='screen' else (2,0)
            fn=CS.make_screenformer_diagnostic(nspins,net.options)
            args=(p,data.positions,data.spins,data.atoms,data.charges)
            production=jax.jit(net.apply)
            before=production(*args)
            x,layers=jax.jit(fn)(*args)
            assert_exact(self,x,layers[-1]['output'])
            assert_exact(self,before,production(*args))
            _,trunk=CS.make_screenformer_layers(nspins,1,net.options)
            geometry=networks.make_periodic_construct_input_features(lattice)
            def ordinary(pp,r):
                ae,ee,ra,re=geometry(r,data.atoms,ndim=2)
                return trunk(pp['layers'],ae=ae,ee=ee,r_ae=ra,r_ee=re,spins=data.spins,charges=data.charges)
            assert_exact(self,jax.jit(lambda pp,r:fn(pp,r,data.spins,data.atoms,data.charges)[0])(p,data.positions),
                         jax.jit(ordinary)(p,data.positions))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),record['sha256'])
        jax.clear_caches()

    def test_dprime_36_conditions_and_all_internal_arrays(self):
        root=Path(__file__).parent
        # 条件別・量別・精度別に固定したD'配列との厳密比較を維持できることを要求。
        # Production-vs-Diagnosticの差をゼロにするテストではない。
        for oldname,newname in (('matrix-v1','matrix'),('matrix-wide-v1','wide')):
            olddir=root/'diagnostic_validation'/oldname
            newdir=root/'formal_d_validation'/newname
            if not (newdir/'summary.json').exists():self.fail('正式APIの36条件計測が未完了: '+newname)
            old=json.loads((olddir/'summary.json').read_text())['cases']
            new=json.loads((newdir/'summary.json').read_text())['cases']
            old_env=json.loads((olddir/'environment.json').read_text())
            new_env=json.loads((newdir/'environment.json').read_text())
            for key in ('versions','python','platform'):self.assertEqual(old_env[key],new_env[key])
            self.assertEqual([r['case'] for r in old],[r['case'] for r in new])
            for before,after in zip(old,new):
                with self.subTest(case=before['case']):
                    self.assertTrue(after['formal_api'])
                    self.assertEqual(before['params_sha256'],after['params_sha256'])
                    with np.load(olddir/(before['case']+'.npz')) as a,np.load(newdir/(after['case']+'.npz')) as b:
                        self.assertEqual(a.files,b.files)
                        for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=before['case']+': '+key)
                    self.assertEqual(before['metrics'],after['metrics'])

    def test_standalone_api_36_conditions(self):
        from periodicwave.pbc import feature_layer
        from tests.measure_diagnostic_layers import benchmark
        self.standalone_measurements=[]
        root=Path(__file__).parent/'diagnostic_validation'
        for dirname in ('matrix-v1','matrix-wide-v1'):
            for case in json.loads((root/dirname/'summary.json').read_text())['cases']:
                jax.config.update('jax_enable_x64',case['x64'])
                with self.subTest(case=case['case']),tempfile.TemporaryDirectory() as tmp:
                    with np.load(root/dirname/(case['case']+'.npz')) as saved:
                        _,_,data,lattice,opts,_=make_case('screen',tmp)
                        ne=case['electrons'];nspins=(ne//2,ne//2)
                        data.positions=jnp.asarray(saved['positions']);data.spins=jnp.asarray(saved['spins'])
                        opts.update(num_layers=case['layers'],use_layer_norm=case['layernorm'],
                            mlp_dim=case['model']['mlp_dim'],num_perceptrons_per_layer=2,attn_dim=3,value_dim=2,
                            use_fixed_value=case['mode']=='fixed',use_softmask=case['mode']!='nomask',
                            feature_layer=feature_layer.make_pbc_feature_layer(
                                natoms=1,nspins=nspins,ndim=2,lattice=lattice,include_r_ae=False))
                        if case['mode'] in ('sin','gauss'):
                            opts['init_softmask_rings']=case['model']['init_softmask_rings']
                        with patch.multiple(CS,**case['effective_settings']):
                            net=CS.make_fermi_net(nspins,data.charges,**opts)
                            p=net.init(jax.random.PRNGKey(7))
                            for i,a in enumerate(jax.tree_util.tree_leaves(p)):
                                np.testing.assert_array_equal(a,saved[f'params_{i}'])
                            args=(p,data.positions,data.spins,data.atoms,data.charges)
                            compiled=jax.jit(CS.make_screenformer_diagnostic(nspins,net.options)).lower(*args).compile()
                            x,layers=compiled(*args)
                            for i,r in enumerate(layers):
                                for path,a in jax.tree_util.tree_flatten_with_path(r)[0]:
                                    key=f'aux_layer{i}_'+''.join(str(k) for k in path)
                                    np.testing.assert_array_equal(a,saved[key],err_msg=case['case']+': '+key)
                            status=CS.screening_diagnostic_status(x)
                            nodal=ne==4 and case['model']['mlp_dim']==4
                            self.assertEqual(bool(status['feature_rank_deficient']),nodal)
                            self.assertIsNone(status['wavefunction_unstable_or_undefined'])
                            self.standalone_measurements.append(dict(case=case['case'],
                                feature_rank_deficient=bool(status['feature_rank_deficient']),
                                performance=benchmark(compiled,args)))
                jax.clear_caches()


if __name__=='__main__':unittest.main()
