"""E1: 実checkpoint/configを使うprobeと正式API、保存した旧解析を比較する。"""
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from tests.support import require_numerics, make_case, batch_data, ROOT
require_numerics()
import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import CustomScreenformer as CS, checkpoint
from periodicwave.pbc import feature_layer
from screening_analysis import attn_probe as probe
from tests.measure_diagnostic_layers import metrics


def fixture(directory, mode='cos', x64=True, replicas=1):
    jax.config.update('jax_enable_x64',x64)
    _,_,data,lattice,kw,_=make_case('screen',directory)
    settings=dict(TRIM_MAX=2.,TRIM_POW=4 if mode=='pow4' else 2,
        WIN_SHAPE='smooth7' if mode=='smooth7' else 'cos',FREEZE_W=mode=='freeze',
        CLIP_R=.95 if mode=='hooks' else None,CLIP_BETA=.3,
        jax_enable_x64=x64,jax_default_matmul_precision='float32')
    o=dict(num_layers=2,num_heads=2,attn_dim=3,value_dim=2,mlp_dim=4,
        num_perceptrons_per_layer=2,use_layer_norm=mode!='noln',mlp_activation_fct='GELU',
        use_softmask=mode!='nomask',use_gate=True,use_tanh_norm=True,init_trim_r=1.9,
        init_softmask_w_factor=.8,init_softmask_w_spread=2.5,use_fixed_value=mode=='fixed',
        use_output_scale=True,init_output_scale=0.)
    if mode=='band':o['init_softmask_centres']=([0.,.6],[.5,.5])
    if mode in ('sin','gauss'):o['init_softmask_rings']=([0.,.5,1.],.16,1.,mode)
    if mode=='features':kw['feature_layer']=feature_layer.make_pbc_feature_layer(
        natoms=1,nspins=(1,1),ndim=2,lattice=lattice,include_r_ae=True)
    kw.update(o)
    with patch.multiple(CS,**{k:settings[k] for k in probe.SETTINGS[:6]}):
        net=CS.make_fermi_net((1,1),data.charges,**kw)
        params=net.init(jax.random.PRNGKey(7))
    if mode=='hooks':
        for ap in params['layers']['attention']:
            ap.update(_v_mode=jnp.array([1.,0.]),_diag_only=jnp.array([0.,1.]))
    batch=batch_data(data)
    replicated=jax.tree_util.tree_map(lambda a:jnp.stack([a]*replicas),params)
    saved_data=jax.tree_util.tree_map(lambda a:jnp.stack([a]*replicas),batch)
    path=checkpoint.save(str(directory),0,saved_data,replicated,None,jnp.ones(replicas)*.2)
    # 旧writer形式。ただし17桁を保持し、旧式との差を格子丸め誤差と混同しない。
    L=np.asarray(lattice)
    text='unserializable object: '+np.array2string(L,precision=17,floatmode='unique')
    cfg=dict(network=dict(network_type='CustomScreenformer',CustomScreenformer=o,
                 make_feature_layer_fn='periodicwave.pbc.feature_layer.make_pbc_feature_layer',
                 make_feature_layer_kwargs=dict(lattice=text,include_r_ae=mode=='features')),
             system=dict(electrons=[1,1],ndim=2,pbc_lattice=text))
    (Path(directory)/'config.json').write_text(json.dumps(cfg))
    return net,params,batch,settings,L,Path(path)


def legacy_probe():
    namespace=dict(__name__='legacy_attn_probe_e1',__file__=str(ROOT/'screening_analysis/attn_probe.py'))
    source=(ROOT/'tests/e1_validation/legacy-attn-probe.txt').read_text()
    exec(compile(source,'<saved legacy attn_probe>','exec'),namespace)
    return namespace['probe']


class AttnProbeChecks(unittest.TestCase):
    def tearDown(self):jax.config.update('jax_enable_x64',True)

    def test_api_and_legacy_by_condition(self):
        self.comparisons=[]
        modes=('cos','pow4','smooth7','fixed','noln','nomask','band','sin','gauss','freeze','hooks','features')
        old=legacy_probe()
        for x64 in (False,True):
            for mode in modes:
                with self.subTest(x64=x64,mode=mode),tempfile.TemporaryDirectory() as tmp:
                    net,p,data,settings,L,ck=fixture(tmp,mode,x64)
                    before={k:getattr(CS,k) for k in probe.SETTINGS[:6]}
                    precision=(jax.config.x64_enabled,jax.config.jax_default_matmul_precision)
                    ckhash=hashlib.sha256(ck.read_bytes()).hexdigest()
                    result=probe.probe(tmp,settings=settings,lattice=L)
                    self.assertEqual(before,{k:getattr(CS,k) for k in before})
                    self.assertEqual(precision,(jax.config.x64_enabled,jax.config.jax_default_matmul_precision))
                    self.assertEqual(hashlib.sha256(ck.read_bytes()).hexdigest(),ckhash)
                    with patch.multiple(CS,**{k:settings[k] for k in probe.SETTINGS[:6]}):
                        fn=jax.jit(CS.make_screenformer_diagnostic((1,1),net.options))
                        for i in range(2):
                            x,layers=fn(p,data.positions[i],data.spins[i],data.atoms[i],data.charges[i])
                            for key,source in (('S','similarity'),('W','aggregation_weights'),('W_window','weights_window'),
                                               ('W_intervened','weights'),('Alpha','alpha')):
                                np.testing.assert_array_equal(result[key][i],np.stack([r['attention'][source] for r in layers]))
                            if mode!='nomask':
                                np.testing.assert_array_equal(result['Window'][i],np.stack([r['attention']['window'] for r in layers]))
                        try:
                            legacy=old(tmp)
                        except (IndexError,KeyError,TypeError) as error:
                            self.assertIn(mode,('noln','nomask','features'))
                            self.comparisons.append(dict(x64=x64,mode=mode,legacy_error=type(error).__name__+': '+str(error),new='PASS'))
                        else:
                            diff=metrics(legacy['W'],result['W'])
                            self.comparisons.append(dict(x64=x64,mode=mode,weights=diff,similarity=metrics(legacy['S'],result['S']),
                                distance=metrics(legacy['d'],result['d']),dref_absolute_difference=abs(float(legacy['dref'])-float(result['dref']))))
                            if mode in ('pow4','smooth7','fixed','hooks'):self.assertGreater(diff['max_abs'],1e-5)
                            if mode=='fixed':
                                np.testing.assert_array_equal(np.diagonal(result['W'],axis1=-2,axis2=-1),0.)
                                self.assertTrue(np.any(np.diagonal(legacy['W'],axis1=-2,axis2=-1)>0))
                    if mode=='cos':
                        # D'の入力/paramsと同じ第一walkerの既存基準に直接照合する。
                        prefix='f64' if x64 else 'f32'
                        with np.load(ROOT/f'tests/diagnostic_validation/matrix-v1/{prefix}-L2-LN1-N2-cos.npz') as saved:
                            for i,a in enumerate(jax.tree_util.tree_leaves(p)):np.testing.assert_array_equal(a,saved[f'params_{i}'])
                            for layer in range(2):
                                np.testing.assert_array_equal(result['W'][0,layer],saved[f"aux_layer{layer}_['attention']['aggregation_weights']"])
                    jax.clear_caches()

    def test_missing_settings_and_lossy_config_do_not_guess(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,_,_,settings,L,_=fixture(tmp)
            with self.assertRaisesRegex(ValueError,'未保存設定'):probe.probe(tmp,lattice=L)
            with self.assertRaisesRegex(ValueError,'元精度'):probe.probe(tmp,settings=settings)
            config=Path(tmp)/'config.json';cfg=json.loads(config.read_text())
            # 先頭の無関係な行列を格子として拾わない。
            cfg={'unrelated':'unserializable object: [[9 0] [0 9]]',**cfg}
            np.testing.assert_array_equal(probe.lattice_from_config(cfg),L)
            cfg['screening_attention_settings']=settings
            config.write_text(json.dumps(cfg))
            result=probe.probe(tmp,lattice=L)
            self.assertEqual(result['metadata']['settings'],settings)
            with self.assertRaisesRegex(ValueError,'矛盾'):
                probe.probe(tmp,lattice=L,settings={'TRIM_POW':4})
            del cfg['screening_attention_settings']
            config.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'boolean'):
                probe.probe(tmp,lattice=L,settings={**settings,'FREEZE_W':'false'})
            cfg['network']['CustomScreenformer']['use_layer_norm']=False
            config.write_text(json.dumps(cfg))
            before={k:getattr(CS,k) for k in probe.SETTINGS[:6]}
            with self.assertRaisesRegex(ValueError,'LayerNorm構造'):
                probe.probe(tmp,lattice=L,settings=settings)
            self.assertEqual(before,{k:getattr(CS,k) for k in before})

    def test_all_devices_subsampling_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,_,_,settings,L,ck=fixture(tmp,replicas=2)
            settings_path=Path(tmp)/'known-settings.json';settings_path.write_text(json.dumps(settings))
            result=probe.probe(tmp,settings=settings_path,lattice=L,nconf=3)
            self.assertEqual(result['S'].shape,(3,2,2,2,2))
            self.assertEqual(result['metadata']['selected_indices'],[0,1,3])
            self.assertEqual(result['metadata']['saved_devices'],2)
            self.assertEqual(result['metadata']['checkpoint_sha256'],hashlib.sha256(ck.read_bytes()).hexdigest())
            self.assertEqual(result['metadata']['settings_provenance']['settings_sha256'],hashlib.sha256(settings_path.read_bytes()).hexdigest())

    def test_saving_keeps_old_and_new_outputs_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,_,_,settings,L,_=fixture(tmp)
            old=Path(tmp)/'attn_probe.npz';old.write_bytes(b'existing research cache')
            result=probe.probe(tmp,settings=settings,lattice=L,nconf=1)
            target=Path(tmp)/'attn_probe_diagnostic_v1.npz'
            probe.save_probe(result,target)
            digest=hashlib.sha256(target.read_bytes()).hexdigest()
            with self.assertRaises(FileExistsError):probe.save_probe(result,target)
            with self.assertRaises(ValueError):probe.save_probe(result,old)
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(),digest)
            self.assertEqual(old.read_bytes(),b'existing research cache')
            with np.load(target,allow_pickle=False) as saved:
                metadata=json.loads(str(saved['metadata_json']))
                self.assertEqual(metadata['method'],'screenformer-diagnostic-v1')
                self.assertEqual(metadata['weight_definition'],'aggregation_weights')
                np.testing.assert_array_equal(saved['W'],result['W'])

    def test_cli_uses_separate_name_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,_,_,settings,L,_=fixture(tmp)
            settings_path=Path(tmp)/'settings.json';settings_path.write_text(json.dumps(settings))
            old=Path(tmp)/'attn_probe.npz';old.write_bytes(b'old cache')
            args=[sys.executable,str(ROOT/'screening_analysis/attn_probe.py'),tmp,
                  '--settings',str(settings_path),'--lattice',json.dumps(L.tolist()),'--nconf','1']
            done=subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            self.assertIn('aggregation_weights',done.stdout)
            out=Path(tmp)/'attn_probe_diagnostic_v1.npz'
            digest=hashlib.sha256(out.read_bytes()).hexdigest()
            again=subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
            self.assertNotEqual(again.returncode,0)
            self.assertIn('上書きしません',again.stderr)
            self.assertEqual(hashlib.sha256(out.read_bytes()).hexdigest(),digest)
            self.assertEqual(old.read_bytes(),b'old cache')

    def test_rank_warning_is_not_a_wavefunction_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,_,data,L,kw,_=make_case('screen',tmp)
            settings=dict(TRIM_MAX=2.,TRIM_POW=2,WIN_SHAPE='cos',FREEZE_W=False,CLIP_R=None,CLIP_BETA=.3,
                          jax_enable_x64=True,jax_default_matmul_precision='float32')
            o=dict(num_layers=2,num_heads=2,attn_dim=3,value_dim=2,mlp_dim=4,num_perceptrons_per_layer=2,
                   use_layer_norm=True,mlp_activation_fct='GELU')
            kw.update(o,feature_layer=feature_layer.make_pbc_feature_layer(
                natoms=1,nspins=(2,2),ndim=2,lattice=L,include_r_ae=False))
            net=CS.make_fermi_net((2,2),data.charges,**kw);p=net.init(jax.random.PRNGKey(7))
            data.positions=jnp.array([.23,.41,1.17,.89,.72,1.61,1.91,1.35]);data.spins=jnp.array([1.,1.,-1.,-1.])
            saved=jax.tree_util.tree_map(lambda a:jnp.stack([jnp.stack([a])]),data)
            checkpoint.save(tmp,0,saved,jax.tree_util.tree_map(lambda a:a[None],p),None,jnp.array([.2]))
            cfg=dict(system=dict(ndim=2,electrons=[2,2],pbc_lattice=np.asarray(L).tolist()),
                network=dict(network_type='CustomScreenformer',CustomScreenformer=o,
                    make_feature_layer_fn='periodicwave.pbc.feature_layer.make_pbc_feature_layer',
                    make_feature_layer_kwargs=dict(lattice=np.asarray(L).tolist(),include_r_ae=False)))
            (Path(tmp)/'config.json').write_text(json.dumps(cfg))
            result=probe.probe(tmp,settings=settings)
            self.assertTrue(bool(result['feature_rank_deficient'][0]))
            self.assertTrue(bool(result['finite_features'][0]))
            self.assertTrue(np.isfinite(result['W']).all())
            self.assertIn('not evaluated',result['metadata']['wavefunction_status'])


if __name__=='__main__':unittest.main()
