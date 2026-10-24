"""E2a: Head分解だけを独立に検算。本体/Diagnosticの既存許容誤差を変更しない。"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.support import require_numerics, ROOT, make_case, batch_data
require_numerics()
import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import CustomScreenformer as CS, checkpoint
from screening_analysis import head_contrib as hc
from screening_analysis import attn_probe as probe
from tests.check_attn_probe import fixture as probe_fixture
from tests.measure_diagnostic_layers import metrics

CASES=(('cos',1,1),('cos',2,2),('cos',3,3),('pow4',2,2),('smooth7',2,2),
       ('nomask',2,2),('fixed',2,2),('noln',1,1),('noln',2,3),
       ('band',2,2),('sin',2,2),('gauss',2,2),('freeze',2,2),('hooks',2,2))


def fixture(directory,mode='cos',x64=True,layers=2,heads=2,bypass=False):
    net,p,data,settings,L,ck=probe_fixture(directory,mode,x64)
    cfgpath=Path(directory)/'config.json';cfg=json.loads(cfgpath.read_text())
    if layers!=2 or heads!=2 or bypass:
        _,_,single,_,kw,_=make_case('screen',directory)
        opts=dict(cfg['network']['CustomScreenformer'])
        opts.update(num_layers=layers,num_heads=heads)
        if bypass:opts.update(use_gate=False,use_tanh_norm=False,use_output_scale=False)
        with patch.multiple(CS,**{k:settings[k] for k in probe.SETTINGS[:6]}):
            net=CS.make_fermi_net((1,1),single.charges,**kw|opts)
            p=net.init(jax.random.PRNGKey(7))
        cfg['network']['CustomScreenformer']=opts;cfgpath.write_text(json.dumps(cfg))
        checkpoint.save(str(directory),0,jax.tree_util.tree_map(lambda a:a[None],data),
                        jax.tree_util.tree_map(lambda a:a[None],p),None,jnp.array([.2]))
    return net,p,data,settings,L,ck


def legacy_compute():
    ns=dict(__name__='legacy_head_contrib_e2a',__file__=str(ROOT/'screening_analysis/head_contrib.py'))
    exec(compile((ROOT/'tests/e2a_validation/legacy-head-contrib.txt').read_text(),'<legacy head_contrib>','exec'),ns)
    return ns['compute']


class HeadContribChecks(unittest.TestCase):
    def tearDown(self):jax.config.update('jax_enable_x64',True)

    def test_api_and_legacy_matrix(self):
        self.comparisons=[];old=legacy_compute()
        for x64 in (False,True):
            for mode,n,h in CASES:
                with self.subTest(x64=x64,mode=mode,layers=n,heads=h),tempfile.TemporaryDirectory() as tmp:
                    net,p,data,settings,L,ck=fixture(tmp,mode,x64,n,h)
                    saved_hash=hashlib.sha256(ck.read_bytes()).hexdigest()
                    before={k:getattr(CS,k) for k in probe.SETTINGS[:6]}
                    r=hc.analyze(tmp,settings=settings,lattice=L)
                    self.assertEqual(before,{k:getattr(CS,k) for k in before})
                    self.assertEqual(saved_hash,hashlib.sha256(ck.read_bytes()).hexdigest())
                    details=dict(x64=x64,mode=mode,layers=n,heads=h,
                        decomposition=metrics(r['attention_output'],r['decomposition_sum']))
                    # 独立スカラー積（NumPy float64）と誤差の前進上界。許容誤差を測定に合わせて拡大しない。
                    for sample in range(2):
                        with patch.multiple(CS,**{k:settings[k] for k in probe.SETTINGS[:6]}):
                            fn=jax.jit(CS.make_screenformer_diagnostic((1,1),net.options))
                            _,records=fn(p,data.positions[sample],data.spins[sample],data.atoms[sample],data.charges[sample])
                        for l,record in enumerate(records):
                            a=record['attention'];ap=p['layers']['attention'][l]
                            for key,expected in (('gate_output',a['gate_output']),('aggregate',a['aggregate']),
                                    ('tanh_output',a['tanh_output']),('aggregation_weights',a['aggregation_weights']),
                                    ('weights_window',a['weights_window']),('layer_input',record['input']),
                                    ('layer_output',record['output']),('attention_output',record['attention_output'])):
                                np.testing.assert_array_equal(r[key][sample,l],expected)
                            if l:np.testing.assert_array_equal(r['layer_input'][sample,l],r['layer_output'][sample,l-1])
                            z=np.asarray(a['gate_output'],np.float64).reshape(2,h,-1)
                            w=np.asarray(ap['attn_output'],np.float64).reshape(h,z.shape[-1],-1)
                            scale=np.exp(np.asarray(ap['out_scale']))
                            expected=np.zeros((2,h,w.shape[-1]))
                            for i in range(2):
                                for head in range(h):
                                    for out in range(w.shape[-1]):
                                        expected[i,head,out]=scale*sum(z[i,head,d]*w[head,d,out] for d in range(z.shape[-1]))
                            eps=np.finfo(r['head_vectors'].dtype).eps
                            bound=16*eps*max(z.shape[-1],1)*max(1.,float(np.max(np.abs(expected))))
                            np.testing.assert_allclose(r['head_vectors'][sample,l],expected,atol=bound,rtol=0)
                            # 分解の和はfull matmulとは別の縮約順。次層に戻さない。
                            bound=32*eps*max(1,h*z.shape[-1])*max(1.,float(np.max(np.abs(expected))))
                            np.testing.assert_allclose(r['decomposition_sum'][sample,l],record['attention_output'],atol=bound,rtol=0)
                            # 自己項の集約は実際のaggregation_weightsの対角で定義する。
                            if mode=='fixed':
                                np.testing.assert_array_equal(r['self_aggregate'][sample,l],0)
                                np.testing.assert_array_equal(r['self_head_vectors'][sample,l],0)
                            else:
                                diagonal=np.diagonal(np.asarray(a['aggregation_weights']),axis1=1,axis2=2).T[...,None]
                                self_ref=diagonal*np.asarray(a['v_effective'])
                                np.testing.assert_allclose(r['self_aggregate'][sample,l],self_ref,atol=16*eps,rtol=16*eps)
                            if mode!='fixed':
                                self_stage=np.asarray(self_ref,dtype=np.float64)
                                norm=np.sqrt(np.sum(self_stage*self_stage,axis=-1,keepdims=True)+1e-12)
                                self_stage=self_stage*np.tanh(norm)/norm
                                gate=np.asarray(record['input'],np.float64)@np.asarray(ap['g_w'],np.float64)
                                gate=np.tanh(gate/(1+np.exp(-gate)))
                                self_stage=self_stage.reshape(2,-1)*gate
                                ref=scale*np.einsum('ihd,hde->ihe',self_stage.reshape(2,h,-1),w)
                                bound=64*eps*max(1,gate.shape[-1])*max(1.,float(np.max(np.abs(ref))))
                                np.testing.assert_allclose(r['self_head_vectors'][sample,l],ref,atol=bound,rtol=0)
                            if mode=='cos' and n==2:
                                prefix='f64' if x64 else 'f32'
                                with np.load(ROOT/f'tests/diagnostic_validation/matrix-v1/{prefix}-L2-LN1-N2-cos.npz') as frozen:
                                    if sample==0:
                                        np.testing.assert_array_equal(a['aggregation_weights'],frozen[f"aux_layer{l}_['attention']['aggregation_weights']"])
                    with patch.multiple(CS,**{k:settings[k] for k in probe.SETTINGS[:6]}):
                        try:previous=old(tmp)
                        except (IndexError,KeyError,TypeError) as error:
                            self.assertIn(mode,('noln','nomask'))
                            details['legacy_error']=type(error).__name__+': '+str(error)
                        else:
                            details['legacy']={key:metrics(previous[j],r[key]) for j,key in enumerate(('C','Cd','Cp','X','R'))}
                            if mode in ('pow4','smooth7','hooks'):
                                self.assertGreater(details['legacy']['C']['max_abs'],1e-6)
                    self.comparisons.append(details)
                    jax.clear_caches()

    def test_projection_bias_scale_and_nonlinear_distinction(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            dtype=jnp.float64 if x64 else jnp.float32
            z=jnp.arange(12,dtype=dtype).reshape(2,6)/7
            w=jnp.arange(24,dtype=dtype).reshape(6,4)/11;b=jnp.arange(4,dtype=dtype)/5
            c,bias=hc.projected_heads(z,dict(w=w,b=b),3,2.)
            eps=np.finfo(np.asarray(z).dtype).eps
            np.testing.assert_allclose(np.sum(c,axis=1)+bias,2*(z@w+b),atol=128*eps,rtol=0)
            np.testing.assert_array_equal(bias,2*b)
            # 非線形な自己応答差 != 他電子だけを独立に通した値。
            a=np.array([.8,.4]);ad=np.array([.5,.1])
            f=lambda x:x*np.tanh(np.sqrt(np.sum(x*x)+1e-12))/np.sqrt(np.sum(x*x)+1e-12)
            self.assertGreater(np.max(np.abs(f(a)-f(ad)-f(a-ad))),.01)

    def test_bypass_flags_and_tuple_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,_,_,s,L,_=fixture(tmp,bypass=True)
            r=hc.analyze(tmp,settings=s,lattice=L)
            np.testing.assert_array_equal(r['aggregate'],r['tanh_output'])
            np.testing.assert_array_equal(r['gate_output'],r['aggregate'].reshape(2,2,2,-1))
            t=hc.compute(tmp,nconf=1,settings=s,lattice=L)
            self.assertEqual(len(t),8);self.assertEqual(t[0].shape,(2,2));self.assertFalse(t[6])
            self.assertIn('metadata',t[-1])

    def test_legacy_derived_statistics_and_all_device_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,_,_,settings,L,_=probe_fixture(tmp,replicas=2)
            r=hc.analyze(tmp,nconf=3,settings=settings,lattice=L)
            self.assertEqual(r['metadata']['selected_indices'],[0,1,3])
            self.assertEqual(r['metadata']['saved_devices'],2)
            for source,target in (('C','q_C'),('Cp','q_Cp'),('Ap','q_Ap')):
                power=r[source]**2
                np.testing.assert_array_equal(r[target],power/power.sum())
            power=(r['Cp']*r['A']/np.maximum(np.tanh(r['A']),1e-12))**2
            np.testing.assert_allclose(r['q_g_heuristic'],power/power.sum(),atol=1e-16,rtol=1e-15)
            np.testing.assert_array_equal(r['alignment_ratio'],r['C'].sum(axis=1)/r['R'])
            np.testing.assert_array_equal(r['layer_index'],[0,1])
            self.assertIn('mean-square',r['metadata']['statistic'])
            # 棄却条件も本体設定を元に戻す。
            cfgpath=Path(tmp)/'config.json';cfg=json.loads(cfgpath.read_text())
            tanh=cfg['network']['CustomScreenformer'].pop('use_tanh_norm')
            cfgpath.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'必須'):
                hc.analyze(tmp,settings=settings,lattice=L)
            cfg['network']['CustomScreenformer']['use_tanh_norm']=tanh
            feature=cfg['network'].pop('make_feature_layer_fn')
            cfgpath.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'feature factory'):
                hc.analyze(tmp,settings=settings,lattice=L)
            cfg['network']['make_feature_layer_fn']=feature
            cfg['network']['CustomScreenformer']['use_layer_norm']=False
            cfgpath.write_text(json.dumps(cfg))
            before={k:getattr(CS,k) for k in probe.SETTINGS[:6]}
            with self.assertRaisesRegex(ValueError,'LayerNorm構造'):
                hc.analyze(tmp,settings=settings,lattice=L)
            self.assertEqual(before,{k:getattr(CS,k) for k in before})

    def test_settings_selection_protection_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,_,_,settings,L,_=fixture(tmp)
            with self.assertRaisesRegex(ValueError,'未保存設定'):hc.analyze(tmp,lattice=L)
            with self.assertRaisesRegex(ValueError,'元精度'):hc.analyze(tmp,settings=settings)
            with self.assertRaisesRegex(ValueError,'sel'):hc.analyze(tmp,sel=[-1],settings=settings,lattice=L)
            r=hc.analyze(tmp,sel=[1,0],settings=settings,lattice=L)
            self.assertEqual(r['metadata']['selected_indices'],[1,0])
            old=Path(tmp)/'head_contrib.npz';old.write_bytes(b'old research cache')
            out=Path(tmp)/'head_contrib_diagnostic_v1.npz';hc.save(r,out)
            with self.assertRaises(FileExistsError):hc.save(r,out)
            with self.assertRaises(ValueError):hc.save(r,old)
            with np.load(out,allow_pickle=False) as z:
                self.assertEqual(json.loads(str(z['metadata_json']))['method'],'head-contrib-diagnostic-v1')
                np.testing.assert_array_equal(z['head_vectors'],r['head_vectors'])
            self.assertEqual(old.read_bytes(),b'old research cache')
            settings_file=Path(tmp)/'settings.json';settings_file.write_text(json.dumps(settings))
            target=Path(tmp)/'cli_diagnostic_v1.npz'
            args=[sys.executable,str(ROOT/'screening_analysis/head_contrib.py'),tmp,'--settings',str(settings_file),
                  '--lattice',json.dumps(L.tolist()),'--nconf','1','--output',str(target)]
            done=subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            self.assertIn('head-contrib-diagnostic-v1',done.stdout)
            self.assertNotEqual(subprocess.run(args,cwd=ROOT,capture_output=True).returncode,0)
            cfgfile=Path(tmp)/'config.json';cfg=json.loads(cfgfile.read_text())
            cfg['network']['network_type']='CustomPsiformer';cfgfile.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'専用'):hc.analyze(tmp,settings=settings,lattice=L)


if __name__=='__main__':unittest.main()
