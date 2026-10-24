"""E2b: 相殺比の独立検算、旧解析比較、正式API/保存・import保護。"""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.support import require_numerics, ROOT
require_numerics()
import attr
import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import CustomScreenformer as CS, checkpoint
from screening_analysis import head_cancel as cancel, head_contrib as contrib, attn_probe as probe
from tests.check_head_contrib import fixture as head_fixture, CASES
from tests.measure_diagnostic_layers import metrics

EXTRA=(('gateoff',2,2),('scaleoff',2,2),('tanhoff',2,2),('bypass',2,2),
       ('diagonly',2,2),('valueonly',2,2),('cliponly',2,2))


def fixture(tmp,mode,x64,n,h):
    base='fixed' if mode=='widefixed' else ('cos' if mode in {x[0] for x in EXTRA} else mode)
    net,p,data,s,L,ck=head_fixture(tmp,base,x64,n,h)
    cfgpath=Path(tmp)/'config.json';cfg=json.loads(cfgpath.read_text())
    overrides={}
    if mode in ('gateoff','bypass'):overrides['use_gate']=False
    if mode in ('scaleoff','bypass'):overrides['use_output_scale']=False
    if mode in ('tanhoff','bypass'):overrides['use_tanh_norm']=False
    for ap in p['layers']['attention']:
        if overrides.get('use_gate') is False:ap.pop('g_w')
        if overrides.get('use_output_scale') is False:ap.pop('out_scale')
        if mode=='diagonly':ap['_diag_only']=jnp.ones(h)
        if mode=='valueonly':ap['_v_mode']=jnp.array([1.,0.])
        if mode in ('wide','widefixed'):ap['mask_omega']=jnp.full_like(ap['mask_omega'],2.)
    if mode=='cliponly':s['CLIP_R']=.95
    cfg['network']['CustomScreenformer'].update(overrides);cfgpath.write_text(json.dumps(cfg))
    checkpoint.save(tmp,0,jax.tree_util.tree_map(lambda x:x[None],data),
                    jax.tree_util.tree_map(lambda x:x[None],p),None,jnp.array([.2]))
    return attr.evolve(net.options,**overrides),p,data,s,L,ck


def legacy_compute():
    ns={'__name__':'legacy_head_cancel_e2b','__file__':str(ROOT/'screening_analysis/head_cancel.py')}
    exec(compile((ROOT/'tests/e2b_validation/legacy-head-cancel.txt').read_text(),'<legacy head_cancel>','exec'),ns)
    return ns['compute']


class HeadCancelChecks(unittest.TestCase):
    def tearDown(self):jax.config.update('jax_enable_x64',True)

    def test_old_new_and_api_matrix(self):
        self.comparisons=[];old=legacy_compute()
        for x64 in (False,True):
            for mode,n,h in CASES+EXTRA:
                with self.subTest(x64=x64,mode=mode,layers=n,heads=h),tempfile.TemporaryDirectory() as tmp:
                    options,p,data,s,L,ck=fixture(tmp,mode,x64,n,h)
                    before={k:getattr(CS,k) for k in probe.SETTINGS[:6]}
                    digest=hashlib.sha256(ck.read_bytes()).hexdigest()
                    r=cancel.analyze(tmp,settings=s,lattice=L)
                    self.assertEqual(before,{k:getattr(CS,k) for k in before})
                    self.assertEqual(digest,hashlib.sha256(ck.read_bytes()).hexdigest())
                    details=dict(x64=x64,mode=mode,layers=n,heads=h,
                        reconstruction=metrics(r['attention_output'],r['decomposition_sum']),
                        rho=r['rho'].tolist(),rho_p=r['rho_p'].tolist(),
                        rho_defined=r['rho_defined'].tolist(),rho_p_defined=r['rho_p_defined'].tolist())
                    # 正式APIの独立JITを直接呼ぶ。層を解析側で更新しない。
                    with patch.multiple(CS,**{k:s[k] for k in probe.SETTINGS[:6]}):
                        _,layers=jax.jit(CS.make_screenformer_diagnostic((1,1),options))(
                            p,data.positions[0],data.spins[0],data.atoms[0],data.charges[0])
                    for l,record in enumerate(layers):
                        a=record['attention'];ap=p['layers']['attention'][l]
                        scale=jnp.exp(ap['out_scale']) if options.use_output_scale else 1.
                        c,b=contrib.projected_heads(a['gate_output'],ap['attn_output'],h,scale)
                        for key,expected in (('head_vectors',c),('bias',b),('aggregation_weights',a['aggregation_weights']),
                                ('layer_input',record['input']),('layer_output',record['output']),
                                ('attention_output',record['attention_output'])):
                            np.testing.assert_array_equal(r[key][0,l],expected)
                        if l:np.testing.assert_array_equal(r['layer_input'][0,l],r['layer_output'][0,l-1])
                    # 独立な実数スカラー/Headペアの内積で平均二乗と交差項を検算する。
                    c=np.asarray(r['head_vectors'],np.float64)
                    cp=c-np.asarray(r['self_head_vectors'],np.float64)
                    ref=np.zeros((n,4))
                    for l in range(n):
                        for b in range(c.shape[0]):
                            for i in range(c.shape[2]):
                                for offset,v in ((0,c),(2,cp)):
                                    ref[l,offset]+=sum(float(v[b,l,i,k,e])**2 for k in range(h) for e in range(c.shape[-1]))
                                    ref[l,offset+1]+=sum(float(v[b,l,i,k,e])*float(v[b,l,i,j,e])
                                        for k in range(h) for j in range(h) for e in range(c.shape[-1]))
                    ref/=c.shape[0]*c.shape[2]
                    eps=np.finfo(r['head_vectors'].dtype).eps
                    bound=64*eps*h*c.shape[-1]*max(1.,float(np.max(np.abs(ref))))
                    np.testing.assert_allclose(r['R'],ref,atol=bound,rtol=0)
                    np.testing.assert_allclose(r['decomposition_sum'],r['attention_output'],
                        atol=64*eps*h*c.shape[-1]*max(1.,float(np.max(np.abs(r['attention_output'])))),rtol=0)
                    for key,den,num,defined in (('rho',0,1,'rho_defined'),('rho_p',2,3,'rho_p_defined')):
                        for l in range(n):
                            if r[defined][l]:
                                self.assertGreaterEqual(r[key][l],-8*eps*h)
                                self.assertLessEqual(r[key][l],h+8*eps*h)
                                np.testing.assert_allclose(r[key][l],ref[l,num]/ref[l,den],rtol=64*eps*h*c.shape[-1],atol=0)
                            else:self.assertTrue(np.isnan(r[key][l]))
                    if mode=='fixed':
                        np.testing.assert_array_equal(r['self_head_vectors'],0.)
                        np.testing.assert_array_equal(r['rho'],r['rho_p'])
                    if mode=='diagonly':
                        # 全Headが自己項だけなら自己応答差は0。微小丸め差も分母へ反映する。
                        details['self_response_difference_max_abs']=float(np.max(np.abs(cp)))
                    with patch.multiple(CS,**{k:s[k] for k in probe.SETTINGS[:6]}):
                        try:previous=old(tmp)
                        except (IndexError,KeyError,TypeError) as error:
                            self.assertIn(mode,('nomask','noln'))
                            details['legacy_error']=type(error).__name__+': '+str(error)
                        else:
                            details['legacy_moments']=metrics(previous[4],r['R'])
                            ratios=[]
                            for offset,key in ((0,'rho'),(2,'rho_p')):
                                den=previous[4][:,offset];num=previous[4][:,offset+1]
                                valid=(den>0)&np.isfinite(den)&r[key+'_defined']
                                legacy_ratio=np.divide(num,den,out=np.full_like(den,np.nan),where=den>0)
                                ratios.append(dict(quantity=key,legacy=legacy_ratio.tolist(),valid=valid.tolist(),
                                    metrics=metrics(legacy_ratio[valid],r[key][valid]) if valid.any() else None))
                            details['legacy_ratios']=ratios
                    self.comparisons.append(details)
                    jax.clear_caches()

    def test_analytic_cancellation_and_ratio_of_means(self):
        for dtype in (np.float32,np.float64):
            # 直交、完全相殺、完全整列。Head数H=2。
            for vectors,rho in (([[1,0],[0,1]],1),([[1,0],[-1,0]],0),([[1,0],[1,0]],2)):
                c=np.array(vectors,dtype=dtype)[None,None,None]
                R,q,qp,ok,okp=cancel.cancellation_statistics(c,np.zeros_like(c))
                np.testing.assert_array_equal(q,[rho]);np.testing.assert_array_equal(qp,q)
                self.assertTrue(ok[0] and okp[0])
                # biasをHead間へ分配すると別の相殺指標になる。共有項は除外する。
                _,with_bias,_,_,_=cancel.cancellation_statistics(c+3,np.zeros_like(c))
                if rho==0:self.assertGreater(with_bias[0],0)
            c=np.array([[[[[1.,0.],[1.,0.]]]],[[[[10.,0.],[-10.,0.]]]]],dtype=dtype)
            R,rho,_,_,_=cancel.cancellation_statistics(c,np.zeros_like(c))
            np.testing.assert_allclose(rho,4/202,rtol=4*np.finfo(dtype).eps)
            self.assertNotEqual(float(rho[0]),1.) # mean of per-sample ratios would be (2+0)/2
            zero=np.zeros((1,1,1,2,2),dtype=dtype)
            _,q,qp,ok,okp=cancel.cancellation_statistics(zero,zero)
            self.assertTrue(np.isnan(q).all() and np.isnan(qp).all());self.assertFalse(ok.any() or okp.any())
            c=np.ones_like(zero)
            _,q,qp,ok,okp=cancel.cancellation_statistics(c,c)
            self.assertTrue(ok.all());self.assertTrue(np.isnan(qp).all());self.assertFalse(okp.any())
        with self.assertRaises(ValueError):cancel.cancellation_statistics(np.zeros((2,2)),np.zeros((2,2)))

    def test_wide_window_nontrivial_cancellation(self):
        # 通常の小さい初期窓ではrho_p≈1に退化しやすい。複数Headの非対角項を明示的に有効化。
        with patch.multiple(sys.modules[__name__],CASES=(),EXTRA=(('wide',3,3),('widefixed',2,3))):
            self.test_old_new_and_api_matrix()
        for row in self.comparisons:
            self.assertTrue(all(row['rho_p_defined']))
            self.assertTrue(any(abs(value-1.)>1e-3 for value in row['rho_p']))

    def test_self_only_warning_preserves_raw_statistics(self):
        self.self_only=[]
        for x64 in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                _,_,_,settings,L,_=fixture(tmp,'diagonly',x64,2,2)
                r=cancel.analyze(tmp,settings=settings,lattice=L)
                self.assertTrue(r['rho_p_self_only_warning'].all())
                moments,rho,rho_p,defined,defined_p=cancel.cancellation_statistics(
                    r['head_vectors'],r['self_head_vectors'])
                np.testing.assert_array_equal(r['R'],moments)
                np.testing.assert_array_equal(r['rho_p'],rho_p)
                self.self_only.append(dict(x64=x64,R=r['R'].tolist(),rho_p=r['rho_p'].tolist(),
                    floating_denominator_positive=r['rho_p_defined'].tolist(),
                    self_only_warning=r['rho_p_self_only_warning'].tolist(),
                    response_difference_max_abs=float(np.max(np.abs(r['head_vectors']-r['self_head_vectors'])))))
        # 非有限値もepsilonで有限化しない。
        c=np.full((1,1,1,2,2),np.nan)
        _,rho,rho_p,ok,okp=cancel.cancellation_statistics(c,c)
        self.assertTrue(np.isnan(rho).all() and np.isnan(rho_p).all())
        self.assertFalse(ok.any() or okp.any())

    def test_io_tuple_provenance_cli_and_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,_,_,s,L,ck=fixture(tmp,'cos',True,2,2)
            with self.assertRaisesRegex(ValueError,'未保存設定'):cancel.analyze(tmp,lattice=L)
            with self.assertRaisesRegex(ValueError,'元精度'):cancel.analyze(tmp,settings=s)
            r=cancel.analyze(tmp,settings=s,lattice=L,sel=[1])
            self.assertEqual(r['metadata']['selected_indices'],[1])
            self.assertEqual(r['metadata']['method'],'head-cancel-diagnostic-v1')
            self.assertEqual(r['metadata']['parent_method'],'head-contrib-diagnostic-v1')
            t=cancel.compute(tmp,nconf=1,settings=s,lattice=L)
            self.assertEqual(len(t),8);self.assertEqual(t[4].shape,(2,4))
            for name in ('head_cancel.npz','head_contrib.npz'):
                target=Path(tmp)/name;target.write_bytes(b'old research data')
                with self.assertRaises(ValueError):cancel.save(r,target)
                self.assertEqual(target.read_bytes(),b'old research data')
            output=Path(tmp)/'head_cancel_diagnostic_v1.npz';cancel.save(r,output)
            digest=hashlib.sha256(output.read_bytes()).hexdigest()
            with self.assertRaises(FileExistsError):cancel.save(r,output)
            self.assertEqual(digest,hashlib.sha256(output.read_bytes()).hexdigest())
            with np.load(output,allow_pickle=False) as z:
                self.assertEqual(json.loads(str(z['metadata_json']))['method'],'head-cancel-diagnostic-v1')
                np.testing.assert_array_equal(z['R'],r['R'])
                self.assertIn(str(ROOT/'screening_analysis/head_cancel.py'),json.loads(str(z['metadata_json']))['source_sha256'])
            sf=Path(tmp)/'settings.json';sf.write_text(json.dumps(s))
            target=Path(tmp)/'cli_head_cancel_diagnostic_v1.npz'
            args=[sys.executable,str(ROOT/'screening_analysis/head_cancel.py'),tmp,'--settings',str(sf),
                  '--lattice',json.dumps(L.tolist()),'--output',str(target),'--nconf','1']
            done=subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr);self.assertIn('rho_p=',done.stdout)
            self.assertNotEqual(subprocess.run(args,cwd=ROOT,capture_output=True).returncode,0)
            before={p.name for p in Path(tmp).iterdir()}
            imported=subprocess.run([sys.executable,'-c','import screening_analysis.head_cancel'],cwd=tmp,
                env={**__import__('os').environ,'PYTHONPATH':str(ROOT)},capture_output=True,text=True)
            self.assertEqual(imported.returncode,0,imported.stderr);self.assertEqual(imported.stdout,'')
            self.assertEqual(before,{p.name for p in Path(tmp).iterdir()})


if __name__=='__main__':unittest.main()
