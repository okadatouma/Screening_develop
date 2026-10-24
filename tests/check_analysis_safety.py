"""Three analysis-boundary fixes. Temporary files only; no research runs."""
from contextlib import ExitStack
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.support import require_numerics, ROOT
require_numerics()
import numpy as np
from screening_analysis import analyze_all as batch
from screening_analysis import fig_channels as channels
from screening_analysis import moire_summary as summary
import head_contrib
import reach
from periodicwave import CustomScreenformer as CS


class AnalysisSafety(unittest.TestCase):
    def test_summary_cache_hit_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp,'summary.json');p.write_text('{"E": 1.25}')
            with patch.object(summary.MA,'load') as load:
                self.assertEqual(summary.summarize(tmp),{'E':1.25})
            load.assert_not_called();self.assertEqual(p.read_text(),'{"E": 1.25}')

    def test_force_recomputes_without_deleting_or_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            old={name:b'old research bytes' for name in (
                'summary.json','samples_3.npy','head_contrib.npz','reach.npz',
                'head_contrib_diagnostic_v1.npz')}
            for name,content in old.items():Path(tmp,name).write_bytes(content)
            d=dict(L=np.eye(2),MO=np.eye(2),ne=2,rs=5.,U=1.,phi=0.)
            z={key:np.ones((1,1)) for key in ('C','Cd','Cp','X','R','w')}
            z.update(X=np.ones(1),R=np.ones(1))
            stack.enter_context(patch.object(summary.MA,'load',return_value=d))
            stack.enter_context(patch.object(summary.MA,'order_parameter',return_value=dict(K_over_bg=1.,G_over_bg=1.,psi_K=1.)))
            stack.enter_context(patch.object(summary.MA,'occupation_summary',return_value=dict(psi=1.,psi_sub=1.,sublattice=[1.],contrast=1.,ideal_contrast=1.,hist=[1.])))
            sample=stack.enter_context(patch.object(summary.SM,'sample',return_value=(np.zeros((3,2,2)),.5)))
            cached=stack.enter_context(patch.object(summary.SM,'cached'))
            analyze=stack.enter_context(patch.object(head_contrib,'analyze',return_value=z))
            hc_cache=stack.enter_context(patch.object(head_contrib,'cached'))
            stack.enter_context(patch.object(CS,'_max_periodic_distance',return_value=1.))
            stack.enter_context(patch.object(reach,'compute',return_value=dict(dbar=np.ones((1,1)),dmax=1.)))
            stack.enter_context(patch.object(summary,'energy',return_value=dict(E=1.,Eerr=.1,steps=1,locstd=1.)))
            result=summary.summarize(tmp,nconf=3,force=True)
            sample.assert_called_once_with(tmp,3);analyze.assert_called_once_with(tmp,nconf=48)
            cached.assert_not_called();hc_cache.assert_not_called()
            p=Path(tmp,'summary_recomputed.json');self.assertEqual(json.loads(p.read_text()),result)
            for name,content in old.items():self.assertEqual(Path(tmp,name).read_bytes(),content)
            # A second call must stop before even loading the run.
            with self.assertRaises(FileExistsError):summary.summarize(tmp,force=True)
            self.assertEqual(sample.call_count,1)
            summary.summarize(tmp,nconf=3,force=True,output=Path(tmp,'another.json'))
            self.assertTrue(Path(tmp,'another.json').exists())
            for name,content in old.items():self.assertEqual(Path(tmp,name).read_bytes(),content)
            cached.return_value=np.zeros((3,2,2));hc_cache.return_value=z
            normal=summary.summarize(tmp,nconf=3,output=Path(tmp,'normal.json'))
            self.assertEqual(normal,result)
            cached.assert_called_once_with(tmp,3);hc_cache.assert_called_once_with(tmp,nconf=48)

    def test_existing_explicit_summary_output_rejected_before_computation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp,'result.json');p.write_bytes(b'keep')
            with patch.object(summary.MA,'load') as load, self.assertRaises(FileExistsError):
                summary.summarize(tmp,output=p)
            load.assert_not_called();self.assertEqual(p.read_bytes(),b'keep')

    def test_subprocess_nonzero_status_propagates(self):
        with self.assertRaises(subprocess.CalledProcessError) as raised:
            batch.sh(sys.executable,'-c','import sys; sys.exit(7)')
        self.assertEqual(raised.exception.returncode,7)

    def test_batch_success_order_and_stop_after_failure(self):
        script=ROOT/'screening_analysis/analyze_all.py'
        for fail in (False,True):
            calls=[]
            def child(cmd,**kwargs):
                self.assertTrue(kwargs['check']);calls.append(cmd[1])
                if fail and len(calls)==2:raise subprocess.CalledProcessError(9,cmd)
                return subprocess.CompletedProcess(cmd,0)
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                # Redirect the script's output guard into an empty temporary ROOT.
                stack.enter_context(patch('os.path.exists',return_value=False))
                stack.enter_context(patch('glob.glob',return_value=[tmp]))
                stack.enter_context(patch('subprocess.run',side_effect=child))
                stack.enter_context(patch.object(sys,'argv',[str(script),'--pair','a','b']))
                if fail:
                    with self.assertRaises(subprocess.CalledProcessError):runpy.run_path(str(script),run_name='__main__')
                else:runpy.run_path(str(script),run_name='__main__')
            expected=['screening_analysis/fig_fig6.py','screening_analysis/fig_qk.py','screening_analysis/fig_kernel.py']
            self.assertEqual(calls,expected[:2] if fail else expected)

    def test_existing_batch_figure_output_is_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp,'figure.png');p.write_bytes(b'old figure')
            with patch.object(batch.subprocess,'run') as child, self.assertRaises(FileExistsError):
                batch.sh('python','plot.py','--out',str(p))
            child.assert_not_called();self.assertEqual(p.read_bytes(),b'old figure')

    def test_batch_force_cannot_silently_plot_old_summary(self):
        script=ROOT/'screening_analysis/analyze_all.py'
        with patch.object(sys,'argv',[str(script),'--scan','a','--force']), patch('subprocess.run') as child:
            with self.assertRaises(SystemExit) as raised:runpy.run_path(str(script),run_name='__main__')
        self.assertEqual(raised.exception.code,2);child.assert_not_called()

    def test_channels_dictionary_and_exact_share_formula(self):
        C=np.array([[1.,2.],[3.,4.]])
        r=dict(C=C,w=np.array([[.3,.6],[.9,1.2]]),metadata={'method':'head-contrib-diagnostic-v1'})
        with patch.object(channels,'RS',[1,5]), patch.object(channels,'analyze',return_value=r) as analyze:
            shares,widths,metadata=channels.collect_channels(settings='known.json',lattice=[[3.,0.],[0.,3.]])
        np.testing.assert_array_equal(shares[1],100*C**2/(C**2).sum())
        np.testing.assert_array_equal(shares[5],shares[1]);np.testing.assert_array_equal(widths,r['w'])
        self.assertEqual(len(metadata),2)
        for call in analyze.call_args_list:
            self.assertEqual(call.kwargs,dict(nconf=48,settings='known.json',lattice=[[3.,0.],[0.,3.]]))
        with patch.object(channels,'RS',[1]), patch.object(channels,'analyze',return_value=r|{'metadata':{'method':'legacy'}}):
            with self.assertRaisesRegex(ValueError,'旧cache'):channels.collect_channels()

    def test_real_diagnostic_and_missing_settings_lattice(self):
        # Reuse one existing N=2 test fixture; no new model or dataset framework.
        from tests.check_attn_probe import fixture
        with tempfile.TemporaryDirectory() as tmp:
            _,_,_,settings,L,ck=fixture(tmp,'pow4')
            old=Path(tmp,'head_contrib.npz');old.write_bytes(b'legacy cache must not be read')
            before={p.name:p.read_bytes() for p in Path(tmp).iterdir() if p.is_file()}
            with patch.object(channels,'RS',[1]),patch.object(channels,'R9',str(Path(tmp,'..')/'%d')):
                # Supply the exact fixture directory without making a second fixture.
                with patch.object(channels,'analyze',wraps=lambda f,**kw:head_contrib.analyze(tmp,**kw)):
                    with self.assertRaisesRegex(ValueError,'settings'):channels.collect_channels()
                    with self.assertRaisesRegex(ValueError,'格子'):channels.collect_channels(settings=settings)
                    result,_,_=channels.collect_channels(settings=settings,lattice=L)
            reference=head_contrib.analyze(tmp,nconf=48,settings=settings,lattice=L)
            C=reference['C'];np.testing.assert_array_equal(result[1],100*C**2/(C**2).sum())
            self.assertEqual(before,{p.name:p.read_bytes() for p in Path(tmp).iterdir() if p.is_file()})

    def test_channels_existing_figure_rejected_before_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp,'old.png');p.write_bytes(b'old plot')
            with patch.object(channels,'analyze') as analyze, self.assertRaises(FileExistsError):
                channels.main(['--output',str(p)])
            analyze.assert_not_called();self.assertEqual(p.read_bytes(),b'old plot')


if __name__=='__main__':unittest.main(verbosity=2)
