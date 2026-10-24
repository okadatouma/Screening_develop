"""G2c: real reporting of complex energies; preserved inference trajectories.

The old train loop is read from the saved source solely as a test reference.
No loss, Hamiltonian, checkpoint, or numerical baseline is patched.
"""
import argparse
import ast
import csv
import hashlib
import json
import logging
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest
from tests.support import require_numerics,make_case,batch_data
require_numerics()
import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import train,train_screen,checkpoint,CustomPsiformer as CP,CustomScreenformer as CS
from tests.test_training_checkpoint import training_config
from tests import check_vmc_gradients as ref

ROOT=Path(__file__).resolve().parents[1]
RECORDS=[]


def previous_loop(module):
    path=ROOT/'tests/g2c_validation'/(module.__name__.split('.')[-1]+'.py-before.txt')
    src=path.read_text();node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='train')
    namespace=dict(vars(module))
    exec(compile(ast.get_source_segment(src,node),str(path),'exec'),namespace)
    return namespace['train']


def run_loop(module,fn,cfg):
    # Exercise the real % formatter on actual log arguments, rather than merely
    # suppressing the former logging errors. Other informational logs are omitted.
    formatted=[];errors=[]
    def log(message,*args,**kwargs):
        if message.startswith('Step %05d:'):
            try:formatted.append(logging.LogRecord('g2c',logging.INFO,'',0,message,args,None).getMessage())
            except TypeError as e:errors.append(str(e))
    with patch.object(module.logging,'info',side_effect=log):fn(cfg)
    with Path(cfg.log.save_path,'inference_stats.csv').open() as f:rows=list(csv.DictReader(f))
    ck=Path(cfg.log.save_path,'qmcjax_ckpt_000002.npz')
    return rows,checkpoint.restore(str(ck),2),formatted,errors


def exact_tree(test,a,b):
    test.assertEqual(jax.tree_util.tree_structure(a),jax.tree_util.tree_structure(b))
    for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
        x=np.asarray(x);y=np.asarray(y)
        test.assertEqual(x.dtype,y.dtype);np.testing.assert_array_equal(x,y)


class ReportingChecks(unittest.TestCase):
    def tearDown(self):
        jax.config.update('jax_enable_x64',True);jax.clear_caches()

    def test_old_new_inference_real_complex_both_trainers_methods_precisions(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for route,module,maker in (('psiformer',train,CP),('screen',train_screen,CS)):
                old=previous_loop(module)
                for complex_output in (False,True):
                    for method in ('default','folx'):
                        with self.subTest(x64=x64,route=route,complex_output=complex_output,method=method),tempfile.TemporaryDirectory() as tmp:
                            net,p,d,_,opts,_=make_case(route,tmp)
                            if not complex_output:
                                net=maker.make_fermi_net((1,1),d.charges,**{**opts,'complex_output':False})
                                p=net.init(jax.random.PRNGKey(7))
                            inp=Path(tmp,'input');inp.mkdir()
                            bd=batch_data(d)
                            replicated=jax.tree_util.tree_map(lambda a:jnp.stack([a]),bd)
                            rp=jax.tree_util.tree_map(lambda a:jnp.stack([a]),p)
                            saved=checkpoint.save(str(inp),0,replicated,rp,None,jnp.asarray([.2]))
                            digest=hashlib.sha256(Path(saved).read_bytes()).hexdigest()
                            results=[]
                            for label,fn in (('old',old),('new',module.train)):
                                cfg=training_config(route,tmp,'none')
                                cfg.network.complex=complex_output
                                cfg.system.make_local_energy_kwargs['kinetic_kwargs']['laplacian_method']=method
                                cfg.optim.iterations=3;cfg.log.restore_path=str(inp);cfg.log.save_path=str(Path(tmp,label))
                                results.append(run_loop(module,fn,cfg))
                            (before,old_ck,_,old_errors),(rows,new_ck,formatted,errors)=results
                            self.assertEqual(len(rows),2);self.assertEqual(len(formatted),2);self.assertEqual(errors,[])
                            exact_tree(self,old_ck,new_ck)
                            self.assertEqual(hashlib.sha256(Path(saved).read_bytes()).hexdigest(),digest)
                            energies=np.asarray([float(r['energy']) for r in rows])
                            for a,b in zip(before,rows):
                                np.testing.assert_array_equal(complex(a['energy']).real,float(b['energy']))
                                for field in ('pmove','locstd'):self.assertEqual(a[field],b[field])
                                for field in ('energy','ewmean','ewvar','locstd'):
                                    self.assertTrue(np.isfinite(float(b[field])))
                                self.assertGreaterEqual(float(b['ewvar']),0.)
                            if complex_output:
                                self.assertEqual(len(old_errors),2)
                            else:
                                self.assertEqual(old_errors,[]);self.assertEqual(before,rows)
                            # Independent moments of the two real observations,
                            # with weights .9/.1, not a copy of the EW recurrence.
                            mean=np.average(energies,weights=[.9,.1])
                            variance=np.average((energies-mean)**2,weights=[.9,.1])
                            ref.close(float(rows[1]['ewmean']),mean,x64)
                            ref.close(float(rows[1]['ewvar']),variance,x64)
                            RECORDS.append(dict(test='old_new_inference',x64=x64,route=route,complex_output=complex_output,method=method,init_seed=7,checkpoint_sha256=digest,old_rows=before,rows=rows,old_log_errors=old_errors,formatted_logs=formatted,expected_final_ewmean=float(mean),expected_final_ewvar=float(variance),all_checkpoint_fields_exact=True))
                        jax.clear_caches()

    def test_complex_variance_real_variance_and_iid_sem_are_distinct(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            p=jnp.zeros(3);d=ref.data_for([0,1,0,1])
            H=np.array([[-1,2j],[-2j,1]])
            f=ref.make_current(H);(energy,aux),g=ref.evaluate(f,p,d,jit=True)
            np.testing.assert_array_equal(aux.local_energy,np.array([-1+2j,1-2j,-1+2j,1-2j]))
            np.testing.assert_array_equal(aux.variance,5.)
            real_var=np.var(np.asarray(aux.local_energy).real);imag_var=np.var(np.asarray(aux.local_energy).imag)
            np.testing.assert_array_equal([real_var,imag_var],[1.,4.]);np.testing.assert_array_equal(g,[2.,-4.,0.])
            for module in (train,train_screen):
                raw=module.make_loss_step(f)(p,d,None,jax.random.PRNGKey(0))[2]
                self.assertTrue(jnp.iscomplexobj(raw))  # K10 raw-primal contract stays unchanged.
            RECORDS.append(ref.serial(dict(test='variance_definitions',x64=x64,gradient=g,energy=energy,complex_variance=aux.variance,real_variance=real_var,imag_variance=imag_var,complex_local_std=jnp.sqrt(aux.variance),reference_distribution_iid_real_mean_SEM=np.sqrt(real_var/4),note='The IID SEM is illustrative, not an uncertainty for correlated MCMC or for this deterministic full-sum batch.')))

    def test_lambda_imaginary_diagnostic_survives_real_reporting(self):
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            old=previous_loop(train_screen)
            with tempfile.TemporaryDirectory() as tmp:
                _,p,d,_,_,_=make_case('lambda',tmp);inp=Path(tmp,'input');inp.mkdir()
                checkpoint.save(str(inp),0,jax.tree_util.tree_map(lambda a:jnp.stack([a]),batch_data(d)),jax.tree_util.tree_map(lambda a:jnp.stack([a]),p),None,jnp.asarray([.2]))
                out=[]
                for label,fn in (('old',old),('new',train_screen.train)):
                    cfg=training_config('lambda',tmp,'none');cfg.optim.iterations=3;cfg.log.restore_path=str(inp);cfg.log.save_path=str(Path(tmp,label))
                    out.append(run_loop(train_screen,fn,cfg))
                (before,old_ck,_,_),(rows,new_ck,formatted,errors)=out
                exact_tree(self,old_ck,new_ck);self.assertEqual(errors,[]);self.assertEqual(len(formatted),2)
                for a,b in zip(before,rows):
                    self.assertEqual(a['eimag'],b['eimag'])
                    self.assertTrue(np.isfinite(float(b['eimag'])))
                    self.assertEqual(complex(a['energy']).real,float(b['energy']))
                RECORDS.append(dict(test='lambda_diagnostic',x64=x64,old_rows=before,rows=rows,all_checkpoint_fields_exact=True,imaginary_diagnostic_unchanged=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    if args.report.exists():parser.error('existing report is protected')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReportingChecks))
    from tests.baseline import metadata
    with args.report.open('x') as f:json.dump(dict(version='g2c-v1',environment=metadata(),harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),methods=result.testsRun,failures=len(result.failures),errors=len(result.errors),records=RECORDS),f,indent=2,allow_nan=False)
    raise SystemExit(0 if result.wasSuccessful() else 1)
