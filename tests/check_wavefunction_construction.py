"""F3: exact old-factory comparisons; no fixed snapshots are regenerated."""
import ast
import inspect
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import unittest

from tests.support import require_numerics
require_numerics()
import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import networks, wavefunction, envelopes, jastrows, network_blocks as NB
from tests.check_window import assert_exact

ARCHIVE=Path(__file__).parent/'f3_validation/networks-before.py.txt'


def previous_factory():
    src=ARCHIVE.read_text()
    node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='make_orbitals')
    ns=dict(vars(networks))
    exec(compile(ast.get_source_segment(src,node),str(ARCHIVE),'exec'),ns)
    return ns['make_orbitals']


def trunk():
    def init(key):return 4, NB.init_linear_layer(key,3,4)
    def apply(p,*,ae,spins,**kwargs):
        f=jnp.stack((jnp.sin(ae[:,0,0]),jnp.cos(ae[:,0,1]),spins),axis=-1)
        return NB.linear_layer(f,**p)
    return init,apply


class WavefunctionConstructionChecks(unittest.TestCase):
    def tearDown(self):
        jax.config.update('jax_enable_x64',True)
        jax.clear_caches()

    def test_reexport_import_orders_signature_and_unchanged_body(self):
        self.assertIs(networks.make_orbitals,wavefunction.make_orbitals)
        def call_signature(fn):
            s=inspect.signature(fn)
            return s.replace(parameters=[p.replace(annotation=inspect.Parameter.empty) for p in s.parameters.values()],return_annotation=inspect.Signature.empty)
        self.assertEqual(call_signature(previous_factory()),call_signature(wavefunction.make_orbitals))
        def body(src):
            n=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='make_orbitals')
            # Only the local geometry import is new. All numerical statements match.
            return ast.dump(ast.Module(body=[a for a in n.body if not isinstance(a,ast.ImportFrom)],type_ignores=[]))
        self.assertEqual(body(ARCHIVE.read_text()),body(Path(wavefunction.__file__).read_text()))
        for order in ('networks, wavefunction','wavefunction, networks'):
            result=subprocess.run([sys.executable,'-c',f'from periodicwave import {order}; assert networks.make_orbitals is wavefunction.make_orbitals'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_exact_initialization_orbitals_determinants_jastrow_jit_and_derivatives(self):
        old=previous_factory()
        for x64 in (False,True):
            jax.config.update('jax_enable_x64',x64)
            for periodic in (False,True):
                for complex_output in (False,True):
                    for use_jastrow in (False,True):
                        with self.subTest(x64=x64,periodic=periodic,complex_output=complex_output,jastrow=use_jastrow):
                            options=networks.BaseNetworkOptions(ndim=2,determinants=2,complex_output=complex_output,
                                pbc_lattice=jnp.eye(2)*3 if periodic else None,
                                envelope=envelopes.make_null_envelope() if periodic else envelopes.make_isotropic_envelope(),
                                jastrow=jastrows.JastrowType.SIMPLE_EE if use_jastrow else jastrows.JastrowType.NONE,
                                jastrow_kwargs={'ndim':2,'interaction_strength':2.},bias_orbitals=False)
                            old_init,old_apply=old((2,1),jnp.zeros(1),options,trunk())
                            new_init,new_apply=wavefunction.make_orbitals((2,1),jnp.zeros(1),options,trunk())
                            key=jax.random.PRNGKey(17);p=new_init(key)
                            assert_exact(self,p,old_init(key))
                            r=jnp.asarray([.23,.41,1.17,.89,.65,1.4]);s=jnp.asarray([1.,1.,-1.]);a=jnp.zeros((1,2));c=jnp.zeros(1)
                            def orbitals(fn):return lambda pp,x:fn(pp,x,s,a,c)
                            before,after=orbitals(old_apply),orbitals(new_apply)
                            assert_exact(self,before(p,r),after(p,r))
                            assert_exact(self,jax.jit(before)(p,r),jax.jit(after)(p,r))
                            assert_exact(self,jnp.linalg.det(before(p,r)[0]),jnp.linalg.det(after(p,r)[0]))
                            def wave(fn):return lambda pp,x:NB.logdet_matmul(fn(pp,x))
                            wb,wa=wave(before),wave(after)
                            assert_exact(self,wb(p,r),wa(p,r))
                            assert_exact(self,jax.jit(wb)(p,r),jax.jit(wa)(p,r))
                            def loss(fn):return lambda pp,x:fn(pp,x)[0]+.7*fn(pp,x)[1]
                            for arg in (0,1):
                                assert_exact(self,jax.jit(jax.grad(loss(wb),argnums=arg))(p,r),
                                             jax.jit(jax.grad(loss(wa),argnums=arg))(p,r))
                            assert_exact(self,jax.jit(jax.hessian(loss(wb),argnums=1))(p,r),
                                         jax.jit(jax.hessian(loss(wa),argnums=1))(p,r))
                            if use_jastrow:
                                # Bypass only Jastrow in test options, retaining the same params/RNG.
                                opts=networks.BaseNetworkOptions(**{**vars(options),'jastrow':jastrows.JastrowType.NONE})
                                _,b0=old((2,1),c,opts,trunk());_,a0=wavefunction.make_orbitals((2,1),c,opts,trunk())
                                assert_exact(self,b0(p,r,s,a,c),a0(p,r,s,a,c))

    def test_float64_executable_kfac_old_new_training_step(self):
        # The frozen float64 snapshot predates KFAC coverage. Compare the two
        # executable float64 routes directly; do not hide the Screenformer errors.
        from tests.test_training_checkpoint import training_config
        from periodicwave import train, checkpoint
        jax.config.update('jax_enable_x64',True)
        for route in ('psiformer','slater'):
            with self.subTest(route=route),tempfile.TemporaryDirectory() as before,tempfile.TemporaryDirectory() as after:
                with patch.object(networks,'make_orbitals',previous_factory()):
                    cfg=training_config(route,before,'kfac')
                    train.train(cfg)
                old=checkpoint.restore(str(Path(cfg.log.save_path)/'qmcjax_ckpt_000000.npz'),2)
                cfg2=training_config(route,after,'kfac')
                train.train(cfg2)
                new=checkpoint.restore(str(Path(cfg2.log.save_path)/'qmcjax_ckpt_000000.npz'),2)
                self.assertEqual(jax.tree_util.tree_structure(old),jax.tree_util.tree_structure(new))
                for a,b in zip(jax.tree_util.tree_leaves(old),jax.tree_util.tree_leaves(new)):
                    a=np.asarray(a);b=np.asarray(b)
                    self.assertEqual(a.dtype,b.dtype)
                    self.assertTrue(np.all(np.isfinite(a)))
                    np.testing.assert_array_equal(a,b)
                self.assertEqual((Path(cfg.log.save_path)/'train_stats.csv').read_text(),
                                 (Path(cfg2.log.save_path)/'train_stats.csv').read_text())
