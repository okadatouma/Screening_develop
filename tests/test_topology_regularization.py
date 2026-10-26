"""Independent PH coordinates, finite differences and MC distribution gradients."""
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from tests.support import require_numerics
require_numerics()
import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import topology_regularization as reg,networks
from screening_analysis import topology as reference


class RegularizationChecks(unittest.TestCase):
    def matrix(self):
        w=np.zeros((4,4));w[np.triu_indices(4,1)]=[.84,.21,.65,.77,.18,.71];return jnp.asarray(w+w.T)

    def test_PH_coordinates_are_actual_flag_diagrams(self):
        b=self.matrix();coords=np.asarray(jax.jit(reg.persistence_coordinates)(b))
        diagrams=reference.persistence(b)
        for dim in (0,1):
            actual=coords[dim];actual=actual[actual[:,1]>actual[:,0]]
            expected=diagrams[dim][np.isfinite(diagrams[dim][:,1])]
            actual=actual[np.lexsort(actual.T[::-1])];expected=expected[np.lexsort(expected.T[::-1])]
            np.testing.assert_allclose(actual,expected,atol=1e-15)

    def test_pair_stable_PH_gradient_finite_difference(self):
        b=self.matrix();direction=jnp.asarray([[0,.12,-.04,.07],[.12,0,.03,-.05],[-.04,.03,0,.08],[.07,-.05,.08,0]])
        fun=lambda x:jnp.sum(reg.ph_landscape(x,3,31)**2)
        grad=jax.jit(jax.grad(fun))(b);ad=float(jnp.sum(grad*direction))
        h=1e-5;fd=float((fun(b+h*direction)-fun(b-h*direction))/(2*h))
        self.assertGreater(abs(ad),1e-5);self.assertAlmostEqual(ad,fd,places=8)

    def test_all_categories_gradients_and_invariances(self):
        rng=np.random.default_rng(29);w=jnp.asarray(rng.uniform(.15,.85,(2,3,4,4)));near=w*.91+.02
        hp=np.array([2,0,1]);p=np.array([3,0,2,1])
        for mode in ('persistence','laplacian','weights'):
            c=reg.configuration(dict(mode=mode,gamma=.1))
            value=reg.consistency_cost(w,near,c);gradient=jax.jit(jax.grad(lambda x:reg.consistency_cost(x,near,c)))(w)
            self.assertTrue(np.isfinite(gradient).all());self.assertGreater(float(jnp.linalg.norm(gradient)),0)
            other=reg.consistency_cost(w[:,hp][:,:,p][:,:,:,p],near[:,hp][:,:,p][:,:,:,p],c)
            np.testing.assert_allclose(value,other,atol=1e-12)
        zero=jnp.zeros((4,4));self.assertTrue(np.isfinite(jax.grad(lambda x:jnp.sum(reg.spectral_moments(x)**2))(zero)).all())

    def test_disabled_callable_identity(self):
        def original(p,k,d):return p,None
        self.assertIs(reg.make_regularized_loss(original,None,None),original)
        self.assertIs(reg.make_regularized_loss(original,None,None,dict(mode='persistence',gamma=0.)),original)

    def test_sampling_score_matches_full_population_derivative(self):
        key=jax.random.PRNGKey(31)
        data=networks.FermiNetData(positions=jnp.asarray([[-1.,.2,.5,.7],[1.,.2,.5,.7]]),
            spins=jnp.ones((2,2)),atoms=jnp.zeros((2,1,2)),charges=jnp.zeros((2,1)))
        def diagnose(p,x,s,a,c):
            value=jax.nn.sigmoid(.4*x[0]+.3*x[0]**2+p*x[0]**2)
            w=jnp.array([[1.,value],[value,1.]])[None]
            return None,({'attention':{'weights_window':w}},)
        net=SimpleNamespace(apply=lambda p,x,s,a,c:(jnp.asarray(0.),p*x[0]))
        base=lambda p,k,d:(jnp.asarray(0.),None)
        cfg=reg.configuration(dict(mode='weights',gamma=1.,max_samples=2,perturb_sigma=.1))
        wrapped=reg.make_regularized_loss(base,net,diagnose,cfg)
        noise=jax.random.normal(jax.random.fold_in(key,7127),data.positions.shape)*cfg['perturb_sigma']
        def population(p):
            local=[]
            for i in range(2):
                args=(data.spins[i],data.atoms[i],data.charges[i]);x=data.positions[i]
                a=diagnose(p,x,*args)[1][0]['attention']['weights_window'][None]
                b=diagnose(p,x+noise[i],*args)[1][0]['attention']['weights_window'][None]
                local.append(reg.consistency_cost(a,b,cfg))
            probabilities=jax.nn.softmax(2*p*data.positions[:,0])
            return jnp.dot(probabilities,jnp.stack(local))
        ad=float(jax.grad(lambda p:wrapped(p,key,data)[0])(jnp.asarray(0.)))
        fd=float((population(1e-5)-population(-1e-5))/2e-5)
        self.assertAlmostEqual(ad,fd,places=8)
        conditional=reg.make_regularized_loss(base,net,diagnose,dict(cfg,sampling_score=False))
        self.assertGreater(abs(ad-float(jax.grad(lambda p:conditional(p,key,data)[0])(jnp.asarray(0.)))),1e-7)

    def test_production_diagnostic_parameter_gradient(self):
        from tests.support import make_case,batch_data
        from periodicwave import CustomScreenformer as CS
        with tempfile.TemporaryDirectory() as tmp:
            net,p,data,_,_,energy=make_case('screen',tmp);d=batch_data(data)
            diagnose=CS.make_screenformer_diagnostic((1,1),net.options)
            base=lambda p,k,d:(jnp.asarray(0.),None)
            for mode in ('persistence','laplacian','weights'):
                loss=reg.make_regularized_loss(base,net,diagnose,dict(mode=mode,gamma=.1,max_samples=2))
                value,grad=jax.jit(jax.value_and_grad(lambda p:loss(p,jax.random.PRNGKey(4),d)[0]))(p)
                self.assertTrue(np.isfinite(value))
                leaves=jax.tree.leaves(grad);self.assertTrue(all(np.isfinite(x).all() for x in leaves))
                self.assertGreater(sum(float(jnp.linalg.norm(x)) for x in leaves),0.)


if __name__=='__main__':unittest.main()
