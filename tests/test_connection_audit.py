import unittest
import numpy as np
import jax
import jax.numpy as jnp
from periodicwave.attention import screening_attention
from screening_analysis.connection_audit import complex_observables,edge_scale,edge_sets,residual_by_bin,angle_difference


class ConnectionAuditChecks(unittest.TestCase):
    def setUp(self):
        context=jax.experimental.enable_x64(True);context.__enter__()
        self.addCleanup(context.__exit__,None,None,None)

    def test_positive_edges_and_tie_rejection(self):
        status=edge_sets([.9,.8,.7,0],[.1,.9,.8,0],2)
        np.testing.assert_array_equal(status,[2,3,1,0])
        self.assertIsNone(edge_sets([.9,.8,.8,0],[.1,.9,.8,0],2))
        self.assertIsNone(edge_sets([.9,0,0],[.9,.8,0],2))

    def test_pair_scale_symmetric_and_self_identity(self):
        z=np.zeros((2,6));z[1,2]=np.log(.7)
        scale=np.asarray(edge_scale(jnp.asarray(z),4))
        np.testing.assert_array_equal(np.diagonal(scale,axis1=-2,axis2=-1),np.ones((2,4)))
        np.testing.assert_array_equal(scale,scale.transpose(0,2,1))
        self.assertAlmostEqual(scale[1,0,3],.7)

    def fixture(self):
        rng=np.random.default_rng(5)
        p={key:jnp.asarray(rng.normal(size=(3,4))) for key in ('q_w','k_w','v_w')}
        p['trim_rho']=jnp.array([2.,2.]);p['attn_output']=jnp.asarray(rng.normal(size=(4,3)))
        x=jnp.asarray(rng.normal(size=(4,3)))
        options=dict(num_electrons=4,num_heads=2,attn_dim=2,value_dim=2,d_ref=2.,use_softmask=False,
                     use_fixed_value=False,use_tanh_norm=False,use_gate=False,use_output_scale=False)
        def forward(params):
            return screening_attention(params,x,jnp.zeros((4,4)),options=options,trim_max=2.,
                trim_pow=4,win_shape='smooth7',freeze_w=False,clip_r=None,clip_beta=.3,return_aux=True)
        return p,forward

    def test_identity_and_zero_intervention_on_production_kernel(self):
        p,forward=self.fixture();base,aux=forward(p)
        identity={**p,'_edge_scale':jnp.ones((2,4,4))}
        out,record=jax.jit(forward)(identity)
        np.testing.assert_allclose(out,base,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(record['aggregation_weights'],aux['aggregation_weights'])
        out,_=forward({**p,'_edge_scale':jnp.zeros((2,4,4))})
        np.testing.assert_allclose(out,0,atol=1e-14)
        with self.assertRaises(ValueError):
            forward({**p,'_edge_scale':jnp.ones((4,4))})

    def test_pair_gradient_against_finite_difference(self):
        p,forward=self.fixture()
        def cost(z):
            out,_=forward({**p,'_edge_scale':edge_scale(z,4)})
            return jnp.sum(out**2)
        z=jnp.zeros((2,6));direction=jnp.arange(12,dtype=float).reshape(2,6)/12
        derivative=jnp.sum(jax.grad(cost)(z)*direction)
        eps=1e-5
        fd=(cost(z+eps*direction)-cost(z-eps*direction))/(2*eps)
        np.testing.assert_allclose(derivative,fd,rtol=1e-7,atol=1e-10)

    def test_residual_and_wrapped_phase(self):
        residual=residual_by_bin([1,3,5,9],np.array([0,0,1,1]))
        np.testing.assert_array_equal(residual,[-1,1,-2,2])
        self.assertAlmostEqual(float(angle_difference(-np.pi+.01,np.pi-.01)),.02)

    def test_complex_api_phase_is_already_an_angle(self):
        from periodicwave.network_blocks import logdet_matmul
        def f(t):
            matrix=jnp.diag(jnp.array([jnp.exp(2*t+3j*t),1+0j]))[None]
            phase,logabs=logdet_matmul([matrix])
            return complex_observables(phase,logabs)
        np.testing.assert_allclose(f(.1),[.2,.3],atol=1e-14)
        np.testing.assert_allclose(jax.jacrev(f)(.1),[2,3],atol=1e-14)


if __name__=='__main__':unittest.main()
