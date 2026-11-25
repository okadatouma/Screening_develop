"""Overlaps of periodic parts in a common Hilbert basis, with explicit quadrature.

A small parameter rectangle does not define a quantized Chern invariant. Full
twist-torus Berry calculations additionally need physical sewing, a tracked
gapped state family and convergence; none are fabricated here.
"""
from itertools import product
import jax
import jax.numpy as jnp
import numpy as np


def twist_overlaps(wave,params,twists,mesh=4):
    n=wave.full['physics']['n']
    if n>2:raise ValueError('product quadrature for twist overlaps guarded at N<=2; use independent state-family estimator at larger N')
    one=np.asarray(list(product(range(mesh),repeat=2)),float)
    one=((one+.173)/mesh)@np.asarray(wave.lattice).T
    configurations=list(product(range(len(one)),repeat=n));spins=np.asarray(list(product([-1.,1.],repeat=n)))
    xx=np.repeat(np.asarray([one[list(i)].ravel() for i in configurations]),len(spins),axis=0)
    ss=np.tile(spins,(len(configurations),1));states=[]
    evaluate=jax.jit(jax.vmap(lambda p,x,s,t:wave.periodic_apply(p,x,s,t),in_axes=(None,0,0,None)))
    for twist in twists:
        phase,logabs=evaluate(params,jnp.asarray(xx),jnp.asarray(ss),jnp.asarray(twist))
        logabs=np.asarray(logabs);phase=np.asarray(phase);valid=np.isfinite(logabs)
        if not valid.any():raise ArithmeticError('zero norm on twist quadrature')
        psi=np.zeros(len(xx),complex);psi[valid]=np.exp(logabs[valid]-logabs[valid].max()+1j*phase[valid]);psi/=np.linalg.norm(psi)
        states.append(psi)
    states=np.asarray(states);overlap=states.conj()@states.T
    # order: (0,0), (dx,0), (dx,dy), (0,dy)
    loop=np.prod([overlap[i,(i+1)%len(twists)]/max(abs(overlap[i,(i+1)%len(twists)]),1e-15) for i in range(len(twists))])
    return dict(overlap=overlap,plaquette_phase=float(np.angle(loop)),quadrature_points=len(xx),
                minimum_overlap=float(min(abs(overlap[i,(i+1)%len(twists)]) for i in range(len(twists))))),states
