"""C3 expectation for lambda-jellium spinors, not a Chern number formula.

h(R_theta k)=U_theta h(k) U_theta^dagger with U_theta=diag(1,e^{i theta})
for h(k)=[[lambda² k²,-lambda k_-],[-lambda k_+,1]]. Thus
(C3 Psi)(R,S)=exp(i theta n_minus) Psi(R_{-theta} R,S).
The periodic cell must itself be invariant under this rotation.
"""
import numpy as np


def c3_local_ratios(evaluate,positions,spins,lattice):
    """Explicit lambda-jellium pseudospin representation, theta=2pi/3.

    A zero-twist rotation eigenvalue alone does NOT determine the many-body Chern
    number. Ground-state symmetry, high-symmetry twists and sewing need separate
    validation before using a symmetry-indicator relation.
    """
    x=np.asarray(positions,dtype=float);s=np.asarray(spins)
    if x.ndim!=3 or x.shape[-1]!=2 or s.shape!=x.shape[:2] or not np.isin(s,[-1,1]).all():
        raise ValueError('positions [M,N,2] and explicit pseudospins +/-1 required')
    L=np.asarray(lattice,dtype=float);theta=2*np.pi/3
    rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    cell_map=np.linalg.inv(L)@rotation@L
    if not np.allclose(cell_map,np.rint(cell_map),atol=1e-10):
        raise ValueError('the supplied periodic cell has no C3 symmetry')
    rotated=x@rotation # column preimage R_-theta; row-vector convention
    fractional=rotated@np.linalg.inv(L).T
    rotated=np.mod(fractional,1.)@L.T
    p0,l0=map(np.asarray,evaluate(x,s));p1,l1=map(np.asarray,evaluate(rotated,s))
    if not np.isfinite(l0).all() or not np.isfinite(p0).all():raise ValueError('node/nonfinite sampled wavefunction')
    finite=np.isfinite(l1)
    if np.isnan(l1).any() or np.isposinf(l1).any() or not np.isfinite(p1[finite]).all():raise ValueError('invalid rotated wavefunction')
    difference=l1-l0
    if np.any(difference[finite]>700):raise FloatingPointError('rotation ratio overflow; no clipping')
    ratio=np.zeros(len(x),complex)
    ratio[finite]=np.exp(difference[finite]+1j*(p1-p0+theta*np.sum(s==-1,axis=1))[finite])
    return ratio


def c3_expectation(local_ratios,sample_weights=None):
    r=np.asarray(local_ratios,dtype=complex)
    weights=np.ones(len(r)) if sample_weights is None else np.asarray(sample_weights,dtype=float)
    if r.ndim!=1 or weights.shape!=r.shape or not np.isfinite(r).all() or np.any(weights<0) or not weights.sum()>0:
        raise ValueError('finite ratios and nonnegative nonzero weights required')
    weights=weights/weights.sum();overlap=np.sum(weights*r)
    roots=np.exp(2j*np.pi*np.arange(3)/3);nearest=roots[np.argmin(abs(roots-overlap))]
    return dict(expectation=overlap,nearest_root_candidate=nearest,
                candidate_eigenstate_residual=float(np.sum(weights*abs(r-nearest)**2)),
                interpretation='operator expectation; eigenvalue only if eigenstate residual and MC error justify it; not Chern')
