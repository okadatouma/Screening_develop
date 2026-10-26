"""Gauge-invariant lattice Berry links, independently checked on a TWIST Hamiltonian.

The lattice-model benchmark below is NOT a Chern calculation for lambda-jellium
or for Screenformer. No boundary phase is appended to an existing trained state.
"""
from itertools import combinations
import numpy as np


def berry_links(states,sewing_x=None,sewing_y=None,overlap_tolerance=1e-8):
    """Normalized state vectors on a periodic parameter grid in a COMMON Hilbert basis.

    Sewing matrices, when necessary, map the state at theta=0 into the theta=2pi
    boundary fiber. Callers must supply physically justified sewing/tracked states.
    Convention: links <psi(theta)|psi(theta+delta)>, CCW parameter plaquettes.
    Returns a bundle invariant; it cannot certify the origin of supplied states.
    """
    z=np.asarray(states,dtype=complex)
    if z.ndim!=3 or min(z.shape[:2])<3 or not np.isfinite(z).all():raise ValueError('finite [nx>=3,ny>=3,Hilbert dimension] required')
    norm=np.linalg.norm(z,axis=-1)
    if np.any(norm==0):raise ValueError('zero state')
    z=z/norm[...,None];D=z.shape[-1]
    sew=[]
    for s in (sewing_x,sewing_y):
        s=np.eye(D) if s is None else np.asarray(s)
        if s.shape!=(D,D) or not np.allclose(s.conj().T@s,np.eye(D),atol=1e-10):raise ValueError('unitary sewing required')
        sew.append(s)
    if not np.allclose(sew[0]@sew[1],sew[1]@sew[0],atol=1e-10):raise ValueError('noncommuting sewing needs an explicit corner cocycle, unsupported here')
    next_x=np.roll(z,-1,axis=0).copy();next_y=np.roll(z,-1,axis=1).copy()
    next_x[-1]=np.einsum('ab,jb->ja',sew[0],z[0]);next_y[:,-1]=np.einsum('ab,ib->ia',sew[1],z[:,0])
    ox=np.sum(z.conj()*next_x,axis=-1);oy=np.sum(z.conj()*next_y,axis=-1)
    minimum=min(np.abs(ox).min(),np.abs(oy).min())
    if minimum<=overlap_tolerance:raise ValueError('vanishing neighboring overlap; refine/track states or use a degenerate-subspace method')
    ux=ox/np.abs(ox);uy=oy/np.abs(oy)
    curvature=np.angle(ux*np.roll(uy,-1,axis=0)*np.conj(np.roll(ux,-1,axis=1))*np.conj(uy))
    value=curvature.sum()/(2*np.pi)
    return dict(curvature=curvature,chern=float(value),nearest_integer=int(np.rint(value)),
        minimum_link_overlap=float(minimum),max_plaquette_phase=float(np.max(np.abs(curvature))),
        branch_margin=float(np.pi-np.max(np.abs(curvature))),
        interpretation='parameter-bundle invariant; physical Hamiltonian, gap, tracking and refinement require independent verification')


SX=np.array([[0,1],[1,0]],complex)
SY=np.array([[0,-1j],[1j,0]],complex)
SZ=np.diag([1.,-1.]).astype(complex)


def slater_berry_links(occupied_grid,overlap_tolerance=1e-8):
    """Exact many-body Slater overlaps det(V(theta)^dagger V(theta+delta)).

    Common FIXED site/orbital basis and periodic Hamiltonian parameterization are
    required. Arbitrary U(Nocc) rotations at each grid point only change the Slater
    phase. This is not a sum of arbitrarily matched one-particle eigenvector phases.
    """
    v=np.asarray(occupied_grid,dtype=complex)
    if v.ndim!=4 or min(v.shape[:2])<3 or v.shape[-1]>v.shape[-2]:raise ValueError('occupied grid [nx,ny,D,Nocc] required')
    gram=np.einsum('ijda,ijdb->ijab',v.conj(),v)
    if not np.allclose(gram,np.eye(v.shape[-1]),atol=1e-10):raise ValueError('orthonormal occupied orbitals required')
    ox=np.linalg.det(np.einsum('ijda,ijdb->ijab',v.conj(),np.roll(v,-1,axis=0)))
    oy=np.linalg.det(np.einsum('ijda,ijdb->ijab',v.conj(),np.roll(v,-1,axis=1)))
    minimum=min(np.abs(ox).min(),np.abs(oy).min())
    if minimum<=overlap_tolerance:raise ValueError('vanishing many-body Slater overlap')
    ux=ox/abs(ox);uy=oy/abs(oy)
    curvature=np.angle(ux*np.roll(uy,-1,axis=0)*np.conj(np.roll(ux,-1,axis=1))*np.conj(uy))
    return dict(chern=float(curvature.sum()/(2*np.pi)),curvature=curvature,
                minimum_link_overlap=float(minimum),max_plaquette_phase=float(np.max(abs(curvature))))


def qwz_twist_hamiltonian(theta,mass=-1.,size=(2,2),derivatives=False):
    """Finite QWZ Chern insulator, twists applied to bonds crossing a torus seam.

    Fixed site/orbital basis; h(k)=sin(kx)sx+sin(ky)sy+(m+cos(kx)+cos(ky))sz.
    Full occupied Slater state follows the actual twisted Hamiltonian.
    """
    nx,ny=size;theta=np.asarray(theta);D=2*nx*ny
    h=np.zeros((D,D),complex);dx=np.zeros_like(h);dy=np.zeros_like(h)
    index=lambda x,y:2*(x*ny+y)
    for x in range(nx):
        for y in range(ny):
            i=index(x,y);h[i:i+2,i:i+2]+=mass*SZ
            for axis,(xx,yy,T) in enumerate(((x+1,y,(SZ-1j*SX)/2),(x,y+1,(SZ-1j*SY)/2))):
                wrap=(xx==nx) if axis==0 else (yy==ny)
                j=index(xx%nx,yy%ny);value=T*np.exp(1j*theta[axis]*wrap)
                h[i:i+2,j:j+2]+=value;h[j:j+2,i:i+2]+=value.conj().T
                derivative=dx if axis==0 else dy
                derivative[i:i+2,j:j+2]+=1j*wrap*value
                derivative[j:j+2,i:i+2]+=(1j*wrap*value).conj().T
    return (h,dx,dy) if derivatives else h


def occupied_fock_state(occupied):
    """Exact normalized Slater amplitudes in the fixed occupation-number basis."""
    v=np.asarray(occupied);D,N=v.shape
    if D>12:raise ValueError('small independent Fock-space benchmark limited to 12 orbitals')
    configs=list(combinations(range(D),N))
    state=np.asarray([np.linalg.det(v[list(rows)]) for rows in configs])
    return state/np.linalg.norm(state)


def qwz_manybody_grid(grid=10,mass=-1.,size=(2,2)):
    """Unique half-filled many-body ground state, min excitation gap recorded."""
    states=[];gap=np.inf;N=np.prod(size)
    for i in range(grid):
        row=[]
        for j in range(grid):
            h=qwz_twist_hamiltonian(2*np.pi*np.array([i,j])/grid,mass,size)
            energy,v=np.linalg.eigh(h);gap=min(gap,energy[N]-energy[N-1])
            row.append(occupied_fock_state(v[:,:N]))
        states.append(row)
    return np.asarray(states),float(gap)


def qwz_occupied_grid(grid=10,mass=-1.,size=(3,3)):
    orbitals=[];gap=np.inf;N=int(np.prod(size))
    for i in range(grid):
        row=[]
        for j in range(grid):
            e,v=np.linalg.eigh(qwz_twist_hamiltonian(2*np.pi*np.array([i,j])/grid,mass,size))
            gap=min(gap,e[N]-e[N-1]);row.append(v[:,:N])
        orbitals.append(row)
    return np.asarray(orbitals),float(gap)


def qwz_kubo_integral(grid=48,mass=-1.,size=(2,2)):
    """Independent spectral-response curvature from analytic dH/dtheta, no links."""
    N=int(np.prod(size));values=[]
    for i in range(grid):
        for j in range(grid):
            theta=2*np.pi*(np.array([i,j])+.5)/grid
            h,dx,dy=qwz_twist_hamiltonian(theta,mass,size,True);e,v=np.linalg.eigh(h)
            if e[N]-e[N-1]<1e-8:raise ValueError('gap closes')
            mx=v[:,:N].conj().T@dx@v[:,N:];my=v[:,N:].conj().T@dy@v[:,:N]
            values.append(2*np.imag(np.sum(mx*my.T/(e[:N,None]-e[None,N:])**2)))
    return float(2*np.pi*np.mean(values))
