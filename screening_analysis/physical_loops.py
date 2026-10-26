"""Fixed physical-basis 1RDM estimator and nonquantized, gauge-invariant loops.

G_ab=<c_a^dagger c_b> = E_|Psi|² sum_i phi_a(x_i)
    integral dy phi_b(y)* Psi(R_i->y)/Psi(R).
x,y include pseudospin when explicitly enabled. Basis labels are NOT electrons.
"""
from itertools import permutations,product
import numpy as np
from screening_analysis.directed_loops import oriented_triangles


class TorusBasis:
    """Orthonormal integer Fourier modes, optionally rotated to fixed spatial orbitals."""
    def __init__(self,lattice,modes,unitary=None,centres=None,components=None):
        self.lattice=np.asarray(lattice,dtype=float);self.area=abs(np.linalg.det(self.lattice))
        modes=np.asarray(modes)
        if self.lattice.shape!=(2,2) or self.area<=0 or not np.isfinite(self.lattice).all():raise ValueError('finite invertible lattice required')
        if modes.ndim!=2 or modes.shape[1]!=2 or not np.array_equal(modes,np.round(modes)):
            raise ValueError('integer reciprocal coordinates required')
        self.modes=modes.astype(int);K=len(modes)
        if len({tuple(k) for k in modes})!=K:raise ValueError('distinct Fourier modes required')
        self.unitary=np.eye(K,dtype=complex) if unitary is None else np.asarray(unitary,dtype=complex)
        if self.unitary.shape!=(K,K) or not np.allclose(self.unitary.conj().T@self.unitary,np.eye(K),atol=1e-12):
            raise ValueError('orthonormal basis rotation required')
        self.components=None if components is None else tuple(components)
        if self.components is not None and (not self.components or len(set(self.components))!=len(self.components)):
            raise ValueError('distinct explicit component labels required')
        nc=1 if components is None else len(components);self.size=K*nc
        self.centres=None if centres is None else np.repeat(np.asarray(centres).reshape(K,2),nc,axis=0)

    def __call__(self,positions,spins=None):
        x=np.asarray(positions,dtype=float)
        frac=x@np.linalg.inv(self.lattice).T
        phi=np.exp(2j*np.pi*frac@self.modes.T)/np.sqrt(self.area)@self.unitary
        if self.components is None:return phi
        if spins is None:raise ValueError('pseudospin labels required')
        mask=np.asarray(spins)[...,None]==np.asarray(self.components)
        return (phi[...,None]*mask[...,None,:]).reshape(x.shape[:-1]+(self.size,))

    def metadata(self):
        return dict(lattice=self.lattice.tolist(),integer_modes=self.modes.tolist(),components=self.components,
            dimension=self.size,centres=None if self.centres is None else self.centres.tolist(),
            basis='fixed finite Fourier subspace, optionally unitary localized DVR',
            trace='projected particle number; equals N only for a complete occupied subspace')


def dvr_basis(lattice,shape=(3,3),components=None):
    """Localized orthonormal finite-Fourier orbitals centred on a fixed odd grid.

    Not delta functions in continuous space. With symmetric odd modes, the
    spatial functions are real; scalar complex conjugation has the usual TR action.
    """
    if len(shape)!=2 or any(type(n) is not int or n<1 or n%2!=1 for n in shape):raise ValueError('two positive odd grid sizes required')
    modes=np.array(list(product(*(range(-(n//2),n//2+1) for n in shape))))
    centres=np.array(list(product(*(np.arange(n)/n for n in shape))))
    rotation=np.exp(-2j*np.pi*modes@centres.T)/np.sqrt(len(modes))
    return TorusBasis(lattice,modes,rotation,centres@np.asarray(lattice).T,components)


def uniform_quadrature(lattice,shape=(8,8),components=None,shift=(.137,.271)):
    """Uniform cell quadrature, sum over explicitly enumerated pseudospins."""
    L=np.asarray(lattice);frac=np.array(list(product(*((np.arange(n)+offset)/n for n,offset in zip(shape,shift)))))
    x=frac@L.T;weight=abs(np.linalg.det(L))/len(x)
    labels=(1,) if components is None else tuple(components)
    return np.repeat(x,len(labels),axis=0),np.tile(labels,len(x)),np.full(len(x)*len(labels),weight)


def one_body_samples(evaluate,positions,spins,basis,points,point_spins,quadrature_weights,chunk=32):
    """Batched callback returns (phase ANGLE, log|Psi|), including zero numerators.

    All electron replacements keep the token's slot, so no extra exchange sign
    appears. The SUM over electrons supplies the factor N. No post-hoc Hermitizing,
    trace renormalization, ratio clipping, or hidden sample reweighting is applied.
    """
    x=np.asarray(positions,dtype=float);s=np.asarray(spins,dtype=float)
    y=np.asarray(points,dtype=float);sy=np.asarray(point_spins);qw=np.asarray(quadrature_weights)
    if x.ndim!=3 or x.shape[-1]!=2 or s.shape!=x.shape[:2]:raise ValueError('positions [M,N,2], spins [M,N] required')
    if y.ndim!=2 or y.shape[1]!=2 or sy.shape!=(len(y),) or qw.shape!=(len(y),) or np.any(qw<0):
        raise ValueError('quadrature positions/components/weights mismatch')
    if basis.components is not None and not set(sy).issubset(set(basis.components)):
        raise ValueError('quadrature components outside fixed basis')
    if type(chunk) is not int or chunk<1:raise ValueError('positive chunk required')
    phi_y=basis(y,sy);result=[]
    for start in range(0,len(x),chunk):
        xx=x[start:start+chunk];ss=s[start:start+chunk];M,N=xx.shape[:2];Q=len(y)
        ph0,la0=map(np.asarray,evaluate(xx,ss))
        if ph0.shape!=(M,) or la0.shape!=(M,) or not np.isfinite(ph0).all() or not np.isfinite(la0).all():
            raise ValueError('sampling at nodes/nonfinite denominators is not supported')
        phi_x=basis(xx,ss);local=np.zeros((M,basis.size,basis.size),complex)
        for i in range(N):
            moved=np.broadcast_to(xx[:,None,:,:],(M,Q,N,2)).copy()
            ms=np.broadcast_to(ss[:,None,:],(M,Q,N)).copy()
            moved[:,:,i,:]=y
            if basis.components is not None:ms[:,:,i]=sy
            ph,la=map(np.asarray,evaluate(moved.reshape(M*Q,N,2),ms.reshape(M*Q,N)))
            ph=ph.reshape(M,Q);la=la.reshape(M,Q)
            finite=np.isfinite(la)
            if np.isnan(la).any() or np.isposinf(la).any() or not np.isfinite(ph[finite]).all():
                raise ValueError('nonfinite replacement amplitudes')
            delta=la-la0[:,None]
            if np.any(delta[finite]>np.log(np.finfo(float).max)-2):raise FloatingPointError('wavefunction ratio overflow; no clipping applied')
            ratio=np.zeros((M,Q),complex)
            angle=ph-ph0[:,None]
            ratio[finite]=np.exp(delta[finite]+1j*angle[finite])
            coefficient=np.einsum('mq,q,qb->mb',ratio,qw,phi_y.conj())
            local+=np.einsum('ma,mb->mab',phi_x[:,i,:],coefficient)
        result.append(local)
    return np.concatenate(result,axis=0)


def density_summary(samples,sample_weights=None,chain_ids=None):
    samples=np.asarray(samples,dtype=complex)
    if samples.ndim!=3 or samples.shape[-1]!=samples.shape[-2] or not np.isfinite(samples).all():raise ValueError('finite [samples,D,D] required')
    weights=np.ones(len(samples)) if sample_weights is None else np.asarray(sample_weights,dtype=float)
    if weights.shape!=(len(samples),) or np.any(weights<0) or not weights.sum()>0:raise ValueError('nonnegative nonzero sample weights required')
    weights=weights/weights.sum();g=np.einsum('m,mab->ab',weights,samples)
    hermitian=(g+g.conj().T)/2
    result=dict(G=g,trace=np.trace(g),hermiticity_residual=np.linalg.norm(g-g.conj().T),
                hermitian_part_eigenvalues=np.linalg.eigvalsh(hermitian),hermitized_for_diagnostics_only=hermitian)
    if chain_ids is not None:
        if sample_weights is not None:raise ValueError('chain error estimates require equal-weight MC samples')
        ids=np.asarray(chain_ids)
        if ids.shape!=(len(samples),):raise ValueError('one chain id per configuration required')
        means=np.stack([samples[ids==i].mean(axis=0) for i in np.unique(ids)])
        result['independent_chain_means']=means
        if len(means)>1:
            result['sem_real']=means.real.std(axis=0,ddof=1)/np.sqrt(len(means))
            result['sem_imag']=means.imag.std(axis=0,ddof=1)/np.sqrt(len(means))
    return result


def independent_loop_products(chain_matrices,triangles):
    """Distinct independent chains for each of the three factors (U statistic).

    Avoids plug-in cubic bias from reusing the same noisy estimate. Independence
    is a sampling requirement, not something this function can establish.
    """
    g=np.asarray(chain_matrices);t=np.asarray(triangles,dtype=int).reshape(-1,3)
    if len(g)<3:raise ValueError('at least 3 independent chains required')
    def estimate(gg):
        total=np.zeros(len(t),complex);count=0
        for a,b,c in permutations(range(len(gg)),3):
            total+=gg[a,t[:,0],t[:,1]]*gg[b,t[:,1],t[:,2]]*gg[c,t[:,2],t[:,0]];count+=1
        return total/count
    value=estimate(g);result=dict(product=value)
    if len(g)>=4:
        leave=np.stack([estimate(np.delete(g,i,axis=0)) for i in range(len(g))])
        result['jackknife_sem_imag']=np.sqrt((len(g)-1)/len(g)*np.sum((leave.imag-leave.imag.mean(axis=0))**2,axis=0))
    return result


def physical_loop_statistics(G,basis,window=None,chain_matrices=None):
    """Im(G_ab G_bc G_ca) on a fixed spatial basis, CCW geometry and distance bins.

    Optional Window is evaluated on BASIS-centre distances, never a copied
    electron-token adjacency matrix. Shapes [...,D,D], typically [L,H,D,D].
    """
    g=np.asarray(G);D=basis.size
    if g.shape!=(D,D) or basis.centres is None:raise ValueError('matching fixed spatial basis and G required')
    geometry=oriented_triangles(basis.centres,basis.lattice);t=geometry['triangles']
    product_value=g[t[:,0],t[:,1]]*g[t[:,1],t[:,2]]*g[t[:,2],t[:,0]]
    result=dict(**geometry,product=product_value,imaginary_loop=product_value.imag,
                orientation='CCW contractible triangles in a fixed physical orbital basis; not a Chern number')
    if chain_matrices is not None:
        result.update({'independent_'+k:v for k,v in independent_loop_products(chain_matrices,t).items()})
    if window is not None:
        w=np.asarray(window,dtype=float)
        if w.shape[-2:]!=(D,D) or np.any(w<0) or not np.isfinite(w).all():raise ValueError('finite nonnegative basis-centre windows required')
        mass=w[...,t[:,0],t[:,1]]*w[...,t[:,1],t[:,2]]*w[...,t[:,2],t[:,0]]
        den=mass.sum(axis=-1)
        result['window_loop_mass']=mass
        loop=result.get('independent_product',product_value).imag
        result['window_weighted_mean']=np.divide((mass*loop).sum(axis=-1),den,
            out=np.full(den.shape,np.nan),where=den>0)
    return result


def make_network_evaluator(network,params,atoms,charges,phase_convention='angle'):
    """Reuse Network.apply, not a reconstructed wavefunction forward.

    For real-output networks explicitly choose 'sign'. Complex-output networks
    in this repository return phase angles. JAX dtype/settings are caller-owned.
    """
    import jax
    import jax.numpy as jnp
    if phase_convention not in ('angle','sign'):raise ValueError('phase convention must be explicit')
    def one(r,s):
        phase,logabs=network.apply(params,r.reshape(-1),s,atoms,charges)
        return (phase if phase_convention=='angle' else jnp.angle(phase+0j)),logabs
    evaluate=jax.jit(jax.vmap(one))
    return lambda positions,spins:tuple(np.asarray(a) for a in evaluate(jnp.asarray(positions),jnp.asarray(spins)))


def sample_wavefunction(evaluate,lattice,particles,seed=1701,chains=4,sweeps=512,burn=256,
                        steps_per_sample=4,width=.2,pseudospin=False):
    """Fresh periodic Metropolis chains targeting |Psi(R,S)|²; fixed trial params.

    Independent RNG streams per chain. Pseudospin flips require a wavefunction
    defined in the full component Hilbert space, not a fixed real-spin sector.
    This read-only sampler does not resume or mutate the trainer or checkpoint.
    """
    if min(particles,chains,sweeps,steps_per_sample)<1 or burn<0 or width<=0:raise ValueError('positive sampling sizes/width and nonnegative burn required')
    L=np.asarray(lattice);rngs=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(chains)]
    frac=np.stack([rng.uniform(size=(particles,2)) for rng in rngs]);x=frac@L.T
    s=np.stack([rng.choice([-1.,1.],size=particles) for rng in rngs]) if pseudospin else np.ones((chains,particles))
    _,la=evaluate(x,s)
    if not np.isfinite(la).all():raise ValueError('initial MCMC configurations are nodes/nonfinite')
    accepted=np.zeros(chains);spin_accepted=np.zeros(chains);positions=[];spins=[]
    for step in range((burn+sweeps)*steps_per_sample):
        proposal_frac=np.mod(frac+np.stack([rng.normal(0,width,size=(particles,2)) for rng in rngs]),1)
        proposal=proposal_frac@L.T;_,new=evaluate(proposal,s)
        if np.isnan(new).any() or np.isposinf(new).any():raise ValueError('invalid proposal wavefunction')
        logu=np.log(np.array([rng.uniform() for rng in rngs]));take=logu<2*(new-la)
        frac[take]=proposal_frac[take];x[take]=proposal[take];la=np.where(take,new,la)
        if step>=burn*steps_per_sample:accepted+=take
        if pseudospin:
            proposed=s.copy()
            for c,rng in enumerate(rngs):proposed[c,rng.integers(particles)]*=-1
            _,new=evaluate(x,proposed)
            if np.isnan(new).any() or np.isposinf(new).any():raise ValueError('invalid spin-flip wavefunction')
            take=np.log(np.array([rng.uniform() for rng in rngs]))<2*(new-la)
            s[take]=proposed[take];la=np.where(take,new,la)
            if step>=burn*steps_per_sample:spin_accepted+=take
        if step>=burn*steps_per_sample and (step+1)%steps_per_sample==0:
            positions.append(x.copy());spins.append(s.copy())
    return dict(positions=np.asarray(positions),spins=np.asarray(spins),
                acceptance=accepted/(sweeps*steps_per_sample),
                spin_acceptance=spin_accepted/(sweeps*steps_per_sample) if pseudospin else None,
                seed=seed,independent_chains=chains,burn=burn,steps_per_sample=steps_per_sample,width=width)


def basis_windows(basis,window_parameters,settings,dref,num_heads):
    """Evaluate the EXISTING radial Window kernel at fixed physical basis centres.

    No token QK/alpha is transplanted to orbital labels. This is a scale filter
    learned by the network, not a new estimator for G or an additive observable.
    """
    import jax
    import jax.numpy as jnp
    from periodicwave import attention,networks
    if basis.centres is None:raise ValueError('spatial basis centres required')
    with jax.experimental.enable_x64(settings['jax_enable_x64']):
        geometry=networks.make_periodic_construct_input_features(jnp.asarray(basis.lattice))
        d=geometry(jnp.asarray(basis.centres.reshape(-1)),jnp.zeros((1,2)),ndim=2)[3][...,0]
        windows=[]
        for record in window_parameters:
            params={k:jnp.asarray(v) for k,v in record.items()}
            windows.append(np.ones((num_heads,basis.size,basis.size)) if 'mask_omega' not in params else
                np.asarray(attention.screening_window(params,d,jnp.asarray(dref),
                           win_shape=settings['WIN_SHAPE'],freeze_w=settings['FREEZE_W'])))
    return np.asarray(windows)
