"""Optional, explicitly distinct PH / spectral / weight consistency penalties.

PH coordinates are piecewise differentiable through selected critical edges.
Discrete pairings and argmax selections are not differentiated. At ties or
pairing changes the classical derivative need not exist. No claim of a globally
smooth PH loss, or of a physical topological invariant, is made.
"""
from itertools import combinations
import numpy as np
import jax
import jax.numpy as jnp
from periodicwave import constants,networks

DEFAULTS=dict(mode='off',gamma=0.,perturb_sigma=.03,max_samples=8,ph_floor=1e-4,
              landscape_levels=3,landscape_grid=21,sampling_score=True)
CATEGORIES={'off':'disabled','persistence':'true PH landscape of epsilon-completed flag complex',
            'laplacian':'differentiable Laplacian spectral moments; NOT PH',
            'weights':'weight smoothness; NOT a topological loss'}


def configuration(values=None):
    c={**DEFAULTS,**dict(values or {})}
    if set(c)!=set(DEFAULTS):raise ValueError('unknown topology regularization keys')
    if c['mode'] not in CATEGORIES:raise ValueError('mode must be off/persistence/laplacian/weights')
    if not np.isfinite(c['gamma']) or c['gamma']<0:raise ValueError('finite gamma>=0 required')
    if not np.isfinite(c['perturb_sigma']) or c['perturb_sigma']<=0:raise ValueError('positive perturb_sigma required')
    if not 0<c['ph_floor']<1:raise ValueError('PH completion floor must be in (0,1)')
    if any(type(c[k]) is not int or c[k]<1 for k in ('max_samples','landscape_levels','landscape_grid')):
        raise ValueError('positive integer sample/landscape dimensions required')
    if type(c['sampling_score']) is not bool:raise ValueError('sampling_score must be boolean')
    return c


def enabled(values=None):
    c=configuration(values);return c['mode']!='off' and c['gamma']>0


def _critical_indices(weights):
    """Host F2 reduction of a complete positive graph, indices of critical edges.

    Fixed shape [dim=0,1; padded E intervals; birth/death]. Sentinel E means
    vertex birth 0. Diagonal padding is harmless for landscapes. Essential H0
    is the same one interval in every completed graph and is excluded.
    """
    b=np.asarray(weights);n=len(b);edges=list(combinations(range(n),2));M=len(edges)
    if n>36:raise ValueError('reference PH regularizer limited to N<=36')
    if not np.isfinite(b).all() or np.any(b[np.triu_indices(n,1)]<=0):raise ValueError('positive completed weights required')
    edge_index={e:i for i,e in enumerate(edges)}
    values=1-b[np.triu_indices(n,1)]
    simplices=[(0.,(i,),M) for i in range(n)]
    simplices.extend((values[i],e,i) for i,e in enumerate(edges))
    for triangle in combinations(range(n),3):
        ids=[edge_index[e] for e in combinations(triangle,2)];critical=ids[int(np.argmax(values[ids]))]
        simplices.append((values[critical],triangle,critical))
    simplices.sort(key=lambda a:(a[0],len(a[1]),a[1]));indices={s:i for i,(_,s,_) in enumerate(simplices)}
    reduced={};pivots={};diagrams=[[],[]]
    for j,(time,simplex,critical) in enumerate(simplices):
        column=0
        if len(simplex)>1:
            for face in combinations(simplex,len(simplex)-1):column^=1<<indices[face]
        while column and column.bit_length()-1 in pivots:column^=reduced[pivots[column.bit_length()-1]]
        reduced[j]=column
        if column:
            i=column.bit_length()-1;pivots[i]=j
            born,face,bc=simplices[i];dim=len(face)-1
            if dim<2 and time>born:diagrams[dim].append((bc,critical))
    result=np.zeros((2,M,2),np.int32)
    for dim,diagram in enumerate(diagrams):
        if diagram:result[dim,:len(diagram)]=diagram
    return result


def persistence_coordinates(weights):
    """JIT/first-gradient compatible critical coordinates; host pairing is stopped."""
    n=weights.shape[-1];M=n*(n-1)//2
    if n<2:raise ValueError('topology regularization requires at least 2 electrons')
    ij=np.triu_indices(n,1)
    ids=jax.pure_callback(_critical_indices,jax.ShapeDtypeStruct((2,M,2),jnp.int32),
                          jax.lax.stop_gradient(weights),vmap_method='sequential')
    edge_times=jnp.concatenate((1-weights[ij],jnp.zeros((1,),weights.dtype)))
    return edge_times[ids]


def ph_landscape(weights,levels=3,grid_size=21):
    diagram=persistence_coordinates(weights);grid=jnp.linspace(0.,1.,grid_size,dtype=weights.dtype)
    tents=jax.nn.relu(jnp.minimum(grid[None,None,:]-diagram[:,:,0,None],diagram[:,:,1,None]-grid[None,None,:]))
    tents=jnp.concatenate((tents,jnp.zeros((2,levels,grid_size),weights.dtype)),axis=1)
    return jnp.sort(tents,axis=1)[:,-levels:,:][:,::-1,:]


def spectral_moments(weights):
    """Tr[(L/N)^k]/N for k=1..4; smooth polynomials even at degenerate spectra."""
    n=weights.shape[-1];lap=(jnp.diag(weights.sum(axis=-1))-weights)/n
    power=jnp.eye(n,dtype=weights.dtype);moments=[]
    for _ in range(4):power=power@lap;moments.append(jnp.trace(power)/n)
    return jnp.stack(moments)


def graph_descriptor(weights,config):
    """[L,H,N,N] -> head-permutation invariant descriptor for each layer.

    PH and moments refer to the graph of MEAN head weights, not to a fixed head
    identity and not to the mean of per-head diagrams. Weight smoothness retains
    direction. Self edges are excluded in every mode.
    """
    n=weights.shape[-1];off=1-jnp.eye(n,dtype=weights.dtype)
    mean=weights.mean(axis=1)*off
    if config['mode']=='weights':return mean
    symmetric=(mean+mean.swapaxes(-1,-2))/2
    if config['mode']=='laplacian':return jax.vmap(spectral_moments)(symmetric)
    if config['mode']=='persistence':
        completed=(config['ph_floor']+(1-config['ph_floor'])*symmetric)*off
        return jax.vmap(lambda b:ph_landscape(b,config['landscape_levels'],config['landscape_grid']))(completed)
    raise ValueError('disabled mode has no graph descriptor')


def consistency_cost(weights,nearby_weights,config):
    a=graph_descriptor(weights,config);b=graph_descriptor(nearby_weights,config)
    return jnp.mean((a-b)**2)/config['perturb_sigma']**2


def make_regularized_loss(base_loss,network,diagnostic,values=None):
    """Return EXACT original callable when off/gamma=0; no extra trace/forward.

    Otherwise optimize E_VMC + gamma E_|psi|²,noise ell(theta,R,noise). Direct
    parameter derivatives AND the 2 cov(ell, d log|psi|) sampling-score term are
    included by default. sampling_score=False is an explicitly conditional,
    fixed-configuration penalty, not the derivative of the full MC expectation.
    AuxiliaryLossData remains the ORIGINAL physical energy/variance structure.
    """
    config=configuration(values)
    if not enabled(config):return base_loss
    def one_weights(p,x,s,a,c):
        _,records=diagnostic(p,x,s,a,c)
        return jnp.stack([record['attention']['weights_window'] for record in records])
    batch_weights=jax.vmap(one_weights,in_axes=(None,0,0,0,0))
    batch_logabs=jax.vmap(lambda p,x,s,a,c:network.apply(p,x,s,a,c)[1],in_axes=(None,0,0,0,0))
    def penalty(params,key,data):
        count=min(config['max_samples'],data.positions.shape[0])
        data=jax.tree.map(lambda a:jax.lax.stop_gradient(a[:count]),data)
        noise=jax.random.normal(key,data.positions.shape,dtype=data.positions.dtype)*config['perturb_sigma']
        original=batch_weights(params,data.positions,data.spins,data.atoms,data.charges)
        nearby=batch_weights(params,data.positions+noise,data.spins,data.atoms,data.charges)
        local=jax.vmap(lambda a,b:consistency_cost(a,b,config))(original,nearby)
        mean=constants.pmean(jnp.mean(local))
        if config['sampling_score']:
            logabs=batch_logabs(params,data.positions,data.spins,data.atoms,data.charges)
            centered=jax.lax.stop_gradient(local-mean)
            mean=mean+2*constants.pmean(jnp.mean(centered*(logabs-jax.lax.stop_gradient(logabs))))
        return mean
    def evaluate(params,key,data):
        # Original energy key is unchanged; no extra PRNG split on the baseline path.
        energy,aux=base_loss(params,key,data)
        regularizer=penalty(params,jax.random.fold_in(key,7127),data)
        return jnp.real(energy)+config['gamma']*regularizer,aux
    return evaluate
