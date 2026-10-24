"""研究用Window。単位は周期ノルムの長さ。既存Windowの既定値は変更しない。"""
import jax
import jax.numpy as jnp
import numpy as np


def input_features(ae, spins, lattice, mode, harmonic=6):
    """p=(sinθ1,sinθ2,cosθ1,cosθ2)。E0/E1/E2/E3の次元は5/11/12/18。

    θ=2π A^-1(r−atom)。χの6はスーパーセルに対する高調波数。
    N36の6×6セルでは結晶周期に対応する。Nを変えても勝手に変更しない。
    """
    theta=2*jnp.pi*jnp.einsum('ab,ib->ia',jnp.linalg.inv(lattice),ae[:,0,:])
    p=jnp.concatenate((jnp.sin(theta),jnp.cos(theta)),axis=-1)
    phases=harmonic*jnp.stack((theta[:,0],theta[:,1],theta[:,0]-theta[:,1]),axis=-1)
    chi=jnp.stack((jnp.cos(phases),jnp.sin(phases)),axis=-1).reshape(len(spins),6)
    onehot=jnp.stack(((1+spins)/2,(1-spins)/2),axis=-1)
    if mode=='E0':return jnp.concatenate((p,spins[:,None]),axis=-1)
    if mode=='E1':return jnp.concatenate((p,spins[:,None],chi),axis=-1)
    if mode=='E2':return jnp.concatenate((p,chi,onehot),axis=-1)
    if mode=='E3':return jnp.concatenate((p,onehot,onehot[:,:1]*chi,onehot[:,1:]*chi),axis=-1)
    raise ValueError('input must be E0/E1/E2/E3')


def partition(eta, min_gap_fraction):
    """正の増分の正規化。fraction=0が提案式。>0は一様混合による間隔下限。

    例fraction=.02では、各gapは全幅の.02/H以上。順序だけで密集を防げるとは限らない。
    """
    delta=jax.nn.softplus(eta)
    p=delta/jnp.sum(delta)
    return (1-min_gap_fraction)*p+min_gap_fraction/len(eta)


def initialize(config,dref,dtype):
    # ここは初期化だけ。W0は独立幅、W1/W2は正増分、W3は基底混合logitsをパラメータにする。
    # 幅・境界・混合係数は異なる量なので、評価時のdetailsでも区別する。
    H=config['heads'];lo=config['w_min'];hi=config['w_max'] or dref
    if not 0<=lo<hi or hi>dref*(1+1e-7):raise ValueError('0<=w_min<w_max<=d_refが必要です')
    if not lo<config['w_first']<=hi:raise ValueError('w_min<w_first<=w_maxが必要です')
    if H<2:raise ValueError('順序/距離比較にはheads>=2が必要です')
    mode=config['window'];fraction=config['min_gap_fraction']
    if mode in ('W2','W3') and not np.isclose(hi,dref,rtol=1e-6,atol=1e-7):
        raise ValueError('分割窓W2/W3の外端はセルのd_refです。--w-maxを省略してください')
    if not 0<=fraction<1:raise ValueError('min_gap_fractionは[0,1)です')
    if mode=='W0':
        widths=np.geomspace(config['w_first'],hi,H)
        p={'omega':jnp.asarray(np.log(np.expm1(widths/dref)),dtype)}
    elif mode in ('W1','W2'):
        if mode=='W1':
            target=(np.geomspace(config['w_first'],hi,H)-lo)/(hi-lo)
            gaps=np.diff(np.r_[0.,target])
        else:gaps=np.ones(H)/H
        positive=(gaps-fraction/H)/(1-fraction)
        if np.any(positive<=0):raise ValueError('初期梯子と最小間隔が矛盾します。min_gap_fractionを小さくしてください')
        p={'increments':jnp.asarray(np.log(np.expm1(positive)),dtype)}
    elif mode=='W3':
        edges=np.asarray(config['basis_edges'],float)
        if edges.ndim!=1 or len(edges)<3 or edges[0]!=0 or not np.all(np.diff(edges)>0) or not np.isclose(edges[-1],hi):
            raise ValueError('basis_edgesは0からw_maxまでの昇順境界です')
        B=len(edges)-1;logits=np.full((H,B),-3.)
        logits[np.arange(H),np.minimum(np.arange(H)*B//H,B-1)]=3.
        p={'logits':jnp.asarray(logits,dtype)}
    else:raise ValueError('window must be W0/W1/W2/W3')
    if config['pair_windows']:
        if mode not in ('W1','W2'):raise ValueError('擬スピン対補正は順序付きW1/W2専用です')
        p['pair_shift']=jnp.zeros((3,H),dtype)  # ++, --, +- (= -+)
    return p


def scales(params,config,dref):
    """返値[3,H]。W1は幅、W2は各ringの外側境界。端はw_maxに固定。"""
    lo=config['w_min'];hi=config['w_max'] or dref;mode=config['window']
    if mode=='W0':base=dref*jax.nn.softplus(params['omega'])
    else:
        c=jnp.cumsum(partition(params['increments'],config['min_gap_fraction']))
        c=c.at[-1].set(1.)
        base=hi*c if mode=='W2' else lo+(hi-lo)*c
    values=jnp.broadcast_to(base,(3,len(base)))
    if 'pair_shift' in params:
        # |δ_h|<.45*近傍log-gapなら、隣同士が交差しない。外端は動かさない。
        logb=jnp.log(base);gap=jnp.diff(logb)
        bound=.45*jnp.minimum(jnp.r_[gap[0],gap],jnp.r_[gap,gap[-1]])
        bound=jnp.minimum(bound,config['pair_max_log_shift']).at[-1].set(0.)
        if mode=='W1' and lo>0:
            bound=bound.at[0].set(jnp.minimum(bound[0],.9*jnp.log(base[0]/lo)))
        values=values*jnp.exp(jnp.tanh(params['pair_shift'])*bound[None,:])
    return values


def ring_basis(distance,boundaries,tau):
    """C(d;b)=sigmoid((b²−d²)/(2bτ))。C0=0,CH=1を固定し、m_h=C_h+1−C_h。

    中間b>0でCはbに関し単調。d²を使い原点の一次微分をゼロにする。
    非負のC∞窓、sum_h m_h=1（丸め誤差内）。最外ringの自己項も削除しない。
    """
    d2=distance[None,:,:]**2
    inside=boundaries[:-1,None,None]
    cumulative=jax.nn.sigmoid((inside**2-d2)/(2*inside*tau))
    return jnp.diff(jnp.concatenate((jnp.zeros_like(d2),cumulative,jnp.ones_like(d2)),axis=0),axis=0)


def evaluate(params,distance,spins,config,dref):
    """学習に使う実Window[H,N,N]と、解釈用の幅/境界/混合係数を返す。"""
    mode=config['window'];tau=config['transition']
    if tau<=0:raise ValueError('transitionは正の長さです')
    if mode=='W3':
        edges=jnp.asarray(config['basis_edges'],distance.dtype)
        basis=ring_basis(distance,edges[1:],tau)
        pi=jax.nn.softmax(params['logits'],axis=-1)
        window=jnp.einsum('hr,rij->hij',pi,basis)
        details={'mixture':pi,'basis_edges':edges}
    else:
        widths=scales(params,config,dref)
        channels=[]
        for c in range(3):
            if mode=='W2':m=ring_basis(distance,widths[c],tau)
            else:
                t=jnp.clip(distance[None,:,:]/widths[c,:,None,None],0.,1.)
                m=.5*(1+jnp.cos(jnp.pi*t))
            channels.append(m)
        pair=jnp.where((spins[:,None]>0)&(spins[None,:]>0),0,
                       jnp.where((spins[:,None]<0)&(spins[None,:]<0),1,2))
        window=jnp.zeros_like(channels[0])
        for c in range(3):window=window+channels[c]*(pair==c)[None]
        details={'scales_pair':widths}
    # 再標本化するR*介入。値/一次〜三次微分が切替点で接続するsmooth7の襟。
    radius=config.get('cut_radius')
    if radius is not None:
        beta=config['cut_beta'];inner=(1-beta)*radius
        t=jnp.clip((distance**2-inner**2)/(radius**2-inner**2),0.,1.)
        smooth=t**4*(35-84*t+70*t**2-20*t**3)
        collar=1-smooth
        head=config.get('cut_head')
        if head is None:window=window*collar[None]
        else:window=window.at[head].multiply(collar)
    return window,details
