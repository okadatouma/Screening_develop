"""Permutation-invariant hidden point clouds, analyzed separately from graph PH."""
import numpy as np
from scipy.spatial.distance import pdist,squareform
from screening_analysis.topology import persistence,summary


def pool_hidden(hidden,method='mean_std'):
    """[...,N,E] -> invariant feature; feature coordinates are NOT aligned across seeds."""
    h=np.asarray(hidden,dtype=float)
    if h.ndim<2 or not np.isfinite(h).all():raise ValueError('finite hidden [...,N,E] required')
    mean=h.mean(axis=-2)
    if method=='mean':return mean
    if method=='mean_std':return np.concatenate((mean,h.std(axis=-2)),axis=-1)
    if method=='moments':
        return np.concatenate((mean,h.std(axis=-2),h.min(axis=-2),h.max(axis=-2)),axis=-1)
    if method=='gram_spectrum':
        centered=h-mean[...,None,:]
        gram=centered@centered.swapaxes(-1,-2)/h.shape[-1]
        spectrum=np.linalg.eigvalsh(gram)[...,::-1]
        return np.concatenate((spectrum,np.linalg.norm(mean,axis=-1)[...,None]),axis=-1)
    raise ValueError('unknown pooling method: '+method)


def point_cloud_ph(points):
    """Euclidean Rips 2-skeleton at its actual distance scale, all edges retained.

    Maps d -> B=1-d/(2*dmax) for the existing reference reducer and converts
    diagrams BACK to Euclidean distance. The scale is saved, not discarded.
    """
    z=np.asarray(points,dtype=float)
    if z.ndim!=2 or not len(z) or not np.isfinite(z).all():raise ValueError('finite nonempty [samples,features] required')
    d=squareform(pdist(z));scale=2*d.max(initial=0.)
    weights=np.ones_like(d) if scale==0 else 1-d/scale
    diagrams=tuple(a.copy() for a in persistence(weights))
    for a in diagrams:a[np.isfinite(a)]*=scale
    return dict(diagrams=diagrams,distance=d,conversion_scale=scale,
                summaries=[summary(a) for a in diagrams])


def analyze_hidden(hidden,methods=('mean','mean_std','moments','gram_spectrum'),max_points=64):
    """[B,L,N,E] -> per-layer clouds and ordinary PH; no layerwise zigzag claims."""
    h=np.asarray(hidden)
    if h.ndim!=4:raise ValueError('hidden [B,L,N,E] required')
    if len(h)>max_points:raise ValueError('explicitly select <=64 configurations for reference PH')
    result={}
    for method in methods:
        clouds=pool_hidden(h,method)
        result[method]=dict(clouds=clouds,layers=[point_cloud_ph(clouds[:,layer]) for layer in range(h.shape[1])])
    return result


def grouped_centroid(features,labels,groups):
    """Leave-group-out; train-only scaling. Caller supplies actual independent groups."""
    x=np.asarray(features,dtype=float);y=np.asarray(labels);g=np.asarray(groups)
    if x.ndim!=2 or len(x)!=len(y) or len(y)!=len(g) or not np.isfinite(x).all():raise ValueError('finite aligned features/labels/groups required')
    classes=np.unique(y);predictions=[];truth=[];heldout=[]
    for group in np.unique(g):
        train=g!=group;test=~train
        if any(not np.any(train&(y==c)) for c in classes):continue
        mean=x[train].mean(axis=0);sd=x[train].std(axis=0);sd=np.where(sd>1e-10,sd,1.)
        z=(x-mean)/sd;centers=np.stack([z[train&(y==c)].mean(axis=0) for c in classes])
        predicted=classes[np.argmin(np.sum((z[test,None]-centers[None])**2,axis=-1),axis=1)]
        predictions.extend(predicted.tolist());truth.extend(y[test].tolist());heldout.extend(g[test].tolist())
    confusion=np.zeros((len(classes),len(classes)),int)
    for a,b in zip(truth,predictions):confusion[np.flatnonzero(classes==a)[0],np.flatnonzero(classes==b)[0]]+=1
    return dict(classes=classes.tolist(),predicted=predictions,actual=truth,groups=heldout,
                accuracy=float(np.mean(np.asarray(predictions)==truth)) if truth else None,
                confusion=confusion.tolist(),independent_groups=len(np.unique(g)))
