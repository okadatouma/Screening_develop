"""Run/seed-group evaluation, cluster uncertainty and label permutation refits.

No phase is inferred from a filename or Hamiltonian parameter. Labels and their
provenance are caller supplied. Confidence intervals from few groups are limited.
"""
from itertools import permutations,product
import numpy as np
from scipy.stats import rankdata
from screening_analysis.topology import diagram_distance


def binary_auc(truth,score):
    y=np.asarray(truth,dtype=bool);score=np.asarray(score,dtype=float);n1=y.sum();n0=len(y)-n1
    if not n1 or not n0:return None
    rank=rankdata(score,method='average')
    return float((rank[y].sum()-n1*(n1+1)/2)/(n1*n0))


def classification_metrics(truth,predicted,scores,classes=None):
    y=np.asarray(truth);p=np.asarray(predicted);s=np.asarray(scores,dtype=float)
    classes=np.unique(np.r_[y,p]) if classes is None else np.asarray(classes)
    if y.ndim!=1 or p.shape!=y.shape or s.shape!=(len(y),len(classes)) or not np.isfinite(s).all():raise ValueError('aligned labels and finite class scores required')
    cm=np.zeros((len(classes),len(classes)),int);mapping={c:i for i,c in enumerate(classes.tolist())}
    for a,b in zip(y.tolist(),p.tolist()):cm[mapping[a],mapping[b]]+=1
    true=cm.sum(axis=1);pred=cm.sum(axis=0);tp=np.diag(cm);count=cm.sum()
    recall=np.divide(tp,true,out=np.full(len(classes),np.nan),where=true>0)
    f1=np.divide(2*tp,true+pred,out=np.zeros(len(classes),float),where=true+pred>0)
    denominator=np.sqrt(float(count**2-np.dot(true,true))*float(count**2-np.dot(pred,pred)))
    mcc=float((np.trace(cm)*count-np.dot(true,pred))/denominator) if denominator>0 else 0.
    if len(classes)==2:
        auroc=binary_auc(y==classes[1],s[:,1]-s[:,0])
    else:
        auc=[binary_auc(y==c,s[:,i]) for i,c in enumerate(classes)]
        auroc=float(np.mean(auc)) if all(a is not None for a in auc) else None
    return dict(classes=classes.tolist(),balanced_accuracy=float(recall.mean()) if np.isfinite(recall).all() else None,
        auroc=auroc,f1_macro=float(f1.mean()),f1_binary=float(f1[1]) if len(classes)==2 else None,
        matthews_correlation=mcc,confusion_matrix=cm.tolist(),test_units=int(count))


def grouped_predictions(features,labels,groups,holdout_factors=None):
    """Leave-group-out or simultaneous held-out group AND factor.

    With factors, test=(group=g AND factor=f), train=(group!=g AND factor!=f).
    Everything in either held-out axis is excluded from training; no seed leak.
    Standardization and centroids are fitted on train only, without tuning.
    """
    x=np.asarray(features,dtype=float);y=np.asarray(labels);g=np.asarray(groups)
    if x.ndim!=2 or len(x)!=len(y) or len(g)!=len(y) or not np.isfinite(x).all():raise ValueError('finite aligned [run,feature], labels, groups required')
    classes=np.unique(y)
    if len(classes)<2:raise ValueError('at least two labels required')
    factors=np.zeros(len(y),int) if holdout_factors is None else np.asarray(holdout_factors)
    if len(factors)!=len(y):raise ValueError('factor length mismatch')
    predicted=np.empty(len(y),dtype=classes.dtype);score=np.full((len(y),len(classes)),np.nan);folds=[]
    for group in np.unique(g):
        for factor in np.unique(factors[g==group]):
            test=g==group;train=~test
            if holdout_factors is not None:test=test&(factors==factor);train=train&(factors!=factor)
            if any(not np.any(train&(y==c)) for c in classes):
                raise ValueError('held-out fold leaves a class absent from training')
            mean=x[train].mean(axis=0);scale=x[train].std(axis=0);scale=np.where(scale>1e-10,scale,1.)
            z=(x-mean)/scale;centroids=np.stack([z[train&(y==c)].mean(axis=0) for c in classes])
            distance=np.sum((z[test,None,:]-centroids[None,:,:])**2,axis=-1)
            score[test]=-distance;predicted[test]=classes[np.argmin(distance,axis=1)]
            folds.append(dict(group=str(group),factor=str(factor),train_indices=np.flatnonzero(train).tolist(),test_indices=np.flatnonzero(test).tolist()))
    return dict(predicted=predicted,scores=score,truth=y,groups=g,classes=classes,folds=folds,
                metrics=classification_metrics(y,predicted,score,classes))


def cluster_bootstrap(predictions,repetitions=1000,seed=202611):
    """95% percentile CI of OOF performance, resampling whole independent groups.

    Conditional on the fitted CV models; does not claim to be a nested training
    bootstrap. Resamples missing any target class are omitted and counted.
    """
    rng=np.random.default_rng(seed);g=predictions['groups'];groups=np.unique(g);values={k:[] for k in
        ('balanced_accuracy','auroc','f1_macro','matthews_correlation')};omitted=0
    for _ in range(repetitions):
        draw=rng.choice(groups,len(groups),replace=True);indices=np.concatenate([np.flatnonzero(g==i) for i in draw])
        m=classification_metrics(predictions['truth'][indices],predictions['predicted'][indices],predictions['scores'][indices],predictions['classes'])
        if m['balanced_accuracy'] is None or m['auroc'] is None:omitted+=1;continue
        for k in values:values[k].append(m[k])
    return dict(interval_95={k:np.quantile(v,[.025,.975]).tolist() if v else None for k,v in values.items()},
        repetitions=repetitions,omitted_missing_class=omitted,independent_groups=len(groups),
        caveat='group bootstrap of out-of-fold predictions; very few groups give coarse/unstable intervals; no refit in bootstrap')


def permutation_refits(features,labels,groups,repetitions=999,seed=9021):
    """Exact small balanced paired-label swaps or homogeneous cluster shuffles.

    Refits standardization and classifier for EVERY permutation. No walker-level
    shuffling. Exact enumeration includes the observed labeling. Monte Carlo uses
    the conservative +1 correction. Exchangeability is an experimental assumption.
    """
    y=np.asarray(labels);g=np.asarray(groups);classes=np.unique(y);unique=np.unique(g);rng=np.random.default_rng(seed)
    observed=grouped_predictions(features,y,g)['metrics']['balanced_accuracy']
    indices=[np.flatnonzero(g==i) for i in unique]
    homogeneous=all(len(np.unique(y[ix]))==1 for ix in indices)
    balanced=all(len(ix)%len(classes)==0 and all(np.sum(y[ix]==c)==len(ix)//len(classes) for c in classes) for ix in indices)
    candidates=[];exact=False
    if homogeneous:
        cluster_labels=[y[ix[0]] for ix in indices]
        if len(unique)<=8:
            assignments=sorted(set(permutations(cluster_labels)));exact=True
        else:assignments=[rng.permutation(cluster_labels) for _ in range(repetitions)]
        for assignment in assignments:
            p=y.copy()
            for ix,label in zip(indices,assignment):p[ix]=label
            candidates.append(p)
    elif balanced:
        maps=list(permutations(classes.tolist()));number=len(maps)**len(unique)
        if number<=1024:choices=product(range(len(maps)),repeat=len(unique));exact=True
        else:choices=rng.integers(len(maps),size=(repetitions,len(unique)))
        for choice in choices:
            p=y.copy()
            for ix,k in zip(indices,choice):
                mapping=dict(zip(classes.tolist(),maps[k]));p[ix]=[mapping[a] for a in y[ix]]
            candidates.append(p)
    else:
        return dict(status='unsupported_exchangeability',reason='need homogeneous groups or balanced paired labels per group')
    null=[]
    for p in candidates:null.append(grouped_predictions(features,p,g)['metrics']['balanced_accuracy'])
    count=int(np.sum(np.asarray(null)>=observed-1e-12));pvalue=count/len(null) if exact else (count+1)/(len(null)+1)
    return dict(status='exploratory',observed=observed,pvalue=float(pvalue),exact=exact,permutations=len(null),
                null_balanced_accuracy=null,scheme='homogeneous group labels' if homogeneous else 'class-name bijections within balanced paired groups')


def evaluate(features,labels,groups,bootstrap_repetitions=1000,permutation_repetitions=999):
    predictions=grouped_predictions(features,labels,groups)
    return dict(**predictions['metrics'],bootstrap=cluster_bootstrap(predictions,bootstrap_repetitions),
        permutation=permutation_refits(features,labels,groups,permutation_repetitions),folds=predictions['folds'],
        independent_groups=len(np.unique(groups)),statistical_status='limited' if len(np.unique(groups))<10 else 'requires physical-label validation',
        classifier='nearest centroid; train-only scaling; leave-group-out; no tuning')


def within_between(diagrams,labels,groups,bottleneck=True):
    """Cross-group PH distances; essential-count mismatch stays infinity.

    Distances share runs and are NOT independent observations. Finite-only
    summaries are explicitly conditional; infinity counts are retained.
    """
    rows=[];within=[];between=[]
    for i in range(len(diagrams)):
        for j in range(i+1,len(diagrams)):
            if groups[i]==groups[j]:continue
            d=sum(diagram_distance(a,b,bottleneck=bottleneck) for a,b in zip(diagrams[i],diagrams[j]))
            same=labels[i]==labels[j];(within if same else between).append(d)
            rows.append(dict(i=i,j=j,same_label=bool(same),distance=float(d)))
    def describe(values):
        a=np.asarray(values);finite=a[np.isfinite(a)]
        return dict(pairs=len(a),infinite=int(np.isinf(a).sum()),finite_mean=float(finite.mean()) if len(finite) else None,
                    finite_values=finite.tolist())
    w,b=describe(within),describe(between);margin=None
    if w['finite_mean'] is not None and b['finite_mean'] is not None:
        total=w['finite_mean']+b['finite_mean'];margin=(b['finite_mean']-w['finite_mean'])/total if total>0 else 0.
    return dict(within=w,between=b,finite_normalized_separation=margin,pairs=rows,
                dependence='shared-run pair distances; no IID pair bootstrap',metric='sum of H0/H1 bottleneck' if bottleneck else 'sum of H0/H1 W1')
