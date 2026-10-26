"""Separate PH, winding and density confounds in synthetic geometry classification.

python -m screening_analysis.topology_evaluate PILOT_JSON --output NEW_JSON
No physical phase labels or statistical significance are inferred.
"""
import argparse
import json
from pathlib import Path
import numpy as np


FEATURES={
    'ph':('H0_total_persistence','H1_total_persistence','H0_essential_intervals',
          'H1_essential_intervals','beta0_auc','beta1_auc'),
    'winding':('rank1_probability','rank2_probability'),
    'density_weight':('support_density','weight_mean','weight_std')}
FEATURES['combined']=FEATURES['ph']+FEATURES['winding']


def evaluate(records):
    results=[]
    for mode in sorted({r['run']['window'] for r in records}):
        for source in ('window','trim','screening','qk'):
            for control in ('raw','rank','shuffle'):
                subset=[r for r in records if r['run']['window']==mode]
                for feature_set,keys in FEATURES.items():
                    X=np.array([[next(a for a in r['aggregates'] if a['source']==source and a['control']==control)[k]
                                 for k in keys] for r in subset])
                    y=np.array([r['run']['synthetic_geometry']=='ordered' for r in subset],int)
                    groups=np.array([r['run']['seed'] for r in subset]);predictions=[];truth=[]
                    for seed in np.unique(groups):
                        train=groups!=seed;test=~train
                        mean=X[train].mean(axis=0);scale=X[train].std(axis=0);scale=np.where(scale>1e-10,scale,1.)
                        z=(X-mean)/scale;centroids=np.stack([z[train&(y==c)].mean(axis=0) for c in (0,1)])
                        pred=np.argmin(np.sum((z[test,None,:]-centroids[None,:,:])**2,axis=-1),axis=-1)
                        predictions.extend(pred.tolist());truth.extend(y[test].tolist())
                    confusion=np.zeros((2,2),int)
                    for actual,predicted in zip(truth,predictions):confusion[actual,predicted]+=1
                    results.append(dict(window=mode,source=source,control=control,feature_set=feature_set,
                        accuracy=float(np.mean(np.asarray(predictions)==truth)),confusion_uniform_ordered=confusion.tolist(),
                        held_out_groups=3,test_units=6))
    return dict(label='synthetic prescribed ordered vs uniform geometry; untrained network, NOT WC/AHC',
        split='leave-network-initialization-seed-out, train-only scaling, no tuning',
        uncertainty='three shared seed groups and one prescribed dataset per label; exploratory, no significance estimate',
        features=FEATURES,results=results)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('pilot',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    data=json.loads(args.pilot.read_text());result=evaluate(data['synthetic_runs'])
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)


if __name__=='__main__':main()
