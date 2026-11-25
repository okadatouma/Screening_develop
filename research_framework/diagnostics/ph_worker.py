"""Isolated Ripser worker: never fork an already initialized JAX process."""
import sys
from pathlib import Path
import numpy as np
from ripser import ripser
from scipy.spatial.distance import pdist,squareform


def main():
    source,target=map(Path,sys.argv[1:])
    points=np.load(source)['points'];d=squareform(pdist(points))
    result=ripser(d,distance_matrix=True,maxdim=1,coeff=2)
    with target.open('xb') as f:np.savez_compressed(f,H0=result['dgms'][0],H1=result['dgms'][1],diameter=d.max())


if __name__=='__main__':main()
