"""Classical Born MI and quantum MI are distinct; all entropies are in nats."""
import numpy as np


def classical_entropy(p,sites,n):
    tensor=np.asarray(p).reshape((2,)*n)
    axes=tuple(i for i in range(n) if i not in sites)
    marginal=tensor.sum(axis=axes) if axes else tensor
    values=marginal.ravel();values=values[values>0]
    return float(-np.sum(values*np.log(values)))


def born_mi(state,n):
    p=abs(state)**2;p=p/p.sum();out=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            out[i,j]=out[j,i]=classical_entropy(p,[i],n)+classical_entropy(p,[j],n)-classical_entropy(p,[i,j],n)
    return out


def reduced_density(state,sites,n):
    order=list(sites)+[i for i in range(n) if i not in sites]
    matrix=np.asarray(state).reshape((2,)*n).transpose(order).reshape(2**len(sites),-1)
    return matrix@matrix.conj().T/np.vdot(state,state).real


def von_neumann(rho):
    p=np.maximum(np.linalg.eigvalsh(rho),0);p=p[p>1e-14]
    return float(-np.sum(p*np.log(p)))


def quantum_mi(state,n):
    out=np.zeros((n,n));single=[von_neumann(reduced_density(state,[i],n)) for i in range(n)]
    for i in range(n):
        for j in range(i+1,n):out[i,j]=out[j,i]=single[i]+single[j]-von_neumann(reduced_density(state,[i,j],n))
    return out
