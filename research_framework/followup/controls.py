"""Global MLP capacity controls: same permitted information and training data."""
import torch
from torch import nn
from .holonomy import inputs


class GlobalMLP(nn.Module):
    def __init__(self,tokens=25,width=24):
        super().__init__();self.blocks=nn.ModuleList()
        self.net=nn.Sequential(nn.Linear(tokens*8,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh())
        self.output=nn.Linear(width,2)

    def forward(self,batch,intervention=None,record=False):
        b=inputs(batch);z=b['z']*b['carrier'].conj()[...,None]
        features=torch.cat([z.real,z.imag,b['coords']],-1).flatten(1)
        h=self.net(features);prediction=self.output(h)
        return (prediction,dict(representation=h,trace=[])) if record else prediction
