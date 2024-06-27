import torch as th
import numpy as np
from space import Space

class Obstacle(Space):
    def __init__(self):
        super().__init__(2, (th.zeros(2), th.ones(2)))
        self.polygon = th.tensor([[0.,0.],[0.,1.],[0.6,1.],[0.6,0.2],[0.8,0.2],[0.8,1.0],[1.,1.],[1.,0.],[0.4,0.],[0.4,0.8],[0.2,0.8],[0.2,0.]])
        self.seg_vec = th.roll(self.polygon, 1, dims=0) - self.polygon
        self.lengs = self.seg_vec.norm(dim=1)
        self.seg_vec = self.seg_vec / self.lengs[:,None]

    def to(self, device):
        super().to(device)
        self.polygon = self.polygon.to(device = device)
        self.lengs = self.lengs.to(device = device)
        self.seg_vec = self.seg_vec.to(device = device)
        return self
    
    def F(self, states, d):
        vs = states[:,None,:]-self.polygon[None,:,:]
        point_dist = vs.norm(dim=2)
        inner = (vs*self.seg_vec[None,:,:]).sum(dim=2)
        line_dist = th.where(th.logical_and(th.gt(inner,0),th.lt(inner,self.lengs[None,:])), th.abs(self.seg_vec[None,:,0]*vs[:,:,1]-self.seg_vec[None,:,1]*vs[:,:,0]), th.inf)
        dist = th.minimum(point_dist.min(dim=1).values, line_dist.min(dim=1).values)
        return (1.+0.01/(dist**2+1e-9))*d.norm(dim=1)

    def F_np(self, state, d):
        return self.F(th.tensor(state,dtype=th.float32).unsqueeze(0), th.tensor(d,dtype=th.float32).unsqueeze(0))[0].numpy()
