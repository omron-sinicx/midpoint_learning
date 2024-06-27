import torch as th
import numpy as np
import tensorflow as tf
import math
from space import Space

import pytorch_kinematics as pk

def cross(a,b):
    return a[:,0]*b[:,1]-a[:,1]*b[:,0]

class Panda2(Space):
    def __init__(self):
        q_mins = th.tensor([-2.8973, -1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
        q_maxs = th.tensor([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])
        super().__init__(7, (q_mins,q_maxs))
        self.chain = pk.build_serial_chain_from_urdf(open("../data/panda.urdf").read(), 'panda_hand')
        self.pole_position = th.tensor([0.4,0.])

    def to(self, device):
        super().to(device)
        self.chain = self.chain.to(device = device)
        self.pole_position = self.pole_position.to(device = device)
        return self

    def F(self, states, d):
        links = self.chain.forward_kinematics(states, end_only = False)
        penalty = th.zeros(states.shape[0], device = states.device)
        link_names = list(links.keys())
        for index in range(len(link_names)-1):
            pos_0 = links[link_names[index]].get_matrix()[:,:2,3]
            pos_1 = links[link_names[index+1]].get_matrix()[:,:2,3]
            dist_0 = th.min((pos_0-self.pole_position).norm(dim=1), (pos_1-self.pole_position).norm(dim=1))
            leng = (pos_1-pos_0).norm(dim=1)
            dist_1 = th.abs(cross(self.pole_position-pos_0,pos_1-pos_0))/leng
            inn = ((pos_1-pos_0)*(self.pole_position-pos_0)).sum(dim=1)
            dist = th.where(th.logical_and(th.gt(inn,0), th.lt(inn, leng**2)), dist_1, dist_0)
            penalty += 1./(dist+1e-9)
        return (1+penalty)*d.norm(dim=1)

    def F_np(self, state, d):
        return self.F(th.tensor(state,dtype=th.float32).unsqueeze(0), th.tensor(d,dtype=th.float32).unsqueeze(0))[0].numpy()

if __name__=='__main__':
    panda = Panda2()
    test = th.tensor([[0.,0.,0.,-1.,0.,0.,0.],[-1.,-1.,-1.,-2.,-1.,1.,1.]])
    links=panda.chain.forward_kinematics(test, end_only=False)
    print(panda.F(test, test))
