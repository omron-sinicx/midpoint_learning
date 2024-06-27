import torch as th
import numpy as np
import tensorflow as tf
import math
from free_space import FreeSpace

import pytorch_kinematics as pk
import kinpy as kp

class Panda5(FreeSpace):
    def __init__(self, wall_x: float = 0.1, wall_y: float = 0.1, penalty_term: float = 10.):
        q_mins = th.tensor([-2.8973, -1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
        q_maxs = th.tensor([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])
        super().__init__(7, (q_mins,q_maxs))
        self.chain = pk.build_serial_chain_from_urdf(open("../data/panda.urdf").read(), 'panda_hand')
        self.chain_np = kp.build_serial_chain_from_urdf(open("../data/panda.urdf").read(), 'panda_hand')
        self.wall_x = wall_x
        self.wall_y = wall_y
        self.penalty_term = penalty_term
        self.device = 'cpu'

    def to(self, device):
        super().to(device)
        self.chain = self.chain.to(device = device)
        return self

    def is_invalid_states(self, states):
        links = self.chain.forward_kinematics(states, end_only = False)
        invalid = th.zeros(states.shape[0], device = states.device, dtype = th.bool)
        link_names = list(links.keys())
        for index in range(len(link_names)-1):
            pos_0 = links[link_names[index]].get_matrix()[:,:2,3]
            pos_1 = links[link_names[index+1]].get_matrix()[:,:2,3]
            ccw_m = (pos_0[:,0]-self.wall_x)*(pos_1[:,1]+self.wall_y)-(pos_1[:,0]-self.wall_x)*(pos_0[:,1]+self.wall_y)
            ccw_p = (pos_0[:,0]-self.wall_x)*(pos_1[:,1]-self.wall_y)-(pos_1[:,0]-self.wall_x)*(pos_0[:,1]-self.wall_y)
            invalid += th.le((pos_0[:,1] + self.wall_y) * (pos_1[:,1] + self.wall_y), 0) \
                * th.lt((pos_0[:,1]-pos_1[:,1]) * ccw_m, 0)
            invalid += th.le((pos_0[:,1] - self.wall_y) * (pos_1[:,1] - self.wall_y), 0) \
                * th.lt((pos_0[:,1]-pos_1[:,1]) * ccw_p, 0)
            invalid += th.le((pos_0[:,0] - self.wall_x) * (pos_1[:,0] - self.wall_x), 0) \
                * th.lt(ccw_m * ccw_p, 0)
        return invalid
    
    def is_invalid_state(self, state):
        links = self.chain_np.forward_kinematics(state, end_only = False)
        invalid = False
        link_names = list(links.keys())
        for index in range(len(link_names)-1):
            pos_0 = links[link_names[index]].pos[:2]
            pos_1 = links[link_names[index+1]].pos[:2]
            ccw_m = (pos_0[0]-self.wall_x)*(pos_1[1]+self.wall_y)-(pos_1[0]-self.wall_x)*(pos_0[1]+self.wall_y)
            ccw_p = (pos_0[0]-self.wall_x)*(pos_1[1]-self.wall_y)-(pos_1[0]-self.wall_x)*(pos_0[1]-self.wall_y)
            invalid |= ((pos_0[1] + self.wall_y) * (pos_1[1] + self.wall_y) <= 0) \
                and ((pos_0[1]-pos_1[1]) * ccw_m < 0)
            invalid |= ((pos_0[1] - self.wall_y) * (pos_1[1] - self.wall_y) <= 0) \
                and ((pos_0[1]-pos_1[1]) * ccw_p < 0)
            invalid |= ((pos_0[0] - self.wall_x) * (pos_1[0] - self.wall_x) <= 0) \
                and (ccw_m * ccw_p < 0)
        return invalid


    def F(self, states, d):
        return d.norm(dim=1)

    def F_np(self, state, d):
        return np.linalg.norm(d)
