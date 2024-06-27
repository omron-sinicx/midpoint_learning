import torch as th
import numpy as np
import tensorflow as tf
import math
from space import Space

import pytorch_kinematics as pk
import kinpy as kp

class Panda3(Space):
    def __init__(self, wall_x: float = 0.4, penalty_term: float = 10.):
        q_mins = th.tensor([-2.8973, -1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
        q_maxs = th.tensor([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])
        super().__init__(7, (q_mins,q_maxs))
        self.chain = pk.build_serial_chain_from_urdf(open("../data/panda.urdf").read(), 'panda_hand')
        self.chain_np = kp.build_serial_chain_from_urdf(open("../data/panda.urdf").read(), 'panda_hand')
        self.wall_x = wall_x
        self.penalty_term = penalty_term

    def to(self, device):
        super().to(device)
        self.chain = self.chain.to(device = device)
        return self

    def is_invalid_states(self, states):
        links = self.chain.forward_kinematics(states, end_only = False)
        invalid = th.zeros(states.shape[0], device = states.device, dtype = th.bool)
        for link in links.values():
            pos_x = link.get_matrix()[:,0,3]
            invalid += th.ge(pos_x, self.wall_x)
        return invalid
    
    def is_invalid_state(self, state):
        links = self.chain_np.forward_kinematics(state, end_only = False)
        invalid = False
        for link in links.values():
            pos_x = link.pos[0]
            invalid |= (pos_x > self.wall_x)
        return invalid


    def F(self, states, d):
        return d.norm(dim=1)

    def F_np(self, state, d):
        return np.linalg.norm(d)

    def calc_deltas(self, states_0, states_1):
        d = states_1 - states_0
        return d.norm(dim=1)+self.penalty_term*th.logical_xor(self.is_invalid_states(states_0), self.is_invalid_states(states_1))

    def calc_delta_np(self, state_0, state_1):
        d = state_1 - state_0
        return np.linalg.norm(d)+self.penalty_term*(self.is_invalid_state(state_0) ^ self.is_invalid_state(state_1))

    def get_random_state_np(self):
        while True:
            state = super().get_random_state_np()
            if not self.is_invalid_state(state):
                break
        return state
    
    def get_random_states(self, batch_size):
        ret = th.tensor([], device = self.device)
        while ret.shape[0] < batch_size:
            states = super().get_random_states(batch_size)
            states = states[th.nonzero(~self.is_invalid_states(states)).squeeze(dim=1)]
            ret = th.concat((ret, states))
        return ret[:batch_size]
    
