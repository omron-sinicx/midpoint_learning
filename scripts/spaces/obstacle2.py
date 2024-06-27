import torch as th
import numpy as np
from space import Space

def cross_prod(vector_0, vector_1):
    return vector_0[:,:,0]*vector_1[:,:,1]-vector_1[:,:,0]*vector_0[:,:,1]

def cross_half(points_0, points_1, points_2, points_3):
    v1 = points_1-points_0
    v2 = points_2-points_0
    v3 = points_3-points_0
    return th.le(cross_prod(v1,v2)*cross_prod(v1,v3), 0.)

def cross(points_0, points_1, points_2, points_3):
    return th.logical_and(cross_half(points_0,points_1,points_2,points_3),
                          cross_half(points_2,points_3,points_0,points_1))

class Obstacle2(Space):
    def __init__(self, segment_0, segment_1, penalty_coef: float = 10.):
        super().__init__(2, (th.zeros(2), th.ones(2)))
        self.segment_0 = segment_0
        self.segment_1 = segment_1
        self.penalty_coef = penalty_coef

    def to(self, device):
        super().to(device)
        self.segment_0 = self.segment_0.to(device = device)
        self.segment_1 = self.segment_1.to(device = device)
        return self

    def calc_deltas(self, states_0, states_1):
        d=states_1-states_0
        crossing = cross(self.segment_0.unsqueeze(0),
                         self.segment_1.unsqueeze(0),
                         states_0.unsqueeze(1), states_1.unsqueeze(1))
        return d.norm(dim=1)+self.penalty_coef*crossing.sum(dim=1)

    def calc_delta_np(self, state_0, state_1):
        return self.calc_deltas(th.tensor(state_0,dtype=th.float32).unsqueeze(0), th.tensor(state_1,dtype=th.float32).unsqueeze(0))[0].numpy()

    def calc_reward(self, state_0, state_1):
        return self.F_np(state_0, state_0-state_1)

    def F_np(self, state_0, d):
        return np.linalg.norm(d)
