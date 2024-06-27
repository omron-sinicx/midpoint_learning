import torch as th
import numpy as np
from free_space import FreeSpace

class MultiAgent(FreeSpace):
    def __init__(self, N : int = 4, thres : float = 2./3.):
        super().__init__(2*N, (-th.ones(2*N), th.ones(2*N)))
        self.N = N
        self.thres2 = thres**2

    def F(self, states, d):
        return sum([d[:,2*i:2*(i+1)].norm(dim=1) for i in range(self.N)])

    def F_np(self, state, d):
        return sum([np.linalg.norm(d[2*i:2*(i+1)]) for i in range(self.N)])

    def is_invalid_states(self, states):
        ret = th.zeros(states.shape[0], device = self.device, dtype = th.bool)
        for i in range(self.N):
            for j in range(i+1, self.N):
                ret |= th.lt(((states[:,2*i:2*(i+1)]-states[:,2*j:2*(j+1)])**2).sum(dim=1), self.thres2)
        return ret

    def is_invalid_state(self, state):
        ret = False
        for i in range(self.N):
            for j in range(i+1, self.N):
                ret |= (((state[2*i:2*(i+1)]-state[2*j:2*(j+1)])**2).sum() < self.thres2)
        return ret
