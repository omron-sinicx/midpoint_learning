import torch as th
import numpy as np
import math

from .disk import Disk

class Matsumoto(Disk):
    def __init__(self, dh):
        super().__init__(dim = 2, radius = 1., symmetric = False)
        self.dh = dh

    def F(self, states, d):
        beta = (self.dh(states)*d).sum(dim=1)
        alpha2 = (d**2).sum(dim=1)+beta**2
        return alpha2/(th.sqrt(alpha2)-beta+1e-9)

    def F_np(self, state, d):
        beta = (self.dh(state)*d).sum()
        alpha2 = (d**2).sum()+beta**2
        return alpha2/(np.sqrt(alpha2)-beta+1e-9)
