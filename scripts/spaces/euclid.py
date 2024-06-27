import torch as th
import numpy as np
from space import Space
from .disk import Disk

class Euclid(Space):
    def F(self, states, d):
        return d.norm(dim=1)

    def F_np(self, state, d):
        return np.linalg.norm(d)

class EuclidDisk(Disk):
    def F(self, states, d):
        return d.norm(dim=1)

    def F_np(self, state, d):
        return np.linalg.norm(d)
