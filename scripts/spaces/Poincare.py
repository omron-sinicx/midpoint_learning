import torch as th
import numpy as np
import math
from spaces.disk import Disk

class Poincare(Disk):
    def F(self, states, d):
        return 2*d.norm(dim=1)/(1-(states**2).sum(axis=1))

    def F_np(self, state, d):
        return 2*np.linalg.norm(d)/(1-(state**2).sum())
