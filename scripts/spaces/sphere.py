import torch as th
import numpy as np
import math
from spaces.disk import Disk

class Sphere(Disk):
    def mapping(self, states):
        X = states[:,0]
        Y = states[:,1]
        N = 1 + X **2 + Y ** 2
        x = 2*X/N
        y = 2*Y/N
        z = 1-2/N
        return th.stack((x,y,z),1)

    def F(self, states, d):
        return 2*d.norm(dim=1)/(1+(states**2).sum(axis=1))

    def F_np(self, state, d):
        return 2*np.linalg.norm(d)/(1+(state**2).sum())
