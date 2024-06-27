import torch as th
import numpy as np
import tensorflow as tf
import math

from space import Space
from .euclid import Euclid, EuclidDisk

class CarLikeDisk(Space):
    def __init__(self, r_min :float = 0.5, c_p :float = 100., c_p2 :float = 100., radius :float = 1.):
        super().__init__(3, (th.tensor([-1,-1,-th.pi]), th.tensor([1,1,th.pi])), False)
        self.r_min = r_min
        self.c_p = c_p
        self.c_p2 = c_p2
        self.disk = EuclidDisk(radius = radius)
        self.angle = Euclid(1, (th.tensor([-th.pi]), th.tensor([th.pi])))

    def F(self, states, d):
        dist=d[:,:2].norm(dim=1)
        theta = states[:,2]
        h  = th.cos(theta)*d[:,1]-th.sin(theta)*d[:,0]
        adb = th.abs(d[:,2])
        h2 = (adb*self.r_min-th.cos(theta)*d[:,0]-th.sin(theta)*d[:,1]).clamp(min=0)
        return th.sqrt(dist**2+self.c_p*h**2+self.c_p2 * h2**2)

    def F_np(self, state, d):
        dist=np.linalg.norm(d[:2])
        theta = state[2]
        h  = np.cos(theta)*d[1]-np.sin(theta)*d[0]
        adb = np.abs(d[2])
        h2 = (adb*self.r_min-np.cos(theta)*d[0]-np.sin(theta)*d[1]).clip(min=0)
        return np.sqrt(dist**2+self.c_p*h**2+self.c_p2*h2**2)
    
    def penalty(self, states):
        return self.disk.penalty(states[:,:2])+self.angle.penalty(states[:,2:])

    def penalty_np(self, state):
        return self.disk.penalty_np(state[:2])+self.angle.penalty_np(state[2:])

    def clamp(self, states):
        return th.concat((self.disk.clamp(states[:,:2]),self.angle.clamp(states[:,2:])), dim=1)

    def clamp_np(self, state):
        return np.concatenate((self.disk.clamp_np(state[:2]),self.angle.clamp_np(state[2:])))

    def clamp_tf(self, state):
        return tf.concat((self.disk.clamp_tf(state[:,:2]),self.angle.clamp_tf(state[:,2:])),1)

    def get_random_state(self):
        return th.concat((self.disk.get_random_state(),self.angle.get_random_state()))

    def get_random_states(self, batch_size):
        return th.concat((self.disk.get_random_states(batch_size),self.angle.get_random_states(batch_size)),1)
    
    def get_random_state_np(self):
        return np.concatenate((self.disk.get_random_state_np(),self.angle.get_random_state_np()))

    def is_free_state(self, state):
        return self.disk.is_free_state(state[:2]) and self.angle.is_free_state(state[2:])

    def to(self, device):
        self.device = device
        self.disk = self.disk.to(device)
        self.angle = self.angle.to(device)
        return self
