import torch as th
import numpy as np
import tensorflow as tf
import math

from space import Space
from .euclid import EuclidDisk

class CarLikeDisk3(Space):
    def __init__(self, r_min :float = 0.5, c_p :float = 100., c_p2 :float = 100., radius :float = 1.):
        super().__init__(4, (-th.ones(4), th.ones(4)), False)
        self.action_dim = 3
        self.r_min = r_min
        self.c_p = c_p
        self.c_p2 = c_p2
        self.disk = EuclidDisk(radius = radius)

    def F(self, states, d):
        dist=d[:,:2].norm(dim=1)
        h  = states[:,2]*d[:,1]-states[:,3]*d[:,0]
        adb = th.abs(d[:,2])
        h2 = (adb*self.r_min-states[:,2]*d[:,0]-states[:,3]*d[:,1]).clamp(min=0)
        return th.sqrt(dist**2+self.c_p*h**2+self.c_p2 * h2**2)

    def F_np(self, state, d):
        dist=np.linalg.norm(d[:2])
        h  = state[2]*d[1]-state[3]*d[0]
        adb = np.abs(d[2])
        h2 = (adb*self.r_min-state[2]*d[0]-state[3]*d[1]).clip(min=0)
        return np.sqrt(dist**2+self.c_p*h**2+self.c_p2*h2**2)
    
    def clamp(self, states):
        clamped = states[:,2:].clamp(min = -1., max = 1.)
        norm = clamped.norm(dim=1)
        return th.concat((self.disk.clamp(states[:,:2]), th.where(th.gt(norm, 0)[:,None], clamped/norm[:,None], np.sqrt(1./2.))), dim=1)

    def add_action(self, state, action):
        c = np.cos(action[2])
        s = np.sin(action[2])
        return np.concatenate((state[:2] + action[:2], [state[2] * c - state[3] * s, state[2] * s + state[3] * c]))

    def clamp_np(self, state):
        assert abs(np.linalg.norm(state[2:]) - 1.) < 1e-3
        return np.concatenate((self.disk.clamp_np(state[:2]), state[2:]))

    def clamp_tf(self, states):
        clamped = tf.clip_by_value(states[:,2:], -1., 1.)
        norm = tf.norm(clamped, axis=1)
        return tf.concat((self.disk.clamp_tf(states[:,:2]), tf.where(tf.math.greater(norm, 0.)[:, None], clamped / norm[:, None], np.sqrt(1./2.))),1)

    def get_random_states(self, batch_size):
        theta = -th.pi + 2 * th.pi * th.rand((batch_size, 1), device = self.device)
        return th.concat((self.disk.get_random_states(batch_size), th.cos(theta), th.sin(theta)),1)
    
    def get_random_state_np(self):
        theta = -np.pi + 2 * np.pi * np.random.random(1)
        return np.concatenate((self.disk.get_random_state_np(), np.cos(theta), np.sin(theta)))

    def calc_deltas(self, states_0, states_1):
        d=states_1[:,:2]-states_0[:,:2]
        theta = th.atan2(-states_0[:,3] *states_1[:,2] + states_0[:,2] * states_1[:,3], states_0[:,2] * states_1[:,2] + states_0[:,3] * states_1[:,3])
        ret = self.F(states_0, th.concat((d, theta.unsqueeze(1)),1))
        return ret

    def calc_delta_np(self, state_0, state_1):
        d=state_1-state_0
        theta = math.atan2(-state_0[3] * state_1[2] + state_0[2] * state_1[3], state_0[2] * state_1[2] + state_0[3] * state_1[3])
        ret = self.F_np(state_0, np.concatenate((d, [theta])))
        return ret
    
    def calc_reward(self, state_0, state_1):
        d=state_0-state_1
        theta = math.atan2(state_0[3] * state_1[2] - state_0[2] * state_1[3], state_0[2] * state_1[2] + state_0[3] * state_1[3])
        ret = self.F_np(state_0, np.concatenate((d, [theta])))
        return ret
    
    def to(self, device):
        self.device = device
        self.disk = self.disk.to(device)
        return self
