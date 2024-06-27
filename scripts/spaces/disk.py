import torch as th
import numpy as np
import tensorflow as tf
import math
from space import Space

class Disk(Space):
    def __init__(self, dim = 2, radius:float = 0.95, symmetric = True):
        super().__init__(dim, bound = (-radius * th.ones(dim), radius * th.ones(dim)), symmetric = symmetric)
        self.radius = radius

    def penalty(self, states):
        return (states.norm(dim=1)-self.radius).clamp(min=0.)**2

    def penalty_np(self, state):
        return (np.linalg.norm(state)-self.radius).clip(min=0.)**2

    def clamp(self, states):
        return (self.radius/states.norm(dim=1).clamp(min=self.radius))[:,None]*states

    def clamp_np(self, state):
        return (self.radius/np.linalg.norm(state).clip(min=self.radius))*state

    def clamp_tf(self, state):
        return (self.radius/tf.math.maximum(tf.norm(state, axis = 1), self.radius))[:,None]*state

    def get_random_state(self):
        while True:
            state = super().get_random_state()
            if state.norm() < self.radius:
                break
        return state

    def get_random_state_np(self):
        while True:
            state = super().get_random_state_np()
            if self.is_free_state(state):
                break
        return state

    def get_random_states(self, batch_size):
        radii = self.radius * th.sqrt(th.rand(batch_size, device = self.device))
        angles = 2 * th.pi * th.rand(batch_size, device = self.device)
        return th.stack((th.cos(angles) * radii, th.sin(angles) * radii), 1)

    def is_free_state(self, state):
        return np.linalg.norm(state) < self.radius
        
