import torch as th
import numpy as np
import tensorflow as tf
import math
from abc import ABC, abstractmethod

class Space(ABC):
    def __init__(self, dim, bound, symmetric :bool = True):
        self.dim = dim
        self.action_dim = dim
        self.bound = bound
        self.bound_np = (bound[0].cpu().numpy(), bound[1].cpu().numpy())
        self.pre_layer = None
        self.symmetric = symmetric

    def to(self, device):
        self.device = device
        self.bound = (self.bound[0].to(device), self.bound[1].to(device))
        return self

    def calc_deltas(self, states_0, states_1):
        d=states_1-states_0
        ret = self.F(states_0, d)
        return ret

    @abstractmethod
    def F(self, states, d):
        pass

    @abstractmethod
    def F_np(self, state, d):
        pass
    
    def calc_delta_np(self, state_0, state_1):
        d=state_1-state_0
        ret = self.F_np(state_0, d)
        return ret
    
    def calc_reward(self, state_0, state_1):
        d=state_0-state_1
        ret = self.F_np(state_0, d)
        return ret
    
    def get_random_state(self):
        return self.bound[0]+(self.bound[1]-self.bound[0])*th.rand(self.dim, device = self.device)

    def get_random_states(self, batch_size):
        return self.bound[0]+(self.bound[1]-self.bound[0])*th.rand((batch_size, self.dim), device = self.device)
    
    def get_random_state_np(self):
        return self.bound_np[0]+(self.bound_np[1]-self.bound_np[0])*np.random.random(self.dim)

    def is_free_state(self, state):
        return (self.bound_np[0]<=state).all() and (state <= self.bound_np[1]).all()

    def penalty(self, states):
        min_diff =  states - states.clamp(min=self.bound[0][None,:])
        max_diff =  states - states.clamp(max=self.bound[1][None,:])
        return (min_diff**2 + max_diff**2).sum(dim=1)

    def penalty_np(self, state):
        min_diff =  state - state.clip(min=self.bound_np[0])
        max_diff =  state - state.clip(max=self.bound_np[1])
        return (min_diff**2 + max_diff**2).sum(dim=1)

    def clamp(self, states):
        return states.clamp(min=self.bound[0][None,:], max=self.bound[1][None,:])

    def clamp_np(self, state):
        return state.clip(min=self.bound_np[0], max=self.bound_np[1])

    def clamp_tf(self, state):
        return tf.clip_by_value(state, self.bound_np[0], self.bound_np[1])

    def add_action(self, state, action):
        return state + action
