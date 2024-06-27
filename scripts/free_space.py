import torch as th
import numpy as np
from space import Space
from abc import ABC, abstractmethod


class FreeSpace(Space):
    
    def __init__(self, dim, bound, symmetric :bool = True, penalty_term: float = 10.):
        super().__init__(dim, bound, symmetric)
        self.penalty_term = penalty_term

    @abstractmethod
    def is_invalid_states(self, states):
        pass

    @abstractmethod
    def is_invalid_state(self, state):
        pass
        
    def calc_deltas(self, states_0, states_1):
        d = states_1 - states_0
        return self.F(states_0, d) + self.penalty_term*th.logical_xor(self.is_invalid_states(states_0), self.is_invalid_states(states_1))

    def calc_delta_np(self, state_0, state_1):
        d = state_1 - state_0
        return self.F_np(state_0, d) + self.penalty_term*(self.is_invalid_state(state_0) ^ self.is_invalid_state(state_1))

    def get_random_state_np(self):
        while True:
            state = super().get_random_state_np()
            if not self.is_invalid_state(state):
                break
        return state
    
    def get_random_states(self, batch_size):
        ret = th.tensor([], device = self.device)
        while ret.shape[0] < batch_size:
            states = super().get_random_states(batch_size)
            states = states[th.nonzero(~self.is_invalid_states(states)).squeeze(dim=1)]
            ret = th.concat((ret, states))
        return ret[:batch_size]
