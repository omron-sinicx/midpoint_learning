import torch as th
import numpy as np
from space import Space

class Obstacle3(Space):
    def __init__(self, obstacles, penalty_term: float = 10.):
        super().__init__(2, (th.zeros(2), th.ones(2)))
        self.obstacles = obstacles
        self.obstacles_np = obstacles.numpy()
        self.penalty_term = penalty_term
        self.non_reflexive = True

    def to(self, device):
        super().to(device)
        self.obstacles = self.obstacles.to(device = device)
        return self

    def is_invalid_states(self, states):
        return (th.le(self.obstacles[None,:,0],states[:,None,0])
                & th.le(self.obstacles[None,:,1],states[:,None,1])
                & th.ge(self.obstacles[None,:,2],states[:,None,0])
                & th.ge(self.obstacles[None,:,3],states[:,None,1])).max(1)[0]
    
    def is_invalid_state(self, state):
        return (np.less_equal(self.obstacles_np[:,0],state[0])
                & np.less_equal(self.obstacles_np[:,1],state[1])
                & np.greater_equal(self.obstacles_np[:,2],state[0])
                & np.greater_equal(self.obstacles_np[:,3],state[1])).max()
    
    def calc_deltas(self, states_0, states_1):
        d = states_1 - states_0
        return d.norm(dim=1)+self.penalty_term*self.is_invalid_states(states_1)

    def F(self, states, d):
        return d.norm(dim=1)

    def F_np(self, state, d):
        return np.linalg.norm(d)
    
    def calc_delta_np(self, state_0, state_1):
        d = state_1 - state_0
        return np.linalg.norm(d)+self.penalty_term * self.is_invalid_state(state_1)

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
    
