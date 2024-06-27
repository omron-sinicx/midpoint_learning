import torch as th
import numpy as np
from free_space import FreeSpace

class Obstacle4(FreeSpace):
    
    def __init__(self, obstacles, penalty_term: float = 10.):
        super().__init__(2, (th.zeros(2), th.ones(2)), penalty_term)
        self.obstacles = obstacles
        self.obstacles_np = obstacles.numpy()
        
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

    def F(self, states, d):
        return d.norm(dim=1)

    def F_np(self, state, d):
        return np.linalg.norm(d)

    def to(self, device):
        super().to(device)
        self.obstacles = self.obstacles.to(device = device)
        return self
