import numpy as np
import torch as th
from torch import nn

class Critic(nn.Module):
    def __init__(self, dim, learning_rate = 1e-6, net_arch = [64,64]):
        super().__init__()
        self.dim = dim
        input_dim = 2*dim
        if len(net_arch) == 2:
            self.seq = nn.Sequential(
                nn.Linear(input_dim, net_arch[0]),
                nn.ReLU(),
                nn.Linear(net_arch[0], net_arch[1]),
                nn.ReLU(),
            )
        elif len(net_arch) == 3:
            self.seq = nn.Sequential(
                nn.Linear(input_dim, net_arch[0]),
                nn.ReLU(),
                nn.Linear(net_arch[0], net_arch[1]),
                nn.ReLU(),
                nn.Linear(net_arch[1], net_arch[2]),
                nn.ReLU(),
            )
        self.optimizer = th.optim.Adam(self.parameters(), lr = learning_rate)

    def forward(self, states_0, states_1):
        inputs = th.cat((states_0, states_1),1)
        return self.seq(inputs)[:,0]
