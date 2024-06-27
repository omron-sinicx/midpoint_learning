import numpy as np
import torch as th
from torch import nn
import math

from torch.distributions import Normal

class StochasticActor4Layer(nn.Module):
    def __init__(self, dim, learning_rate = 1e-6, unscaled = False, net_arch = [64,64]):
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
        self.mu = nn.Linear(net_arch[-1], dim)
        self.log_std = nn.Linear(net_arch[-1], dim)
        self.unscaled = unscaled
        self.optimizer = th.optim.Adam(self.parameters(), lr = learning_rate)

    def forward(self, states_0, states_1, deterministic:bool = False,):
        inputs = th.cat((states_0, states_1),1)
        latent_pi = self.seq(inputs)
        mean = self.mu(latent_pi)
        if deterministic:
            return mean
        distribution = Normal(th.zeros(self.dim, device = states_0.device), th.ones(self.dim, device = states_0.device))
        log_std = self.log_std(latent_pi)
        batch_size = mean.shape[0]
        samples = mean +  log_std.exp() * distribution.sample((batch_size,))
        return samples
