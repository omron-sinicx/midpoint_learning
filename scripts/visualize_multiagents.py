import numpy as np
import torch as th
from stochastic_actor import StochasticActor4Layer
from optimize import calc_true_trajectory
import sys
import math
import random
random.seed(0)
th.manual_seed(0)
np.random.seed(0)

import argparse
import pickle
parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", action="store", default="../exp/MultiAgent-3-0.5/ACDQT-11")
parser.add_argument("--depth", action="store", default=2, type = int)

args = parser.parse_args()

log_dir = args.log_dir
space_name = "MultiAgent-3-0.5"

from pick_space import pick_space
space, _ = pick_space(space_name)
dim = space.dim

actor = StochasticActor4Layer(dim, unscaled = True, net_arch = [400, 300, 300])
actor.load_state_dict(th.load(log_dir+'/actor_model.pt', map_location = 'cpu'))
count=0
depth = args.depth

start = th.tensor([-0.75,   -0.5,   0.0,  -0.5, 0.75,   -0.5 ])

goal = th.tensor([0.75,   0.5,   0.0,  0.5, -0.75,   0.5 ])

states = th.zeros((2,dim))
states[0] = start
states[1] = goal
for rev_dep in range(depth):
    with th.no_grad():
        middles = actor(states[:-1],states[1:], deterministic = True)
        middles = space.clamp(middles)
        new_states = th.zeros((states.shape[0]+middles.shape[0], dim))
        new_states[::2] = states
        new_states[1::2] = middles
        states = new_states

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.size"] = 0
plt.rcParams["figure.figsize"] = (7,7)

states = states.numpy()
fig = plt.figure(figsize = (14, 3))

r = 0.25

for index, state in enumerate(states):
    ax = fig.add_subplot(1,len(states),index+1)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect(1.)
    ax.add_patch(plt.Circle((state[0], state[1]), r, color = 'C0'))
    ax.add_patch(plt.Circle((state[2], state[3]), r, color = 'C1'))
    ax.add_patch(plt.Circle((state[4], state[5]), r, color = 'C2'))

    plt.tick_params(bottom = False, top = False, left = False, right = False)
    plt.tick_params(labelbottom = False, labeltop = False, labelleft = False, labelright = False)

fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace = 0.1)
plt.savefig("../figures/trajectories_multi.pdf")
plt.show()
