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
parser.add_argument("--log_dir", action="store", default="../exp/Panda5/ACDQT-11")
parser.add_argument("--depth", action="store", default=2, type = int)

args = parser.parse_args()

log_dir = args.log_dir
space_name = 'Panda5'

episode_num = 0

from pick_space import pick_space
space, eval_episodes = pick_space(space_name)
dim = space.dim

actor = StochasticActor4Layer(dim, unscaled = True, net_arch = [400, 300, 300])
actor.load_state_dict(th.load(log_dir+'/actor_model.pt', map_location = 'cpu'))
count=0
depth = args.depth

start = th.tensor([-1.2279485,   0.5184796,   0.9566655,  -0.44470763, -0.9326868,   1.8705176, 1.4915984 ])

goal = th.tensor([ 0.8752949, 0.9677279,  -0.36568952, -1.5134895,   0.67131805,  3.0369098, 2.7819703 ])

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
deltas=space.calc_deltas(states[:-1], states[1:])
# print(deltas.max())

import roboticstoolbox as rp
import numpy as np

panda = rp.models.DH.Panda()

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.size"] = 0
plt.rcParams["figure.figsize"] = (7,7)

cube_points = [[0.1, -0.1, 0.],
               [0.1, 0.1, 0.],
               [3.0, 0.1, 0.],
               [3.0, -0.1, 0.]]

states = states.numpy()
azim = -90
elev = 90

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import itertools
               
for index, state in enumerate(states):
    fig = panda.plot(state, jointaxes = False, eeframe = False, name = False)
    ax = fig.ax
    
    ax.set_xlim(-0.3, 0.7)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0., 1.0)
    ax.view_init(azim = azim, elev = elev)
    ax.dist = 7.7
    ax.set_box_aspect((1.,1.,1.))
    
    ax.add_collection3d(Poly3DCollection([cube_points], color='b', alpha=0.6))
    #fig.hold()
    plt.savefig('../figure/panda5_'+str(index)+'.png', bbox_inches='tight')
    plt.clf()
