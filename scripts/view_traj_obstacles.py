import numpy as np
import torch as th
from stochastic_actor import StochasticActor4Layer
from critic import Critic
from optimize import calc_true_trajectory
import sys
import math
import random
random.seed(0)
th.manual_seed(0)
np.random.seed(0)
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.size"] = 24

space_name = "Obstacle4Outer"

episode_num = 0
seed=12

from pick_space import pick_space
space, eval_episodes = pick_space(space_name)
dim = space.dim

fig = plt.figure(figsize = (7, 8.3))

ax = fig.add_subplot(1,1,1)
ax.set_xlim(0., 1.)
ax.set_ylim(0., 1.)
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle(xy=(0.2,0.), width = 0.2, height = .8, color='black'))
ax.add_patch(Rectangle(xy=(0.6,0.2), width = 0.2, height = .8, color='black'))
ax.set_aspect(1.)


log_dir = '../exp/Obstacle4Outer/ACDQT-'+str(seed)

actor = StochasticActor4Layer(dim, net_arch = [400, 300, 300])
actor.load_state_dict(th.load(log_dir+'/actor_model.pt', map_location = 'cpu'))

count=0

eval_episodes = [(th.tensor([0.1,0.1]), th.tensor([0.9,0.9]))]

depth = 6

for start, goal in eval_episodes:
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

states = states.numpy()
ax.plot(states[:,0], states[:,1], marker = 'o', ms = 12, label = 'Our-T', linestyle = 'solid', markevery = 8)

# Seq

from stable_baselines3 import PPO
from gym_env_wrapper import SpaceEnv

log_dir = "../exp/Obstacle4Outer/"

model = PPO.load(log_dir+"Seq-"+str(seed)+"/best_model.zip")
eval_env = SpaceEnv(space, step_length = 0.1, max_step = 64)

done = False

obs = eval_env.reset(options={"start": eval_episodes[0][0].numpy(), "goal": eval_episodes[0][1].numpy()})

traj = [eval_episodes[0][0].numpy()]
dist = 0
while not done:
    action, _states = model.predict(obs, deterministic = True)
    obs, reward, done, info = eval_env.step(action)
    traj.append(obs[:space.dim])
    dist+=space.calc_delta_np(traj[-2], traj[-1])

traj.append(eval_episodes[0][1].numpy())
states = np.array(traj)
ax.plot(states[:,0], states[:,1], 'P', ms = 12, label = "Seq", linestyle = 'solid', markevery = 8, color = 'C2')

# Inter

log_dir = '../exp/Obstacle4Outer/Inter-'+str(seed)

actor = StochasticActor4Layer(dim, net_arch = [400, 300, 300])
actor.load_state_dict(th.load(log_dir+'/actor_model.pt', map_location = 'cpu'))

eval_episodes = [(th.tensor([0.1,0.1]), th.tensor([0.9,0.9]))]

depth = 6

for start, goal in eval_episodes:
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

states = states.numpy()

ax.plot(states[:,0], states[:,1], marker = 's', ms = 12, label = 'Inter', linestyle = 'solid', markevery = 8, color = 'C4')

fig.subplots_adjust(left=0.13, right=0.96, bottom=0.21, top=0.99)
plt.legend(bbox_to_anchor = (-0.16,-0.05), loc = 'upper left', ncol = 3)

plt.savefig("../figures/obstacles_trajectories.pdf")
plt.show()
