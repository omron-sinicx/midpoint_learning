from SGT_PG.path_helper import deserialize_uncompress

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.size"] = 24

import sys
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

space_name = "Matsumoto_-1"

episode_num = 0
depth = 6
seed = 14

from pick_space import pick_space
space, eval_episodes = pick_space(space_name)
dim = space.dim



ax = fig.add_subplot(1,1,1)
ax.set_xlim(-0.5, 0.4)
ax.set_ylim(-0.7, 0.2)
ax.set_aspect(1.)

start, goal = eval_episodes[episode_num]

true_traj = calc_true_trajectory(space, start, goal, N=63, timesteps=1000)
true_traj = true_traj.numpy()

ax.plot(true_traj[:,0], true_traj[:,1], '*', ms = 12, color = 'black', label = "Truth", linestyle = 'solid', markevery = 8)

def mt_plot(folder, name, marker):
    log_dir = "../exp/Matsumoto_-1/"

    actor = StochasticActor4Layer(dim, net_arch = [64, 64])
    actor.load_state_dict(th.load(log_dir+folder+'-'+str(seed)+'/actor_model.pt', map_location = 'cpu'))
    critic = Critic(dim)
    critic.load_state_dict(th.load(log_dir+folder+'-'+str(seed)+'/critic_model.pt', map_location = 'cpu'))

    start, goal = eval_episodes[episode_num]

    states = th.zeros((2,dim))
    states[0] = start
    states[1] = goal
    for _ in range(depth):
        with th.no_grad():
            middles = actor(states[:-1],states[1:], deterministic = True)
            middles = space.clamp(middles)
            dist = critic(states[:-1],states[1:])
            new_states = th.zeros((states.shape[0]+middles.shape[0], dim))
            new_states[::2] = states
            new_states[1::2] = middles
            states = new_states

    states = states.numpy()

    ax.plot(states[:,0], states[:,1], marker, ms = 12, label = name, linestyle = 'solid', markevery = 8)

mt_plot("ACDQT", "Our-T", 'o')
mt_plot("ACDQC", "Our-C", 'D')

from stable_baselines3 import PPO
from gym_env_wrapper import SpaceEnv

dim = space.dim
states = th.zeros((2,dim))
states[0] = eval_episodes[episode_num][0]
states[1] = eval_episodes[episode_num][1]

initial_dist=0.
for start, goal in eval_episodes:
    initial_dist = space.F_np(goal.numpy(), goal.numpy()- start.numpy())

log_dir = "../exp/Matsumoto_-1/"

model = PPO.load(log_dir+"Seq-"+str(seed)+"/final_model.zip")
eval_env = SpaceEnv(space, pre_layer = space.pre_layer, step_length = 0.1, max_step = 2** depth)

done = False

obs = eval_env.reset(options={"start": states[0].numpy(), "goal": states[1].numpy()})

traj = [states[0].numpy()]
dist = 0
while not done:
    action, _states = model.predict(obs, deterministic = True)
    obs, reward, done, info = eval_env.step(action)
    traj.append(obs[:space.dim])
    dist+=space.calc_delta_np(traj[-2], traj[-1])

traj.append(states[1].numpy())
states = np.array(traj)
ax.plot(states[:,0], states[:,1], 'P', ms = 12, label = "Seq", linestyle = 'solid', markevery = 8)

traj_log_dir = "../exp/Matsumoto_-1/SGT_"+str(seed)+"/test_trajectories/level6_all.txt"

states = deserialize_uncompress(traj_log_dir)

states = states[episode_num][0]

ax.plot(states[:,0], states[:,1], 'x', ms = 12, label = "PG", linestyle = 'solid', markevery = 8)

mt_plot("Inter", "Inter", 's')
mt_plot("Alpha2", "2:1", 'X')
mt_plot("Cut", "Cut", 'p')

fig.subplots_adjust(left=0.13, right=0.96, bottom=0.21, top=0.99)
plt.legend(bbox_to_anchor = (-0.16,-0.05), loc = 'upper left', ncol = 3)

for r in range(1,20):
    ax.add_artist(plt.Circle((0., 0.), 1./3.*np.sqrt(r), fill = False, alpha = 0.1))

plt.savefig("../figures/trajectories.pdf")
plt.show()
