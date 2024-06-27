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

space_name = "CarLikeDisk3-0.2"

from pick_space import pick_space
space, eval_episodes = pick_space(space_name)
space = space.to('cpu')
dim = space.dim

eval_episodes = [(th.tensor([-0.4,0.,0.,-1.]), th.tensor([0.4,0.,0.,-1.]))]

start = eval_episodes[0][0]
goal = eval_episodes[0][1]

depth = 6

def generate_path(log_dir):
    actor = StochasticActor4Layer(dim, unscaled = False, net_arch = [400,300,300])
    
    actor.load_state_dict(th.load(log_dir+'/actor_model.pt', map_location = 'cpu'))
    
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
    return states.numpy()

timesteps = 100000
# timesteps = 1000
lr = 1e-3
N = 63

traj = np.zeros((N, 3))
for i in range(N):
    traj[i,:2] = ((N-i)*start[:2].numpy()+(i+1)*goal[2:].numpy())/(N+1)
traj[i,2] = -np.pi/2
traj = th.tensor(traj, dtype = th.float32, requires_grad=True)
optimizer = th.optim.Adam([traj], lr = lr)
for _ in range(timesteps):
    states = th.cat((traj[:, :2],th.cos(traj[:,2:]), th.sin(traj[:,2:])),1) 
    length = (space.calc_deltas(th.cat((start.unsqueeze(0), states)), th.cat((states, goal.unsqueeze(0))))**2).sum()
    optimizer.zero_grad()
    length.backward()
    optimizer.step()

states = th.cat((traj[:, :2],th.cos(traj[:,2:]), th.sin(traj[:,2:])),1) 
true_traj = th.cat((start.unsqueeze(0), states.detach(), goal.unsqueeze(0)))

fig = plt.figure(figsize = (7, 8.3))

ax = fig.add_subplot(1,1,1)
ax.set_xlim(-.5, .5)
ax.set_ylim(-.5, .5)
ax.set_aspect(1.)
d = 0.1

ax.plot(true_traj[:,0], true_traj[:,1], marker = '*', color = 'black', markevery = 8, label = 'Truth', ms = 12)
for state in true_traj[::8]:
    ax.arrow(state[0], state[1], d * state[2], d * state[3], color = 'black', head_width = 0.02, head_length = 0.01)

    states = generate_path("../exp/CarLikeDisk3-0.2/ACDQT-11/")
ax.plot(states[:,0], states[:,1], marker = 'o', markevery = 8, color = 'C0', label = "Our-T", ms = 12)
for state in states[::8]:
    ax.arrow(state[0], state[1], d * state[2], d * state[3], color = 'C0', head_width = 0.02, head_length = 0.01)

states = generate_path("../exp/CarLikeDisk3-0.2/ACDQC-11/")
ax.plot(states[:,0], states[:,1], marker = 'D', markevery = 8, color = 'C1', label = "Our-C", ms = 12)
for state in states[::8]:
    ax.arrow(state[0], state[1], d * state[2], d * state[3], color = 'C1', head_width = 0.02, head_length = 0.01)

from stable_baselines3 import PPO
from gym_env_wrapper import SpaceEnv

seed = 11

log_dir = "../exp/CarLikeDisk3-0.2/"

model = PPO.load(log_dir+"Seq-"+str(seed)+"/final_model.zip")
eval_env = SpaceEnv(space, step_length = 0.2, max_step = 64)

done = False

obs = eval_env.reset(options={"start": start.numpy(), "goal": goal.numpy()})

traj = [start.numpy()]
dist = 0
while not done:
    action, _states = model.predict(obs, deterministic = True)
    obs, reward, done, info = eval_env.step(action)
    traj.append(obs[:space.dim])
    dist+=space.calc_delta_np(traj[-2], traj[-1])

if info['goaled']:
    traj.append(states[1].numpy())
states = np.array(traj)
ax.plot(states[:,0], states[:,1], marker = 'P', label = "Seq", color = 'C2', markevery = 8, ms = 12)
for state in states[::8]:
    ax.arrow(state[0], state[1], d * state[2], d * state[3], color = 'C2', head_width = 0.02, head_length = 0.01)

states = generate_path("../exp/CarLikeDisk3-0.2/Inter-11/")
ax.plot(states[:,0], states[:,1], marker = 's', markevery = 8, color = 'C4', label = "Inter", ms = 12)
for state in states[::8]:
    ax.arrow(state[0], state[1], d * state[2], d * state[3], color = 'C4', head_width = 0.02, head_length = 0.01)

states = generate_path("../exp/CarLikeDisk3-0.2/Alpha2-11/")
ax.plot(states[:,0], states[:,1], marker = 'X', markevery = 8, color = 'C5', label = "2:1", ms = 12)
for state in states[::8]:
    ax.arrow(state[0], state[1], d * state[2], d * state[3], color = 'C5', head_width = 0.02, head_length = 0.01)

states = generate_path("../exp/CarLikeDisk3-0.2/Cut-11/")
ax.plot(states[:,0], states[:,1], marker = 'p', markevery = 8, color = 'C6', label = "Cut", ms = 12)
for state in states[::8]:
    ax.arrow(state[0], state[1], d * state[2], d * state[3], color = 'C6', head_width = 0.02, head_length = 0.01)

fig.subplots_adjust(left=0.13, right=0.96, bottom=0.21, top=0.99)
plt.legend(bbox_to_anchor = (-0.16,-0.05), loc = 'upper left', ncol = 3)

plt.savefig("../figures/trajectories_car3.pdf")
plt.show()
