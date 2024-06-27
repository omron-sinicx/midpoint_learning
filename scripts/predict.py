import numpy as np
import torch as th
#from actor import Actor
from stochastic_actor import StochasticActor4Layer
#from timed_stochastic_actor import TimedStochasticActor
#from timed_actor import TimedActor
from critic import Critic
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
parser.add_argument("--log_dir", action="store", default="tmp")
parser.add_argument("--space", action="store", default="CarLike2")
parser.add_argument("--stochastic", action="store_true", default=True)
parser.add_argument("--multiactors", action="store_true", default=False)
parser.add_argument("--depth", action="store", default=6, type = int)
parser.add_argument("--optimize_steps", action="store", default=1000, type = int)
parser.add_argument("--unscaled", action="store_true", default=False)
parser.add_argument("--net_arch", action="store", default=[64,64], type = int, nargs = '*')
parser.add_argument("--eps", action="store", default = 0.1, type = float)
parser.add_argument("--truth", action="store_true", default=False)
parser.add_argument("--episode_num", action="store", default=0, type = int)

args = parser.parse_args()

log_dir = args.log_dir
space_name = args.space

episode_num = args.episode_num

from pick_space import pick_space
space, eval_episodes = pick_space(space_name)
space = space.to('cpu')
dim = space.dim

if args.multiactors:
    actors = th.load(log_dir+'/actor_model.pt', map_location = 'cpu')
    critics = th.load(log_dir+'/critic_model.pt', map_location = 'cpu')
elif args.stochastic:
    actor = StochasticActor4Layer(dim, unscaled = args.unscaled, net_arch = args.net_arch)
    
    # actor = TimedStochasticActor(dim)
    actor.load_state_dict(th.load(log_dir+'/actor_model.pt', map_location = 'cpu'))
    critic = Critic(dim, net_arch = args.net_arch)
    critic.load_state_dict(th.load(log_dir+'/critic_model.pt', map_location = 'cpu'))
else:
    actor = Actor(dim)
    # actor = TimedActor(dim, scaling = True)
    actor.load_state_dict(th.load(log_dir+'/actor_model.pt', map_location = 'cpu'))

count=0

#

if args.episode_num == -1:
    if space_name[:12] == "CarLikeDisk3":
        eval_episodes = [(th.tensor([-0.4,0.,0.,-1.]), th.tensor([0.4,0.,0.,-1.]))]
    elif space_name[:11] == "CarLikeDisk":
        eval_episodes = [(th.tensor([-0.4,0.,-th.pi/2]), th.tensor([0.4,0.,-th.pi/2]))]
elif args.episode_num >= 0:
    eval_episodes = eval_episodes[episode_num:episode_num + 1]
#    eval_episodes = [(th.tensor([-0.2,0.,-th.pi/2]), th.tensor([0.2,0.,th.pi/2]))]
#elif space_name == "Matsumoto_-1":
#    eval_episodes = [(th.tensor([-0.51,-0.5]), th.tensor([0.5,0.5]))]
#

depth = args.depth


for start, goal in eval_episodes:
    states = th.zeros((2,dim))
    states[0] = start
    states[1] = goal
    for rev_dep in range(depth):
        #if space.calc_deltas(states[:-1], states[1:]).max() < 1e-1:
        #    break
        with th.no_grad():
            if args.multiactors:
                middles = actors[depth-rev_dep-1](states[:-1],states[1:], deterministic = True)
                '''
                if rev_dep != 0:
                    dist = critics[depth-rev_dep-1](states[:-1],states[1:])
                    print(dist)
                '''
            else:
                middles = actor(states[:-1],states[1:], deterministic = True)
            middles = space.clamp(middles)
            # print(middles)
            # dist = critic(states[:-1],states[1:])
            # print(dist)
            new_states = th.zeros((states.shape[0]+middles.shape[0], dim))
            new_states[::2] = states
            new_states[1::2] = middles
            states = new_states
        #print(deltas.sum())
    # states = space.clamp(states)
    deltas=space.calc_deltas(states[:-1], states[1:])
    print(start, goal, deltas.max(), deltas.sum())
    #print(critic(states[:-1], states[1:]))
    #print(deltas)
    if deltas.max() < args.eps:
        count+=1
print(count)

if args.truth:
    true_traj = calc_true_trajectory(space, states[0], states[-1], N=2**depth-1, timesteps=args.optimize_steps)
    true_dist=(space.calc_deltas(true_traj[:-1],true_traj[1:])).sum().item()
    #print(space.calc_deltas(true_traj[:-1],true_traj[1:]))
    print("True dist:", true_dist)
    true_traj = true_traj.numpy()

states = states.numpy()

#print(states)

#print(true_traj)
#for i in range(7):
    #traj = true_traj[::1<<(6-i)]
    #print(space.pre_layer(traj[:-1], traj[1:]))
    #print(critic(traj[:-1],traj[1:]))
    #print(th.abs((traj[:-1]-traj[1:])[:,2])*0.5-(traj[:-1]-traj[1:])[:,:2].norm(dim=1))


import matplotlib.pyplot as plt


fig = plt.figure(figsize = (10, 10))

if space_name == "Plane" or space_name[:9] == "Matsumoto":
    ax = fig.add_subplot(2,2,1)
    ax.plot(states[:,0], states[:,1], marker = 'o')
    ax.plot(true_traj[:,0], true_traj[:,1], marker = 'o', color = 'r')
    ax.set_xlim(-1., 1.)
    ax.set_ylim(-1., 1.)
    ax.set_aspect(1.)

elif space_name[:8] == "Obstacle":
    ax = fig.add_subplot(1,1,1)
    ax.plot(states[:,0], states[:,1], marker = 'o')
    ax.plot(true_traj[:,0], true_traj[:,1], marker = 'o', color = 'r')
    #ax.set_xlim(0., 1.)
    #ax.set_ylim(0., 1.)
    ax.set_aspect(1.)

elif space_name == "Poincare":
    ax = fig.add_subplot(2,2,1)
    ax.plot(true_traj[:,0], true_traj[:,1], marker = 'o', color = 'r')
    ax.plot(states[:,0], states[:,1], marker = 'o')
    ax.set_xlim(-1., 1.)
    ax.set_ylim(-1., 1.)
    ax.set_aspect(1.)

elif space_name[:7] == "CarLike" and False:
    ax = fig.add_subplot(1,1,1,projection='3d')
    ax.set_xlim(-1., 1.)
    ax.set_ylim(-1., 1.)
    ax.set_zlim(-th.pi, th.pi)
    ax.plot(states[:,0], states[:,1], states[:,2], marker = 'o')
    ax.plot(true_traj[:,0], true_traj[:,1], true_traj[:,2], marker = 'o', color = 'r')

elif space_name[:12] == "CarLikeDisk3":
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim(-1., 1.)
    ax.set_ylim(-1., 1.)
    ax.plot(states[:,0], states[:,1], marker = 'o')
    d = 0.1
    for state in states:
        ax.arrow(state[0], state[1], d * state[2], d * state[3], color = 'C0')
    #ax.plot(true_traj[:,0], true_traj[:,1], marker = 'o', color = 'r')
    #for state in true_traj:
    #    ax.arrow(state[0], state[1], d * state[2], d * state[3], color = 'r')

elif space_name[:7] == "CarLike":
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim(-1., 1.)
    ax.set_ylim(-1., 1.)
    ax.plot(states[:,0], states[:,1], marker = 'o')
    d = 0.1
    for state in states:
        ax.arrow(state[0], state[1], d * np.cos(state[2]), d * np.sin(state[2]), color = 'C0')
    ax.plot(true_traj[:,0], true_traj[:,1], marker = 'o', color = 'r')
    for state in true_traj:
        ax.arrow(state[0], state[1], d * np.cos(state[2]), d * np.sin(state[2]), color = 'r')
    
elif space_name[:10] == "MultiAgent":
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim(-1., 1.)
    ax.set_ylim(-1., 1.)
    for i in range(space.N):
        ax.plot(states[:,2*i], states[:,2*i+1], marker = 'o')
    
elif space_name == "Sphere":
    ax = fig.add_subplot(2,2,1,projection='3d')
    mapped = space.mapping(th.tensor(states))
    mapped = mapped.numpy()

    ax.plot(mapped[:,0], mapped[:,1], mapped[:,2], marker = 'o')

    start = mapped[0]
    goal = mapped[-1]
    N = 100
    middles = [(1-i/N)*start+i/N*goal for i in range(N+1)]
    middles = np.array([p/np.linalg.norm(p) for p in middles])
    ax.plot(middles[:,0], middles[:,1], middles[:,2], color = 'r')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
elif space_name == "Sphere3":
    ax = fig.add_subplot(2,2,1,projection='3d')

    ax.plot(states[:,0], states[:,1], states[:,2], marker = 'o')
    ax.plot(true_traj[:,0], true_traj[:,1], true_traj[:,2], marker = 'o', color = 'r')
    
    r=math.sqrt(1/3)
    ax.set_xlim(-r,r)
    ax.set_ylim(-r,r)
    ax.set_zlim(-r,r)
elif False and space_name == 'PolarGravity':
    ax = fig.add_subplot(2,2,1,projection='3d')
    lower, upper = space.bound
    ax.set_xlim(-upper[0], upper[0])
    ax.set_ylim(-upper[0], upper[0])
    ax.set_zlim(lower[2], upper[2])
    ax.plot(states[:,0]*np.cos(states[:,1]), states[:,0]*np.sin(states[:,1]), states[:,2], marker = 'o')
    ax.plot(true_traj[:,0]*np.cos(true_traj[:,1]), true_traj[:,0]*np.sin(true_traj[:,1]), true_traj[:,2], marker = 'o', color = 'r')
elif space.dim == 3:
    ax = fig.add_subplot(2,2,1,projection='3d')
    lower, upper = space.bound
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.plot(states[:,0], states[:,1], states[:,2], marker = 'o')
    ax.plot(true_traj[:,0], true_traj[:,1], true_traj[:,2], marker = 'o', color = 'r')
plt.show()
'''
ax = fig.add_subplot(2,2,2)

evaluations = np.load(log_dir+"/evaluations.npz")

ax.plot(evaluations["timesteps"],evaluations["success_rates"])
#ax.set_yscale('log')
ax = fig.add_subplot(2,2,3)

losses = np.load(log_dir+"/train_losses.npz")

ave_num = 1

ax.plot(np.convolve(losses['critic_losses'], np.ones(ave_num)/ave_num, mode='valid'))
ax.set_yscale('log')

ax = fig.add_subplot(2,2,4)

ax.plot(np.convolve(losses['actor_losses'], np.ones(ave_num)/ave_num, mode='valid'))

#plt.savefig(log_dir+"/result.png")
'''
