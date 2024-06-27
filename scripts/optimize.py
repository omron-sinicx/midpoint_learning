import torch as th
import numpy as np
import math

def calc_true_trajectory(space, start, goal, N =63, timesteps = 1000, lr = 1e-3):
    traj = np.zeros((N, space.dim))
    for i in range(N):
        traj[i] = ((N-i)*start.numpy()+(i+1)*goal.numpy())/(N+1)
    traj = th.tensor(traj, dtype = th.float32, requires_grad=True)
    optimizer = th.optim.Adam([traj], lr = lr)
    for _ in range(timesteps):
        length = (space.calc_deltas(th.cat((start.unsqueeze(0), traj)), th.cat((traj,goal.unsqueeze(0))))**2).sum()
        optimizer.zero_grad()
        length.backward()
        optimizer.step()
    return th.cat((start.unsqueeze(0), traj.detach(), goal.unsqueeze(0)))

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import sys
    # space_name = sys.argv[1]
    space_name = 'CarLikeDisk'

    from pick_space import pick_space
    space, eval_episodes = pick_space(space_name)
    dim = space.dim
    for start, goal in eval_episodes:
        states = th.zeros((2,dim))
        states[0] = start
        states[1] = goal
    
        true_traj = calc_true_trajectory(space, states[0], states[-1], timesteps = 1000, N=256)
        true_dist = space.calc_deltas(true_traj[:-1], true_traj[1:])
        print(true_dist.sum().item(), true_dist.max().item())
