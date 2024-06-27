import matplotlib.pyplot as plt
import numpy as np
import torch as th
from stochastic_actor import StochasticActor4Layer
from pick_space import pick_space
from stable_baselines3 import PPO
from gym_env_wrapper import SpaceEnv

prob = "Matsumoto_-1"

def get_dire(prob, label):
    return "../exp/"


probs = [("Matsumoto_-1", 6, 10, [64, 64], 0.1),
         ("CarLikeDisk3-0.2", 6, 5, [400, 300, 300], 0.2),
         ("Obstacle4Outer", 6, 5, [400, 300, 300], 0.1),
         ("Panda5", 6, 5, [400, 300, 300], 0.2),
         ("MultiAgent-3-0.5", 6, 5, [400, 300, 300], 0.2)
]

algos = [("ACDQT-","Our-T"),
         ("ACDQC-","Our-C"),
         ("Seq-", "Seq"),
         ("Inter-","Inter"),
         ("Alpha2-","2:1"),
         ("Cut-","Cut"),
]         


for prob, depth, N, net_arch, eps in probs:
    space, eval_episodes = pick_space(prob)
    space.to('cpu')
    ress = []
    for seed in range(11, 11+N):
        res = []
        for name, label in algos:
            re = []
            log_dir = get_dire(prob, label)+prob+"/"+name+str(seed)
            if label == "Seq":
                model = PPO.load(log_dir+"/final_model.zip")
                eval_env = SpaceEnv(space, pre_layer = space.pre_layer, step_length = eps, max_step = 2**depth)
                for start, goal in eval_episodes:
                    done = False

                    obs = eval_env.reset(options={"start": start.numpy(), "goal": goal.numpy()})

                    dist = 0
                    traj = [start.numpy()]
                    while not done:
                        action, _states = model.predict(obs, deterministic = True)
                        obs, reward, done, info = eval_env.step(action)
                        traj.append(obs[:space.dim])
                        dist+=space.F_np(traj[-2], traj[-1]-traj[-2])

                    traj.append(goal.numpy())
                    dist+=space.F_np(traj[-2], traj[-1]-traj[-2])
                    re.append([info['goaled'], dist])
            else:
                dim = space.dim
                actor = StochasticActor4Layer(dim, net_arch = net_arch)
                actor.load_state_dict(th.load(log_dir+'/actor_model.pt'))
                for start, goal in eval_episodes:
                    states = th.zeros((2,dim))
                    states[0] = start
                    states[1] = goal
                    for _ in range(depth):
                        with th.no_grad():
                            middles = actor(states[:-1],states[1:], deterministic = True)
                        middles = space.clamp(middles)
                        new_states = th.zeros((states.shape[0]+middles.shape[0], dim))
                        new_states[::2] = states
                        new_states[1::2] = middles
                        states = new_states
                    deltas=space.calc_deltas(states[:-1], states[1:])
                    re.append([deltas.max().item() < eps, deltas.sum().item()])
            res.append(re)
        ress.append(res)
    ress = np.array(ress)
    np.save("../data/"+prob+"_compare.npy", ress)
    print(prob)
    for i in range(len(algos)):
        for j in range(i, len(algos)):
            win_c = 0
            all_c = 0
            for seed in range(N):
                for k in range(len(eval_episodes)):
                    if ress[seed][i][k][0] and ress[seed][j][k][0]:
                        all_c+=1
                        if ress[seed][i][k][1] < ress[seed][j][k][1]:
                            win_c+=1
            print(algos[i][1], algos[j][1], ":", win_c,'/',all_c, '=', win_c/all_c if all_c>0 else 'Nan')
