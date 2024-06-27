import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gym

from stable_baselines3 import PPO
from pick_space import pick_space
from gym_env_wrapper import SpaceEnv

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", action="store", default="tmp")
parser.add_argument("--space", action="store", default="CarLike2")
parser.add_argument("--seed", action="store", default=0, type = int)
parser.add_argument("--eval_freq", action="store", default=250000, type = int)
parser.add_argument("--total_timesteps", action="store", default=10000000, type = int)
parser.add_argument("--device", action="store", default='cuda:0', type = str)
parser.add_argument("--eps", action="store", default=0.1, type = float)
parser.add_argument("--ent_coef", action="store", default=0.0, type = float)
parser.add_argument("--max_step", action="store", default=64, type = int)
parser.add_argument("--learning_rate", action="store", default=0.003, type = float)
parser.add_argument("--batch_size", action="store", default=128, type = int)
parser.add_argument("--net_arch", action="store", default=[64,64], type = int, nargs = '*')

args = parser.parse_args()

import random
import torch as th
import numpy as np
from ppo_call_back import PPOCallback
random.seed(args.seed)
th.manual_seed(args.seed)
np.random.seed(args.seed)

os.makedirs(args.log_dir, exist_ok = True)
sys.stdout = open(args.log_dir+"/log.txt", "w")
sys.stderr = open(args.log_dir+"/error_log.txt", "w")

space, eval_states = pick_space(args.space)
env = SpaceEnv(space, pre_layer = space.pre_layer, step_length = args.eps, max_step = args.max_step)

eval_callback = PPOCallback(env, eval_episodes = eval_states,
                            log_path=args.log_dir, eval_freq=args.eval_freq, verbose = 0)

#kwargs = {"device": args.device, "gamma": 1.0, "ent_coef": args.ent_coef, "learning_rate": args.learning_rate, "batch_size": args.batch_size, "policy_kwargs":{"net_arch":args.net_arch}}
#kwargs = {"device": args.device, "gamma": 1.0, "ent_coef": args.ent_coef, "learning_rate": args.learning_rate, "batch_size": args.batch_size}
kwargs = {"device": args.device, "gamma": 1.0, "ent_coef": args.ent_coef, "learning_rate": args.learning_rate, "batch_size": args.batch_size, "policy_kwargs":{"net_arch":[dict(pi=args.net_arch, vf=args.net_arch)]}}
#kwargs = {"device": args.device, "gamma": 1.0}

if os.path.isfile(args.log_dir+"/best_model.zip"):
    model = PPO.load(args.log_dir+"/best_model.zip", env, verbose = 0, **kwargs)
else:
    model = PPO("MlpPolicy", env, verbose = 0, **kwargs)
model.learn(total_timesteps = args.total_timesteps, callback = eval_callback)

model.save(args.log_dir+"/final_model")
