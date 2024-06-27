import numpy as np
import torch as th
from torch import nn
import math

from stochastic_actor import StochasticActor4Layer
from critic import Critic
import os
from stable_baselines3.common.utils import polyak_update
from torch.distributions import Normal

from torch.utils.tensorboard import SummaryWriter

class TraininingModel(nn.Module):
    def __init__(self, space, batch_size = 256, device = 'cpu', eps = 0.1, learning_rate = 0.0003, log_dir = 'log', eval_freq = 500, total_timesteps = 100000, eval_episodes = [], depth = 6, 
                 gae_lambda: float = 1.0,
                 n_epochs: int = 10,
                 not_midpoint: bool = False,
                 alpha: float = 1.0,
                 penalty_coef: float = 0.,
                 unscaled: bool = False,
                 net_arch: list = [64,64,64],
                 random_actor_train:bool = False,
                 relation_train: bool = False,
                 unproject: bool = False,
                 timestep_depth_schedule: bool = False,
                 unlog_critic: bool = False,
                 load_model: bool = False,
                 cut_deltas: bool = False,
                 cut_deltas_dist: float = 30.0,
    ):
        super().__init__()
        self.space = space
        space.to(device)
        self.eps = eps
        self.actor = StochasticActor4Layer(space.dim, learning_rate = learning_rate, unscaled = unscaled, net_arch = net_arch).to(device)
        self.critic = Critic(space.dim, learning_rate = learning_rate, net_arch=net_arch).to(device)
        if load_model:
            self.actor.load_state_dict(th.load(log_dir+'/actor_model.pt', map_location = device))
            self.critic.load_state_dict(th.load(log_dir+'/critic_model.pt', map_location = device))
        self.batch_size = batch_size
        self.device = device
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        self.eval_episodes = [[ep[0].to(device), ep[1].to(device)] for ep in eval_episodes]
        self.total_timesteps = total_timesteps
        self.depth = depth
        self.gae_lambda = gae_lambda
        self.not_midpoint = not_midpoint
        self.alpha = alpha
        self.penalty_coef = penalty_coef
        self.random_actor_train = random_actor_train
        self.relation_train = relation_train
        self.unproject = unproject
        self.timestep_depth_schedule = timestep_depth_schedule
        self.unlog_critic = unlog_critic
        self.cut_deltas = cut_deltas
        self.cut_deltas_dist = cut_deltas_dist

        self.n_epochs = n_epochs

        self.summary_writer = SummaryWriter(log_dir + "/tensorboard")
        self.best_success_count = 0
        self.best_average_cost = -np.inf


    def get_depth(self):
        if self.timestep_depth_schedule:
            divisor = self.total_timesteps // self.depth + 1
            return min(self.timesteps // divisor, self.depth)
        divisor = self.total_timesteps // (2**(self.depth+1)-1) + 1
        return min(self.num_cycle // divisor, self.depth)
                
    def train_critic_target(self, states_0, states_1, target):
        current_v_values = self.critic(states_0, states_1)
        if not self.unlog_critic:
            target = th.log1p(target)
        critic_loss = nn.functional.mse_loss(current_v_values, target)
        if self.space.symmetric:
            reverse_v_values = self.critic(states_1, states_0)
            critic_loss += nn.functional.mse_loss(reverse_v_values, current_v_values)
            
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        return critic_loss.item()

    def get_critic_value(self, states_0, states_1):
        ret = self.critic(states_0, states_1)
        if not self.unlog_critic:
            ret = th.expm1(ret)
        return ret
    
    def train_actor(self, states_0, states_1):
        middles = self.actor(states_0, states_1)
        # penalty = self.penalty_coef * self.space.penalty(middles).mean()
        if not self.unproject:
            middles = self.space.clamp(middles)
        q_value_0 = self.get_critic_value(states_0, middles)
        q_value_1 = self.get_critic_value(middles, states_1)
        if self.not_midpoint:
            actor_loss = (q_value_0+q_value_1).mean()
            
            if self.space.symmetric:
                actor_loss += self.get_critic_value(middles, self.actor(states_1, states_0)).mean()
        else:
            actor_loss = (q_value_0**2).mean()+self.alpha*(q_value_1**2).mean()
            
            if self.space.symmetric and self.alpha == 1.0:
                sym_critic = self.get_critic_value(middles, self.actor(states_1, states_0))
                actor_loss += (sym_critic**2).mean()

        # actor_loss+=penalty
        if self.relation_train and self.alpha == 1.0 and not self.not_midpoint:
            quad_0 = self.actor(states_0, middles)
            quad_1 = self.actor(middles, states_1)
            actor_loss+= (self.get_critic_value(middles, self.actor(quad_0, quad_1))**2).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        return actor_loss.item()

    def predict_midpoints_batch(self, states_0, states_1, deterministic = False):
        shape = states_0.shape
        middles = self.actor(states_0.reshape(-1,shape[-1]), states_1.reshape(-1,shape[-1]), deterministic = deterministic)
        if not self.unproject:
            middles = self.space.clamp(middles)
        return middles.view(shape)
    
    def predict_dists_batch(self, states_0, states_1):
        shape = states_0.shape
        dists = self.critic(states_0.reshape(-1,shape[-1]), states_1.reshape(-1,shape[-1]))
        return dists.view(shape[:-1])
    
    def calc_deltas_batch(self, states_0, states_1):
        shape = states_0.shape
        deltas = self.space.calc_deltas(states_0.reshape(-1,shape[-1]), states_1.reshape(-1,shape[-1]))
        if self.cut_deltas:
            deltas = th.where(th.lt(deltas, self.eps), deltas, self.cut_deltas_dist)
        return deltas.view(shape[:-1])
    
    def rollout(self):
        depth = self.get_depth()
        sz = max(self.batch_size // 2**depth, 1)
        states = self.space.get_random_states(2*sz).view(sz,2,self.space.dim)
        states_0 = th.tensor([], device = self.device)
        states_1 = th.tensor([], device = self.device)

        for dep in range(depth):
            states_0 = th.cat((states_0, states[:,:-1]),1)
            states_1 = th.cat((states_1, states[:,1:]),1)                
            with th.no_grad():
                middles = self.predict_midpoints_batch(states[:,:-1], states[:,1:], deterministic = False)
            new_states = th.zeros((sz, states.shape[1]+middles.shape[1], self.space.dim), device=self.device)
            new_states[:,::2] = states
            new_states[:,1::2] = middles
            states = new_states
            
        states_0 = th.cat((states_0, states[:,:-1]),1)
        states_1 = th.cat((states_1, states[:,1:]),1)                
        states_0 = th.cat((states_0, states),1)
        states_1 = th.cat((states_1, states),1)                
        self.num_cycle+=sz

        return states_0, states_1

    def calc_TD(self, states_0, states_1):
        advantages = th.zeros(states_0.shape[:-1], device = self.device)
        depth = round(math.log2(states_0.shape[1]//3))
        S = (2**depth-1)
        S2 = (2**(depth+1)-1)
        advantages[:,S:S2] = self.calc_deltas_batch(states_0[:,S:S2], states_1[:,S:S2])
        self.timesteps+=states_0.shape[0]*(S2-S)
        for t in range(depth):
            d = depth - t - 1
            S = 2**d-1
            s=2**(d+1)-1
            t=s+2**(d+1)
            with th.no_grad():
                # advantages[:,S:S+2**(d)]= (1-self.gae_lambda)*(self.predict_dists_batch(states_0[:,s:t-1:2], states_1[:,s:t-1:2])\
                #                                              + self.predict_dists_batch(states_0[:,s+1:t:2], states_1[:,s+1:t:2]))\
                #                                              + self.gae_lambda * (advantages[:,s:t-1:2]+advantages[:,s+1:t:2])
                advantages[:,S:S+2**(d)] = advantages[:,s:t-1:2]+advantages[:,s+1:t:2]
                '''
                if hasattr(self.space, 'is_invalid_states'):
                    invalid = self.space.is_invalid_states(states_1[:,s:t-1:2].view(-1,self.space.dim)).view(-1,2**d,self.space.dim)
                    advantages[:,S:S+2**(d)]+=th.where(invalid, self.space.penalty_term, 0)
                '''

        return advantages
     
    def learn(self):
        self.critic_losses = []
        self.actor_losses = []
        self.evaluations = []
        self.min_deltas = []
        self.success_rates = []
        self.average_costs = []
        self.num_cycle=0
        self.timesteps=0
        self.loss_timesteps = []
        self.eval_timesteps = []
        self.best_evaluation = np.inf
        iteration = 0
        while self.timesteps < self.total_timesteps:
            states_0, states_1 = self.rollout()
            target_v_values = self.calc_TD(states_0, states_1)
            states_0 = states_0.view(-1,self.space.dim)
            states_1 = states_1.view(-1,self.space.dim)
            target_v_values = target_v_values.view(-1)
            num_data = states_0.shape[0]
            for _ in range(self.n_epochs):
                for batch in range((num_data-1) //self.batch_size+1):
                    s = batch*self.batch_size
                    t = (batch+1)*self.batch_size
                    critic_loss = self.train_critic_target(states_0[s:t], states_1[s:t], target_v_values[s:t])
                    if self.random_actor_train:
                        random_states = self.space.get_random_states(2*self.batch_size)
                        actor_loss = self.train_actor(random_states[:self.batch_size], random_states[self.batch_size:])
                    else:
                        actor_loss = self.train_actor(states_0[s:t], states_1[s:t])
            self.critic_losses.append(critic_loss)
            self.actor_losses.append(actor_loss)
            self.loss_timesteps.append(self.timesteps)
            iteration+=1
            if iteration % 100 == 0:
                self.summary_writer.add_scalar('critic_loss', critic_loss, self.timesteps)
                self.summary_writer.add_scalar('actor_loss', actor_loss, self.timesteps)
            if iteration % self.eval_freq == 0:
                evaluation, min_delta, success_count, average_cost = self.evaluate()
                self.summary_writer.add_scalar('evaluation', evaluation, self.timesteps)
                self.summary_writer.add_scalar('min_delta', min_delta, self.timesteps)
                self.summary_writer.add_scalar('success_rate', success_count / len(self.eval_episodes), self.timesteps)
                self.summary_writer.add_scalar('average_cost', average_cost, self.timesteps)
                self.evaluations.append(evaluation)
                self.min_deltas.append(min_delta)
                self.success_rates.append(success_count / len(self.eval_episodes))
                self.average_costs.append(average_cost)
                self.eval_timesteps.append(self.timesteps)
                if success_count > self.best_success_count or (success_count == self.best_success_count and average_cost < self.best_average_cost):
                    self.best_success_count = success_count
                    self.best_average_cost = average_cost
                self.save_model()
                self.save_evals()

    def save_model(self):
        th.save(self.actor.state_dict(), self.log_dir + '/actor_model.pt')
        th.save(self.critic.state_dict(), self.log_dir + '/critic_model.pt')

    def save_evals(self):
        np.savez(self.log_dir + '/train_losses.npz',
                 timesteps = self.loss_timesteps,
                 critic_losses = self.critic_losses,
                 actor_losses = self.actor_losses)
        np.savez(self.log_dir + '/evaluations.npz',
                 timesteps = self.eval_timesteps,
                 evaluations = self.evaluations,
                 success_rates = self.success_rates,
                 min_deltas = self.min_deltas,
                 average_costs = self.average_costs)

    def evaluate(self):
        length_sum = 0.
        min_delta_sum = 0.
        success_count = 0
        success_sum = 0.
        for state_0, state_1 in self.eval_episodes:
            states = th.stack((state_0, state_1),0)
            for _ in range(self.depth):
                with th.no_grad():
                    middles = self.actor(states[:-1], states[1:], deterministic = True)
                if not self.unproject:
                    middles = self.space.clamp(middles)
                new_states = th.zeros((states.shape[0]+middles.shape[0],self.space.dim), device=self.device)
                new_states[::2] = states
                new_states[1::2] = middles
                states = new_states
            deltas = self.space.calc_deltas(states[:-1], states[1:])
            #if deltas.max() < self.eps and not (hasattr(self.space, 'is_invalid_states') and self.space.is_invalid_states(states).max()):
            if deltas.max() < self.eps:
                success_count+=1
                success_sum+=deltas.sum().item()
            length_sum+=deltas.sum().item()
            min_delta_sum+=deltas.max().item()
        N = len(self.eval_episodes)
        if success_count == 0.:
            success_sum=np.inf
        else:
            success_sum /= success_count
        return length_sum/N, min_delta_sum/N, success_count, success_sum

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", action="store", default="tmp")
parser.add_argument("--space", action="store", default="CarLike2")
parser.add_argument("--device", action="store", default="cuda:0")
parser.add_argument("--batch_size", action="store", default=256, type = int)
parser.add_argument("--eval_freq", action="store", default=2000, type = int)
parser.add_argument("--total_timesteps", action="store", default=10000000, type = int)
parser.add_argument("--depth", action = "store", default = 6, type = int)
parser.add_argument("--eps", action = "store", default = 0.1, type = float)
parser.add_argument("--gae_lambda", action = "store", default = 1.0, type = float)
parser.add_argument("--n_epochs", action = "store", default = 10, type = int)
parser.add_argument("--seed", action="store", default=0, type = int)
parser.add_argument("--learning_rate", action="store", default=1e-6, type = float)
parser.add_argument("--not_midpoint", action="store_true", default=False)
parser.add_argument("--alpha", action="store", default=1.0, type = float)
parser.add_argument("--penalty_coef", action="store", default=0., type = float)
parser.add_argument("--unscaled", action="store_true", default=True)
parser.add_argument("--net_arch", action="store", default=[64,64], type = int, nargs = '*')
parser.add_argument("--random_actor_train", action="store_true", default=False)
parser.add_argument("--relation_train", action="store_true", default=False)
parser.add_argument("--unproject", action="store_true", default=False)
parser.add_argument("--timestep_depth_schedule", action="store_true", default=False)
parser.add_argument("--unlog_critic", action="store_true", default=False)
parser.add_argument("--load_model", action="store_true", default=False)
parser.add_argument("--cut_deltas", action="store_true", default=False)
parser.add_argument("--cut_deltas_dist", action="store", default=30.0, type = float)

args = parser.parse_args()

import random
import numpy as np
random.seed(args.seed)
th.manual_seed(args.seed)
np.random.seed(args.seed)

os.makedirs(args.log_dir, exist_ok = True)
sys.stdout = open(args.log_dir+"/log.txt", "w")
sys.stderr = open(args.log_dir+"/error_log.txt", "w")

with open(args.log_dir+"/args.plk", 'wb') as f:
    pickle.dump(vars(args), f)
    
from pick_space import pick_space
space, eval_episodes = pick_space(args.space)

model = TraininingModel(space = space,
                        eval_episodes=eval_episodes,
                        log_dir = args.log_dir,
                        device = args.device,
                        batch_size = args.batch_size,
                        eval_freq = args.eval_freq,
                        total_timesteps = args.total_timesteps,
                        depth = args.depth,
                        eps = args.eps,
                        gae_lambda = args.gae_lambda,
                        n_epochs = args.n_epochs,
                        learning_rate = args.learning_rate,
                        not_midpoint = args.not_midpoint,
                        alpha = args.alpha,
                        penalty_coef = args.penalty_coef,
                        unscaled = args.unscaled,
                        net_arch = args.net_arch,
                        random_actor_train = args.random_actor_train,
                        relation_train = args.relation_train,
                        unproject = args.unproject,
                        timestep_depth_schedule = args.timestep_depth_schedule,
                        unlog_critic = args.unlog_critic,
                        load_model = args.load_model,
                        cut_deltas = args.cut_deltas,
                        cut_deltas_dist = args.cut_deltas_dist,
)
model.learn()
