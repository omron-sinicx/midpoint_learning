import gym
import numpy as np
import torch as th
import math

class SpaceEnv(gym.Env):
    " class to wrap space to gym environments for sequential RL"

    def __init__(self, space,
                 step_length: float = 1e-1,
                 reach_threshold : float = 1,
                 max_step: int = 256,
                 pre_layer = None,
                 eval_states = None,
                 stay_penalty:float = -100.,
    ):
        super().__init__()
        self.space = space
        self.action_space = gym.spaces.Box(low = -np.ones(space.action_dim),
                                           high = np.ones(space.action_dim),
                                           shape = (space.action_dim,))
        self.pre_layer = pre_layer
        if pre_layer == None:
            input_dim = 2*space.dim
        else:
            input_dim = pre_layer.output_dim
        self.observation_space = gym.spaces.Box(low = -np.inf,
                                                high = np.inf, shape = (input_dim,))

        self.step_length = step_length
        self.reach_threshold = reach_threshold * step_length
        self.max_step = max_step

        self.EPS = 1e-9
        if eval_states is not None:
            self.eval_states = [(s.numpy(), g.numpy()) for (s,g) in eval_states]
        else:
            self.eval_states = None
        self.eval_state_count = 0

        self.stay_penalty = stay_penalty

    def get_next_state(self, action):
        norm_action = (self.step_length / self.space.F_np(self.current_state, action))*action
        next_state = self.space.add_action(self.current_state, norm_action)
        return next_state

    def get_obs(self):
        if self.pre_layer != None:
            obs = self.pre_layer.s(self.current_state, self.goal_state)
        else:
            obs = np.concatenate((self.current_state, self.goal_state))
        return obs
        
    def step(self, action):
        self.n_step+=1
        
        if np.count_nonzero(action) ==0:
            return self.get_obs(), self.stay_penalty, (self.n_step >= self.max_step), {'goaled': False}

        if self.pre_layer != None:
            action = self.pre_layer.s_inverse(self.current_state, action, self.goal_state)-self.current_state
        next_state=self.get_next_state(action)
        
        self.current_state = self.space.clamp_np(next_state)

        if hasattr(self.space, 'is_invalid_state') and self.space.is_invalid_state(self.current_state):
            reward = -self.space.penalty_term
            done = True
            info={'goaled': False}
            return self.get_obs(), reward, done, info
        
        reward = -self.step_length
        
        next_dist = self.space.calc_reward(self.goal_state, self.current_state)
        reward += self.current_dist - next_dist
        self.current_dist = next_dist

        if self.space.calc_delta_np(self.current_state, self.goal_state) < self.reach_threshold:
            done = True
            info={'goaled': True}
        elif self.n_step >= self.max_step:
            done = True
            info={'goaled': False}
        else:
            done = False
            info = {}

        return self.get_obs(), reward.item(), done, info

    def reset(self, options = None):
        if options is not None and "start" in options and "goal" in options:
            self.current_state = options["start"]
            self.goal_state = options["goal"]
        else:
            while True:
                self.current_state = self.space.get_random_state_np()
                self.goal_state = self.space.get_random_state_np()
                if self.space.calc_delta_np(self.current_state, self.goal_state) >= self.reach_threshold:
                    break
        self.n_step = 0
        self.current_dist = self.space.calc_reward(self.goal_state, self.current_state)
        return self.get_obs()
