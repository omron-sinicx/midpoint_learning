import torch as th
import numpy as np
import os
import gym

from stable_baselines3.common.callbacks import EventCallback
from torch.utils.tensorboard import SummaryWriter

from typing import Any, Callable, Dict, List, Optional, Union
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

class PPOCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env,
        eval_episodes,
        eval_freq: int = 5000,
        log_path: Optional[str] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)

        self.eval_episodes = [(s.numpy(), g.numpy()) for (s,g) in eval_episodes]
        self.eval_freq = eval_freq
        self.best_success_count = 0
        self.best_average_cost = -np.inf
        
        self.eval_env = eval_env
        self.log_path = log_path
        self.summary_writer = SummaryWriter(log_path+"/tensorboard")

        

    def _init_callback(self) -> None:
        
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.evaluations, self.average_costs, self.success_rates, self.eval_timesteps = [], [], [], []

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            
            evaluation = 0.
            average_cost = 0.
            success_count = 0

            for start, goal in self.eval_episodes:
                obs = self.eval_env.reset(options= {"start": start, "goal": goal})
                cost = self.eval_env.space.F_np(goal, goal-start)
                done = False
                while not done:
                    action, _states = self.model.predict(obs, deterministic = True)
                    obs, reward, done, info = self.eval_env.step(action)
                    cost -= reward
                if info['goaled']:
                    success_count+=1.
                    average_cost+=cost
                evaluation+=cost
            if success_count == 0.:
                average_cost = np.inf
            else:
                average_cost/= success_count
            success_rate = success_count / len(self.eval_episodes)
            evaluation /= len(self.eval_episodes)

            timesteps = self.model.num_timesteps
            self.summary_writer.add_scalar('evaluation', evaluation, timesteps)
            self.summary_writer.add_scalar('success_rate', success_rate, timesteps)
            self.summary_writer.add_scalar('average_cost', average_cost, timesteps)

            self.evaluations.append(evaluation)
            self.success_rates.append(success_rate)
            self.average_costs.append(average_cost)
            self.eval_timesteps.append(timesteps)
            np.savez(self.log_path + '/evaluations.npz',
                     timesteps = self.eval_timesteps,
                     evaluations = self.evaluations,
                     success_rates = self.success_rates,
                     average_costs = self.average_costs)
            

            if success_count > self.best_success_count or (success_count == self.best_success_count and average_cost < self.best_average_cost):
                if self.verbose >= 1:
                    print("New best mean reward!")
                self.model.save(os.path.join(self.log_path, "best_model"))
                self.best_success_count = success_count
                self.best_average_cost = average_cost

        return continue_training
