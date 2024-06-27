import gym
import numpy as np
import torch as th
import math

from SGT_PG.abstract_motion_planning_game_subgoal import AbstractMotionPlanningGameSubgoal

class SpaceGame(AbstractMotionPlanningGameSubgoal):
    " class to wrap game_subgoal to gym environments for SGT-PG"

    def __init__(self, space, eval_states):
        self.space = space
        self.eval_states = [(s.numpy(), g.numpy()) for (s,g) in eval_states]

    def get_fixed_start_goal_pairs(self, challenging=False):
        return self.eval_states

    def get_start_goals(self, number_of_pairs, curriculum_coefficient, get_free_states):
        pairs = []
        for _ in range(number_of_pairs):
            pairs.append((self.space.get_random_state_np(), self.space.get_random_state_np()))
        return pairs

    def test_predictions(self, predictions):
        results = {}
        for path_id in predictions:
            results[path_id] = {}
            for i, start, goal in predictions[path_id]:
                is_start_free = self.is_free_state(start)
                is_goal_free = self.is_free_state(goal)
                collision_length = 0.
                free_length = self.space.calc_delta_np(self.space.clamp_np(start),
                                                       self.space.clamp_np(goal))
                results[path_id][i] = (start, goal, is_start_free, is_goal_free, free_length, collision_length)
        return results
            

    def is_free_state(self, state):
        return True

    def get_state_size(self):
        return self.space.dim
    
