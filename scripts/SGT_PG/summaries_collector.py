import os
import tensorflow as tf
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from path_helper import get_base_directory

class SummariesCollector:
    def __init__(self, working_dir, model_name):
        summaries_dir = os.path.join(working_dir, 'tensorboard', model_name)
        #self._train_summary_writer = tf.summary.create_file_writer(os.path.join(summaries_dir, 'train_' + model_name))
        self._train_summary_writer = SummaryWriter(os.path.join(summaries_dir, 'train_' + model_name))
        init_res = self._init_episode_summaries('train', self._train_summary_writer)
        self.write_train_success_summaries = init_res[0]
        self.write_train_optimization_summaries = init_res[1]

        #self._test_summary_writer = tf.summary.create_file_writer(os.path.join(summaries_dir, 'test_' + model_name))
        self._test_summary_writer = SummaryWriter(os.path.join(summaries_dir, 'test_' + model_name))
        init_res = self._init_episode_summaries('test', self._test_summary_writer, working_dir)
        self.write_test_success_summaries = init_res[0]
        self.write_test_optimization_summaries = init_res[1]

    @staticmethod
    def _init_episode_summaries(prefix, summary_writer, log_dir = ""):
        success_rate_var = tf.Variable(0, trainable=False, dtype=tf.float32)
        accumulated_cost_var = tf.Variable(0, trainable=False, dtype=tf.float32)
        curriculum_coefficient_var = tf.Variable(0, trainable=False, dtype=tf.float32)

        if prefix == 'test':
            timestepses = []
            success_rates = []
            costs = []
            levels = []
            success_costs = []
            min_deltas = []

        def write_success_summaries(sess, timesteps, success_rate, current_level, cost, success_cost, min_delta, curriculum_coefficient=None):
            #feed_dict = {
            #    success_rate_var: success_rate,
            #    accumulated_cost_var: cost,
            #}
            #if curriculum_coefficient is not None:
            #    feed_dict[curriculum_coefficient_var] = curriculum_coefficient
            #summary_str = sess.run(summaries, feed_dict=feed_dict)

            #summary_writer.add_summary(summary_str, timesteps)
            summary_writer.add_scalar(prefix+'_cost', cost, timesteps)
            summary_writer.add_scalar(prefix+'_success_rate', success_rate, timesteps)
            summary_writer.add_scalar(prefix+'_success_cost', success_cost, timesteps)
            summary_writer.add_scalar(prefix+'_level', current_level, timesteps)
            summary_writer.add_scalar(prefix+'_min_delta', min_delta, timesteps)

            if prefix == 'test':
                timestepses.append(timesteps)
                success_rates.append(success_rate)
                costs.append(cost)
                levels.append(current_level)
                success_costs.append(success_cost)
                min_deltas.append(min_delta)
                np.savez(log_dir + '/evaluations.npz',
                         timesteps = timestepses,
                         success_rates = success_rates,
                         evaluations = costs,
                         average_costs = success_costs,
                         levels = levels,
                         min_deltas = min_deltas
                )


        def write_optimization_summaries(summaries, global_step):
            #summary_writer.add_summary(summaries, global_step)
            ## for s in summaries:
            ##     if s is not None:
            ##         summary_writer.add_summary(s, global_step)
            #summary_writer.flush()
            pass
            
        return write_success_summaries, write_optimization_summaries




