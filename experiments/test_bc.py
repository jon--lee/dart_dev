"""
    Experiment script intended to test Behavior Cloning
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import gym
import numpy as np
from tools.expert import load_policy
from tools import statistics, utils
import argparse
import scipy.stats
import time as timer
import framework
import IPython
from tools.utils import log


def main():
    title = 'test_bc'

    ap = framework.get_args()
    args = vars(ap.parse_args())
    args = framework.load_config(args)

    framework.startup(title, args, Test)


class Test(framework.Test):


    def run_iters(self):
        T = self.params['t']

        results = {
            'rewards': [],
            'sup_rewards': [],
            'surr_losses': [],
            'sup_losses': [],
            'sim_errs': [],
            'data_used': [],
        }

        start_time = timer.time()
        data_states = []
        data_actions = []

        iteration = 0
        while len(data_states) < self.params['max_data']:
            log("\tIteration: " + str(iteration))
            log("\tData states: " + str(len(data_states)))
            assert(len(data_states) == len(data_actions))


            states, i_actions, _, _ = statistics.collect_traj(self.env, self.sup, T, False)
            states, i_actions, _ = utils.filter_data(self.params, states, i_actions)
            
            data_states += states
            data_actions += i_actions

            self.lnr.set_data(data_states, data_actions)

            iteration += 1

        end_time = timer.time()

        for sr in self.snapshot_ranges:

            # # Uncomment for actual evaluations
            snapshot_states = data_states[:sr]
            snapshot_actions = data_actions[:sr]

            self.lnr.set_data(snapshot_states, snapshot_actions)
            self.lnr.train(verbose=True)
            log("\nData from snapshot: " + str(sr))
            it_results = self.iteration_evaluation()
            
            results['sup_rewards'].append(it_results['sup_reward_mean'])
            results['rewards'].append(it_results['reward_mean'])
            results['surr_losses'].append(it_results['surr_loss_mean'])
            results['sup_losses'].append(it_results['sup_loss_mean'])
            results['sim_errs'].append(it_results['sim_err_mean'])
            results['data_used'].append(sr)

            # Uncomment for time trials
            # results['sup_rewards'].append(0)
            # results['rewards'].append(0)
            # results['surr_losses'].append(0)
            # results['sup_losses'].append(0)
            # results['sim_errs'].append(0)
            # results['data_used'].append(0)



        for key in results.keys():
            results[key] = np.array(results[key])
        results['total_time'] = end_time - start_time

        return results




if __name__ == '__main__':
    main()

