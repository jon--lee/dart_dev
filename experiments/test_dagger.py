"""
    Experiment script intended to test DAgger
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import gym
import numpy as np
from tools import statistics, noise, utils
from tools.supervisor import GaussianSupervisor
import argparse
import scipy.stats
import time as timer
import framework

def main():
    title = 'test_dagger'
    
    ap = framework.get_args()
    ap.add_argument('--update_period', required=True, type=int)         # period between updates to the policy
    ap.add_argument('--beta', required=True, type=float)                # beta term, see Ross et al.
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
        trajs = []

        beta = self.params['beta']

        data_states = []
        data_actions = []

        iteration = 0

        while len(data_states) < self.params['max_data']:
            print "\tIteration: " + str(iteration)
            print "\tData states: " + str(len(data_states))
            assert(len(data_states) == len(data_actions))

            if iteration == 0:
                states, i_actions, _, _ = statistics.collect_traj(self.env, self.sup, T, False)
                states, i_actions, _ = utils.filter_data(self.params, states, i_actions)
            else:
                states, _, _, _ = statistics.collect_traj_beta(self.env, self.sup, self.lnr, T, beta, False)
                i_actions = [self.sup.intended_action(s) for s in states]
                states, i_actions, _ = utils.filter_data(self.params, states, i_actions)
                beta = beta * self.params['beta']

            data_states += states
            data_actions += i_actions

            self.lnr.set_data(data_states, data_actions)

            if iteration % self.params['update_period'] == 0:
                self.lnr.train(verbose=True)

            iteration += 1


        for sr in self.snapshot_ranges:
            snapshot_states = data_states[:sr]
            snapshot_actions = data_actions[:sr]

            self.lnr.set_data(snapshot_states, snapshot_actions)
            self.lnr.train(verbose=True)
            print "\nData from snapshot: " + str(sr)
            it_results = self.iteration_evaluation()
            
            results['sup_rewards'].append(it_results['sup_reward_mean'])
            results['rewards'].append(it_results['reward_mean'])
            results['surr_losses'].append(it_results['surr_loss_mean'])
            results['sup_losses'].append(it_results['sup_loss_mean'])
            results['sim_errs'].append(it_results['sim_err_mean'])
            results['data_used'].append(sr)


        for key in results.keys():
            results[key] = np.array(results[key])
        return results




if __name__ == '__main__':
    main()

