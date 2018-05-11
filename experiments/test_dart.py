"""
    Experiment script intended to test DART
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
from tools.utils import log

def main():
    title = 'test_dart'

    ap = framework.get_args()
    ap.add_argument('--update_period', required=True, type=int)         # period between updates to the policy
    ap.add_argument('--partition', required=True, type=float)             # Integer between 1 and 450 (exclusive),
    args = vars(ap.parse_args())
    args = framework.load_config(args)
    
    assert args['partition'] < 1.0 and args['partition'] > 0.0

    framework.startup(title, args, Test)


class Test(framework.Test):

    def count_states(self, trajs):
        count = 0
        for states, actions in trajs:
            count += len(states)
        return count



    def update_noise(self, i, trajs):
        assert i % self.params['update_period'] == 0

        self.lnr.train()
        new_cov = noise.sample_covariance_trajs(self.env, self.lnr, trajs, self.params['t'])
        log("Estimated covariance matrix: ")
        log(new_cov)
        log("Trace: " + str(np.trace(new_cov)))
        # d = env.action_space.shape[0]
        self.sup = GaussianSupervisor(self.net_sup, new_cov)
        return self.sup



    def run_iters(self):
        T = self.params['t']
        partition = self.params['partition']

        results = {
            'rewards': [],
            'sup_rewards': [],
            'surr_losses': [],
            'sup_losses': [],
            'sim_errs': [],
            'data_used': [],
        }

        trajs = []
        traj_snapshots = []
        self.optimized_data = 0

        data_states = []
        data_actions = []

        train_states = []
        train_i_actions = []

        iteration = 0

        while len(data_states) < self.params['max_data']:
            log("\tIteration: " + str(iteration))
            log("\tData states: " + str(len(data_states)))
            assert(len(data_states) == len(data_actions))

            states, i_actions, _, _ = statistics.collect_traj(self.env, self.sup, T, False)
            states, i_actions, _ = utils.filter_data(self.params, states, i_actions)

            data_states += states
            data_actions += i_actions

            rang = np.arange(0, len(states))
            np.random.shuffle(rang)

            partition_cutoff = int(partition * len(states))
            noise_states, noise_actions = [states[k] for k in rang[:partition_cutoff]], [i_actions[k] for k in rang[:partition_cutoff]]
            states, i_actions = [states[k] for k in rang[partition_cutoff:]], [i_actions[k] for k in rang[partition_cutoff:]]

            train_states += states
            train_i_actions += i_actions

            self.lnr.set_data(train_states, train_i_actions)
            trajs.append((noise_states, noise_actions))

            if iteration % self.params['update_period'] == 0:
                self.sup = self.update_noise(iteration, trajs)

            iteration += 1


        for sr in self.snapshot_ranges:
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
        
        log("\tTrain data: " + str(len(train_i_actions)))
        log("\tNoise opt data: " + str(self.count_states(trajs)))

        for key in results.keys():
            results[key] = np.array(results[key])
        return results




if __name__ == '__main__':
    main()

