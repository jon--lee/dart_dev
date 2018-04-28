import numpy as np
import os
import argparse
import pandas as pd
import scipy.stats
from tools import statistics, utils
import matplotlib.pyplot as plt
import itertools
marker = itertools.cycle((',', '+', '.', 'o', '*', 's')) 
import IPython


def main():

    # In the event that you change the sub_directory within results, change this to match it.
    color = itertools.cycle(( "#FCB716", "#2D3956", "#A0B2D8", "#988ED5", "#F68B20"))

    sub_dir = 'experts'

    ap = argparse.ArgumentParser()
    ap.add_argument('--envname', required=True)
    ap.add_argument('--t', required=True, type=int)
    ap.add_argument('--save', action='store_true', default=False)
    ap.add_argument('--normalize', action='store_true', default=False)
    ap.add_argument('--num_evals', required=True, type=int)             # number of evaluations
    ap.add_argument('--max_data', required=True, type=int)              # maximum amount of data


    params = vars(ap.parse_args())
    params['arch'] = [64, 64]
    params['lr'] = .01
    params['epochs'] = 100

    should_save = params['save']
    should_normalize = params['normalize']
    del params['save']
    del params['normalize']

    snapshot_ranges = utils.compute_snapshot_ranges(params)

    plt.style.use('ggplot')

    # Best supervisor reward
    title = 'test_bc'
    ptype = 'sup_reward'
    params_bc = params.copy()
    means, sems = utils.extract_data(params_bc, title, sub_dir, ptype)
    if not should_normalize:
        plt.plot(snapshot_ranges, means, label='Supervisor', color='green')

    sup_means, sup_sems = means, sems
    def normalize(means, sems):
        if should_normalize:
            means = means / sup_means
            sems = sems / sup_means
            return means, sems
        else:
            return means, sems



    # Noisy supervisor reward using DART
    title = 'test_dart'
    ptype = 'sup_reward'
    params_dart = params.copy()
    try:
        means, sems = utils.extract_data(params_dart, title, sub_dir, ptype)
        means, sems = normalize(means, sems)
        plt.plot(snapshot_ranges, means, label='DART Noisy Supervisor', color='green', linestyle='--')
    except IOError:
        pass

    # BC
    title = 'test_bc'
    ptype = 'reward'
    params_bc = params.copy()
    c = next(color)
    try:
        means, sems = utils.extract_data(params_bc, title, sub_dir, ptype)
        means, sems = normalize(means, sems)
        plt.plot(snapshot_ranges, means, label='Behavior Cloning', color=c)
        plt.fill_between(snapshot_ranges, (means - sems), (means + sems), alpha=.3, color=c)
    except IOError:
        pass


    # DAgger
    title = 'test_dagger'
    ptype = 'reward'
    params_dagger = params.copy()
    params_dagger['beta'] = .5
    params_dagger['update_period'] = 5
    c = next(color)
    try:
        means, sems = utils.extract_data(params_dagger, title, sub_dir, ptype)
        means, sems = normalize(means, sems)
        plt.plot(snapshot_ranges, means, label='DAgger', color=c)
        plt.fill_between(snapshot_ranges, (means - sems), (means + sems), alpha=.3, color=c)
    except IOError:
        pass


    # Isotropic noise
    title = 'test_iso'
    ptype = 'reward'
    params_iso = params.copy()
    params_iso['scale'] = 1.0
    c = next(color)
    try:
        means, sems = utils.extract_data(params_iso, title, sub_dir, ptype)
        means, sems = normalize(means, sems)
        plt.plot(snapshot_ranges, means, color=c, label='Isotropic')
        plt.fill_between(snapshot_ranges, (means - sems), (means + sems), alpha=.3, color=c)
    except IOError:
        pass


    # DART
    partition = .1
    title = 'test_dart'
    ptype = 'reward'
    params_dart = params.copy()
    params_dart['partition'] = partition
    params_dart['update_period'] = 5
    c = next(color)
    try: 
        means, sems = utils.extract_data(params_dart, title, sub_dir, ptype)
        means, sems = normalize(means, sems)
        plt.plot(snapshot_ranges, means, label='DART ' + str(partition), color=c)
        plt.fill_between(snapshot_ranges, (means - sems), (means + sems), alpha=.3, color=c)
    except IOError:
        pass

    plt.title("Reward on " + str(params['envname']))
    plt.legend()
    plt.xticks(snapshot_ranges)
    if should_normalize:
        plt.ylim(0, 1.05)
        plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    plt.title(params['envname'][:-3])

    save_path = 'images/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if should_save == True:
        plt.savefig(save_path + str(params['envname']) + "_reward.pdf")
    else:
        plt.legend()
        plt.show()



if __name__ == '__main__':
    main()


