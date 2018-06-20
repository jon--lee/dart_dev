import matplotlib
matplotlib.use('Agg')
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
from framework import load_config
from tools.utils import log

def main():

    sub_dir = 'experts'

    ap = argparse.ArgumentParser()
    ap.add_argument('--envname', required=True)
    ap.add_argument('--t', required=True, type=int)
    ap.add_argument('--save', action='store_true', default=False)
    ap.add_argument('--normalize', action='store_true', default=False)
    ap.add_argument('--num_evals', required=True, type=int)             # number of evaluations
    ap.add_argument('--max_data', required=True, type=int)              # maximum amount of data
    ap.add_argument('--config', required=True, type=str)

    params = vars(ap.parse_args())
    params = load_config(params)


    should_save = params['save']
    should_normalize = params['normalize']
    del params['save']
    del params['normalize']

    snapshot_ranges = utils.compute_snapshot_ranges(params)

    update_periods = [50, 300]
    update_periods_dart = [300]
    update_periods_dagger = [50, 300]

    if params['envname'] == 'Humanoid-v1':
        update_periods = [200, 1000]
        update_periods_dart = [1000]
        update_periods_dagger = [200, 1000]

    plt.style.use('ggplot')

    def normalize(means, sems):
        return means, sems

    all_means = []

    # BC
    title = 'test_bc'
    ptype = 'total_time'
    params_bc = params.copy()
    try:
        means, sems = utils.extract_data(params_bc, title, sub_dir, ptype)
        means, sems = normalize(means, sems)
        all_means.append(means[0])
        # p = plt.plot(snapshot_ranges, means, label='Behavior Cloning')
        # plt.fill_between(snapshot_ranges, (means - sems), (means + sems), alpha=.3, color=p[0].get_color())
    except IOError:
        log("Not found.")
        pass


    # DAgger
    title = 'test_dagger'
    ptype = 'total_time'
    params_dagger = params.copy()
    params_dagger['beta'] = .5

    for update_period in update_periods_dagger:
        params_dagger['update_period'] = update_period
        try:
            means, sems = utils.extract_data(params_dagger, title, sub_dir, ptype)
            means, sems = normalize(means, sems)
            all_means.append(means[0])
            # p = plt.plot(snapshot_ranges, means, label='DAgger ' + str(update_period))
            # plt.fill_between(snapshot_ranges, (means - sems), (means + sems), alpha=.3, color=p[0].get_color())
        except IOError:
            log("Not found.")
            pass


    # Isotropic noise
    # scales = [1.0, 10.0, 20.0]
    scales = [1.0]
    
    for scale in scales: 
        title = 'test_iso'
        ptype = 'total_time'
        params_iso = params.copy()
        params_iso['scale'] = scale
        try:
            means, sems = utils.extract_data(params_iso, title, sub_dir, ptype)
            means, sems = normalize(means, sems)
            all_means.append(means[0])
            # p = plt.plot(snapshot_ranges, means, label='Iso ' + str(scale))
            # plt.fill_between(snapshot_ranges, (means - sems), (means + sems), alpha=.3, color=p[0].get_color())
        except IOError:
            log("Not found.")
            pass


    # DART
    partition = .1

    title = 'test_dart'
    ptype = 'total_time'
    params_dart = params.copy()
    params_dart['partition'] = partition
    for update_period in update_periods_dart:
        params_dart['update_period'] = update_period
        try:
            means, sems = utils.extract_data(params_dart, title, sub_dir, ptype)
            means, sems = normalize(means, sems)
            all_means.append(means[0])
            # p = plt.plot(snapshot_ranges, means, label='DART part: ' + str(partition) + ", per: " + str(update_period))
            # plt.fill_between(snapshot_ranges, (means - sems), (means + sems), alpha=.3, color=p[0].get_color())
        except IOError:
            log("Not found.")
            pass

    all_means = np.array(all_means)
    inds = np.arange(len(all_means))
    inds[0] = 3
    inds[1] = 0
    inds[2] = 1
    inds[3] = 2
    inds[4] = 4

    for ind, mean in zip(inds, all_means):
        plt.bar([ind], [mean])

    # plt.legend()
    save_path = 'images/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.xticks([])
    if should_save == True:
        plt.savefig(save_path + str(params['envname']) + "_time.pdf")
        plt.savefig(save_path + str(params['envname']) + "_time.svg")
    else:
        plt.legend()
        plt.show()



if __name__ == '__main__':
    main()


