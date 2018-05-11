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
from matplotlib import colors as mcolors
from framework import load_config


def main():

    # In the event that you change the sub_directory within results, change this to match it.
    sub_dir = 'experts'

    ap = argparse.ArgumentParser()
    ap.add_argument('--envname', required=True)
    ap.add_argument('--t', required=True, type=int)
    ap.add_argument('--save', action='store_true', default=False)
    ap.add_argument('--num_evals', required=True, type=int)             # number of evaluations
    ap.add_argument('--max_data', required=True, type=int)              # maximum amount of data
    ap.add_argument('--config', required=True, type=str)

    params = vars(ap.parse_args())
    params = load_config(params)

    should_save = params['save']
    del params['save']
    snapshot_ranges = utils.compute_snapshot_ranges(params)


    plt.style.use('ggplot')

    # Behavior Cloning loss on sup distr
    title = 'test_bc'
    ptype = 'sup_loss'
    params_bc = params.copy()
    try:
        means, sems = utils.extract_data(params_bc, title, sub_dir, ptype)
        p = plt.plot(snapshot_ranges, means, linestyle='--')

        ptype = 'surr_loss'
        means, sems = utils.extract_data(params_bc, title, sub_dir, ptype)
        plt.plot(snapshot_ranges, means, label='Behavior Cloning', color=p[0].get_color())
        plt.fill_between(snapshot_ranges, (means - sems), (means + sems), alpha=.3, color=p[0].get_color())
    except IOError:
        pass



    # DAgger
    update_periods = [4]
    beta = .5

    title = 'test_dagger'
    ptype = 'sup_loss'
    params_dagger = params.copy()
    params_dagger['beta'] = .5      # You may adjust the prior to whatever you chose.
    for update_period in update_periods:
        params_dagger['update_period'] = update_period
        try:
            means, sems = utils.extract_data(params_dagger, title, sub_dir, ptype)
            p = plt.plot(snapshot_ranges, means, linestyle='--')

            ptype = 'surr_loss'
            means, sems = utils.extract_data(params_dagger, title, sub_dir, ptype)
            plt.plot(snapshot_ranges, means, label='DAgger', color=p[0].get_color())
            plt.fill_between(snapshot_ranges, (means - sems), (means + sems), alpha=.3, color=p[0].get_color())
        except IOError:
            pass


    # Isotropic noise
    scale = 1.0

    title = 'test_iso'
    ptype = 'sup_loss'
    params_iso = params.copy()
    params_iso['scale'] = scale
    try:
        means, sems = utils.extract_data(params_iso, title, sub_dir, ptype)
        p = plt.plot(snapshot_ranges, means, linestyle='--')

        ptype = 'surr_loss'
        means, sems = utils.extract_data(params_iso, title, sub_dir, ptype)
        plt.plot(snapshot_ranges, means, label='Isotropic Noise', color=p[0].get_color())
        plt.fill_between(snapshot_ranges, (means - sems), (means + sems), alpha=.3, color=p[0].get_color())
    except IOError:
        pass



    # DART
    update_periods = [4]
    partition = .5

    title = 'test_dart'
    ptype = 'sup_loss'
    params_dart = params.copy()
    params_dart['partition'] = partition

    for update_period in update_periods:
        params_dart['update_period'] = update_period
        try:
            means, sems = utils.extract_data(params_dart, title, sub_dir, ptype)
            p = plt.plot(snapshot_ranges, means, linestyle='--')
            
            ptype = 'surr_loss'
            means, sems = utils.extract_data(params_dart, title, sub_dir, ptype)
            plt.plot(snapshot_ranges, means, label='DART ' + str(partition), color=p[0].get_color())
            plt.fill_between(snapshot_ranges, (means - sems), (means + sems), alpha=.3, color=p[0].get_color())
        except IOError:
            pass



    plt.title("Loss on " + str(params['envname']))
    plt.legend()
    plt.xticks(snapshot_ranges)
    plt.legend(loc='upper right')

    save_path = 'images/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if should_save == True:
        plt.savefig(save_path + str(params['envname']) + "_loss.pdf")
    else:
        plt.show()



if __name__ == '__main__':
    main()


