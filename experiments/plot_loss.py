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
color = itertools.cycle(( "#FCB716", "#2D3956", "#A0B2D8", "#988ED5", "#F68B20"))
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

    params = vars(ap.parse_args())
    params['arch'] = [64, 64]
    params['lr'] = .01
    params['epochs'] = 100
    update_period = 2

    should_save = params['save']
    del params['save']
    snapshot_ranges = utils.compute_snapshot_ranges(params)


    plt.style.use('ggplot')

    # Behavior Cloning loss on sup distr
    title = 'test_bc'
    ptype = 'sup_loss'
    params_bc = params.copy()
    c = next(color)
    try:
        means, sems = utils.extract_data(params_bc, title, sub_dir, ptype)
        plt.plot(snapshot_ranges, means, color=c, linestyle='--')

        ptype = 'surr_loss'
        means, sems = utils.extract_data(params_bc, title, sub_dir, ptype)
        plt.plot(snapshot_ranges, means, label='Behavior Cloning', color=c)
        plt.fill_between(snapshot_ranges, (means - sems), (means + sems), alpha=.3, color=c)
    except IOError:
        pass



    # DAgger
    beta = .5
    title = 'test_dagger'
    ptype = 'sup_loss'
    params_dagger = params.copy()
    params_dagger['beta'] = .5      # You may adjust the prior to whatever you chose.
    params_dagger['update_period'] = update_period
    c = next(color)
    try:
        means, sems = utils.extract_data(params_dagger, title, sub_dir, ptype)
        plt.plot(snapshot_ranges, means, color=c, linestyle='--')

        ptype = 'surr_loss'
        means, sems = utils.extract_data(params_dagger, title, sub_dir, ptype)
        plt.plot(snapshot_ranges, means, label='DAgger', color=c)
        plt.fill_between(snapshot_ranges, (means - sems), (means + sems), alpha=.3, color=c)
    except IOError:
        pass


    # Isotropic noise
    title = 'test_iso'
    ptype = 'sup_loss'
    params_iso = params.copy()
    params_iso['scale'] = 1.0
    c = next(color)
    try:
        means, sems = utils.extract_data(params_iso, title, sub_dir, ptype)
        plt.plot(snapshot_ranges, means, color=c, linestyle='--')

        ptype = 'surr_loss'
        means, sems = utils.extract_data(params_iso, title, sub_dir, ptype)
        plt.plot(snapshot_ranges, means, label='Isotropic Noise', color=c)
        plt.fill_between(snapshot_ranges, (means - sems), (means + sems), alpha=.3, color=c)
    except IOError:
        pass



    # DART
    partition = .5
    title = 'test_dart'
    ptype = 'sup_loss'
    params_dart = params.copy()
    params_dart['partition'] = partition
    params_dart['update_period'] = update_period
    c = next(color)
    try:
        means, sems = utils.extract_data(params_dart, title, sub_dir, ptype)
        plt.plot(snapshot_ranges, means, color=c, linestyle='--')
        
        ptype = 'surr_loss'
        means, sems = utils.extract_data(params_dart, title, sub_dir, ptype)
        plt.plot(snapshot_ranges, means, label='DART ' + str(partition), color=c)
        plt.fill_between(snapshot_ranges, (means - sems), (means + sems), alpha=.3, color=c)
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


