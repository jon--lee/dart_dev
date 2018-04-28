import numpy as np
import os
import argparse
import pandas as pd
import scipy.stats
from tools import statistics, utils
import matplotlib.pyplot as plt
import itertools
marker = itertools.cycle((',', '+', '.', 'o', '*', 's')) 
color = itertools.cycle(( "#FCB716", "#2D3956", "#A0B2D8", "#988ED5", "#F68B20", 'purple'))
import IPython


def main():

    # In the event that you change the sub_directory within results, change this to match it.
    sub_dir = 'experts'

    ap = argparse.ArgumentParser()
    ap.add_argument('--envname', required=True)
    ap.add_argument('--t', required=True, type=int)
    ap.add_argument('--iters', required=True, type=int, nargs='+')
    ap.add_argument('--update', required=True, nargs='+', type=int)
    ap.add_argument('--save', action='store_true', default=False)
    ap.add_argument('--normalize', action='store_true', default=False)
    
    params = vars(ap.parse_args())
    params['arch'] = [64, 64]
    params['lr'] = .01
    params['epochs'] = 100

    should_save = params['save']
    should_normalize = params['normalize']
    del params['save']
    del params['normalize']

    plt.style.use('ggplot')

    iters = params['iters']
    ptype = 'data_used'
    



    parts = [5, 10, 50, 450][::-1]
    dart_names = ['DART ' + str(part) for part in parts]
    dart_data = []
    dart_errs = []
    for part in parts:
        title = 'test_dart'
        ptype = 'data_used'
        params_dart = params.copy()
        params_dart['partition'] = part
        try: 
            means, sems = utils.extract_data(params_dart, iters, title, sub_dir, ptype)
            dart_data.append(means[-1])
            dart_errs.append(sems[-1])
        except IOError:
            pass

    labels = dart_names
    data = dart_data
    errs = dart_errs
    plt.bar(labels, data, yerr=errs)
    plt.title(params['envname'][:-3])

    save_path = 'images/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if should_save == True:
        plt.savefig(save_path + str(params['envname']) + "_data.pdf")
        plt.savefig(save_path + "svg_" + str(params['envname']) + "_data.svg")
    else:
        plt.show()



if __name__ == '__main__':
    main()


