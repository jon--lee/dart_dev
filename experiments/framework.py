import time as timer
import numpy as np
from tools.expert import load_policy
from tools import statistics, noise, utils
from tools import learner
from tools.utils import log
from tools.supervisor import GaussianSupervisor, Supervisor
import tensorflow as tf
from models.net import knet
from models import linear, kernel_poly, poly
import gym
import os
import pandas as pd
import scipy.stats
import asserts
import sys
import IPython
import argparse
import yaml
sys.path.append("..")

TRIALS = 20

def load_config(args):
    directory = 'experiments/configs/'
    filename = args['config'] + '.yaml'
    filepath = os.path.join(directory, filename)
    f = open(filepath)
    config = yaml.load(f)
    f.close()

    for key in config:
        args[key] = config[key]
    return args


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--envname', required=True)                         # OpenAI gym environment
    ap.add_argument('--t', required=True, type=int)                     # time horizon
    ap.add_argument('--num_evals', required=True, type=int)             # number of evaluations
    ap.add_argument('--max_data', required=True, type=int)              # maximum amount of data
    ap.add_argument('--config', required=True, type=str)
    return ap



def startup(title, args, TestClass):
    test = TestClass(args)
    start_time = timer.time()
    test.run_trials(title, TRIALS)
    end_time = timer.time()
    utils.log("\n\n\nTotal time: " + str(end_time - start_time) + '\n\n')



class Test(object):

    def __init__(self, params):
        self.params = params
        self.params['trials'] = TRIALS
        
        asserts.enforce(params)
        # utils.log("\n\n")
        # utils.log("######################################")
        # utils.log("### REMEMBER TO ENFORCE ASSERTIONS ###")
        # utils.log("######################################")
        # utils.log("\n\n")

        # utils.log("\n\n")
        # utils.log("########################################")
        # utils.log("### REMEMBER TO TURN OFF TERMINATION ###")
        # utils.log("########################################")
        # utils.log("\n\n")

        del self.params['trials']
        return

    def reset_learner(self, params):
        """
            Initializes new neural network and learner wrapper
        """
        if params['type'] == 'linear':
            est = linear.Linear()
        elif params['type'] == 'net':
            est = knet.Network(params['arch'], learning_rate=params['lr'], epochs=params['epochs'])
        elif params['type'] == 'poly':
            est = poly.Poly(params['degree'])
        elif params['type'] == 'kernel_poly':
            est = kernel_poly.KernelPoly(params['degree'])
        else:
            raise Exception("Learner type not found, see config files.")

        lnr = learner.Learner(est)
        return est, lnr


    def prologue(self):
        """
            Preprocess hyperparameters and initialize learner and supervisor
        """
        self.params['filename'] = './experts/' + self.params['envname'] + '.pkl'
        self.env = gym.envs.make(self.params['envname'])

        self.params['d'] = self.env.action_space.shape[0]

        sess = tf.Session()
        policy = load_policy.load_policy(self.params['filename'])
        net_sup = Supervisor(policy, sess)
        init_cov = np.zeros((self.params['d'], self.params['d']))
        sup = GaussianSupervisor(net_sup, init_cov)
        est, lnr = self.reset_learner(self.params)

        self.lnr, self.sup, self.net_sup = lnr, sup, net_sup
        return self.params


    def run_iters(self):
        """
            To be implemented by learning methods (e.g. behavior cloning, dart, dagger...)
        """
        raise NotImplementedError


    def run_trial(self):
        """
            Run a trial by first preprocessing the parameters and initializing
            the supervisor and learner. Then run each iterations (not implemented here)
        """

        self.prologue()
        results = self.run_iters()

        return results



    def iteration_evaluation(self):
        """
            Evaluate learner and supervisor given the current amount of data
            Supervisor is averaged over p trajectories
            Learner is averaged over q trajectories
        """
        # Asserting limited data per iteration. 
        # See experiments from Ho and Ermon, 2016 for sampling method
        log("Data: " + str(len(self.lnr.X)))
        assert len(self.lnr.X) <= self.params['max_data']
        
        it_results = {}

        p = 1
        q = 3
        sup_rewards = np.zeros(p)
        sup_losses = np.zeros(q)
        rewards = np.zeros(q)
        surr_losses = np.zeros(q)
        sim_errs = np.zeros(q)

        for j in range(p):
            sup_rewards[j] = statistics.eval_rewards(self.env, self.sup, self.params['t'], 1)
        
        for j in range(q):
            eval_results = self.evals()
            rewards[j] = eval_results['rewards']
            surr_losses[j] = eval_results['surr_losses']
            sup_losses[j] = eval_results['sup_losses']
            sim_errs[j] = eval_results['sim_errs']

        it_results['sup_reward_mean'], it_results['sup_reward_std'] = np.mean(sup_rewards), np.std(sup_rewards)
        it_results['reward_mean'], it_results['reward_std'] = np.mean(rewards), np.std(rewards)
        it_results['surr_loss_mean'], it_results['surr_loss_std'] = np.mean(surr_losses), np.std(surr_losses)
        it_results['sup_loss_mean'], it_results['sup_loss_std'] = np.mean(sup_losses), np.std(sup_losses)
        it_results['sim_err_mean'], it_results['sim_err_std'] = np.mean(sim_errs), np.std(sim_errs)

        utils.log("\t\tSup reward: " + str(it_results['sup_reward_mean']) + " +/- " + str(it_results['sup_reward_std']))
        utils.log("\t\tLnr_reward: " + str(it_results['reward_mean']) + " +/- " + str(it_results['reward_std']))
        utils.log("\t\tSurr loss: " + str(it_results['surr_loss_mean']) + " +/- " + str(it_results['surr_loss_std']))
        utils.log("\t\tSup loss: " + str(it_results['sup_loss_mean']) + "+/-" + str(it_results['sup_loss_std']))
        utils.log("\t\tSim err: " + str(it_results['sim_err_mean']) + " +/- " + str(it_results['sim_err_std']))
        utils.log("\t\tTrace: " + str(np.trace(self.sup.cov)))

        return it_results



    def evals(self):
        """
            Evaluate on all metrics including 
            reward, loss, and simulated error of the supervisor
        """
        results = {}
        results['rewards'] = statistics.eval_rewards(self.env, self.lnr, self.params['t'], 1)
        results['surr_losses'] = statistics.evaluate_lnr_cont(self.env, self.lnr, self.sup, self.params['t'], 1)
        results['sup_losses'] = statistics.evaluate_sup_cont(self.env, self.lnr, self.sup, self.params['t'], 1)
        results['sim_errs'] = statistics.evaluate_sim_err_cont(self.env, self.sup, self.params['t'], 1)
        return results



    @property
    def snapshot_ranges(self):
        snapshot_ranges = np.linspace(0, self.params['max_data'], 1+self.params['num_evals'])[1:].astype(int)
        return snapshot_ranges


    def run_trials(self, title, TRIALS):
        """
            Runs and saves all trials. Generates directories under 'results/experts/'
            where sub-directory names are based on the parameters. Data is saved after
            every trial, so it is safe to interrupt program.
        """
        sub_dir = 'experts'
        paths = {}
        for sr in self.snapshot_ranges:
            self.params['sr'] = sr
            parent_data_dir = utils.generate_data_dir(title, sub_dir, self.params)
            save_path = parent_data_dir + 'data.csv'
            paths[sr] = save_path
            if not os.path.exists(parent_data_dir):
                os.makedirs(parent_data_dir)
            utils.log("Creating directory at " + str(save_path))


        m = len(self.snapshot_ranges)
        self.rewards_all, self.sup_rewards_all = np.zeros((TRIALS, m)), np.zeros((TRIALS, m))       # reward obtained from learner and (noisy) supervisor
        self.surr_losses_all, self.sup_losses_all = np.zeros((TRIALS, m)), np.zeros((TRIALS, m))    # loss obtained on supervisor's distribution and on learner's distribution
        self.sim_errs_all = np.zeros((TRIALS, m))                                                   # Empirical simulated error of supervisor (trace of covariance matrix)
        self.data_used_all = np.zeros((TRIALS, m))
        self.total_times_all = np.zeros((TRIALS))

        for t in range(TRIALS):
            utils.log("\n\nTrial: " + str(t))
            results = self.run_trial()
            total_time = results['total_time']

            self.rewards_all[t, :], self.sup_rewards_all[t, :] = results['rewards'], results['sup_rewards']
            self.surr_losses_all[t, :], self.sup_losses_all[t, :] = results['surr_losses'], results['sup_losses']
            self.sim_errs_all[t, :] = results['sim_errs']
            self.data_used_all[t, :] = results['data_used']
            self.total_times_all[t] = results['total_time']

            utils.log("trial time: " + str(total_time))
            self.save_all(t + 1, paths)



    def save_all(self, t, paths):
        rewards_all = self.rewards_all[:t, :]
        surr_losses_all = self.surr_losses_all[:t, :]
        sup_rewards_all = self.sup_rewards_all[:t, :]
        sup_losses_all = self.sup_losses_all[:t, :]
        sim_errs_all = self.sim_errs_all[:t, :]
        data_used_all = self.data_used_all[:t, :]
        total_times_all = self.total_times_all[:t]

        for i in range(len(self.snapshot_ranges)):
            sr = self.snapshot_ranges[i]
            rewards = rewards_all[:, i]
            surr_losses = surr_losses_all[:, i]
            sup_rewards = sup_rewards_all[:, i]
            sup_losses = sup_losses_all[:, i]
            sim_errs = sim_errs_all[:, i]
            data_used = data_used_all[:, i]
            total_time = total_times_all[:]
            save_path = paths[self.snapshot_ranges[i]]

            utils.log("Saving to: " + str(save_path))


            d = {'reward': rewards, 'surr_loss': surr_losses, 
                'sup_reward': sup_rewards, 'sup_loss': sup_losses,
                'sim_err': sim_errs, 'data_used': data_used, 'total_time': total_time}
            df = pd.DataFrame(d)
            df.to_csv(save_path)

            reward_mean, reward_sem = np.mean(rewards), scipy.stats.sem(rewards)
            surr_loss_mean, surr_loss_sem = np.mean(surr_losses), scipy.stats.sem(surr_losses)
            sup_reward_mean, sup_reward_sem = np.mean(sup_rewards), scipy.stats.sem(sup_rewards)
            sup_loss_mean, sup_loss_sem = np.mean(sup_losses), scipy.stats.sem(sup_losses)
            sim_err_mean, sim_err_sem = np.mean(sim_errs), scipy.stats.sem(sim_errs)
            data_used_mean, data_used_sem = np.mean(data_used), scipy.stats.sem(data_used)
            total_time_mean, total_time_sem = np.mean(total_time), scipy.stats.sem(total_time)

            log("Iteration " + str(sr) + " results:")
            log("For iteration: " + str(sr))
            log("Lnr reward: " + str(reward_mean) + ' +/- ' + str(reward_sem))
            log("Surr loss: " + str(surr_loss_mean) + " +/- " + str(surr_loss_sem))
            log("Sup reward: " + str(sup_reward_mean) + " +/- " + str(sup_reward_sem))
            log("Sup loss: " + str(sup_loss_mean) + " +/- " + str(sup_loss_sem))
            log("Sim err: " + str(sim_err_mean) + " +/- " + str(sim_err_sem))
            log("Data used: " + str(data_used_mean) + " +/- " + str(data_used_sem))
            log("Total time: " + str(total_time_mean) + " +/- " + str(total_time_sem))
            log("\n\n\n")



        return





