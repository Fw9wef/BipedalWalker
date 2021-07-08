from agent import Actor
from env import *
import torch.multiprocessing as mp
from copy import deepcopy
import numpy as np


class PPO:
    def __init__(self, per_gpu_workers=1, gpus=[0], lam=0.95, gamma=0.99, epsilon=0.2):
        self.per_gpu_workers = per_gpu_workers
        self.gpus = gpus
        self.n_workers = self.per_gpu_workers * len(self.gpus)
        self.lam = lam
        self.gamma = gamma
        self.epsilon = epsilon
        self.workers = list()
        if gpus != 'cpu':
            for gpu in self.gpus:
                for _ in range(self.per_gpu_workers):
                    self.workers.append(Actor(gpu_id=gpu, l=self.lam, gamma=self.gamma, epsilon=self.epsilon))
        else:
            for _ in range(self.per_gpu_workers):
                self.workers.append(Actor(gpu_id='cpu', l=self.lam, gamma=self.gamma, epsilon=self.epsilon))

        policy_state_dict, value_state_dict = self.workers[0].get_weights()
        for worker in self.workers[1:]:
            worker.sync_nets(policy_state_dict, value_state_dict)

    def gather_sards(self, n_episodes, n_sards):
        procs = list()
        queue = mp.Queue()
        event = mp.Event()
        for worker in self.workers:
            procs.append(mp.Process(target=worker.run, args=(n_episodes, n_sards, queue, event)))
        for proc in procs:
            proc.start()
        ret_sards = list()
        ret_stats = list()
        for _ in range(n_episodes * len(procs)):
            sards, stats = queue.get()
            ret_sards += sards
            ret_stats.append(stats)
        sards = deepcopy(ret_sards)
        stats = deepcopy(ret_stats)
        del ret_sards, ret_stats
        event.set()
        for proc in procs:
            proc.join()
        return sards, stats

    def gather_gradients(self, batch):
        per_worker_batch = len(batch) // self.n_workers
        batches = [batch[i*per_worker_batch:(i+1)*per_worker_batch] for i in range(self.n_workers)]
        procs = list()
        for mini_batch, worker in zip(batches, self.workers):
            procs.append(mp.Process(target=worker.get_grads, args=(mini_batch, self.queue)))
        for proc in procs:
            proc.start()

        policy_grads = list()
        value_grads = list()
        for i, _ in enumerate(procs):
            p_grad, v_grad = self.queue.get()
            policy_grads.append(p_grad)
            value_grads.append(v_grad)

        for proc in procs:
            proc.join()
        return policy_grads, value_grads

    def update_and_spread(self, policy_grads, value_grads):
        self.workers[0].apply_grads(policy_grads, value_grads)
        policy_state_dict, value_state_dict = self.workers[0].get_weights()
        for worker in self.workers[1:]:
            worker.sync_nets(policy_state_dict, value_state_dict)

    def spread(self):
        policy_state_dict, value_state_dict = self.workers[0].get_weights()
        for i, worker in enumerate(self.workers[1:]):
            self.workers[i+1].sync_nets(policy_state_dict, value_state_dict)

    def train_multiproc(self, iterations, ppo_epochs, batch_size, n_batch, n_episodes):
        n_sards = int(batch_size*n_batch/n_episodes/len(self.workers))
        print("Per episode sards: ", n_sards)
        for iteration in range(1, iterations + 1):
            sards, stats = self.gather_sards(n_episodes, n_sards)
            sards = np.random.permutation(sards)
            #print("-" * 100)
            #print("Gathered")
            #print("-" * 100)
            Episode.average_stats_and_show(stats)

            for ppo_iter in range(ppo_epochs):
                for batch in range(len(sards) // batch_size):
                    mini_batch = sards[batch * batch_size: (batch + 1) * batch_size]
                    policy_grads, value_grads = self.workers[0].get_grads(mini_batch)
                    self.workers[0].apply_grads([policy_grads], [value_grads])
            self.spread()

    def train(self, iterations, ppo_epochs, batch_size, n_steps):
        for iteration in range(1, iterations + 1):
            episodes = [self.workers[0].run_n_steps(n_steps)]
            episodes[0].show_stats()

            for ppo_iter in range(ppo_epochs):
                sards = np.random.permutation(episodes[0].sards)
                for batch in range(n_steps // batch_size):
                    sards_batch = sards[batch * batch_size: (batch + 1) * batch_size]
                    policy_grads, value_grads = self.workers[0].get_grads(sards_batch)
                    self.workers[0].apply_grads([policy_grads], [value_grads])
