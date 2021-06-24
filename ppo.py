from agent import Actor
from env import *
import torch.multiprocessing as mp
from time import time


class PPO:
    def __init__(self, per_gpu_workers=1, gpus=[0], lam=0.95, gamma=0.99, epsilon=0.2):
        self.per_gpu_workers = per_gpu_workers
        self.gpus = gpus
        self.n_workers = self.per_gpu_workers * len(self.gpus)
        self.lam = lam
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes_queue = mp.Queue()
        self.grads_queue = mp.Queue()
        self.workers = list()
        for gpu in self.gpus:
            for _ in range(self.per_gpu_workers):
                self.workers.append(Actor(gpu_id=gpu, l=self.lam, gamma=self.gamma, epsilon=self.epsilon))
        policy_state_dict, value_state_dict = self.workers[0].get_weights()
        procs = list()
        for worker in self.workers[1:]:
            procs.append(mp.Process(target=worker.sync_nets, args=(policy_state_dict, value_state_dict)))
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()

    def gather_episodes(self, n_episodes):
        procs = list()

        qwe = time()
        for worker in self.workers:
            procs.append(mp.Process(target=worker.run, args=(n_episodes, self.episodes_queue)))
        ewq = time()
        for proc in procs:
            proc.start()
        for proc in procs:
            print("JOINED")
            proc.join()

        episodes = list()
        for i in procs:
            print(i)
            episodes += self.episodes_queue.get()
        return episodes

    def gather_gradients(self, batch):
        per_worker_batch = len(batch) // self.n_workers
        batches = [batch[i*per_worker_batch:(i+1)*per_worker_batch] for i in range(self.n_workers)]
        procs = list()
        for mini_batch, worker in zip(batches, self.workers):
            procs.append(mp.Process(target=worker.get_grads, args=(mini_batch, self.grads_queue)))
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()

        policy_grads = list()
        value_grads = list()
        for _ in procs:
            p_grad, v_grad = self.grads_queue.get()
            policy_grads.append(p_grad)
            value_grads.append(v_grad)
        return policy_grads, value_grads

    def update_and_spread(self, policy_grads, value_grads):
        self.workers[0].apply_grads(policy_grads, value_grads)
        policy_state_dict, value_state_dict = self.workers[0].get_weights()
        procs = list()
        for worker in self.workers[1:]:
            procs.append(mp.Process(target=worker.sync_nets, args=(policy_state_dict, value_state_dict)))
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()

    def train(self, iterations, ppo_epochs, batch_size, n_batch,  n_episodes):
        for iteration in range(1, iterations + 1):
            episodes = self.gather_episodes(n_episodes)
            sards = Episode.permute_episodes(episodes)
            sards = sards[:n_batch*batch_size]
            Episode.show_episodes_stats(episodes)

            for ppo_iter in range(ppo_epochs):
                for batch in range(len(sards) // batch_size):
                    mini_batch = sards[batch * batch_size: (batch + 1) * batch_size]
                    policy_grads, value_grads = self.gather_gradients(mini_batch)
                    self.update_and_spread(policy_grads, value_grads)
