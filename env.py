import torch
import numpy as np


class SARD:
    def __init__(self, state, state_value, action, reward, done, action_log_prob, advantage=None):
        self.state = state
        self.state_value = state_value
        self.action = action
        self.reward = reward
        self.done = done
        self.action_log_prob = action_log_prob
        self.advantage = advantage
        self.target_value = None

    @staticmethod
    def sards2tensors(sards):
        states, actions, values, target_values, adv, old_log_policy = [list() for _ in range(6)]
        for sard in sards:
            states.append(sard.state)
            actions.append(sard.action)
            values.append(sard.state_value)
            target_values.append(sard.target_value)
            adv.append(sard.advantage)
            old_log_policy.append(sard.action_log_prob)
        states = torch.stack(states, dim=0)
        actions = torch.stack(actions, dim=0)
        values = torch.stack(values, dim=0)
        target_values = torch.stack(target_values, dim=0)
        adv = torch.stack(adv, dim=0)
        old_log_policy = torch.stack(old_log_policy, dim=0)
        return states, actions, values, target_values, adv, old_log_policy


class Episode:
    def __init__(self):
        self.sards = list()
        self.ep_reward = 0

    def __getitem__(self, idx):
        return self.sards[idx]

    def __len__(self):
        return len(self.sards)

    def add_sard(self, sard):
        self.sards.append(sard)
        self.ep_reward += self.sards[-1].reward

    def compute_advantages(self, l=0.95, gamma=0.99):
        run_adv = 0
        for t in reversed(range(len(self))):
            if self.sards[t].done:
                run_adv = self.sards[t].reward - self.sards[t].state_value
            else:
                sigma = self.sards[t].reward + gamma * self.sards[t + 1].state_value - self.sards[t].state_value
                run_adv = sigma + run_adv * gamma * l

            self.sards[t].advantage = run_adv
            self.sards[t].target_value = self.sards[t].state_value + run_adv

    def get_random_sards(self, n):
        if n < len(self):
            chosen_idx = np.random.choice(len(self.sards), n, replace=False)
            chosen_sards = [self.sards[i] for i in chosen_idx]
        else:
            chosen_sards = self.sards
        return chosen_sards

    def send_episode(self, send_conn):
        pass

    @staticmethod
    def get_random_sard_from_episodes(episodes, per_episode_sards):
        sards = list()
        for episode in episodes:
            sards += episode.get_random_sards(per_episode_sards)
        sards = np.random.permutation(sards)
        return sards

    @staticmethod
    def permute_episodes(episodes):
        sards = list()
        for episode in episodes:
            sards += episode.sards
        sards = np.random.permutation(sards)
        return sards

    @staticmethod
    def get_episodes_stats(episodes):
        rewards, ls = list(), list()
        for episode in episodes:
            rewards.append(episode.ep_reward)
            ls.append(len(episode))
        stats = dict()
        stats['reward_mean'] = np.mean(rewards)
        stats['reward_std'] = np.std(rewards)
        stats['len_mean'] = np.mean(ls)
        stats['len_std'] = np.std(ls)
        return stats

    @staticmethod
    def show_episodes_stats(episodes):
        stats = Episode.get_episodes_stats(episodes)
        print("Reward mean: %.6f  |  Reward std: %.6f  |  Len mean: %.6f  |  Len std: %.6f" % \
              (stats["reward_mean"], stats["reward_std"], stats["len_mean"], stats["len_std"]))


class Sequence:
    def __init__(self):
        self.sards = list()
        self.ep_reward = 0
        self.ep_length = 0
        self.eps_l = list()
        self.eps_r = list()

    def __getitem__(self, idx):
        return self.sards[idx]

    def __len__(self):
        return len(self.sards)

    def add_sard(self, sard):
        self.sards.append(sard)
        self.ep_reward += 100 * self.sards[-1].reward
        self.ep_length += 1
        if self.sards[-1].done:
            self.eps_r.append(self.ep_reward)
            self.ep_reward = 0
            self.eps_l.append(self.ep_length)
            self.ep_length = 0

    def compute_advantages(self, l=0.95, gamma=0.99):
        advs = list()
        run_adv = 0
        for t in reversed(range(len(self))):
            if self.sards[t].done:
                run_adv = self.sards[t].reward - self.sards[t].state_value
            else:
                sigma = self.sards[t].reward + gamma * self.sards[t + 1].state_value - self.sards[t].state_value
                run_adv = sigma + run_adv * gamma * l

            self.sards[t].advantage = run_adv
            self.sards[t].target_value = self.sards[t].state_value + run_adv
            advs.append(run_adv)

        # adv_mean = np.mean(advs)
        # adv_std = np.std(advs)
        # for t in range(len(self)):
        #    self.sards[t].advantage = (self.sards[t].advantage - adv_mean) / adv_std

    def get_stats(self):
        stats = dict()
        stats['reward_mean'] = np.mean(self.eps_r)
        stats['reward_std'] = np.std(self.eps_r)
        stats['len_mean'] = np.mean(self.eps_l)
        stats['len_std'] = np.std(self.eps_l)
        return stats

    def show_stats(self):
        stats = self.get_stats()
        print("Reward mean: %.6f  |  Reward std: %.6f  |  Len mean: %.6f  |  Len std: %.6f" % \
              (stats["reward_mean"], stats["reward_std"], stats["len_mean"], stats["len_std"]))
