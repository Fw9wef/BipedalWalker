import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import gym
from env import *
import math
import sys
from time import time

gym.logger.set_level(40)


class A2C_policy(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(A2C_policy, self).__init__()

        self.lp = nn.Sequential(
            nn.Linear(input_shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU())

        self.mean_l = nn.Linear(64, n_actions[0])
        #self.mean_l.weight.data.mul_(0.1)

        #self.logstd = nn.Parameter(torch.zeros(n_actions[0]))
        self.register_buffer("logstd", -0.5*torch.ones(n_actions[0]))

    def forward(self, x):
        ot_n = self.lp(x.float())
        return torch.tanh(self.mean_l(ot_n))


class A2C_value(nn.Module):
    def __init__(self, input_shape):
        super(A2C_value, self).__init__()

        self.lp = nn.Sequential(
            nn.Linear(input_shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x):
        return self.lp(x.float())


class Actor(object):

    def __init__(self, actor_net=None, critic_net=None, env_name="BipedalWalker-v3", gpu_id=0,
                 max_iters=1600, l=0.95, gamma=0.99, epsilon=0.2):
        if gpu_id != 'cpu':
            self.device = torch.device("cuda:" + str(gpu_id))
        else:
            self.device = torch.device("cpu")
        self.env = gym.make(env_name)
        self.max_iters = max_iters
        self.l = l
        self.gamma = gamma
        self.epsilon = epsilon

        if actor_net is None:
            self.policy = A2C_policy(self.env.observation_space.shape, self.env.action_space.shape)
        else:
            self.policy = actor_net
        self.policy.to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=0.0004)

        if critic_net is None:
            self.value = A2C_value(self.env.observation_space.shape)
        else:
            self.value = critic_net
        self.value.to(self.device)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=0.001)

        self.pi = torch.ones((1,)).to(self.device) * math.pi

    @torch.no_grad()
    def run_episode(self, test_mode=False):
        self.policy.eval()
        self.value.eval()
        episode = Episode()
        state = self.env.reset()

        for i in range(self.max_iters):
            state = torch.Tensor(state).to(self.device)
            action_mean = self.policy(state)
            state_value = self.value(state)

            action = action_mean + torch.exp(self.policy.logstd) * torch.randn_like(self.policy.logstd)
            action = torch.clamp(action, -1, 1)
            action_log_prob = self.log_policy_prob(action_mean, self.policy.logstd, action)

            action = action.detach().cpu()
            new_state, reward, done, _ = self.env.step(action)
            state, state_value, action_log_prob = state.detach().cpu(), state_value.detach().cpu(), \
                                                  action_log_prob.detach().cpu()

            reward = 0.01 * reward
            if done:
                episode.add_sard(SARD(state, state_value, action, reward, done, action_log_prob))
                break
            else:
                episode.add_sard(SARD(state, state_value, action, reward, done, action_log_prob))

            state = new_state

        episode.compute_advantages()
        return episode

    def run(self, n_episodes=10, n_sards=None, queue=None, event=None):
        if queue is None:
            episodes = [self.run_episode() for _ in range(n_episodes)]
            return episodes

        for _ in range(n_episodes):
            episode = self.run_episode()
            stats = episode.get_episodes_stats([episode])
            sards = episode.get_random_sards(n_sards)

            #a = time()
            queue.put((sards, stats))
            #print("Put to queue")
        event.wait()

    def run_n_steps(self, n_steps):
        self.policy.eval()
        self.value.eval()
        episode = Sequence()
        state = self.env.reset()
        ep_reward = 0
        last_ep_reward = None
        steps_passed = 0

        while True:
            steps_passed += 1
            state = torch.Tensor(state).to(self.device)
            action_mean = self.policy(state)
            state_value = self.value(state)

            action = action_mean + torch.exp(self.policy.logstd) * torch.randn_like(self.policy.logstd)
            action = torch.clamp(action, -1, 1)
            action_log_prob = self.log_policy_prob(action_mean, self.policy.logstd, action)

            action = action.detach().cpu()
            new_state, reward, done, _ = self.env.step(action)
            state, state_value, action_log_prob = state.detach().cpu(), state_value.detach().cpu(), \
                                                  action_log_prob.detach().cpu()

            reward = 0.01 * reward
            if done:
                episode.add_sard(SARD(state, state_value, action, reward, done, action_log_prob))
                # episode.add_sard(SARD(state, state_value, action, 0, done, action_log_prob))
                state = self.env.reset()
                if steps_passed > n_steps:
                    break
            else:
                episode.add_sard(SARD(state, state_value, action, reward, done, action_log_prob))
                state = new_state

        episode.compute_advantages()
        return episode

    def log_policy_prob(self, mean, logstd, actions):
        # policy log probability
        std = torch.exp(logstd).clamp(min=1e-4)
        act_log_softmax = -0.5 * (((actions - mean) / std) ** 2) - torch.log(std) - 0.5 * torch.log(2 * self.pi)
        return act_log_softmax

    def get_grads(self, sards, queue=None):
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        self.policy.train()
        self.value.train()

        states, actions, values, target_values, adv, old_log_policy = SARD.sards2tensors(sards)
        states, actions, values, target_values, adv, old_log_policy = states.to(self.device), \
                                                                      actions.to(self.device), \
                                                                      values.to(self.device), \
                                                                      target_values.to(self.device), \
                                                                      adv.to(self.device), \
                                                                      old_log_policy.to(self.device)

        pred_mean = self.policy(states)
        new_log_policy = self.log_policy_prob(pred_mean, self.policy.logstd, actions)
        rt_theta = torch.exp(new_log_policy - old_log_policy)
        pg_loss = -torch.mean(
            torch.min(rt_theta * adv, torch.clamp(rt_theta, 1 - self.epsilon, 1 + self.epsilon) * adv))
        pg_loss.backward()
        policy_grads = list()
        for param in self.policy.parameters():
            policy_grads.append(torch.Tensor(param.grad.cpu().clone()))

        pred_values = self.value(states)
        loss_v = F.mse_loss(target_values, pred_values)
        loss_v.backward()
        #print("Loss p: %.6f\tLoss v: %.6f" % (pg_loss.cpu().item(), loss_v.cpu().item()))
        value_grads = list()
        for param in self.value.parameters():
            value_grads.append(torch.Tensor(param.grad.cpu().clone()))

        if queue is None:
            return policy_grads, value_grads
        else:
            queue.put((policy_grads, value_grads))

    def apply_grads(self, policy_grads, value_grads):
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        n_p_grads, n_v_grads = len(policy_grads), len(value_grads)

        for grads in policy_grads:
            for grad, param in zip(grads, self.policy.parameters()):
                param.grad += grad.to(self.device) / n_p_grads

        for grads in value_grads:
            for grad, param in zip(grads, self.value.parameters()):
                param.grad += grad.to(self.device) / n_v_grads

        self.policy_optimizer.step()
        self.value_optimizer.step()

    def get_weights(self):
        return self.policy.state_dict(), self.value.state_dict()

    def sync_nets(self, policy_state_dict, value_state_dict):
        self.policy.load_state_dict(policy_state_dict)
        self.value.load_state_dict(value_state_dict)
