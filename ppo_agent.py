from torch.distributions.categorical import Categorical
import random
import logging
import numpy as np
import torch
from torch import optim
from datetime import datetime
import os
import copy

class ppo_agent:
    def __init__(self, envs, args, policy_net, env_net):
        self.envs = envs
        self.args = args
        self.main_net = policy_net
        self.target_net = copy.deepcopy(policy_net)
        self.env_net = env_net
        self.policy_optimizer = optim.Adam(self.main_net.parameters(), lr=1e-5)
        self.env_optimizer = optim.Adam(self.env_net.parameters(), lr=1e-5)
        self.obs_init = envs.reset()
        self.obs_shape = envs.observation_space.shape
        if not os.path.exists(args.policy_model_dir):
            os.makedirs(args.policy_model_dir)
        if not os.path.exists(args.env_model_dir):
            os.makedirs(args.env_model_dir)

    # replay
    def collect_train_data(self, num_hist_max):
        reward_hist, actions_hist, value_hist = [], [], []
        m, n, p = self.envs.observation_space.shape
        obs_hist = np.zeros([num_hist_max, p, m, n])
        obs = self.obs_init
        obs = np.random.randn(*self.envs.observation_space.shape)
        dones = 0
        num_hist = 0
        for iters in range(num_hist_max):
            if dones==1:
                self.envs.reset()
            num_hist += 1
            with torch.no_grad():
                obs = self._get_tensors(obs)
                values, pis = self.main_net(obs)
                obs_hist[iters, :, :, :] = obs
            # exploration and exploitation tradeoff
            # if np.random.choice(np.arange(10))/10 < 0.3:
            #     actions = np.zeros([obs.shape[0], 1])
            #     for i in range(obs.shape[0]):
            #         actions[i, :] = np.int(np.random.choice(self.envs.action_space.n))
            try:
                actions = np.int(self.select_actions(pis))
            except:
                pis = torch.tensor(np.ones_like(pis.detach().cpu().numpy())/2)
                actions = np.int(self.select_actions(pis))
            actions_hist.append(actions)
            value_hist.append(values.detach().cpu().numpy().squeeze())
            obs, rewards, dones, _ = self.envs.step(actions)
            reward_hist.append(rewards)

        reward_hist = np.array(reward_hist, dtype=np.float32)
        actions_hist = np.array(actions_hist, dtype=np.float32)
        value_hist = np.array(value_hist, dtype=np.float32)
        with torch.no_grad():
            obs_tensor = self._get_tensors(obs)
            last_values, _ = self.main_net(obs_tensor)
            last_values = last_values.detach().cpu().numpy().squeeze()
        # start to compute advantages...
        adv_hist = np.zeros_like(reward_hist)
        lastgaelam = 0
        for t in reversed(range(num_hist)):
            if t == num_hist - 1:
                nextnonterminal = 0
                nextvalues = last_values
            else:
                nextnonterminal = 1.0
                nextvalues = value_hist[t + 1]
            delta = reward_hist[t] + self.args.gamma * nextvalues * nextnonterminal - value_hist[t]
            adv_hist[t] = lastgaelam = delta + self.args.gamma * self.args.tau * nextnonterminal * lastgaelam
        returns = adv_hist + value_hist
        # after compute the returns, let's process the rollouts
        obs_hist = obs_hist[:num_hist, :, :, :]
        obs_hist = obs_hist.reshape(num_hist, -1)
        actions_hist = actions_hist.reshape(num_hist, 1)
        returns = returns.reshape(num_hist, 1)
        adv_hist = adv_hist.reshape(num_hist, 1)
        # before update the network, the old network will try to load the weights
        return obs_hist, actions_hist, returns, adv_hist, reward_hist


    # update the network
    def _update_network_by_env_net(self, obs, actions, reward):
        inds = np.arange(obs.shape[0])
        nbatch_train = obs.shape[0] // self.args.batch_size
        BCE_loss = torch.nn.BCELoss()
        L1_loss = torch.nn.L1Loss()
        if np.mean(reward)>0:
            env_train_time = 10
        else:
            env_train_time = 1
        policy_train_time = 1
        for _ in range(self.args.epoch):
            np.random.shuffle(inds)
            for start in range(0, obs.shape[0], nbatch_train):
                # get the mini-batchs
                end = start + nbatch_train
                binds = inds[start:end]
                obs_temp = obs[binds]
                actions_temp = actions[binds]
                reward_temp = reward[binds]
                obs_temp = torch.tensor(obs_temp, dtype=torch.float32)
                actions_temp = torch.tensor(actions_temp, dtype=torch.float32)
                reward_temp = torch.tensor(reward_temp, dtype=torch.float32)

                self.env_optimizer.zero_grad()
                pred_reward = self.env_net(obs_temp, actions_temp)
                env_loss = BCE_loss(pred_reward, reward_temp.reshape(-1, 1))
                torch.nn.utils.clip_grad_norm_(self.env_net.parameters(), self.args.max_grad_norm)
                env_loss.backward()
                self.env_optimizer.step()
                # if np.mean(reward) > 0:

                obs_shape = (-1, ) + self.obs_shape
                obs_temp2 = np.reshape(obs_temp.detach().cpu().numpy(), obs_shape)
                obs_temp2 = np.transpose(obs_temp2, (0, 3, 1, 2))
                obs_temp2 = torch.tensor(obs_temp2, dtype=torch.float32)

                _, pis = self.main_net(obs_temp2)
                # if np.random.choice(np.arange(10)) / 10 < 0.3:
                #     pred_action = np.zeros([obs_temp.shape[0], 1])
                #     for i in range(obs_temp.shape[0]):
                #         pred_action[i, :] = np.int(np.random.choice(self.envs.action_space.n))
                # else:
                try:
                    pred_action = self.select_actions(pis)
                except:
                    pis = torch.tensor(np.ones_like(pis.detach().cpu().numpy())/2)
                    pred_action = self.select_actions(pis)
                pred_action = np.reshape(pred_action, [-1, 1])
                pred_action = torch.tensor(pred_action, dtype=torch.float32)
                corresponding_reward = self.env_net(obs_temp, pred_action)
                policy_loss = BCE_loss(corresponding_reward, torch.ones_like(corresponding_reward))
                torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), self.args.max_grad_norm)
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
                torch.save(self.main_net.state_dict(), self.args.policy_model_dir + '/policy_net.pth')
        # if np.mean(reward) > 0:
        return env_loss.item(), policy_loss.item()
        # else:
        #     return env_loss.item()


    def _update_network_wo_env_net(self, obs, actions, returns, advantages):
        inds = np.arange(obs.shape[0])
        nbatch_train = obs.shape[0] // self.args.batch_size
        for _ in range(self.args.epoch):
            np.random.shuffle(inds)
            for start in range(0, obs.shape[0], nbatch_train):
                # get the mini-batchs
                end = start + nbatch_train
                binds = inds[start:end]
                obs_temp = obs[binds]
                actions_temp = actions[binds]
                returns_temp = returns[binds]
                advs_temp = advantages[binds]
                # convert minibatches to tensor
                obs_temp = self._get_tensors_ppo(obs_temp)
                actions_temp = torch.tensor(actions_temp, dtype=torch.float32)
                returns_temp = torch.tensor(returns_temp, dtype=torch.float32).unsqueeze(1)
                advs_temp = torch.tensor(advs_temp, dtype=torch.float32).unsqueeze(1)
                # normalize adv
                advs_temp = (advs_temp - advs_temp.mean()) / (advs_temp.std() + 1e-8)
                # start to get values
                values, pis = self.main_net(obs_temp)
                # start to calculate the value loss...
                value_loss = (returns_temp - values).pow(2).mean()
                # start to calculate the policy loss
                with torch.no_grad():
                    _, old_pis = self.target_net(obs_temp)
                    # get the old log probs
                    old_log_prob, _ = self.evaluate_actions(old_pis, actions_temp)
                    old_log_prob = old_log_prob.detach()
                # evaluate the current policy
                log_prob, ent_loss = self.evaluate_actions(pis, actions_temp)
                prob_ratio = torch.exp(log_prob - old_log_prob)
                # surr1
                surr1 = prob_ratio * advs_temp
                surr2 = torch.clamp(prob_ratio, 1 - self.args.clip, 1 + self.args.clip) * advs_temp
                policy_loss = -torch.min(surr1, surr2).mean()
                # final total loss
                total_loss = policy_loss + self.args.vloss_coef * value_loss - ent_loss * self.args.ent_coef
                # clear the grad buffer
                self.policy_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), self.args.max_grad_norm)
                # update
                self.policy_optimizer.step()
        return policy_loss.item(), value_loss.item(), ent_loss.item()

    # convert the numpy array to tensors
    def _get_tensors(self, obs):
        obs_tensor = torch.tensor(np.transpose(obs, (2, 0, 1)), dtype=torch.float32)
        m, n, p = obs_tensor.shape
        obs_tensor = obs_tensor.reshape([1, m, n, p])
        # decide if put the tensor on the GPU
        return obs_tensor
    def _get_tensors_ppo(self, obs):
        b, _ = obs.shape
        obs_shape = (b, ) + self.obs_shape
        obs_tensor = torch.tensor(np.transpose(np.reshape(obs, obs_shape), (0, 3, 1, 2)), dtype=torch.float32)
        return obs_tensor
    # select actions
    def select_actions(self, pi):
        actions = Categorical(pi).sample()
        # return actions
        return actions.detach().cpu().numpy().squeeze()

    # evaluate actions
    def evaluate_actions(self, pi, actions):
        cate_dist = Categorical(pi)
        log_prob = cate_dist.log_prob(actions).unsqueeze(-1)
        entropy = cate_dist.entropy().mean()
        return log_prob, entropy

