import numpy as np
import torch
from torch import optim
from utils import select_actions, evaluate_actions, config_logger
from datetime import datetime
import os
import copy

class Compation_agent:
    def __init__(self, envs, args, net, env_net=None):
        self.envs = envs
        self.args = args
        # define the newtork...
        self.player1 = net
        self.player1_old = copy.deepcopy(self.player1)
        self.player2 = net
        self.player2_old = copy.deepcopy(self.player2)
        # if use the cuda...
        if self.args.cuda:
            self.player1.cuda()
            self.player1_old.cuda()
            self.player2.cuda()
            self.player2_old.cuda()
        # define the optimizer...
        self.optimizer_player1 = optim.Adam(self.player1.parameters(), self.args.lr, eps=self.args.eps)
        self.optimizer_player2 = optim.Adam(self.player2.parameters(), self.args.lr, eps=self.args.eps)

        # check saving folder..
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # env folder..
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        # logger folder
        if not os.path.exists(self.args.log_dir):
            os.mkdir(self.args.log_dir)
        self.log_path = self.args.log_dir + self.args.env_name + '.log'
        # get the observation
        self.batch_ob_shape = (self.args.num_workers * self.args.nsteps, ) + self.envs.observation_space.shape
        self.obs1 = np.zeros((self.args.num_workers, ) + self.envs.observation_space.shape, dtype=self.envs.observation_space.dtype.name)
        self.obs1[:] = self.envs.reset()
        self.dones1 = [False for _ in range(self.args.num_workers)]
        self.obs2 = np.zeros((self.args.num_workers,) + self.envs.observation_space.shape,
                             dtype=self.envs.observation_space.dtype.name)
        self.obs2[:] = self.envs.reset()
        self.dones2 = [False for _ in range(self.args.num_workers)]
        self.logger = config_logger(self.log_path)
        self.obs_shape = envs.observation_space.shape

    # start to train the network...
    def learn(self):
        num_updates = self.args.total_frames // (self.args.nsteps * self.args.num_workers)
        # get the reward to calculate other informations
        episode_rewards = torch.zeros([self.args.num_workers, 1])
        final_rewards = torch.zeros([self.args.num_workers, 1])
        reward_hist = []
        player1_loss_hist = []
        player2_loss_hist = []
        players = ['player1', 'player2']
        for update in range(num_updates):
            for player in players:
                if player == 'player1':
                    mb_obs1, mb_rewards1, mb_actions1, mb_dones1, mb_values1 = [], [], [], [], []
                    mb_rewards2, mb_values2, mb_dones2 = [], [], []
                    if self.args.lr_decay:
                        self._adjust_learning_rate(update, num_updates)
                    for step in range(self.args.nsteps):
                        with torch.no_grad():
                            # get tensors
                            obs_tensor = self._get_tensors(self.obs1)
                            values1, pis1 = self.player1(obs_tensor)
                            values2, pis2 = self.player2(obs_tensor)
                        # select actions
                        actions1 = select_actions(pis1)
                        actions2 = select_actions(pis2)

                        # get the input actions
                        input_actions1 = actions1
                        input_actions2 = actions2

                        # start to store information
                        mb_obs1.append(np.copy(self.obs1))
                        mb_actions1.append(actions1)
                        mb_dones1.append(self.dones1)
                        mb_dones2.append(self.dones2)
                        mb_values1.append(values1.detach().cpu().numpy().squeeze())
                        mb_values2.append(values2.detach().cpu().numpy().squeeze())
                        # start to excute the actions in the environment
                        obs1, rewards1, dones1, _ = self.envs.step(input_actions1)
                        obs2, rewards2, dones2, _ = self.envs.step(input_actions2)

                        # update dones
                        self.dones1 = dones1
                        mb_rewards1.append(rewards1)
                        # clear the observation
                        for n, done in enumerate(dones1):
                            if done:
                                self.obs1[n] = self.obs1[n] * 0
                        self.obs1 = obs1

                        self.dones2 = dones2
                        mb_rewards2.append(rewards2)
                        # clear the observation
                        for n, done in enumerate(dones2):
                            if done:
                                self.obs2[n] = self.obs2[n] * 0
                        self.obs2 = obs2
                        # process the rewards part -- display the rewards on the screen
                        rewards1 = torch.tensor(np.expand_dims(np.stack(rewards1), 1), dtype=torch.float32)
                        episode_rewards += rewards1
                        masks1 = torch.tensor([[0.0] if done_ else [1.0] for done_ in dones1], dtype=torch.float32)
                        final_rewards *= masks1
                        final_rewards += (1 - masks1) * episode_rewards
                        episode_rewards *= masks1
                    # process the rollouts
                    mb_obs1 = np.asarray(mb_obs1, dtype=np.float32)
                    mb_rewards1 = np.asarray(mb_rewards1, dtype=np.float32)
                    mb_rewards2 = np.asarray(mb_rewards2, dtype=np.float32)
                    mb_actions1 = np.asarray(mb_actions1, dtype=np.float32)
                    mb_dones1 = np.asarray(mb_dones1, dtype=np.bool)
                    mb_values1 = np.asarray(mb_values1, dtype=np.float32)
                    mb_values2 = np.asarray(mb_values2, dtype=np.float32)
                    mb_dones2 = np.asarray(mb_dones2, dtype=np.float32)
                    # compute the last state value
                    with torch.no_grad():
                        obs_tensor = self._get_tensors(self.obs1)
                        last_values1, _ = self.player1(obs_tensor)
                        last_values1 = last_values1.detach().cpu().numpy().squeeze()
                    with torch.no_grad():
                        obs_tensor = self._get_tensors(self.obs2)
                        last_values2, _ = self.player2(obs_tensor)
                        last_values2 = last_values2.detach().cpu().numpy().squeeze()
                    # start to compute advantages...
                    mb_advs1 = np.zeros_like(mb_rewards1)
                    lastgaelam = 0
                    for t in reversed(range(self.args.nsteps)):
                        if t == self.args.nsteps - 1:
                            nextnonterminal = 1.0 - self.dones1
                            nextvalues = last_values1
                        else:
                            nextnonterminal = 1.0 - mb_dones1[t + 1]
                            nextvalues = mb_values1[t + 1]
                        delta = mb_rewards1[t] + self.args.gamma * nextvalues * nextnonterminal - mb_values1[t]
                        mb_advs1[t] = lastgaelam = delta + self.args.gamma * self.args.tau * nextnonterminal * lastgaelam
                    mb_returns1 = mb_advs1 + mb_values1

                    mb_advs2 = np.zeros_like(mb_rewards2)
                    lastgaelam = 0
                    for t in reversed(range(self.args.nsteps)):
                        if t == self.args.nsteps - 1:
                            nextnonterminal = 1.0 - self.dones2
                            nextvalues = last_values2
                        else:
                            nextnonterminal = 1.0 - mb_dones2[t + 1]
                            nextvalues = mb_values2[t + 1]
                        delta = mb_rewards2[t] + self.args.gamma * nextvalues * nextnonterminal - mb_values2[t]
                        mb_advs2[t] = lastgaelam = delta + self.args.gamma * self.args.tau * nextnonterminal * lastgaelam
                    mb_returns2 = mb_advs2 + mb_values2
                    adv_1over2 = np.maximum(mb_returns1 - mb_returns2, 0)
                    # after compute the returns, let's process the rollouts
                    mb_obs1 = mb_obs1.swapaxes(0, 1).reshape(self.batch_ob_shape)
                    mb_actions1 = mb_actions1.swapaxes(0, 1).flatten()
                    mb_returns1 = mb_returns1.swapaxes(0, 1).flatten()
                    mb_advs1 = mb_advs1.swapaxes(0, 1).flatten()
                    adv_1over2 = adv_1over2.swapaxes(0, 1).flatten()
                    # before update the network, the old network will try to load the weights
                    self.player1_old.load_state_dict(self.player1.state_dict())
                    # start to update the network
                    pl, vl, ent = self._update_player1(mb_obs1, mb_actions1, mb_returns1, mb_advs1, adv_1over2)
                    # display the training information
                    reward_hist.append(final_rewards.mean().detach().cpu().numpy())
                    player1_loss_hist.append(pl)

                else:
                    mb_obs2, mb_rewards2, mb_actions2, mb_dones2, mb_values2 = [], [], [], [], []
                    mb_rewards1, mb_values1, mb_dones1 = [], [], []
                    if self.args.lr_decay:
                        self._adjust_learning_rate(update, num_updates)
                    for step in range(self.args.nsteps):
                        with torch.no_grad():
                            # get tensors
                            obs_tensor = self._get_tensors(self.obs1)
                            values1, pis1 = self.player1(obs_tensor)
                            values2, pis2 = self.player2(obs_tensor)
                        # select actions
                        actions1 = select_actions(pis1)
                        actions2 = select_actions(pis2)

                        # get the input actions
                        input_actions1 = actions1
                        input_actions2 = actions2

                        # start to store information
                        mb_obs2.append(np.copy(self.obs2))
                        mb_actions2.append(actions2)
                        mb_dones1.append(self.dones1)
                        mb_dones2.append(self.dones2)
                        mb_values1.append(values1.detach().cpu().numpy().squeeze())
                        mb_values2.append(values2.detach().cpu().numpy().squeeze())
                        # start to excute the actions in the environment
                        obs1, rewards1, dones1, _ = self.envs.step(input_actions1)
                        obs2, rewards2, dones2, _ = self.envs.step(input_actions2)

                        # update dones
                        self.dones1 = dones1
                        mb_rewards1.append(rewards1)
                        # clear the observation
                        for n, done in enumerate(dones1):
                            if done:
                                self.obs1[n] = self.obs1[n] * 0
                        self.obs1 = obs1

                        self.dones2 = dones2
                        mb_rewards2.append(rewards2)
                        # clear the observation
                        for n, done in enumerate(dones2):
                            if done:
                                self.obs2[n] = self.obs2[n] * 0
                        self.obs2 = obs2
                        # process the rewards part -- display the rewards on the screen
                        rewards2 = torch.tensor(np.expand_dims(np.stack(rewards2), 1), dtype=torch.float32)
                        episode_rewards += rewards2
                        masks2 = torch.tensor([[0.0] if done_ else [1.0] for done_ in dones2], dtype=torch.float32)
                        final_rewards *= masks2
                        final_rewards += (1 - masks2) * episode_rewards
                        episode_rewards *= masks2
                    # process the rollouts
                    mb_obs2 = np.asarray(mb_obs2, dtype=np.float32)
                    mb_rewards2 = np.asarray(mb_rewards2, dtype=np.float32)
                    mb_actions2 = np.asarray(mb_actions2, dtype=np.float32)
                    mb_dones1 = np.asarray(mb_dones1, dtype=np.bool)
                    mb_dones2 = np.asarray(mb_dones2, dtype=np.bool)
                    mb_values1 = np.asarray(mb_values1, dtype=np.float32)
                    mb_values2 = np.asarray(mb_values2, dtype=np.float32)
                    mb_rewards1 = np.asarray(mb_rewards1, dtype=np.float32)
                    # compute the last state value
                    with torch.no_grad():
                        obs_tensor = self._get_tensors(self.obs1)
                        last_values1, _ = self.player1(obs_tensor)
                        last_values1 = last_values1.detach().cpu().numpy().squeeze()
                    with torch.no_grad():
                        obs_tensor = self._get_tensors(self.obs2)
                        last_values2, _ = self.player2(obs_tensor)
                        last_values2 = last_values2.detach().cpu().numpy().squeeze()
                    # start to compute advantages...
                    mb_returns1 = np.zeros_like(mb_rewards1)
                    mb_advs1 = np.zeros_like(mb_rewards1)
                    lastgaelam = 0
                    for t in reversed(range(self.args.nsteps)):
                        if t == self.args.nsteps - 1:
                            nextnonterminal = 1.0 - self.dones1
                            nextvalues = last_values1
                        else:
                            nextnonterminal = 1.0 - mb_dones1[t + 1]
                            nextvalues = mb_values1[t + 1]
                        delta = mb_rewards1[t] + self.args.gamma * nextvalues * nextnonterminal - mb_values1[t]
                        mb_advs1[t] = lastgaelam = delta + self.args.gamma * self.args.tau * nextnonterminal * lastgaelam
                    mb_returns1 = mb_advs1 + mb_values1

                    mb_advs2 = np.zeros_like(mb_rewards2)
                    lastgaelam = 0
                    for t in reversed(range(self.args.nsteps)):
                        if t == self.args.nsteps - 1:
                            nextnonterminal = 1.0 - self.dones2
                            nextvalues = last_values2
                        else:
                            nextnonterminal = 1.0 - mb_dones2[t + 1]
                            nextvalues = mb_values2[t + 1]
                        delta = mb_rewards2[t] + self.args.gamma * nextvalues * nextnonterminal - mb_values2[t]
                        mb_advs2[t] = lastgaelam = delta + self.args.gamma * self.args.tau * nextnonterminal * lastgaelam
                    mb_returns2 = mb_advs2 + mb_values1
                    adv_2over1 = np.maximum(mb_returns2 - mb_returns1, 0)
                    # after compute the returns, let's process the rollouts
                    mb_obs2 = mb_obs2.swapaxes(0, 1).reshape(self.batch_ob_shape)
                    mb_actions2 = mb_actions2.swapaxes(0, 1).flatten()
                    mb_returns2 = mb_returns2.swapaxes(0, 1).flatten()
                    mb_advs2 = mb_advs2.swapaxes(0, 1).flatten()
                    adv_2over1 = adv_2over1.swapaxes(0, 1).flatten()
                    # before update the network, the old network will try to load the weights
                    self.player2_old.load_state_dict(self.player2.state_dict())
                    # start to update the network
                    pl, vl, ent = self._update_player2(mb_obs2, mb_actions2, mb_returns2, mb_advs2, adv_2over1)
                    # display the training information
                    reward_hist.append(final_rewards.mean().detach().cpu().numpy())
                    player2_loss_hist.append(pl)
            if update % self.args.display_interval == 0:
                self.logger.info('[{}] Update: {} / {}, Frames: {}, Rewards: {:.3f}, Min: {:.3f}, Max: {:.3f}'
                                 .format(datetime.now(), update, num_updates, (update + 1)*self.args.nsteps*self.args.num_workers, \
                                final_rewards.mean().item(), final_rewards.min().item(), final_rewards.max().item()))
                # save the model
                torch.save(self.player1.state_dict(), self.model_path + '/model.pt')
        return reward_hist, player1_loss_hist, player2_loss_hist



    # convert the numpy array to tensors
    def _get_tensors(self, obs):
        obs_tensor = torch.tensor(np.transpose(obs, (0, 3, 1, 2)), dtype=torch.float32)
        # decide if put the tensor on the GPU
        if self.args.cuda:
            obs_tensor = obs_tensor.cuda()
        return obs_tensor

    # adjust the learning rate
    def _adjust_learning_rate(self, update, num_updates):
        lr_frac = 1 - (update / num_updates)
        adjust_lr = self.args.lr * lr_frac
        for param_group in self.optimizer_player1.param_groups:
             param_group['lr'] = adjust_lr
        for param_group in self.optimizer_player2.param_groups:
             param_group['lr'] = adjust_lr

        # update the network
    def _update_player1(self, obs, actions, returns, advantages, adv_1over2):
        inds = np.arange(obs.shape[0])
        nbatch_train = obs.shape[0] // self.args.batch_size

        for _ in range(self.args.epoch):
            np.random.shuffle(inds)
            for start in range(0, obs.shape[0], nbatch_train):
                # get the mini-batchs
                end = start + nbatch_train
                mbinds = inds[start:end]
                mb_obs = obs[mbinds]
                mb_actions = actions[mbinds]
                mb_returns = returns[mbinds]
                mb_advs = advantages[mbinds]
                mb_adv_1over2 = adv_1over2[mbinds]
                # convert minibatches to tensor
                mb_obs = self._get_tensors(mb_obs)
                mb_actions = torch.tensor(mb_actions, dtype=torch.float32)
                mb_returns = torch.tensor(mb_returns, dtype=torch.float32).unsqueeze(1)
                mb_advs = torch.tensor(mb_advs, dtype=torch.float32).unsqueeze(1)
                mb_adv_1over2 = torch.tensor(mb_adv_1over2, dtype=torch.float32).unsqueeze(1)
                # normalize adv
                mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)
                if self.args.cuda:
                    mb_actions = mb_actions.cuda()
                    mb_returns = mb_returns.cuda()
                    mb_advs = mb_advs.cuda()
                    mb_adv_1over2 = mb_adv_1over2.cuda()
                # start to get values
                mb_values, pis = self.player1(mb_obs)
                # start to calculate the value loss...
                value_loss = (mb_returns - mb_values).pow(2).mean()
                # start to calculate the policy loss
                with torch.no_grad():
                    _, old_pis = self.player1_old(mb_obs)
                    # get the old log probs
                    old_log_prob, _ = evaluate_actions(old_pis, mb_actions)
                    old_log_prob = old_log_prob.detach()
                # evaluate the current policy
                log_prob, ent_loss = evaluate_actions(pis, mb_actions)
                prob_ratio = torch.exp(log_prob - old_log_prob)
                # surr1
                surr1 = prob_ratio * mb_advs
                surr2 = torch.clamp(prob_ratio, 1 - self.args.clip, 1 + self.args.clip) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()
                # final total loss
                total_loss = - ent_loss * self.args.ent_coef - mb_adv_1over2 * 0.5 # + policy_loss + self.args.vloss_coef * value_loss
                # clear the grad buffer
                self.optimizer_player1.zero_grad()
                total_loss.sum().backward()
                torch.nn.utils.clip_grad_norm_(self.player1.parameters(), self.args.max_grad_norm)
                # update
                self.optimizer_player1.step()
        return policy_loss.item(), value_loss.item(), ent_loss.item()


    def _update_player2(self, obs, actions, returns, advantages, adv_2over1):
        inds = np.arange(obs.shape[0])
        nbatch_train = obs.shape[0] // self.args.batch_size

        for _ in range(self.args.epoch):
            np.random.shuffle(inds)
            for start in range(0, obs.shape[0], nbatch_train):
                # get the mini-batchs
                end = start + nbatch_train
                mbinds = inds[start:end]
                mb_obs = obs[mbinds]
                mb_actions = actions[mbinds]
                mb_returns = returns[mbinds]
                mb_advs = advantages[mbinds]
                mb_adv_2over1 = adv_2over1[mbinds]
                # convert minibatches to tensor
                mb_obs = self._get_tensors(mb_obs)
                mb_actions = torch.tensor(mb_actions, dtype=torch.float32)
                mb_returns = torch.tensor(mb_returns, dtype=torch.float32).unsqueeze(1)
                mb_advs = torch.tensor(mb_advs, dtype=torch.float32).unsqueeze(1)
                mb_adv_2over1 = torch.tensor(mb_adv_2over1, dtype=torch.float32).unsqueeze(1)
                # normalize adv
                mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)
                if self.args.cuda:
                    mb_actions = mb_actions.cuda()
                    mb_returns = mb_returns.cuda()
                    mb_advs = mb_advs.cuda()
                    mb_adv_2over1 = mb_adv_2over1.cuda()
                # start to get values
                mb_values, pis = self.player2(mb_obs)
                # start to calculate the value loss...
                value_loss = (mb_returns - mb_values).pow(2).mean()
                # start to calculate the policy loss
                with torch.no_grad():
                    _, old_pis = self.player2_old(mb_obs)
                    # get the old log probs
                    old_log_prob, _ = evaluate_actions(old_pis, mb_actions)
                    old_log_prob = old_log_prob.detach()
                # evaluate the current policy
                log_prob, ent_loss = evaluate_actions(pis, mb_actions)
                prob_ratio = torch.exp(log_prob - old_log_prob)
                # surr1
                surr1 = prob_ratio * mb_advs
                surr2 = torch.clamp(prob_ratio, 1 - self.args.clip, 1 + self.args.clip) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()
                # final total loss
                total_loss = policy_loss + self.args.vloss_coef * value_loss - ent_loss * self.args.ent_coef + (mb_values - mb_adv_2over1) * 0.5
                # clear the grad buffer
                self.optimizer_player2.zero_grad()
                total_loss.sum().backward()
                torch.nn.utils.clip_grad_norm_(self.player1.parameters(), self.args.max_grad_norm)
                # update
                self.optimizer_player1.step()
        return policy_loss.item(), value_loss.item(), ent_loss.item()
