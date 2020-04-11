import numpy as np
import torch
from torch import optim
from utils import select_actions, evaluate_actions, config_logger
from datetime import datetime
import os
import copy
import torch.nn as nn

class ddqn_agent:
    def __init__(self, envs, args, net, env_net=None):
        self.envs = envs
        self.args = args
        # define the newtork...
        self.net = net
        self.old_net = copy.deepcopy(self.net)
        # if use the cuda...
        if self.args.cuda:
            self.net.cuda()
            self.old_net.cuda()
        # define the optimizer...
        self.optimizer = optim.Adam(self.net.parameters(), self.args.lr, eps=self.args.eps)
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
        self.obs = np.zeros((self.args.num_workers, ) + self.envs.observation_space.shape, dtype=self.envs.observation_space.dtype.name)
        self.obs[:] = self.envs.reset()
        self.dones = [False for _ in range(self.args.num_workers)]
        self.logger = config_logger(self.log_path)
        self.env_net = env_net
        self.policy_optimizer = optim.Adam(self.net.parameters(), lr=1e-5)
        self.env_optimizer = optim.Adam(self.env_net.parameters(), lr=1e-5)
        self.obs_shape = envs.observation_space.shape

    # start to train the network...
    def learn(self):
        num_updates = self.args.total_frames // (self.args.nsteps * self.args.num_workers)
        # get the reward to calculate other informations
        episode_rewards = torch.zeros([self.args.num_workers, 1])
        final_rewards = torch.zeros([self.args.num_workers, 1])
        reward_hist = []
        policy_loss_hist = []
        env_loss_hist = []
        for update in range(num_updates):
            mb_obs, mb_rewards, mb_actions, mb_dones, mb_values, Qvalues = [], [], [], [], [], []
            if self.args.lr_decay:
                self._adjust_learning_rate(update, num_updates)
            for step in range(self.args.nsteps):
                with torch.no_grad():
                    # get tensors
                    obs_tensor = self._get_tensors(self.obs)
                    _, q = self.net(obs_tensor)
                # select actions
                actions = np.zeros([8, ])
                # for i in range(8):
                    # if np.abs(np.random.randn(1)) < 0.3:
                    #     actions[i] = np.int(np.random.choice(np.arange(self.envs.action_space.n)))
                    # else:
                actions = select_actions(q)
                # get the input actions
                input_actions = actions
                # start to store information
                mb_obs.append(np.copy(self.obs))
                mb_actions.append(actions)
                mb_dones.append(self.dones)
                # start to excute the actions in the environment
                obs, rewards, dones, _ = self.envs.step(input_actions)
                # update dones
                self.dones = dones
                mb_rewards.append(rewards)
                # clear the observation
                for n, done in enumerate(dones):
                    if done:
                        self.obs[n] = self.obs[n] * 0
                self.obs = obs
                # process the rewards part -- display the rewards on the screen
                rewards = torch.tensor(np.expand_dims(np.stack(rewards), 1), dtype=torch.float32)
                episode_rewards += rewards
                masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in dones], dtype=torch.float32)
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks
            # process the rollouts
            mb_obs = np.asarray(mb_obs, dtype=np.float32)
            Qvalues = np.asarray(Qvalues, dtype=np.float32)
            mb_obs = mb_obs.swapaxes(0, 1).reshape(self.batch_ob_shape)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            # before update the network, the old network will try to load the weights
            self.old_net.load_state_dict(self.net.state_dict())
            # start to update the network
            policy_loss = self._update_network(mb_obs, mb_rewards)
            policy_loss_hist.append(policy_loss)
            # display the training information
            reward_hist.append(final_rewards.mean().detach().cpu().numpy())
            if update % self.args.display_interval == 0:
                self.logger.info('[{}] Update: {} / {}, Frames: {}, Rewards: {:.3f}, Min: {:.3f}, Max: {:.3f}'
                                 .format(datetime.now(), update, num_updates, (update + 1)*self.args.nsteps*self.args.num_workers, \
                                final_rewards.mean().item(), final_rewards.min().item(), final_rewards.max().item()))
                # save the model
                torch.save(self.net.state_dict(), self.model_path + '/model.pt')

        return reward_hist, env_loss_hist, policy_loss_hist



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
        for param_group in self.policy_optimizer.param_groups:
             param_group['lr'] = adjust_lr

        # update the network
    def _update_network(self, obs, rewards):
        lens_data = np.minimum(len(obs), len(rewards))
        inds = np.arange(lens_data)
        nbatch_train = lens_data // self.args.batch_size
        criterion = nn.SmoothL1Loss()
        for _ in range(self.args.epoch):
            np.random.shuffle(inds)
            for start in range(0, lens_data, nbatch_train):
                # get the mini-batchs
                end = start + nbatch_train
                mbinds = inds[start:end]
                mb_obs = obs[mbinds]
                mb_rewards = rewards[mbinds]
                # convert minibatches to tensor
                mb_obs = self._get_tensors(mb_obs)
                _, q = self.net(mb_obs)
                action = select_actions(q)
                _, Q_sa_temp = self.old_net(mb_obs)
                Q_sa_temp = Q_sa_temp.detach().cpu().numpy()
                Q_sa = np.zeros_like(Q_sa_temp)
                Q_sa[np.arange(nbatch_train), action] = Q_sa_temp[np.arange(nbatch_train), action]
                target = mb_rewards + 0.9 * Q_sa
                _, qpred = self.net(mb_obs)
                loss = criterion(qpred, torch.tensor(target, dtype=torch.float32))
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.max_grad_norm)
                # update
                self.optimizer.step()
        return loss.detach().cpu().numpy()
