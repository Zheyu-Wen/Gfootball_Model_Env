import numpy as np
import torch
from torch import optim
from utils import select_actions, evaluate_actions, config_logger
from datetime import datetime
import os
import copy

class PPO_agent:
    def __init__(self, envs, args, net):
        """
        PPO agent init
        Including networks and saving configs
        """
        self.envs = envs 
        self.args = args
        self.net = net
        self.prev_net = copy.deepcopy(self.net)

        if self.args.cuda:
            self.net.cuda()
            self.prev_net.cuda()
        """
        Optimizer innitialization
        """
        self.optimizer = optim.Adam(self.net.parameters(), self.args.lr, eps=self.args.eps)
        # check saving folder
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # env folder
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        # logger folder
        if not os.path.exists(self.args.log_dir):
            os.mkdir(self.args.log_dir)
        self.log_path = self.args.log_dir + self.args.env_name + '.log'
        """
        PPO init
        """
        self.batch_ob_shape = (self.args.num_workers * self.args.nsteps, ) + self.envs.observation_space.shape
        self.obs = np.zeros((self.args.num_workers, ) + self.envs.observation_space.shape, dtype=self.envs.observation_space.dtype.name)
        self.obs[:] = self.envs.reset()
        self.dones = [False for _ in range(self.args.num_workers)]
        self.logger = config_logger(self.log_path)

    def _update_network(self, obs, actions, returns, advantages):
        inds = np.arange(obs.shape[0])
        nbatch_train = obs.shape[0] // self.args.batch_size
        for _ in range(self.args.epoch):
            """
            generate random indeces to be minibatch sample index
            """
            np.random.shuffle(inds)
            for start in range(0, obs.shape[0], nbatch_train):
                """
                sample minibatches
                """
                end = start + nbatch_train
                inds_vector = inds[start:end]
                obs_vec = obs[inds_vector]
                actions_vec = actions[inds_vector]
                returns_vec = returns[inds_vector]
                advs_vec = advantages[inds_vector]
                """
                Convert to tensor
                """
                obs_vec = self._get_tensors(obs_vec)
                actions_vec = torch.tensor(actions_vec, dtype=torch.float32)
                returns_vec = torch.tensor(returns_vec, dtype=torch.float32).unsqueeze(1)
                advs_vec = torch.tensor(advs_vec, dtype=torch.float32).unsqueeze(1)
                """
                advantage normalization
                """
                advs_vec = (advs_vec - advs_vec.mean()) / (advs_vec.std() + 1e-8)
                if self.args.cuda:
                    actions_vec = actions_vec.cuda()
                    returns_vec = returns_vec.cuda()
                    advs_vec = advs_vec.cuda()
                values_vec, pis = self.net(obs_vec)
                """
                value loss and policy loss computation
                """
                value_loss = (returns_vec - values_vec).pow(2).mean()
                with torch.no_grad():
                    _, prev_pis = self.prev_net(obs_vec)
                    prev_log_prob, _ = evaluate_actions(prev_pis, actions_vec)
                    prev_log_prob = prev_log_prob.detach()
                # evaluate the current policy
                log_prob, ent_loss = evaluate_actions(pis, actions_vec)
                prob_ratio = torch.exp(log_prob - prev_log_prob)
                surrogate1 = prob_ratio * advs_vec
                surrogate2 = torch.clamp(prob_ratio, 1 - self.args.clip, 1 + self.args.clip) * advs_vec
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                total_loss = policy_loss + self.args.vloss_coef * value_loss - ent_loss * self.args.ent_coef
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.max_grad_norm)
                # update
                self.optimizer.step()
        return policy_loss.item(), value_loss.item(), ent_loss.item()

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
        for param_group in self.optimizer.param_groups:
             param_group['lr'] = adjust_lr


    def train(self):
        num_updates = self.args.total_frames // (self.args.nsteps * self.args.num_workers)
        # get the reward to calculate other informations
        episode_rewards = torch.zeros([self.args.num_workers, 1])
        final_rewards = torch.zeros([self.args.num_workers, 1])
        for update in range(num_updates):
            """
            lr setup
            """
            if self.args.lr_decay:
                self._adjust_learning_rate(update, num_updates)
            """
            initialize contrainer for training date
            """
            obs_vector, rewards_vector, actions_vector, dones_vector, values_vector = [], [], [], [], []
            """
            training starts
            """
            for step in range(self.args.nsteps):
                with torch.no_grad():
                    obs_tensor = self._get_tensors(self.obs)
                    values, pis = self.net(obs_tensor)
                """
                sample actions from distribution
                """
                actions = select_actions(pis)
                input_actions = actions 
                """
                take selected actions and interact with the env 
                """
                obs_vector.append(np.copy(self.obs))
                actions_vector.append(actions)
                dones_vector.append(self.dones)
                values_vector.append(values.detach().cpu().numpy().squeeze())
                """
                store observations by taking the action
                """
                obs, rewards, dones, _ = self.envs.step(input_actions)
                # update dones
                self.dones = dones
                rewards_vector.append(rewards)
                """
                If obs = done, clear observation
                """
                for n, done in enumerate(dones):
                    if done:
                        self.obs[n] = self.obs[n] * 0
                self.obs = obs
                """
                reward processing 
                """
                rewards = torch.tensor(np.expand_dims(np.stack(rewards), 1), dtype=torch.float32)
                episode_rewards += rewards
                masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in dones], dtype=torch.float32)
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks


            """
            Update network using collected data
            """
            obs_vector = np.asarray(obs_vector, dtype=np.float32)
            rewards_vector = np.asarray(rewards_vector, dtype=np.float32)
            actions_vector = np.asarray(actions_vector, dtype=np.float32)
            dones_vector = np.asarray(dones_vector, dtype=np.bool)
            values_vector = np.asarray(values_vector, dtype=np.float32)
            # compute the last state value
            with torch.no_grad():
                obs_tensor = self._get_tensors(self.obs)
                last_values, _ = self.net(obs_tensor)
                last_values = last_values.detach().cpu().numpy().squeeze()
            # start to compute advantages...
            returns_vector = np.zeros_like(rewards_vector)
            advs_vector = np.zeros_like(rewards_vector)
            lastgaelam = 0
            """
            PPO loss calculation
            """
            for t in reversed(range(self.args.nsteps)):
                if t == self.args.nsteps - 1:
                    next_non_terminal = 1.0 - self.dones
                    next_values = last_values
                else:
                    next_non_terminal = 1.0 - dones_vector[t + 1]
                    next_values = values_vector[t + 1]
                delta = rewards_vector[t] + self.args.gamma * next_values * next_non_terminal - values_vector[t]
                advs_vector[t] = lastgaelam = delta + self.args.gamma * self.args.tau * next_non_terminal * lastgaelam

            returns_vector = advs_vector + values_vector
            """
            reshape data, update network
            """
            obs_vector = obs_vector.swapaxes(0, 1).reshape(self.batch_ob_shape)
            actions_vector = actions_vector.swapaxes(0, 1).flatten()
            returns_vector = returns_vector.swapaxes(0, 1).flatten()
            advs_vector = advs_vector.swapaxes(0, 1).flatten()
            self.prev_net.load_state_dict(self.net.state_dict())
            pl, vl, ent = self._update_network(obs_vector, actions_vector, returns_vector, advs_vector)
            """
            Info display
            """
            if update % self.args.display_interval == 0:
                self.logger.info('[{}] Update: {} / {}, Frames: {}, Rewards: {:.3f}, Min: {:.3f}, Max: {:.3f}, PL: {:.3f},'\
                    'VL: {:.3f}, Ent: {:.3f}'.format(datetime.now(), update, num_updates, (update + 1)*self.args.nsteps*self.args.num_workers, \
                    final_rewards.mean().item(), final_rewards.min().item(), final_rewards.max().item(), pl, vl, ent))
                torch.save(self.net.state_dict(), self.model_path + '/model.pt')


