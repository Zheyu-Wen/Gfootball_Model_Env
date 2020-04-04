import torch
from policy_net import cnn_net
from ppo_agent import ppo_agent
from ENV_net import env_net
import numpy as np

class evaluation:
    def __init__(self, model_dir, num_actions, envs, args):
        self.policy_net = cnn_net(num_actions)
        self.env_net = env_net(envs.observation_space.shape)
        self.policy_net.load_state_dict(torch.load(model_dir+"/policy_net.pth"))
        self.policy_net.eval()
        self.ppo_datacollector = ppo_agent(envs, args, self.policy_net, self.env_net)
        self.num_collected_data = np.int(1e3)
        envs.reset()
    def eval(self):
        obs_hist, actions_hist, returns, adv_hist, reward_hist = self.ppo_datacollector.collect_train_data(self.num_collected_data)
        return np.mean(reward_hist)

