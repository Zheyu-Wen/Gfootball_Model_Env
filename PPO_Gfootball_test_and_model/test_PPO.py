from arguments import get_args
from PPO_agent import PPO_agent
import gfootball.env as football_env
import numpy as np
from Networks_PPO import PPO_CNN
import torch
import os
# get the tensors
def get_tensors(obs):
    return torch.tensor(np.transpose(obs, (0, 3, 1, 2)), dtype=torch.float32)
"""
to run code
MESA_GL_VERSION_OVERRIDE=3.2 MESA_GLSL_VERSION_OVERRIDE=150 python3 test_PPO.py
"""

if __name__ == '__main__':
    args = get_args()
    model_path = args.save_dir + args.env_name + '/model.pt'
    env = football_env.create_environment(env_name=args.env_name, stacked=True, render=True)#, write_full_episode_dumps=True,write_video=True, logdir='trace_'+args.env_name)
    network = PPO_CNN(env.action_space.n)
    network.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    test_obs = []
    is_same = []
    rewards = np.zeros(1)
    accumulate_reward = np.zeros(1)
    num_epoch = np.zeros(1)
    # start to do the test
    obs = env.reset()
    for iter in range(1000):
        obs_tensor = get_tensors(np.expand_dims(obs, 0))
        with torch.no_grad():
            _, pi = network(obs_tensor)
        actions = torch.argmax(pi, dim=1).item()
        tmp = obs
        obs, reward, done, _ = env.step(actions)
        # is_same.append(np.array_equal(tmp,obs))
        # test_obs.append(obs)
        # print(obs)
        accumulate_reward = accumulate_reward+reward
        if done:
            obs = env.reset()
    env.close()
    average_reward = accumulate_reward/1000
print(average_reward)
