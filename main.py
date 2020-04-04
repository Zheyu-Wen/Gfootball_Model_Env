import argparse
from ppo_agent import ppo_agent
from policy_net import cnn_net
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gfootball.env as football_env
from ENV_net import env_net
import os
import numpy as np
from evaluate_performance import evaluation
import matplotlib.pyplot as plt

def init_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--env_name', type=str, default='academy_empty_goal_close')
    parse.add_argument('--seed', type=int, default=999, help='the random seed')
    parse.add_argument('--batch_size', type=int, default=5, help='training batch size')
    parse.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parse.add_argument('--epoch', type=int, default=20, help='training epoch')
    parse.add_argument('--vloss_coef', type=float, default=1, help='value loss coef for PPO')
    parse.add_argument('--entloss_coef', type=float, default=1,help='entropy loss coef')
    parse.add_argument('--gae_coef', type=float, default=1, help='gae loss coef')
    parse.add_argument('--clip', type=float, default=0.1, help='clip in ppo loss')
    parse.add_argument('--policy_model_dir', type=str, default='policy_model_dict')
    parse.add_argument('--env_model_dir', type=str, default='env_model_dict')
    parse.add_argument('--gamma', type=float, default=0.993)
    parse.add_argument('--tau', type=float, default=0.95)
    parse.add_argument('--ent_coef', type=float, default=0.01)
    parse.add_argument('--max-grad-norm', type=float, default=0.5)

    args = parse.parse_args()

    return args

# create the environment
def create_single_football_env(args):
    """Creates gfootball environment."""
    env = football_env.create_environment(env_name=args.env_name, stacked=True)
    return env

if __name__ == '__main__':
    # get the arguments
    args = init_args()
    # create environments
    envs = create_single_football_env(args)
    train = False
    if train == True:
        # create networks
        policy_net = cnn_net(envs.action_space.n)
        env_net = env_net(envs.observation_space.shape)
        # create the ppo agent
        ppo_trainer = ppo_agent(envs, args, policy_net, env_net)
        num_collected_data = np.int(1e5)
        num_epoch = 1000
        poloss_hist = []
        envloss_hist = []
        reward_history = []
        for iters in range(num_epoch):
            envs.reset()
            obs_hist, actions_hist, returns, adv_hist, reward_hist = ppo_trainer.collect_train_data(num_collected_data)
            if np.mean(reward_hist) > 0:
                policy_loss, env_loss = ppo_trainer._update_network_by_env_net(obs_hist, actions_hist, reward_hist)
                print('In epoch {}\n policy loss is: {}\n env loss is: {}'.format(iters, policy_loss, env_loss))
                poloss_hist.append(policy_loss)
                envloss_hist.append(env_loss)
            else:
                env_loss = ppo_trainer._update_network_by_env_net(obs_hist, actions_hist, reward_hist)
                print('In epoch {}\n env loss is: {}'.format(iters, env_loss))
                envloss_hist.append(env_loss)
            if iters % 5 == 0:
                _ = ppo_trainer._update_network_wo_env_net(obs_hist, actions_hist, returns, adv_hist)

            evaluation_process = evaluation(args.policy_model_dir, envs.action_space.n, envs, args)
            reward_history.append(evaluation_process.eval())
        plt.plot(np.arange(len(reward_history)), reward_history, label="reward")
        plt.xlabel("iter")
        plt.ylabel("reward")
        plt.show()
        # close the environments
        envs.close()
    else:
        evaluation_process = evaluation(args.policy_model_dir, envs.action_space.n, envs, args)
        reward = evaluation_process.eval()
        print("trained model gives average reward: {}".format(reward))


