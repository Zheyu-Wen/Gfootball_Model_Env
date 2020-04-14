from arguments import get_args
from models import cnn_net
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gfootball.env as football_env
from ENV_net import env_net
import matplotlib.pyplot as plt
import numpy as np
from ppo_agent_GAN import ppo_agent_gan
from ppo_agent import ppo_agent
from A2C_agent import a2c_agent
from DDQN_agent import ddqn_agent
from Compatition_learning_agent import Compation_agent
import os

# create the environment
def create_single_football_env(args):
    """Creates gfootball environment."""
    env = football_env.create_environment(\
            env_name=args.env_name, stacked=True#, with_checkpoints=False,
            )
    return env


if __name__ == '__main__':
    mode = 'COMPATITION'
    if mode == 'COMPATITION':
        # get the arguments
        args = get_args()
        # create environments
        envs = SubprocVecEnv([(lambda _i=i: create_single_football_env(args)) for i in range(args.num_workers)])
        # create networks
        network = cnn_net(envs.action_space.n)
        # create the ppo agent
        Compation_trainer = Compation_agent(envs, args, network)
        reward_hist, player1_loss_hist, player2_loss_hist = Compation_trainer.learn()
        np.save('reward_comp.npy', reward_hist)
        np.save('comp_player1_loss_hist.npy', player1_loss_hist)
        np.save('comp_player2_loss_hist.npy', player2_loss_hist)

    if mode == 'DDQN':
        # get the arguments
        args = get_args()
        # create environments
        envs = SubprocVecEnv([(lambda _i=i: create_single_football_env(args)) for i in range(args.num_workers)])
        # create networks
        network = cnn_net(envs.action_space.n)
        env_net = env_net(envs.observation_space.shape)
        # create the ppo agent
        ddqn_trainer = ddqn_agent(envs, args, network, env_net)
        reward_hist, _, policy_loss_hist = ddqn_trainer.learn()
        np.save('reward_ddqn.npy', reward_hist)
        np.save('policy_loss_ddqn.npy', policy_loss_hist)

    if mode == 'A2C':
        # get the arguments
        args = get_args()
        # create environments
        envs = SubprocVecEnv([(lambda _i=i: create_single_football_env(args)) for i in range(args.num_workers)])
        # create networks
        network = cnn_net(envs.action_space.n)
        env_net = env_net(envs.observation_space.shape)
        # create the ppo agent
        a2c_trainer = a2c_agent(envs, args, network, env_net)
        reward_hist, _, policy_loss_hist = a2c_trainer.learn()
        np.save('reward_a2c.npy', reward_hist)
        np.save('policy_loss_a2c.npy', policy_loss_hist)

    if mode == 'PPO_GAN':
        # get the arguments
        args = get_args()
        # create environments
        envs = SubprocVecEnv([(lambda _i=i: create_single_football_env(args)) for i in range(args.num_workers)])
        # create networks
        network = cnn_net(envs.action_space.n)
        env_net = env_net(envs.observation_space.shape)
        # create the ppo agent
        gan_trainer = ppo_agent_gan(envs, args, network, env_net)
        reward_hist, env_loss_hist, policy_loss_hist = gan_trainer.learn()
        np.save('reward_gan.npy', reward_hist)
        np.save('env_loss_gan.npy', env_loss_hist)
        np.save('policy_loss_gan.npy', policy_loss_hist)

    if mode == 'PPO':
        args = get_args()
        envs = SubprocVecEnv([(lambda _i=i: create_single_football_env(args)) for i in range(args.num_workers)])
        network = cnn_net(envs.action_space.n)
        # create the ppo agent
        env_net = env_net(envs.observation_space.shape) # we don't use it in ppo
        ppo_trainer = ppo_agent(envs, args, network, env_net)
        reward_hist, _, policy_loss_hist = ppo_trainer.learn()
        np.save('reward_ppo.npy', reward_hist)
        np.save('policy_loss_ppo.npy', policy_loss_hist)

    # close the environments
    envs.close()
