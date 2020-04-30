from arguments import get_args
from PPO_agent import PPO_agent
from Networks_PPO import PPO_CNN
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gfootball.env as football_env
import os

# create the environment
def create_single_football_env(args):
    env = football_env.create_environment(env_name=args.env_name, stacked=True)
    return env

if __name__ == '__main__':
    params = get_args()
    """
    create env for individual workers
    """
    envs = SubprocVecEnv([(lambda _i=i: create_single_football_env(params)) for i in range(params.num_workers)])
    network = PPO_CNN(envs.action_space.n)
    """
    Init PPO agent instance
    """
    PPO_trainer = PPO_agent(envs, params, network)
    PPO_trainer.train()
    envs.close()
