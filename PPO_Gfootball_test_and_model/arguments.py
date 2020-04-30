import argparse

def get_args():
    arg = argparse.ArgumentParser()
    """
    Basic parameter setup
    """
    arg.add_argument('--env-name', type=str, default='5_vs_5')
    arg.add_argument('--total-frames', type=int, default=int(2e6))
    # while training 11V11 mode, set steps to be 2500 or 3000
    # for academy mode 128 is enough
    arg.add_argument('--nsteps', type=int, default=128)
    # while training 11V11 mode, set num_workers to 2
    arg.add_argument('--num-workers', type=int, default=8)
    arg.add_argument('--batch-size', type=int, default=8)
    arg.add_argument('--epoch', type=int, default=4)
    arg.add_argument('--cuda', action='store_true')
    """
    PPO hyperparameter setup
    """
    # the discount factor of RL
    arg.add_argument('--gamma', type=float, default=0.995)
    # the random seeds for the env to reset
    arg.add_argument('--seed', type=int, default=1)
    # learning rate of the algorithm
    arg.add_argument('--lr', type=float, default=0.00008)
    # the coefficient of value loss
    arg.add_argument('--vloss-coef', type=float, default=0.5)
    # the entropy loss coefficient
    arg.add_argument('--ent-coef', type=float, default=0.01)
    # gae coefficient
    arg.add_argument('--tau', type=float, default=0.95)
    # param for adam optimizer
    arg.add_argument('--eps', type=float, default=1e-5)
    # the ratio clip param
    arg.add_argument('--clip', type=float, default=0.27)
    # if using the learning rate decay during decay
    arg.add_argument('--lr-decay', action='store_true')
    # grad norm
    arg.add_argument('--max-grad-norm', type=float, default=0.5)
    """
    Log parameter setup
    """
    arg.add_argument('--save-dir', type=str, default='saved_models/')
    arg.add_argument('--display-interval', type=int, default=10)
    arg.add_argument('--log-dir', type=str, default='logs/')

    args = arg.parse_args()

    return args
