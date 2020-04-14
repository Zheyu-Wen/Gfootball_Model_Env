import matplotlib.pyplot as plt
import numpy as np

reward_a2c = np.load('reward_a2c.npy')
reward_ddqn = np.load('reward_ddqn.npy')
reward_gan = np.load('reward_gan.npy')
reward_ppo = np.load('reward_ppo.npy')
reward_comp = np.load('reward_comp.npy')


plt.figure(1)
plt.plot(np.arange(len(np.array(reward_a2c))), np.array(reward_a2c), label='A2C reward')
plt.plot(np.arange(len(np.array(reward_ddqn))), np.array(reward_ddqn), label='DDQN reward')
# plt.plot(np.arange(len(np.array(reward_gan))), np.array(reward_gan), label='GAN reward')
plt.plot(np.arange(len(np.array(reward_ppo))), np.array(reward_ppo), label='PPO reward')
plt.plot(np.arange(len(np.array(reward_comp))/2), np.array(reward_comp[0:-1:2]), label='COMP reward')

plt.legend(['A2C', 'DDQN', 'PPO', 'COMP'])
plt.xlabel('iter')
plt.ylabel('reward')

policy_loss_a2c = np.load('policy_loss_a2c.npy')
policy_loss_ddqn = np.load('policy_loss_ddqn.npy')
# policy_loss_gan = np.load('policy_loss_gan.npy')
policy_loss_ppo = np.load('policy_loss_ppo.npy')
policy_loss_comp_player1 = np.load('comp_player1_loss_hist.npy')
policy_loss_comp_player2 = np.load('comp_player2_loss_hist.npy')


plt.figure(2)
# plt.plot(np.arange(len(np.array(policy_loss_a2c))), np.array(policy_loss_a2c), label='A2C policy net loss')
plt.plot(np.arange(len(np.array(policy_loss_ddqn))), np.array(policy_loss_ddqn), label='DDQN policy net loss')
# plt.plot(np.arange(len(np.array(policy_loss_gan))), np.array(policy_loss_gan), label='GAN policy net loss')
plt.plot(np.arange(len(np.array(policy_loss_ppo))), np.array(policy_loss_ppo), label='PPO policy net loss')
plt.plot(np.arange(len(np.array(policy_loss_comp_player1))), np.array(policy_loss_comp_player1), label='COMP player1 policy net loss')
plt.plot(np.arange(len(np.array(policy_loss_comp_player2))), np.array(policy_loss_comp_player2), label='COMP player2 net loss')

plt.legend(['DDQN', 'PPO', 'COMP player1', 'COMP player2'])
plt.xlabel('iter')
plt.ylabel('policy net loss')

env_loss_gan = np.load('env_loss_gan.npy')
plt.figure(3)
plt.plot(np.arange(len(np.array(env_loss_gan))), np.array(env_loss_gan), label='env net loss')
plt.xlabel('iter')
plt.ylabel('main net loss')
plt.show()