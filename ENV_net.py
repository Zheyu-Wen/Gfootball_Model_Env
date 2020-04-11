import torch.nn as nn
import torch

class env_net(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        self.obs_shape = obs_shape
        m, n, p = obs_shape
        self.action_shape = 1
        self.fco1 = nn.Linear(m * n * p, 64)
        self.fca1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, obs, action):
        out1 = torch.relu(self.fco1(obs))
        out2 = torch.relu(self.fca1(action))
        out = torch.cat([out1, out2], dim=1)
        out = torch.relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))
        return out
