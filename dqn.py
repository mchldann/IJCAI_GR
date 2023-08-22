import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn import functional as F


class DQN_Config(object):

    def __init__(self, state_size, n_actions, gpu=-1, noisy_nets=False, n_latent=64):
        self.state_size = state_size
        self.n_actions = n_actions
        self.gpu = gpu
        self.noisy_nets = noisy_nets
        self.n_latent = n_latent


class NoisyLinear(nn.Module):

    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module):

    def __init__(self, config:DQN_Config):
        super(DQN, self).__init__()

        self.device = torch.device("cuda" if config.gpu >= 0 else "cpu")

        if config.noisy_nets:
            lin = NoisyLinear
        else:
            lin = nn.Linear

        self.fc1 = lin(in_features=config.state_size, out_features=config.n_latent).to(self.device)

        self.fc2_a = lin(in_features=config.n_latent, out_features=int(config.n_latent/2)).to(self.device)
        self.fc2_v = lin(in_features=config.n_latent, out_features=int(config.n_latent/2)).to(self.device)

        self.fc3_a = lin(in_features=int(config.n_latent/2), out_features=config.n_actions).to(self.device)
        self.fc3_v = lin(in_features=int(config.n_latent/2), out_features=config.n_actions).to(self.device)

        self.relu = nn.ReLU().to(self.device)


    def forward(self, x):

        x = self.relu(self.fc1(x))

        a = self.relu(self.fc2_a(x))
        a = self.relu(self.fc3_a(a))

        v = self.relu(self.fc2_v(x))
        v = self.relu(self.fc3_v(v))

        x = v + a - a.mean(a.dim() - 1, keepdim=True)

        return x
