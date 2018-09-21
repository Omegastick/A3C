"""
Collection of models for A3C
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_columns_initializer(weights, std=1.0):
    """
    Normalises each layer's weights when initialising.
    Based on: https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py
    """
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(module):
    """
    Weight initialisation function magic.
    Based on: https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py
    """
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(module.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        module.weight.data.uniform_(-w_bound, w_bound)
        module.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(module.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        module.weight.data.uniform_(-w_bound, w_bound)
        module.bias.data.fill_(0)


class Flatten(nn.Module):
    """
    Module for flattening an n-dimensional matrix down to a 2-dimensional one.
    """

    def forward(self, x) -> torch.Tensor:  # pylint:disable=arguments-differ
        x = x.view(x.shape[0], -1)
        return x


class NatureCNN(nn.Module):
    """
    The convolutional neural network used in the Nature DQN paper.
    """

    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.conv_1 = nn.Conv2d(input_shape[0], 32, 8, 4)
        self.conv_2 = nn.Conv2d(32, 64, 4, 2)
        self.conv_3 = nn.Conv2d(64, 64, 3, 1)

    def forward(self, x) -> torch.Tensor:  # pylint:disable=arguments-differ
        x = x.view(-1, *self.input_shape).float()
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = self.conv_3(x)
        x = F.relu(x)
        return x


class MultiLayerPerceptron(nn.Module):
    """
    A Multi Layer Perceptron for extracting features from an input.
    """

    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.layer_1 = nn.Linear(input_shape[0], 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 256)

    def forward(self, x) -> torch.Tensor:  # pylint:disable=arguments-differ
        x = x.view(-1, *self.input_shape).float()
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        x = F.relu(x)

        return x


class ActorCritic(nn.Module):
    """
    Model for an actor-critic agent.
    """

    def __init__(self, input_shape, action_space, features='cnn'):
        super().__init__()

        self.action_space = action_space

        if features == 'cnn':
            self.features = NatureCNN(input_shape)
        elif features == 'mlp':
            self.features = MultiLayerPerceptron(input_shape)
        else:
            raise ValueError("Type must be one of: cnn, mlp")

        self.flatten = Flatten()

        feature_size = self.flatten(
            self.features(torch.zeros(1, *input_shape))).size(1)

        self.gru = nn.GRUCell(feature_size, 256)

        self.actor = nn.Linear(256, action_space)
        self.critic = nn.Linear(256, 1)

        self.apply(weights_init)

        self.actor.weight.data = normalized_columns_initializer(
            self.actor.weight.data, 0.01)
        self.actor.bias.data.fill_(0)
        self.critic.weight.data = normalized_columns_initializer(
            self.critic.weight.data, 1.0)
        self.critic.bias.data.fill_(0)

    def forward(self, x, hx) -> torch.Tensor:  # pylint:disable=arguments-differ
        x = self.features(x)
        x = self.flatten(x)

        hx = self.gru(x, hx)
        x = hx

        actor = self.actor(x)
        critic = self.critic(x)

        return actor, critic, hx
