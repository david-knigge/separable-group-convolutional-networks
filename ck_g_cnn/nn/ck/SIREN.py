from torch import nn
import torch

import numpy as np


class SIREN(nn.Module):

    def __init__(self, in_features, hidden_features, hidden_layers, first_omega_0, hidden_omega_0):
        super().__init__()

        self.net = []

        self.net.append(
            SineLayerLinear(
                in_features=in_features,
                out_features=hidden_features,
                is_first=True,
                omega_0=first_omega_0,
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                SineLayerLinear(
                    in_features=hidden_features, out_features=hidden_features, is_first=False,
                    omega_0=hidden_omega_0
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """ Propagate the coordinates through SIREN.
        """
        x = self.net(x)
        return x


class SineLayerBase(nn.Module):

    def __init__(self, in_features, out_features, bias, is_first, omega_0):
        super().__init__()

        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.out_features = out_features

        self.bias = bias
        self.is_first = is_first

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)
                # Important! Bias is not defined in original SIREN implementation
                if self.linear.bias is not None:
                    # self.linear.bias.data.uniform_(-1.0, 1.0)
                    self.linear.bias.data.fill_(0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SineLayerLinear(SineLayerBase):

    def __init__(self, in_features, omega_0, out_features, bias=True, is_first=False):
        super(SineLayerLinear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            is_first=is_first,
            omega_0=omega_0
        )

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
