import torch
from torch import nn

import numpy as np

from ck_g_cnn.nn.ck.SIREN import SIREN


class CKNetBase(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, hidden_features, hidden_layers, implementation,
                 first_omega_0, hidden_omega_0):
        """ Base class for convolutional kernel net, extended in different implementations below (1dConv, 2dConv, 3dConv,
         Linear) to allow for efficient kernel parameterization.

        :param in_channels: Number of input channels (equal to number of dimensions in kernel grid)
        :param out_channels: Number of output channels to parameterise, this is equal to the number of parameters
            we need for a single spatial coordinate of the convolution kernel we are parameterizing:
                either [no_in_channels * no_out_channels] in case of a lifting convolution
                or     [no_in_channels * no_group_elem * no_out_channels] in case of a group convolution
        :param kernel_size: Spatial kernel size
        :param hidden_features: Number of hidden features to use in CKNet
        :param hidden_layers: Number of hidden layers ot use in CKNet
        :param implementation: Which implementation of the CKNet to use, choices are from: ["SIREN", "RFF", "MFN"]
        :param first_omega_0: Value of omega_0 used in the first sine layer
        :param hidden_omega_0: Value of omega_0 used in subsequent sine layers
        """
        super(CKNetBase, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers

        if implementation not in ["SIREN"]:
            raise ValueError("Specified CKNet implementation not found.")

        self.implementation = implementation

        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0

    def forward(self, coords):
        """ Propagate the coordinates through CKNet. """

        output = self.net(coords)
        return output

    def extra_repr(self) -> str:
        er = ''
        er += f"(first_omega_0) {self.first_omega_0}\n"
        er += f"(hidden_omega_0) {self.hidden_omega_0}\n"
        er += f"(kernel_size) {self.kernel_size}"
        return er


class CKNetLinear(CKNetBase):

    def __init__(self, in_features, in_channels, out_channels, kernel_size, hidden_features, hidden_layers,
                 first_omega_0, hidden_omega_0, implementation="SIREN"):
        """ Implements CKNet using linear layers. This is useful when we are parameterizing kernels of arbitrary sizes,
        and are unable to batch all coordinate grids together along a channel dimension (as is the case in spatial
        equivariance where kernels along the group axis have different spatial extents.

        :param in_features: the number of dimensions of the convolution kernel we are parameterizing.
        """
        super(CKNetLinear, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            implementation=implementation,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )

        if implementation == "SIREN":
            self.net = SIREN(
                in_features=in_features,
                hidden_features=hidden_features,
                hidden_layers=hidden_layers,
                first_omega_0=first_omega_0,
                hidden_omega_0=hidden_omega_0
            )

        self.final_linear = nn.Linear(hidden_features, in_channels * out_channels)

        with torch.no_grad():

            fan_in = in_channels * (kernel_size ** 2)
            uniform_variance = np.sqrt(6 / hidden_features) / (np.sqrt(fan_in))

            # uniform initialization
            self.final_linear.weight.data.uniform_(-uniform_variance, uniform_variance)
            self.final_linear.bias.data.zero_()

    def forward(self, x):
        """ Propagate the coordinates through CKNet.
        """
        x = self.net(x)
        x = self.final_linear(x)
        return x


class CKNetLinearSplit(nn.Module):

    def __init__(self, in_features, in_channels, out_channels, kernel_size, hidden_features, hidden_layers, implementation,
                 first_omega_0, hidden_omega_0):
        """ Base class for convolutional kernel net, extended in different implementations below (1dConv, 2dConv, 3dConv,
         Linear) to allow for efficient kernel parameterization.

        :param in_channels: Number of input channels (equal to number of dimensions in kernel grid)
        :param out_channels: Number of output channels to parameterise, this is equal to the number of parameters
            we need for a single spatial coordinate of the convolution kernel we are parameterizing:
                either [no_in_channels * no_out_channels] in case of a lifting convolution
                or     [no_in_channels * no_group_elem * no_out_channels] in case of a group convolution
        :param kernel_size: Spatial kernel size
        :param hidden_features: Number of hidden features to use in CKNet
        :param hidden_layers: Number of hidden layers ot use in CKNet
        :param implementation: Which implementation of the CKNet to use, choices are from: ["SIREN", "RFF", "MFN"]
        :param first_omega_0: Value of omega_0 used in the first sine layer
        :param hidden_omega_0: Value of omega_0 used in subsequent sine layers
        """
        super(CKNetLinearSplit, self).__init__()

        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.kernel_size = kernel_size

        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0

        self.net = SIREN(
            in_features=in_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0
        )

        self.final_linear_group = nn.Linear(
            in_features=hidden_features,
            out_features=in_channels * out_channels
        )

        self.final_linear_spatial = nn.Linear(
            in_features=hidden_features,
            out_features=out_channels
        )

        with torch.no_grad():

            fan_in = in_channels
            uniform_variance = np.sqrt(6 / hidden_features) / (np.sqrt(fan_in))
            # uniform initialization
            self.final_linear_group.weight.data.uniform_(-uniform_variance, uniform_variance)
            self.final_linear_group.bias.data.zero_()

            fan_in = kernel_size ** 2
            uniform_variance = np.sqrt(6 / hidden_features) / (np.sqrt(fan_in))
            # uniform initialization
            self.final_linear_spatial.weight.data.uniform_(-uniform_variance, uniform_variance)
            self.final_linear_spatial.bias.data.zero_()

    def forward_spatial(self, x):
        x = self.net(x)
        return self.final_linear_spatial(x)

    def forward_group(self, x):
        x = self.net(x)
        return self.final_linear_group(x)

    def extra_repr(self) -> str:
        er = ''
        er += f"(first_omega_0) {self.first_omega_0}\n"
        er += f"(hidden_omega_0) {self.hidden_omega_0}\n"
        er += f"(kernel_size) {self.kernel_size}"
        return er

