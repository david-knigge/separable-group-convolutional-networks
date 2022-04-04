import torch
from torch import nn

from ck_g_cnn.nn import CKGCNNBase
from ck_g_cnn.nn.conv.group_conv_block import GroupConvBlock

from ck_g_cnn.nn.activation import ReLU
from ck_g_cnn.nn.dropout import Dropout
from ck_g_cnn.nn.pooling import GroupMaxPoolingRn


class CKGAllCNNC(CKGCNNBase):

    def __init__(
            self,
            group,
            in_channels,
            out_channels,
            spatial_in_size,
            implementation,
            num_group_elem,
            kernel_size,
            bias,
            padding,
            hidden_sizes,
            ck_net_num_hidden,
            ck_net_hidden_size,
            ck_net_implementation,
            ck_net_first_omega_0,
            ck_net_omega_0,
            stride,
            dropout,
            sampling_method,
            pooling,
            normalisation,
            widen_factor=1
    ):
        super(CKGAllCNNC, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_in_size=spatial_in_size,
            implementation=implementation,
            kernel_size=kernel_size,
            hidden_sizes=hidden_sizes,
            normalisation=normalisation
        )

        # Based on (Springenberg et al, 2014) and (Cohen 2016)
        # https://github.com/tscohen/gconv_experiments/blob/master/gconv_experiments/CIFAR10/models/AllCNNC.py.
        # c: [3, 96, POOL, 96, 96, 192, 192, 192, 10]
        # k: [5,  5,    2,  5,  5,   5,   5,   5,  1]
        self.hidden_sizes = [96, 96, 96, 192, 192, 192, 192, 192]

        # widening factor
        self.hidden_sizes = [int(h * widen_factor) for h in self.hidden_sizes]

        self.layers.append(Dropout(p=0.2))

        # layer 1
        self.lifting = self.lifting_impl(
            group=group,
            kernel_size=self.kernel_size,
            stride=1,
            in_channels=self.in_channels,
            padding=padding,
            out_channels=self.hidden_sizes[0],
            num_group_elem=num_group_elem,
            ck_net_num_hidden=ck_net_num_hidden,
            ck_net_hidden_size=ck_net_hidden_size,
            ck_net_implementation=ck_net_implementation,
            ck_net_first_omega_0=ck_net_first_omega_0,
            ck_net_omega_0=ck_net_omega_0,
            sampling_method=sampling_method,
        )

        self.layers.append(self.normalisation_fn(channels=self.hidden_sizes[0]))

        self.layers.append(ReLU())

        # layer 2
        self.layers.append(
            self.groupconv_impl(
                group=group,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=True,
                in_channels=self.hidden_sizes[0],
                out_channels=self.hidden_sizes[1],
                num_group_elem=num_group_elem,
                ck_net_num_hidden=ck_net_num_hidden,
                ck_net_hidden_size=ck_net_hidden_size,
                ck_net_implementation=ck_net_implementation,
                ck_net_first_omega_0=ck_net_first_omega_0,
                ck_net_omega_0=ck_net_omega_0,
                sampling_method=sampling_method
            )
        )

        # instead of stride, to ensure equivariance
        self.layers.append(GroupMaxPoolingRn())

        self.layers.append(self.normalisation_fn(channels=self.hidden_sizes[1]))

        self.layers.append(ReLU())

        # layer 3
        self.layers.append(
            self.groupconv_impl(
                group=group,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=True,
                in_channels=self.hidden_sizes[1],
                out_channels=self.hidden_sizes[2],
                num_group_elem=num_group_elem,
                ck_net_num_hidden=ck_net_num_hidden,
                ck_net_hidden_size=ck_net_hidden_size,
                ck_net_implementation=ck_net_implementation,
                ck_net_first_omega_0=ck_net_first_omega_0,
                ck_net_omega_0=ck_net_omega_0,
                sampling_method=sampling_method
            )
        )

        self.layers.append(self.normalisation_fn(channels=self.hidden_sizes[2]))

        self.layers.append(Dropout(p=0.5))

        self.layers.append(ReLU())

        # layer 4
        self.layers.append(
            self.groupconv_impl(
                group=group,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=True,
                in_channels=self.hidden_sizes[2],
                out_channels=self.hidden_sizes[3],
                num_group_elem=num_group_elem,
                ck_net_num_hidden=ck_net_num_hidden,
                ck_net_hidden_size=ck_net_hidden_size,
                ck_net_implementation=ck_net_implementation,
                ck_net_first_omega_0=ck_net_first_omega_0,
                ck_net_omega_0=ck_net_omega_0,
                sampling_method=sampling_method
            )
        )

        self.layers.append(self.normalisation_fn(channels=self.hidden_sizes[3]))

        self.layers.append(ReLU())

        # layer 5
        self.layers.append(
            self.groupconv_impl(
                group=group,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=True,
                in_channels=self.hidden_sizes[3],
                out_channels=self.hidden_sizes[4],
                num_group_elem=num_group_elem,
                ck_net_num_hidden=ck_net_num_hidden,
                ck_net_hidden_size=ck_net_hidden_size,
                ck_net_implementation=ck_net_implementation,
                ck_net_first_omega_0=ck_net_first_omega_0,
                ck_net_omega_0=ck_net_omega_0,
                sampling_method=sampling_method
            )
        )

        # instead of stride, to ensure equivariance
        self.layers.append(GroupMaxPoolingRn())

        self.layers.append(self.normalisation_fn(channels=self.hidden_sizes[4]))

        self.layers.append(ReLU())

        # layer 6
        self.layers.append(
            self.groupconv_impl(
                group=group,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=True,
                in_channels=self.hidden_sizes[4],
                out_channels=self.hidden_sizes[5],
                num_group_elem=num_group_elem,
                ck_net_num_hidden=ck_net_num_hidden,
                ck_net_hidden_size=ck_net_hidden_size,
                ck_net_implementation=ck_net_implementation,
                ck_net_first_omega_0=ck_net_first_omega_0,
                ck_net_omega_0=ck_net_omega_0,
                sampling_method=sampling_method
            )
        )

        self.layers.append(self.normalisation_fn(channels=self.hidden_sizes[5]))

        self.layers.append(Dropout(p=0.5))

        self.layers.append(ReLU())

        # layer 7
        self.layers.append(
            self.groupconv_impl(
                group=group,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=True,
                in_channels=self.hidden_sizes[5],
                out_channels=self.hidden_sizes[6],
                num_group_elem=num_group_elem,
                ck_net_num_hidden=ck_net_num_hidden,
                ck_net_hidden_size=ck_net_hidden_size,
                ck_net_implementation=ck_net_implementation,
                ck_net_first_omega_0=ck_net_first_omega_0,
                ck_net_omega_0=ck_net_omega_0,
                sampling_method=sampling_method
            )
        )

        self.layers.append(self.normalisation_fn(channels=self.hidden_sizes[6]))

        self.layers.append(ReLU())

        # layer 8
        self.layers.append(
            self.groupconv_impl(
                group=group,
                kernel_size=1,
                stride=stride,
                padding=False,
                in_channels=self.hidden_sizes[6],
                out_channels=self.hidden_sizes[7],
                num_group_elem=num_group_elem,
                ck_net_num_hidden=ck_net_num_hidden,
                ck_net_hidden_size=ck_net_hidden_size,
                ck_net_implementation=ck_net_implementation,
                ck_net_first_omega_0=ck_net_first_omega_0,
                ck_net_omega_0=ck_net_omega_0,
                sampling_method=sampling_method
            )
        )

        self.layers.append(self.normalisation_fn(channels=self.hidden_sizes[7]))

        self.layers.append(ReLU())

        # layer 9
        # final gconv to reduce spatial dims
        self.final_linear = self.groupconv_impl(
            group=group,
            kernel_size=1,
            stride=1,
            in_channels=self.hidden_sizes[7],
            out_channels=self.out_channels,
            num_group_elem=num_group_elem,
            padding=False,
            ck_net_num_hidden=ck_net_num_hidden,
            ck_net_hidden_size=ck_net_hidden_size,
            ck_net_implementation=ck_net_implementation,
            ck_net_first_omega_0=ck_net_first_omega_0,
            ck_net_omega_0=ck_net_omega_0,
            sampling_method=sampling_method
        )

    def forward(self, x):
        """ Forward pass of CKGCNN, sequentially applies lifting and group convolutional layers. Uses normalisation,
        activation and optionally pooling functions defined in initialisation.

        @param x: Input sample.
        """

        x, grid_H = self.lifting(x)

        for mod in self.layers:
            x, grid_H = mod(x, grid_H)

        # take max over spatial and groups dimension to ensure invariance
        x, _ = self.final_linear(x, grid_H)

        x = torch.amax(x, dim=(-3, -2, -1))
        x = x.view(-1, x.shape[1])

        return x