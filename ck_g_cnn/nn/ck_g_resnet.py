import torch
from torch import nn

from ck_g_cnn.nn import CKGCNNBase
from ck_g_cnn.nn.conv.group_conv_block import GroupConvBlock

from ck_g_cnn.nn.activation import ReLU
from ck_g_cnn.nn.dropout import Dropout
from ck_g_cnn.nn.pooling import GroupMaxPoolingRn


class CKGResNet(CKGCNNBase):

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
            widen_factor
    ):
        super(CKGResNet, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_in_size=spatial_in_size,
            implementation=implementation,
            kernel_size=kernel_size,
            hidden_sizes=hidden_sizes,
            normalisation=normalisation,
            widen_factor=widen_factor
        )

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

        if dropout:
            self.layers.append(Dropout(p=dropout))

        self.layers.append(ReLU())

        # define groups convolutions
        for h_index in range(1, len(self.hidden_sizes)):
            self.layers.append(
                GroupConvBlock(
                    groupconv=self.groupconv_impl,
                    group=group,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    in_channels=self.hidden_sizes[h_index - 1],
                    out_channels=self.hidden_sizes[h_index],
                    num_group_elem=num_group_elem,
                    ck_net_num_hidden=ck_net_num_hidden,
                    ck_net_hidden_size=ck_net_hidden_size,
                    ck_net_implementation=ck_net_implementation,
                    ck_net_first_omega_0=ck_net_first_omega_0,
                    ck_net_omega_0=ck_net_omega_0,
                    sampling_method=sampling_method
                )
            )

            self.layers.append(self.normalisation_fn(channels=self.hidden_sizes[h_index]))

            if pooling:
                self.layers.append(GroupMaxPoolingRn())

            if dropout:
                self.layers.append(Dropout(p=dropout))

            self.layers.append(ReLU())

        if pooling is not None:
            last_kernel_size = self.spatial_in_size // (2 ** (len(hidden_sizes)-1)) - kernel_size
            if last_kernel_size > 11:
                last_kernel_size = 11
            if last_kernel_size < 3:
                last_kernel_size = 3
        else:
            last_kernel_size = self.kernel_size

        # final gconv to reduce spatial dims
        self.layers.extend([
            self.groupconv_impl(
                group=group,
                kernel_size=last_kernel_size,
                stride=1,
                in_channels=self.hidden_sizes[-1],
                out_channels=self.hidden_sizes[-1],
                num_group_elem=num_group_elem,
                padding=False,
                ck_net_num_hidden=ck_net_num_hidden,
                ck_net_hidden_size=ck_net_hidden_size,
                ck_net_implementation=ck_net_implementation,
                ck_net_first_omega_0=ck_net_first_omega_0,
                ck_net_omega_0=ck_net_omega_0,
                sampling_method=sampling_method
            ),
            self.normalisation_fn(channels=self.hidden_sizes[-1]),
            ReLU()
        ])

        if dropout:
            self.layers.append(Dropout(p=dropout))
