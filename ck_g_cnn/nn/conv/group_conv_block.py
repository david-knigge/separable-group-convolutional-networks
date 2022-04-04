import torch
import torch.nn.functional as F

from ck_g_cnn.nn.activation import ReLU
from ck_g_cnn.nn.normalisation import LayerNorm, InstanceNorm, BatchNorm


class GroupConvBlock(torch.nn.Module):

    def __init__(self,
                 groupconv,
                 group,
                 in_channels,
                 out_channels,
                 num_group_elem,
                 kernel_size,
                 ck_net_num_hidden,
                 ck_net_hidden_size,
                 ck_net_implementation,
                 ck_net_first_omega_0,
                 ck_net_omega_0,
                 stride,
                 sampling_method):
        """ Group convolution block.

        :param groupconv: Group convolution operation implementation.
        :param group: Group implementation.
        :param in_channels: Number of channels in input feature map.
        :param out_channels: Number of channels in output feature map.
        :param num_group_elem: Number of group elements to sample or discretise group by.
        :param kernel_size: Convolution kernel size.
        :param ck_net_num_hidden: Number of layers of continuous kernel net.
        :param ck_net_hidden_size: Hidden size of continuous kernel net.
        :param ck_net_implementation: Type of implementation to use for continuous kernel.
        :param ck_net_first_omega_0: SIREN first layer omega_0 parameter.
        :param ck_net_omega_0: SIREN omega_0 parameter.
        :param stride: Convolution stride.
        :param sampling_method: Sampling method over the group. Can be either 'discretise' or 'uniform'.
        """
        super().__init__()

        self.group = group

        self.gconv_1 = groupconv(
            group=group,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=in_channels,
            num_group_elem=num_group_elem,
            ck_net_num_hidden=ck_net_num_hidden,
            ck_net_hidden_size=ck_net_hidden_size,
            ck_net_implementation=ck_net_implementation,
            ck_net_first_omega_0=ck_net_first_omega_0,
            ck_net_omega_0=ck_net_omega_0,
            padding=True,
            stride=stride,
            sampling_method=sampling_method,
        )

        self.gconv_2 = groupconv(
            group=group,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_group_elem=num_group_elem,
            ck_net_num_hidden=ck_net_num_hidden,
            ck_net_hidden_size=ck_net_hidden_size,
            ck_net_implementation=ck_net_implementation,
            ck_net_first_omega_0=ck_net_first_omega_0,
            ck_net_omega_0=ck_net_omega_0,
            padding=True,
            stride=1,
            sampling_method=sampling_method
        )

        if in_channels != out_channels:

            self.skip_align = groupconv(
                group=group,
                kernel_size=1,
                in_channels=in_channels,
                out_channels=out_channels,
                num_group_elem=num_group_elem,
                ck_net_num_hidden=ck_net_num_hidden,
                ck_net_hidden_size=ck_net_hidden_size // 2,
                ck_net_implementation=ck_net_implementation,
                ck_net_first_omega_0=ck_net_first_omega_0,
                ck_net_omega_0=ck_net_omega_0,
                padding=True,
                stride=stride,
                sampling_method=sampling_method
            )

        else:

            self.skip_align = None

        self.activation_1 = ReLU()
        self.activation_2 = ReLU()
        self.activation_3 = ReLU()

        self.normalisation_1 = BatchNorm(in_channels)
        self.normalisation_2 = BatchNorm(out_channels)

    def forward(self, x_0, grid_H_0):
        """ Apply group convolution block. Note that with random sampling over the group, this operation needs special
        care since in and output feature maps need to be defined over the same group elements for the shortcut
        connection to be valid.

        :param x: Function on a homogeneous space of the groups
            [batch_dim, in_channels, num_group_elements, spatial_dim_1, spatial_dim_2]
        :return: Function on a homogeneous space of the groups
            [batch_dim, out_channels, num_group_elements, spatial_dim_1, spatial_dim_2]
        """

        if self.gconv_1.in_channels != self.gconv_2.out_channels:

            # first conv
            x, grid_H = self.gconv_1(x_0, grid_H=grid_H_0)
            x, grid_H = self.normalisation_1(x, grid_H=grid_H)
            x, grid_H = self.activation_1(x, grid_H=grid_H)

            # second conv
            x, grid_H = self.gconv_2(x, grid_H=grid_H)
            x, grid_H = self.normalisation_2(x, grid_H=grid_H)
            x, grid_H = self.activation_2(x, grid_H=grid_H)

            skip_x, _ = self.skip_align(x_0, grid_H=grid_H_0, sampled_group_elements=grid_H)

            # skip connection
            return self.activation_3(x + skip_x, grid_H)

        else:

            # first conv
            x, grid_H = self.gconv_1(x_0, grid_H=grid_H_0, sampled_group_elements=grid_H_0)
            x, grid_H = self.normalisation_1(x, grid_H=grid_H)
            x, grid_H = self.activation_1(x, grid_H=grid_H)

            # second conv
            x, grid_H = self.gconv_2(x, grid_H=grid_H, sampled_group_elements=grid_H)
            x, grid_H = self.normalisation_2(x, grid_H=grid_H)
            x, grid_H = self.activation_2(x, grid_H=grid_H)

            # skip connection
            return self.activation_3(x + x_0, grid_H)
