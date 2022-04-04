import torch
import torch.nn.functional as F

from ck_g_cnn.nn.conv.ck_g_conv import ContinuousKernelGroupConv
from ck_g_cnn.nn.kernels.SE2_kernel import SE2GroupKernelSeparable
# from ck_g_cnn.nn.kernels.SE2_kernel import SE2GroupKernelSeparableSplit as SE2GroupKernelSeparable
from ck_g_cnn.nn.kernels.R2xRplus_kernel import R2xRplusGroupKernelSeparable
from ck_g_cnn.nn.kernels.Sim2_kernel import Sim2GroupKernelSeparable

from ck_g_cnn.groups import SE2, Rplus, Sim2


class GroupConvSeparable(ContinuousKernelGroupConv):

    def __init__(self,
                 group,
                 kernel_size,
                 in_channels,
                 out_channels,
                 num_group_elem,
                 ck_net_num_hidden,
                 ck_net_hidden_size,
                 ck_net_implementation,
                 ck_net_first_omega_0,
                 ck_net_omega_0,
                 sampling_method,
                 stride,
                 padding=True):
        """ Create a groups convolution layer, performing a convolution on (a homogeneous space of) the groups. Let i,j
        be in and ouput channels, implementation defines the group convolution kernel as:

            k_{ij}(x, g) = k_{ij}(g) k_j(x)

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
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_group_elem=num_group_elem,
            kernel_size=kernel_size,
            padding=padding,
            sampling_method=sampling_method,
            group=group,
            stride=stride
        )

        # construct separable kernel
        if type(group) == SE2:
            self.kernel = SE2GroupKernelSeparable(
                group=group,
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=out_channels,
                ck_net_num_hidden=ck_net_num_hidden,
                ck_net_hidden_size=ck_net_hidden_size,
                ck_net_implementation=ck_net_implementation,
                ck_net_first_omega_0=ck_net_first_omega_0,
                ck_net_omega_0=ck_net_omega_0,
                sampling_method=sampling_method
            )
        elif type(group) == Rplus:
            self.kernel = R2xRplusGroupKernelSeparable(
                group=group,
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=out_channels,
                ck_net_num_hidden=ck_net_num_hidden,
                ck_net_hidden_size=ck_net_hidden_size,
                ck_net_implementation=ck_net_implementation,
                ck_net_first_omega_0=ck_net_first_omega_0,
                ck_net_omega_0=ck_net_omega_0,
                sampling_method=sampling_method
            )
        elif type(group) == Sim2:
            self.kernel = Sim2GroupKernelSeparable(
                group=group,
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=out_channels,
                ck_net_num_hidden=ck_net_num_hidden,
                ck_net_hidden_size=ck_net_hidden_size,
                ck_net_implementation=ck_net_implementation,
                ck_net_first_omega_0=ck_net_first_omega_0,
                ck_net_omega_0=ck_net_omega_0,
                sampling_method=sampling_method
            )

    def forward(self, x, grid_H, sampled_group_elements=None):
        """ Perform groups convolution

        :param x: Function on a homogeneous space of the groups
            [batch_dim, in_channels, num_group_elements, spatial_dim_1, spatial_dim_2]
        :grid_H: Coordinates on the group of input data
        :return: Function on a homogeneous space of the groups
            [batch_dim, out_channels, num_group_elements, spatial_dim_1, spatial_dim_2]
        """
        # Reshape x into [batch_dim, channels * angular_dim, spatial_1, spatial_2], expanding the angle dimension into
        # the input channel dimension.

        # Sample group elements to apply to filters.
        if sampled_group_elements is None:
            sampled_group_elements = self.group.sample(num_elements=self.num_group_elem, method=self.sampling_method)

        H_conv_kernel, Rn_conv_kernel = self.kernel.sample(sampled_group_elements, grid_H)

        # Apply convolution operation over group and input channel dimensions of feature map.
        x = F.conv2d(
            input=x.reshape(-1, x.shape[1] * x.shape[2], x.shape[3], x.shape[4]),
            weight=H_conv_kernel.reshape(
                self.kernel.out_channels * sampled_group_elements.shape[0],  # num_group_elem in output
                self.kernel.in_channels * grid_H.shape[0],  # num group elem in input
                1,
                1
            ),
            padding=0
        )

        # Apply convolution operation over spatial dimensions of feature map.
        x = F.conv2d(
            input=x,
            weight=Rn_conv_kernel.reshape(
                self.kernel.out_channels * sampled_group_elements.shape[0],
                1,
                self.kernel.kernel_size,
                self.kernel.kernel_size
            ),
            padding=self.padding,
            stride=self.stride,
            groups=(self.kernel.out_channels * sampled_group_elements.shape[0])  # each out channel and elem gets own weight
        )

        x = x.reshape(-1, self.kernel.out_channels, sampled_group_elements.shape[0], x.shape[-1], x.shape[-2])

        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1, 1)

        return x, sampled_group_elements
