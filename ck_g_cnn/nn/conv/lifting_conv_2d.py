import torch
import torch.nn.functional as F

from ck_g_cnn.nn.conv.ck_g_conv import ContinuousKernelGroupConv

from ck_g_cnn.nn.kernels.Sim2_kernel import Sim2LiftingKernel

from ck_g_cnn.groups import Sim2


class LiftingConv2D(ContinuousKernelGroupConv):

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
                 padding):
        """ Defines a lifting convolution, lifting from the input space to a homogeneous subgroup. This
        implementation supports two-dimensional groups (e.g. roto-dilations).

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

        if type(group) == Sim2:
            self.kernel = Sim2LiftingKernel(
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
        else:
            raise ValueError(f"LiftingConv2D does not support groups of dimensionality {group.dimension}.")

    def forward(self, x, sampled_group_elements=None):
        """ Perform lifting convolution

        :param x: Input sample [batch_dim, in_channels, spatial_dim_1, spatial_dim_2].
        :param sampled_group_elements: Group elements on which to define output grid of this convolution operation.
        :return: Function on a homogeneous space of the groups
            [batch_dim, out_channels, num_group_elements, spatial_dim_1, spatial_dim_2].
        """

        # Obtain group elements to sample kernels for.
        if sampled_group_elements is None:
            SO2_elem, Rplus_elem = self.group.sample(
                num_elements=self.num_group_elem,
                method=self.sampling_method,
                separable=True
            )

        else:
            SO2_elem, Rplus_elem = sampled_group_elements

        # Stack separate group elements into a meshgrid.
        sampled_group_elements = torch.stack(torch.meshgrid(SO2_elem, Rplus_elem)).view(self.group.dimension, -1).T
        conv_kernels = self.kernel.sample(sampled_group_elements)

        x = F.conv2d(
            input=x,
            weight=conv_kernels.reshape(
                self.kernel.out_channels * sampled_group_elements.shape[0],
                self.kernel.in_channels,
                self.kernel.kernel_size,
                self.kernel.kernel_size
            ),
            padding=self.padding,
            stride=self.stride,
        )

        x = x.reshape(-1, self.kernel.out_channels, SO2_elem.shape[0] * Rplus_elem.shape[0], x.shape[-2], x.shape[-1])

        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1, 1)

        return x, (SO2_elem, Rplus_elem)
