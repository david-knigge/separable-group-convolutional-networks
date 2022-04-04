import torch
import torch.nn.functional as F

from ck_g_cnn.nn.conv.ck_g_conv import ContinuousKernelGroupConv
from ck_g_cnn.nn.kernels.Sim2_kernel import Sim2GroupKernelSeparable2D

from ck_g_cnn.groups import Sim2


class GroupConvSeparable2DEinsum(ContinuousKernelGroupConv):

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
        """ Create a group convolution layer, performing a convolution on (a homogeneous space of) the group. This
        implementation is tailored to 2-dimensional groups, where the kernel is defined as separable over the input
        group. Let i,j be in and output channels, this implementation defines the group convolution kernel as:

            k_{ij}(x, g1, g2) = k_{ij}(g_1) k_j(g2) k_j(x)

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
            self.kernel = Sim2GroupKernelSeparable2D(
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
        :grid_H: Coordinates on the group of input data.
        :param sampled_group_elements: Group elements on which to define output grid of this convolution operation.
        :return: Function on a homogeneous space of the groups
            [batch_dim, out_channels, num_group_elements, spatial_dim_1, spatial_dim_2]
        """
        # separate grids for both rotation and scale subgroups
        grid_SO2, grid_Rplus = grid_H

        # sample group elements to apply to filters
        if sampled_group_elements is None:

            sampled_elem_SO2, sampled_elem_Rplus = self.group.sample(
                num_elements=self.num_group_elem,
                method=self.sampling_method,
                separable=True
            )

        else:
            sampled_elem_SO2, sampled_elem_Rplus = sampled_group_elements

        SO2_kernel, Rplus_kernel, Rn_conv_kernel = self.kernel.sample(
            sampled_elem_SO2=sampled_elem_SO2,
            sampled_elem_Rplus=sampled_elem_Rplus,
            grid_SO2=grid_SO2,
            grid_Rplus=grid_Rplus
        )

        # [batch, channel_in, rot_in, scale_in, width, height] *
        # [channel_out, rot_out, channel_in, rot_in] ->
        #   [batch, channel_out, rot_out, scale_in, width, height]
        x = torch.einsum(
            'bcrswh,ogcr->bogswh',
            x.reshape(-1, self.kernel.in_channels, grid_SO2.shape[0], grid_Rplus.shape[0],
                      x.shape[3], x.shape[4]),
            SO2_kernel.reshape(
                self.kernel.out_channels,
                sampled_elem_SO2.shape[0],  # num_group_elem in output
                self.kernel.in_channels,
                grid_SO2.shape[0]  # num group elem in input
            )
        )

        # [batch, channel_out, rot_out, scale_in, width, height] *
        # [channel_out, scale_out, scale_in] ->
        #   [batch, channel_out, rot_out, scale_out, width, height]
        x = torch.einsum(
            'bogswh,ofs->bogfwh',  # sum over input channel dim
            x,
            Rplus_kernel.reshape(
                self.kernel.out_channels,
                sampled_elem_Rplus.shape[0],  # num_group_elem in output
                grid_Rplus.shape[0]  # num group elem in input
            )
        ).reshape(
            -1,
            self.kernel.out_channels * sampled_elem_SO2.shape[0] * sampled_elem_Rplus.shape[0],
            x.shape[-2],
            x.shape[-1]
        )

        x = F.conv2d(
            input=x,
            weight=Rn_conv_kernel.reshape(
                self.kernel.out_channels * sampled_elem_SO2.shape[0] * sampled_elem_Rplus.shape[0],
                1,
                self.kernel.kernel_size,
                self.kernel.kernel_size
            ),
            padding=self.padding,
            stride=self.stride,
            groups=(self.kernel.out_channels * sampled_elem_SO2.shape[0] * sampled_elem_Rplus.shape[0])  # Each out channel and group element gets its own weight.
        )

        x = x.reshape(-1, self.kernel.out_channels, sampled_elem_SO2.shape[0] * sampled_elem_Rplus.shape[0], x.shape[-1], x.shape[-2])

        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1, 1)

        return x, (sampled_elem_SO2, sampled_elem_Rplus)


class GroupConvSeparable2D(ContinuousKernelGroupConv):

    def __init__(self, group, kernel_size, in_channels, out_channels, num_group_elem, ck_net_num_hidden,
                 ck_net_hidden_size, ck_net_implementation, ck_net_first_omega_0, ck_net_omega_0, sampling_method,
                 stride, padding=True):
        """ Create a groups convolution layer, performing a convolution on (a homogeneous space of) the groups

        :param group: instance of a Group object
        :param kernel_size: integer defining the size of the convolution kernel
        :param in_channels: size of input channel dimension
        :param out_channels: size of output channel dimension
        :param padding: boolean indicating whether or not to use implicit padding on the input
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
            self.kernel = Sim2GroupKernelSeparable2D(
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
        # separate grids for both rotation and scale subgroups
        grid_SO2, grid_Rplus = grid_H
        batch_size = x.shape[0]
        width = x.shape[-1]

        # sample group elements to apply to filters
        if sampled_group_elements is None:

            sampled_elem_SO2, sampled_elem_Rplus = self.group.sample(
                num_elements=self.num_group_elem,
                method=self.sampling_method,
                separable=True
            )

        else:
            sampled_elem_SO2, sampled_elem_Rplus = sampled_group_elements

        SO2_kernel, Rplus_kernel, Rn_conv_kernel = self.kernel.sample(
            sampled_elem_SO2=sampled_elem_SO2,
            sampled_elem_Rplus=sampled_elem_Rplus,
            grid_SO2=grid_SO2,
            grid_Rplus=grid_Rplus
        )

        # reshape x into [batch_dim, channels * rot_dim, scale_dim * spatial_1, spatial_2],
        # expanding the group into separate dimensions
        x = x.reshape(
            -1,
            x.shape[1] * grid_SO2.shape[0],
            grid_Rplus.shape[0] * width,
            width
        )

        # convolve over rotation group elements, reshape into [batch_dim, channels_out, rot_out,
        # scale_in, spatial_1, spatial_2]
        x = F.conv2d(
            input=x,
            weight=SO2_kernel.reshape(
                self.kernel.out_channels * sampled_elem_SO2.shape[0],  # num_group_elem in output
                self.kernel.in_channels * grid_SO2.shape[0],  # num group elem in input
                1,
                1
            ),
            padding=0
        ).reshape(
            batch_size,
            self.kernel.out_channels,
            sampled_elem_SO2.shape[0],
            grid_Rplus.shape[0],
            width,
            width
        )

        # Fold rotation out dimension out of the way by reshaping it into the batch dimension
        x = x.transpose(1, 2).reshape(
            batch_size * sampled_elem_SO2.shape[0],
            self.kernel.out_channels * grid_Rplus.shape[0],
            width,
            width
        )

        # convolve over input scale elements
        x = F.conv2d(
            input=x,
            weight=Rplus_kernel.reshape(
                self.kernel.out_channels * sampled_elem_Rplus.shape[0],
                grid_Rplus.shape[0],
                1,
                1
            ),
            padding=0,
            groups=self.kernel.out_channels
        ).reshape(
            batch_size,
            sampled_elem_SO2.shape[0],
            self.kernel.out_channels,
            sampled_elem_Rplus.shape[0],
            width,
            width
        ).transpose(1, 2) # fold rotation element back into correct position

        # convolve over spatial dimensions
        x = F.conv2d(
            input=x.reshape(
                batch_size,
                self.kernel.out_channels * sampled_elem_SO2.shape[0] * sampled_elem_Rplus.shape[0],
                width,
                width
            ),
            weight=Rn_conv_kernel.reshape(
                self.kernel.out_channels * sampled_elem_SO2.shape[0] * sampled_elem_Rplus.shape[0],
                1,
                self.kernel.kernel_size,
                self.kernel.kernel_size
            ),
            padding=self.padding,
            stride=self.stride,
            groups=(self.kernel.out_channels * sampled_elem_SO2.shape[0] *
                    sampled_elem_Rplus.shape[0])  # each out channel and elem gets own weight
        )

        # reshape in into [batch_dim, channels_out, group_out, spatial_1, spatial_2]
        x = x.reshape(
            -1,
            self.kernel.out_channels,
            sampled_elem_SO2.shape[0] * sampled_elem_Rplus.shape[0],
            x.shape[-1],
            x.shape[-2]
        )

        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1, 1)

        return x, (sampled_elem_SO2, sampled_elem_Rplus)