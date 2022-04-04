import torch
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity

from ck_g_cnn.nn.kernels.group_kernel import GroupKernel

from ck_g_cnn.nn.ck import CKNetLinear
import ck_g_cnn.nn.functional as gF


class R2xRPlusLiftingKernel(GroupKernel):

    def __init__(self, group, kernel_size, in_channels, out_channels, ck_net_num_hidden,
                 ck_net_hidden_size, ck_net_implementation, ck_net_first_omega_0, ck_net_omega_0, sampling_method):
        super().__init__(
            group=group,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            sampling_method=sampling_method
        )

        self.cknet = CKNetLinear(
            in_features=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            hidden_features=ck_net_hidden_size,
            hidden_layers=ck_net_num_hidden,
            first_omega_0=ck_net_first_omega_0,
            hidden_omega_0=ck_net_omega_0,
            implementation=ck_net_implementation
        )

        # Spatial grid for scale equivariance depends on the maximum scale of the group.
        self.grid_R2 = torch.stack(torch.meshgrid(
            torch.linspace(-self.group.max_scale, self.group.max_scale, self.kernel_size,
                           dtype=torch.float),
            torch.linspace(-self.group.max_scale, self.group.max_scale, self.kernel_size,
                           dtype=torch.float),
        )).to(self.group.identity.device)

    def transform_grid_by_group(self, grid, elements):
        return self.group.left_action_on_Rd(self.group.inverse(elements), grid)

    def sample(self, sampled_group_elements):
        """ Create a lifting convolution kernel for a given number of rotation group elements.
        @param num_group_elements: number of group elements to transform the kernel by.
        @return:
        """

        num_group_elements = sampled_group_elements.shape[0]

        # obtain kernel grid, localized over lie algebra
        grid = self.transform_grid_by_group(self.grid_R2, sampled_group_elements)

        # coordinates in last dimension
        flattened_grid = grid.transpose(1, -1).reshape(-1, 2)

        # obtain convolution kernel, place out_channels in front, kernel size in back
        # [out_channels, out_group_dim, in_channels, kernel_size, kernel_size]
        kernels = self.cknet(flattened_grid).view(
            num_group_elements,
            self.kernel_size,
            self.kernel_size,
            self.out_channels,
            self.in_channels,
        ).transpose(0, 1).transpose(0, 3).transpose(2, 4)

        # apply a smooth distance mask to ensure scale equivariance
        kernels = gF.distance_mask_smooth(kernels=kernels, kernel_grids=grid, max_rel_dist=1.0, slope=2.0)

        # weight kernels by determinant of corresponding group element, this amounts to scalar multiplication along the
        # output group dimension
        inv_det = 1 / (self.group.determinant(sampled_group_elements))
        kernels = kernels * inv_det[None, :, None, None, None]

        return kernels


class R2xRplusGroupKernel(GroupKernel):

    def __init__(self, group, kernel_size, in_channels, out_channels, ck_net_num_hidden,
                 ck_net_hidden_size, ck_net_implementation, ck_net_first_omega_0, ck_net_omega_0, sampling_method):
        super().__init__(
            group=group,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            sampling_method=sampling_method
        )

        self.cknet = CKNetLinear(
            in_features=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            hidden_features=ck_net_hidden_size,
            hidden_layers=ck_net_num_hidden,
            first_omega_0=ck_net_first_omega_0,
            hidden_omega_0=ck_net_omega_0,
            implementation=ck_net_implementation
        )

        # spatial grid for scale equivariance depends on the maximum scale of the group
        self.grid_R2 = torch.stack(torch.meshgrid(
            torch.linspace(-self.group.max_scale, self.group.max_scale, self.kernel_size,
                           dtype=torch.float),
            torch.linspace(-self.group.max_scale, self.group.max_scale, self.kernel_size,
                           dtype=torch.float),
        )).to(self.group.identity.device)

    def transform_grid_by_group(self, grid_Rn, grid_H, sampled_group_elements):
        """ Create kernel grid and transform grid by all sampled group elements

        :param grid_Rn: spatial kernel grid [2, kernel_size, kernel_size]
        :param grid_H: groups dimension kernel grid [1, num_group_elements]
        :param sampled_group_elements: group element to transform grids by
        :return: tensor of transformed grids with length num_group_elements
        """
        no_output_group_elem = sampled_group_elements.shape[0]
        no_input_group_elem = grid_H.shape[0]
        kernel_size = grid_Rn.shape[-1]

        # transform grids for Rn and H separately
        transformed_grid_H = self.group.logarithmic_map(
            self.group.left_action_on_H(self.group.inverse(sampled_group_elements), grid_H))

        # transformed_grid_H = self.group.normalize(transformed_grid_H)

        transformed_grid_R2 = self.group.left_action_on_Rd(self.group.inverse(sampled_group_elements), grid_Rn)

        # repeat Rn along the angle dimension, and repeat H along the spatial dimension
        # to create a [output_group_elem, 3, num_group_elements, kernel_size, kernel_size] grid
        return torch.cat(
            (
                transformed_grid_R2.view(
                    no_output_group_elem,
                    2,
                    1,
                    kernel_size,
                    kernel_size
                ).repeat(1, 1, no_input_group_elem, 1, 1),
                transformed_grid_H.view(
                    no_output_group_elem,
                    1,
                    no_input_group_elem,
                    1,
                    1
                ).repeat(1, 1, 1, kernel_size, kernel_size)
            ),
            dim=1
        )

    def sample(self, grid_H, sampled_group_elements):
        """ Create a lifting convolution kernel for a given number of rotation group elements.

        @param num_group_elements: number of group elements to transform the kernel by.
        """
        # number of group elements in the output feature map
        num_group_elements = sampled_group_elements.shape[0]

        grid = self.transform_grid_by_group(self.grid_R2, grid_H, sampled_group_elements)

        # coordinates in last dimension. Reshape [out_group, coords, in_group, kernel_size, kernel_size]
        # to [out_group, in_group, kernel_size, kernel_size, coords] to [*, coords]
        flattened_grid = grid.transpose(1, 2).transpose(2, 3).transpose(3, 4).reshape(-1, 3)

        # obtain convolution kernel
        # [out_channels, out_group_dim, in_channels, in_group_dim, kernel_size, kernel_size]
        kernels = self.cknet(flattened_grid).view(
            num_group_elements,  # allows for batching all grids at once
            grid_H.shape[0], # number of group elements in input
            self.kernel_size,  # kernel size is in last dimension
            self.kernel_size,
            self.out_channels,
            self.in_channels
        ).permute(4, 0, 5, 1, 2, 3) # nog omschrijven naar transposeje

        # apply a smooth distance mask to ensure scale equivariance
        kernels = gF.distance_mask_smooth(
            kernels=kernels,
            kernel_grids=grid,
            max_rel_dist=1.0,
            slope=3.0
        )

        # apply smooth mask over group axis to regulate inter-scale interactions
        kernels = gF.group_distance_mask_smooth(
            kernels=kernels,
            kernel_grids=grid,
            max_dist=torch.log(self.group.max_scale) / num_group_elements,
            slope=10.0
        )

        # weight kernels by determinant, this amounts to scalar multiplication along the output group dimension
        inv_det = 1 / self.group.determinant(sampled_group_elements)
        kernels = kernels * inv_det[None, :, None, None, None, None]

        return kernels


class R2xRplusGroupKernelSeparable(GroupKernel):

    def __init__(self, group, kernel_size, in_channels, out_channels, ck_net_num_hidden,
                 ck_net_hidden_size, ck_net_implementation, ck_net_first_omega_0, ck_net_omega_0, sampling_method):
        super(R2xRplusGroupKernelSeparable, self).__init__(
            group=group,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            sampling_method=sampling_method
        )

        self.cknet_H = CKNetLinear(
            in_features=1,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            hidden_features=ck_net_hidden_size,
            hidden_layers=ck_net_num_hidden,
            first_omega_0=ck_net_first_omega_0,
            hidden_omega_0=ck_net_omega_0,
            implementation=ck_net_implementation,
        )

        self.cknet_Rn = CKNetLinear(
            in_features=2,
            in_channels=1,  # is now a convolution over a single channel
            out_channels=out_channels,
            kernel_size=kernel_size,
            hidden_features=ck_net_hidden_size,
            hidden_layers=ck_net_num_hidden,
            first_omega_0=ck_net_first_omega_0,
            hidden_omega_0=ck_net_omega_0,
            implementation=ck_net_implementation,
        )

        # spatial grid for scale equivariance depends on the maximum scale of the group
        self.grid_R2 = torch.stack(torch.meshgrid(
            torch.linspace(-self.group.max_scale, self.group.max_scale, self.kernel_size,
                           dtype=torch.float),
            torch.linspace(-self.group.max_scale, self.group.max_scale, self.kernel_size,
                           dtype=torch.float),
        )).to(self.group.identity.device)

    def transform_grid_by_group(self, grid_Rn, grid_H, sampled_group_elements):
        """ Transform the kernel grid by all groups elements

        :param grid_Rn: spatial kernel grid [2, kernel_size, kernel_size]
        :param grid_H: groups dimension kernel grid [1, num_group_elements]
        :param sampled_group_elements: group element to transform grids by
        :return: two stacks of transformed grids with length num_group_elements
        """
        transformed_grid_H = self.group.logarithmic_map(
            self.group.left_action_on_H(self.group.inverse(sampled_group_elements), grid_H))

        # transformed_grid_H = self.group.normalize(transformed_grid_H)

        transformed_grid_Rn = self.group.left_action_on_Rd(self.group.inverse(sampled_group_elements), grid_Rn)

        return transformed_grid_Rn, transformed_grid_H

    def sample(self, grid_H, sampled_group_elements):
        """ Create a convolutional kernel over the group given an input grid.

        :param num_group_elements: number of group elements to sample from the group
        :param grid_Rn: stack of grids of dimensions [no_grids, 2, kernel_size, kernel_size] containing kernel coordinates
        :param grid_H: stack of grids of dimensions [no_grids, 1, num_group_elements] containing kernel coordinates
        :return:
        """

        # number of group elements in the output feature map
        num_group_elements = sampled_group_elements.shape[0]

        transformed_grid_Rn, transformed_grid_H = self.transform_grid_by_group(self.grid_R2, grid_H, sampled_group_elements)

        # put coordinates in last dimension, flatten into list
        flattened_grid_Rn = transformed_grid_Rn.transpose(1, -1).reshape(-1, 2)

        # flatten into list
        flattened_grid_H = transformed_grid_H.reshape(-1, 1)

        # get g conv separate
        kernels_H = self.cknet_H(flattened_grid_H).view(
            num_group_elements,
            grid_H.shape[0],  # no_group_elem in the input
            self.out_channels,
            self.in_channels,
        ).transpose(0, 1).transpose(2, 3).transpose(0, 3).unsqueeze(-1).unsqueeze(-1)  # .permute(2, 0, 3, 1)

        # get spatial conv separate
        kernels_Rn = self.cknet_Rn(flattened_grid_Rn).view(
            num_group_elements,
            self.kernel_size,
            self.kernel_size,
            self.out_channels,
            1,
        ).transpose(1, 3).transpose(2, 4).transpose(0, 1)

        # apply circular mask to kernels to ensure rotation equivariance
        kernels_Rn = gF.distance_mask_smooth(
            kernels=kernels_Rn,
            kernel_grids=transformed_grid_Rn,
            max_rel_dist=1.0,
            slope=3.0
        )

        # apply smooth mask over group axis to regulate inter-scale interactions
        kernels_H = gF.group_distance_mask_smooth(
            kernels=kernels_H,
            kernel_grids=transformed_grid_H.view(num_group_elements, 1, grid_H.shape[0], 1, 1), # fully expand the grid
            max_dist=torch.log(self.group.max_scale) / num_group_elements,
            slope=10.0
        )

        # weight kernels by determinant, this amounts to scalar multiplication along the output group dimension
        inv_det = 1 / self.group.determinant(sampled_group_elements)
        kernels_Rn = kernels_Rn * inv_det[None, :, None, None, None]

        return kernels_H, kernels_Rn


class R2xRplusGroupKernelGSeparable(GroupKernel):

    def __init__(self, group, kernel_size, in_channels, out_channels, ck_net_num_hidden,
                 ck_net_hidden_size, ck_net_implementation, ck_net_first_omega_0, ck_net_omega_0, sampling_method):
        super(R2xRplusGroupKernelGSeparable, self).__init__(
            group=group,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            sampling_method=sampling_method
        )

        self.cknet_H = CKNetLinear(
            in_features=1,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            hidden_features=ck_net_hidden_size,
            hidden_layers=ck_net_num_hidden,
            first_omega_0=ck_net_first_omega_0,
            hidden_omega_0=ck_net_omega_0,
            implementation=ck_net_implementation,
        )

        self.cknet_Rn = CKNetLinear(
            in_features=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            hidden_features=ck_net_hidden_size,
            hidden_layers=ck_net_num_hidden,
            first_omega_0=ck_net_first_omega_0,
            hidden_omega_0=ck_net_omega_0,
            implementation=ck_net_implementation,
        )

        # spatial grid for scale equivariance depends on the maximum scale of the group
        self.grid_R2 = torch.stack(torch.meshgrid(
            torch.linspace(-self.group.max_scale, self.group.max_scale, self.kernel_size,
                           dtype=torch.float),
            torch.linspace(-self.group.max_scale, self.group.max_scale, self.kernel_size,
                           dtype=torch.float),
        )).to(self.group.identity.device)

    def transform_grid_by_group(self, grid_Rn, grid_H, sampled_group_elements):
        """ Transform the kernel grid by all groups elements

        :param grid_Rn: spatial kernel grid [2, kernel_size, kernel_size]
        :param grid_H: groups dimension kernel grid [1, num_group_elements]
        :param sampled_group_elements: group element to transform grids by
        :return: two stacks of transformed grids with length num_group_elements
        """
        transformed_grid_H = self.group.logarithmic_map(
            self.group.left_action_on_H(self.group.inverse(sampled_group_elements), grid_H))

        # transformed_grid_H = self.group.normalize(transformed_grid_H)

        transformed_grid_Rn = self.group.left_action_on_Rd(self.group.inverse(sampled_group_elements), grid_Rn)

        return transformed_grid_Rn, transformed_grid_H

    def sample(self, grid_H, sampled_group_elements):
        """ Create a convolutional kernel over the group given an input grid.

        :param num_group_elements: number of group elements to sample from the group
        :param grid_Rn: stack of grids of dimensions [no_grids, 2, kernel_size, kernel_size] containing kernel coordinates
        :param grid_H: stack of grids of dimensions [no_grids, 1, num_group_elements] containing kernel coordinates
        :return:
        """

        # number of group elements in the output feature map
        num_group_elements = sampled_group_elements.shape[0]

        transformed_grid_Rn, transformed_grid_H = self.transform_grid_by_group(self.grid_R2, grid_H, sampled_group_elements)

        # put coordinates in last dimension, flatten into list
        flattened_grid_Rn = transformed_grid_Rn.transpose(1, -1).reshape(-1, 2)

        # flatten into list
        flattened_grid_H = transformed_grid_H.reshape(-1, 1)

        # get g conv separate
        kernels_H = self.cknet_H(flattened_grid_H).view(
            num_group_elements,
            grid_H.shape[0],  # no_group_elem in the input
            self.out_channels,
            self.in_channels,
        ).transpose(0, 1).transpose(2, 3).transpose(0, 3).unsqueeze(-1).unsqueeze(-1)  # .permute(2, 0, 3, 1)

        # get spatial conv separate
        kernels_Rn = self.cknet_Rn(flattened_grid_Rn).view(
            num_group_elements,
            self.kernel_size,
            self.kernel_size,
            self.out_channels,
            self.in_channels,
        ).transpose(1, 3).transpose(2, 4).transpose(0, 1)

        # apply circular mask to kernels to ensure rotation equivariance
        kernels_Rn = gF.distance_mask_smooth(
            kernels=kernels_Rn,
            kernel_grids=transformed_grid_Rn,
            max_rel_dist=1.0,
            slope=3.0
        )

        # apply smooth mask over group axis to regulate inter-scale interactions
        kernels_H = gF.group_distance_mask_smooth(
            kernels=kernels_H,
            kernel_grids=transformed_grid_H.view(num_group_elements, 1, grid_H.shape[0], 1, 1), # fully expand the grid
            max_dist=torch.log(self.group.max_scale) / num_group_elements,
            slope=10.0
        )

        # weight kernels by determinant, this amounts to scalar multiplication along the output group dimension
        inv_det = 1 / self.group.determinant(sampled_group_elements)
        kernels_Rn = kernels_Rn * inv_det[None, :, None, None, None]

        return kernels_H, kernels_Rn

