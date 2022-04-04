import torch
import numpy as np

from ck_g_cnn.nn.kernels.group_kernel import GroupKernel

from ck_g_cnn.nn.ck import CKNetLinear
import ck_g_cnn.nn.functional as gF


class Sim2LiftingKernel(GroupKernel):

    def __init__(self, group, kernel_size, in_channels, out_channels, ck_net_num_hidden,
                 ck_net_hidden_size, ck_net_implementation, ck_net_first_omega_0, ck_net_omega_0, sampling_method):
        super(Sim2LiftingKernel, self).__init__(
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

        # spatial grid for scale equivariance depends on the maximum scale of the group
        self.grid_R2 = torch.stack(torch.meshgrid(
            torch.linspace(-self.group.Rplus.max_scale, self.group.Rplus.max_scale, self.kernel_size,
                           dtype=torch.float),
            torch.linspace(-self.group.Rplus.max_scale, self.group.Rplus.max_scale, self.kernel_size,
                           dtype=torch.float),
        )).to(self.group.identity.device)

    def transform_grid_by_group(self, grid, elements):
        """ Transform the kernel grid by all groups elements

        :param grid: kernel grid [2, kernel_size, kernel_size]
        :param elements: tensor of group elements to apply transformations for, [num_elements, 2]. The rotation group
            is in the first dimension, the scale in the second.
        :return: list of transformed grids with length num_group_elements
        """
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
        batched_grid = grid.transpose(1, -1).reshape(-1, 2)

        # sample kernel values
        kernels = self.cknet(batched_grid).view(
            num_group_elements,
            self.kernel_size,
            self.kernel_size,
            self.out_channels,
            self.in_channels,
        ).transpose(0, 1).transpose(0, 3).transpose(2, 4)

        # apply circular mask to ensure rotation equivariance
        kernels = gF.circular_mask_smooth(kernels, grid, max_rel_dist=1.0, slope=3.0)

        # weight kernels by determinant, this amounts to scalar multiplication along the output group dimension
        inv_det = 1 / self.group.determinant(sampled_group_elements)
        kernels = kernels * inv_det[None, :, None, None, None]

        return kernels


class Sim2GroupKernel(GroupKernel):

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
            in_features=4,
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
            torch.linspace(-self.group.Rplus.max_scale, self.group.Rplus.max_scale, self.kernel_size,
                           dtype=torch.float),
            torch.linspace(-self.group.Rplus.max_scale, self.group.Rplus.max_scale, self.kernel_size,
                           dtype=torch.float),
        )).to(self.group.identity.device)

    def transform_grid_by_group(self, grid_Rn, grid_H, sampled_group_elements):
        """ Create kernel grid and transform grid by all sampled group elements

        :param grid_Rn: spatial kernel grid [2, kernel_size, kernel_size]
        :param grid_H: groups dimension kernel grid [1, num_group_elements]
        :param sampled_group_elements: group element to transform grids by
        :return: list of transformed grids with length num_group_elements
        """

        # number of sample points is equal in each group dimension
        no_input_group_elem = grid_H.shape[0]
        no_output_group_elem = sampled_group_elements.shape[0]
        kernel_size = grid_Rn.shape[-1]

        # batch transform the input grid over the group with all sampled group elements
        # -> [out_group_dim, in_group_dim, 2]
        transformed_grid_H = self.group.logarithmic_map(self.group.left_action_on_H(
            self.group.inverse(sampled_group_elements), grid_H))

        # normalize group dim values
        transformed_grid_H = self.group.normalize(transformed_grid_H)

        # perform group action on Rd, for all sampled group elements
        # -> [out_group_dim, 2, kernel_size, kernel_size]
        transformed_grid_Rn = self.group.left_action_on_Rd(self.group.inverse(sampled_group_elements), grid_Rn)

        # swap in_group_dim and group coordinates, reshape into 2d meshgrid, add spatial dimensions
        transformed_grid_H = transformed_grid_H.transpose(1, 2).view(
            no_output_group_elem,
            self.group.dimension,
            no_input_group_elem,
            1,
            1
        )

        # expand into spatial dimensions
        transformed_grid_H = transformed_grid_H.expand(
            no_output_group_elem,
            2,
            no_input_group_elem,
            kernel_size,
            kernel_size
        )

        # add group dimensions to Rn grid
        transformed_grid_Rn = transformed_grid_Rn.view(
            no_output_group_elem,
            2,
            1,
            kernel_size,
            kernel_size
        )

        # repeat Rn grid along group dimensions
        transformed_grid_Rn = transformed_grid_Rn.expand(
            no_output_group_elem,
            2,
            no_input_group_elem,
            kernel_size,
            kernel_size
        )

        # concatenate the Rn and H grids
        return torch.cat((transformed_grid_Rn, transformed_grid_H), dim=1)

    def sample(self, grid_H, sampled_group_elements):
        """ Create a lifting convolution kernel for a given number of rotation group elements.

        @param num_group_elements: number of group elements to transform the kernel by.
        """

        num_group_elements = sampled_group_elements.shape[0]

        # obtain kernel grid, localized over lie algebra
        grid = self.transform_grid_by_group(self.grid_R2, grid_H, sampled_group_elements)

        # coordinates in last dimension. Reshape [out_group, coords, in_group, kernel_size, kernel_size]
        # to [out_group, in_group, kernel_size, kernel_size, coords] to [*, coords]
        flattened_grid = grid.transpose(1, 2).transpose(2, 3).transpose(3, 4).reshape(-1, 4)

        # sample kernel values
        kernels = self.cknet(flattened_grid).view(
            num_group_elements,  # allows for batching all grids at once
            grid_H.shape[0], # number of group elements in input
            self.kernel_size,  # kernel size is in last dimension
            self.kernel_size,
            self.out_channels,
            self.in_channels
        ).permute(4, 0, 5, 1, 2, 3) # nog omschrijven naar transposeje

        # apply circular mask to ensure rotation equivariance
        kernels = gF.circular_mask_smooth(kernels, grid, max_rel_dist=1.0, slope=3.0)

        # apply smooth mask over group axis to regulate inter-scale interactions
        kernels = gF.group_distance_mask_smooth(
            kernels=kernels,
            kernel_grids=grid,
            max_dist=torch.log(self.group.Rplus.max_scale) / num_group_elements,
            slope=5.0,
            dim=-1 # only localise over the scale group axis
        )

        # weight kernels by determinant, this amounts to scalar multiplication along the output group dimension
        inv_det = 1 / self.group.determinant(sampled_group_elements)
        kernels = kernels * inv_det[None, :, None, None, None, None]

        return kernels


class Sim2GroupKernelSeparable2D(GroupKernel):

    def __init__(self, group, kernel_size, in_channels, out_channels, ck_net_num_hidden,
                 ck_net_hidden_size, ck_net_implementation, ck_net_first_omega_0, ck_net_omega_0, sampling_method):
        super(Sim2GroupKernelSeparable2D, self).__init__(
            group=group,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            sampling_method=sampling_method
        )

        self.cknet_SO2 = CKNetLinear(
            in_features=1,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            hidden_features=ck_net_hidden_size,
            hidden_layers=ck_net_num_hidden,
            first_omega_0=ck_net_first_omega_0,
            hidden_omega_0=ck_net_omega_0,
            implementation=ck_net_implementation
        )

        self.cknet_Rplus = CKNetLinear(
            in_features=1,
            in_channels=1, # Is now a convolution over a single channel.
            out_channels=out_channels,
            kernel_size=1,
            hidden_features=ck_net_hidden_size,
            hidden_layers=ck_net_num_hidden,
            first_omega_0=ck_net_first_omega_0,
            hidden_omega_0=ck_net_omega_0,
            implementation=ck_net_implementation
        )

        self.cknet_Rn = CKNetLinear(
            in_features=2,
            in_channels=1,  # Is now a convolution over a single channel.
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
            torch.linspace(-self.group.Rplus.max_scale, self.group.Rplus.max_scale, self.kernel_size,
                           dtype=torch.float),
            torch.linspace(-self.group.Rplus.max_scale, self.group.Rplus.max_scale, self.kernel_size,
                           dtype=torch.float),
        )).to(self.group.identity.device)

    def transform_grid_by_group(self, grid_Rn, grid_SO2, grid_Rplus, sampled_elem_SO2, sampled_elem_Rplus):
        """ Transform the kernel grid by all groups elements

        :param grid_Rn: spatial kernel grid [2, kernel_size, kernel_size]
        :param grid_H: groups dimension kernel grid [1, num_group_elements]
        :param sampled_group_elements: group element to transform grids by
        :return: two stacks of transformed grids with length num_group_elements
        """
        transformed_grid_SO2 = self.group.SO2.normalize(self.group.SO2.logarithmic_map(
            self.group.SO2.left_action_on_H(self.group.SO2.inverse(sampled_elem_SO2), grid_SO2)
        ))

        transformed_grid_Rplus = self.group.Rplus.logarithmic_map(
            self.group.Rplus.left_action_on_H(self.group.Rplus.inverse(sampled_elem_Rplus), grid_Rplus)
        )

        transformed_grid_Rn = self.group.left_action_on_Rd(
            self.group.inverse(
                torch.stack(torch.meshgrid(sampled_elem_SO2, sampled_elem_Rplus)).view(self.group.dimension, -1).T
            ),
            grid_Rn
        )

        return transformed_grid_Rn, transformed_grid_SO2, transformed_grid_Rplus

    def sample(self, grid_SO2, grid_Rplus, sampled_elem_SO2, sampled_elem_Rplus):
        """ Create a convolutional kernel over the group given an input grid.

        :param num_group_elements: number of group elements to sample from the group
        :param grid_Rn: stack of grids of dimensions [no_grids, 2, kernel_size, kernel_size] containing kernel coordinates
        :param grid_H: stack of grids of dimensions [no_grids, 1, num_group_elements] containing kernel coordinates
        :return:
        """

        # obtain kernel grid, localized over lie algebra
        transformed_grid_Rn, transformed_grid_SO2, transformed_grid_Rplus = self.transform_grid_by_group(
            self.grid_R2, grid_SO2, grid_Rplus, sampled_elem_SO2, sampled_elem_Rplus)

        # put coordinates in last dimension, flatten into list
        flattened_grid_Rn = transformed_grid_Rn.transpose(1, -1).reshape(-1, 2)

        # flatten into list
        flattened_grid_SO2 = transformed_grid_SO2.reshape(-1, 1)

        flattened_grid_Rplus = transformed_grid_Rplus.reshape(-1, 1)

        # get g conv separate
        kernels_SO2 = self.cknet_SO2(flattened_grid_SO2).view(
            sampled_elem_SO2.shape[0],
            grid_SO2.shape[0],  # no_group_elem in the input
            self.out_channels,
            self.in_channels,
        ).transpose(0, 1).transpose(2, 3).transpose(0, 3).unsqueeze(-1).unsqueeze(-1)  # .permute(2, 0, 3, 1)

        kernels_Rplus = self.cknet_Rplus(flattened_grid_Rplus).view(
            sampled_elem_Rplus.shape[0],
            grid_Rplus.shape[0],
            self.out_channels,
            1,
        ).transpose(0, 1).transpose(2, 3).transpose(0, 3).unsqueeze(-1).unsqueeze(-1)

        # get spatial conv separate
        kernels_Rn = self.cknet_Rn(flattened_grid_Rn).view(
            int(sampled_elem_SO2.shape[0] * sampled_elem_Rplus.shape[0]),
            self.kernel_size,
            self.kernel_size,
            self.out_channels,
            1,
        ).transpose(1, 3).transpose(2, 4).transpose(0, 1)

        # apply circular mask to kernels to ensure rotation equivariance
        kernels_Rn = gF.circular_mask_smooth(kernels_Rn, transformed_grid_Rn)

        # apply smoothed distance mask over the scale group axis
        kernels_Rplus = gF.group_distance_mask_smooth(
            kernels_Rplus,
            transformed_grid_Rplus.view(sampled_elem_Rplus.shape[0], 1, grid_Rplus.shape[0], 1, 1),
            max_dist=torch.log(self.group.Rplus.max_scale) / sampled_elem_Rplus.shape[0],
            slope=10.0,
            dim=-1  # only localise over the scale group axis
        )

        # weight kernels by determinant, this amounts to scalar multiplication along the output group dimension
        inv_det = 1 / self.group.Rplus.determinant(sampled_elem_Rplus)
        kernels_Rn = kernels_Rn * inv_det[None, :, None, None, None].repeat(1, sampled_elem_SO2.shape[0], 1, 1, 1)

        return kernels_SO2, kernels_Rplus, kernels_Rn


class Sim2GroupKernelGSeparable2D(GroupKernel):

    def __init__(self, group, kernel_size, in_channels, out_channels, ck_net_num_hidden,
                 ck_net_hidden_size, ck_net_implementation, ck_net_first_omega_0, ck_net_omega_0, sampling_method):
        super(Sim2GroupKernelGSeparable2D, self).__init__(
            group=group,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            sampling_method=sampling_method
        )

        self.cknet_SO2 = CKNetLinear(
            in_features=1,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            hidden_features=ck_net_hidden_size,
            hidden_layers=ck_net_num_hidden,
            first_omega_0=ck_net_first_omega_0,
            hidden_omega_0=ck_net_omega_0,
            implementation=ck_net_implementation
        )

        self.cknet_Rplus = CKNetLinear(
            in_features=1,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            hidden_features=ck_net_hidden_size,
            hidden_layers=ck_net_num_hidden,
            first_omega_0=ck_net_first_omega_0,
            hidden_omega_0=ck_net_omega_0,
            implementation=ck_net_implementation
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
            implementation=ck_net_implementation
        )

        # spatial grid for scale equivariance depends on the maximum scale of the group
        self.grid_R2 = torch.stack(torch.meshgrid(
            torch.linspace(-self.group.Rplus.max_scale, self.group.Rplus.max_scale, self.kernel_size,
                           dtype=torch.float),
            torch.linspace(-self.group.Rplus.max_scale, self.group.Rplus.max_scale, self.kernel_size,
                           dtype=torch.float),
        )).to(self.group.identity.device)

    def transform_grid_by_group(self, grid_Rn, grid_SO2, grid_Rplus, sampled_elem_SO2, sampled_elem_Rplus):
        """ Transform the kernel grid by all groups elements

        :param grid_Rn: spatial kernel grid [2, kernel_size, kernel_size]
        :param grid_H: groups dimension kernel grid [1, num_group_elements]
        :param sampled_group_elements: group element to transform grids by
        :return: two stacks of transformed grids with length num_group_elements
        """
        transformed_grid_SO2 = self.group.SO2.normalize(self.group.SO2.logarithmic_map(
            self.group.SO2.left_action_on_H(self.group.SO2.inverse(sampled_elem_SO2), grid_SO2)
        ))

        transformed_grid_Rplus = self.group.Rplus.logarithmic_map(
            self.group.Rplus.left_action_on_H(self.group.Rplus.inverse(sampled_elem_Rplus), grid_Rplus)
        )

        transformed_grid_Rn = self.group.left_action_on_Rd(
            self.group.inverse(
                torch.stack(torch.meshgrid(sampled_elem_SO2, sampled_elem_Rplus)).view(self.group.dimension, -1).T
            ),
            grid_Rn
        )

        return transformed_grid_Rn, transformed_grid_SO2, transformed_grid_Rplus

    def sample(self, grid_SO2, grid_Rplus, sampled_elem_SO2, sampled_elem_Rplus):
        """ Create a convolutional kernel over the group given an input grid.

        :param num_group_elements: number of group elements to sample from the group
        :param grid_Rn: stack of grids of dimensions [no_grids, 2, kernel_size, kernel_size] containing kernel coordinates
        :param grid_H: stack of grids of dimensions [no_grids, 1, num_group_elements] containing kernel coordinates
        :return:
        """

        # obtain kernel grid, localized over lie algebra
        transformed_grid_Rn, transformed_grid_SO2, transformed_grid_Rplus = self.transform_grid_by_group(
            self.grid_R2, grid_SO2, grid_Rplus, sampled_elem_SO2, sampled_elem_Rplus)

        # put coordinates in last dimension, flatten into list
        flattened_grid_Rn = transformed_grid_Rn.transpose(1, -1).reshape(-1, 2)

        # flatten into list
        flattened_grid_SO2 = transformed_grid_SO2.reshape(-1, 1)

        flattened_grid_Rplus = transformed_grid_Rplus.reshape(-1, 1)

        # get g conv separate
        kernels_SO2 = self.cknet_SO2(flattened_grid_SO2).view(
            sampled_elem_SO2.shape[0],
            grid_SO2.shape[0],  # no_group_elem in the input
            self.out_channels,
            self.in_channels,
        ).transpose(0, 1).transpose(2, 3).transpose(0, 3).unsqueeze(-1).unsqueeze(-1)  # .permute(2, 0, 3, 1)

        kernels_Rplus = self.cknet_Rplus(flattened_grid_Rplus).view(
            sampled_elem_Rplus.shape[0],
            grid_Rplus.shape[0],
            self.out_channels,
            self.in_channels,
        ).transpose(0, 1).transpose(2, 3).transpose(0, 3).unsqueeze(-1).unsqueeze(-1)

        # get spatial conv separate
        kernels_Rn = self.cknet_Rn(flattened_grid_Rn).view(
            int(sampled_elem_SO2.shape[0] * sampled_elem_Rplus.shape[0]),
            self.kernel_size,
            self.kernel_size,
            self.out_channels,
            self.in_channels,
        ).transpose(1, 3).transpose(2, 4).transpose(0, 1)

        # apply circular mask to kernels to ensure rotation equivariance
        kernels_Rn = gF.circular_mask_smooth(kernels_Rn, transformed_grid_Rn)

        # apply smoothed distance mask over the scale group axis
        kernels_Rplus = gF.group_distance_mask_smooth(
            kernels_Rplus,
            transformed_grid_Rplus.view(sampled_elem_Rplus.shape[0], 1, grid_Rplus.shape[0], 1, 1),
            max_dist=torch.log(self.group.Rplus.max_scale) / sampled_elem_Rplus.shape[0],
            slope=10.0,
            dim=-1  # only localise over the scale group axis
        )

        # weight kernels by determinant, this amounts to scalar multiplication along the output group dimension
        inv_det = 1 / self.group.Rplus.determinant(sampled_elem_Rplus)
        kernels_Rn = kernels_Rn * inv_det[None, :, None, None, None].repeat(1, sampled_elem_SO2.shape[0], 1, 1, 1)

        return kernels_SO2, kernels_Rplus, kernels_Rn


class Sim2GroupKernelSeparable(GroupKernel):

    def __init__(self, group, kernel_size, in_channels, out_channels, ck_net_num_hidden,
                 ck_net_hidden_size, ck_net_implementation, ck_net_first_omega_0, ck_net_omega_0, sampling_method):
        super(Sim2GroupKernelSeparable, self).__init__(
            group=group,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            sampling_method=sampling_method
        )

        self.cknet_H = CKNetLinear(
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

        self.cknet_Rn = CKNetLinear(
            in_features=2,
            in_channels=1,  # is now a convolution over a single channel
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
            torch.linspace(-self.group.Rplus.max_scale, self.group.Rplus.max_scale, self.kernel_size,
                           dtype=torch.float),
            torch.linspace(-self.group.Rplus.max_scale, self.group.Rplus.max_scale, self.kernel_size,
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

        transformed_grid_H = self.group.normalize(transformed_grid_H)

        transformed_grid_Rn = self.group.left_action_on_Rd(self.group.inverse(sampled_group_elements), grid_Rn)

        return transformed_grid_Rn, transformed_grid_H

    def sample(self, grid_H, sampled_group_elements):
        """ Create a convolutional kernel over the group given an input grid.

        :param num_group_elements: number of group elements to sample from the group
        :param grid_Rn: stack of grids of dimensions [no_grids, 2, kernel_size, kernel_size] containing kernel coordinates
        :param grid_H: stack of grids of dimensions [no_grids, 1, num_group_elements] containing kernel coordinates
        :return:
        """

        num_group_elements = sampled_group_elements.shape[0]

        # obtain kernel grid, localized over lie algebra
        transformed_grid_Rn, transformed_grid_H = self.transform_grid_by_group(self.grid_R2, grid_H, sampled_group_elements)

        # put coordinates in last dimension, flatten into list
        flattened_grid_Rn = transformed_grid_Rn.transpose(1, -1).reshape(-1, 2)

        # flatten into list
        flattened_grid_H = transformed_grid_H.reshape(-1, 2)

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
        kernels_Rn = gF.circular_mask_smooth(kernels_Rn, transformed_grid_Rn)

        # apply smoothed distance mask over the group axis
        kernels_H = gF.group_distance_mask_smooth(
            kernels_H,
            transformed_grid_H.view(num_group_elements, 2, grid_H.shape[0], 1, 1),
            max_dist=torch.log(self.group.Rplus.max_scale) / num_group_elements,
            slope=10.0,
            dim=-1  # only localise over the scale group axis
        )

        # # weight kernels by determinant, this amounts to scalar multiplication along the output group dimension
        inv_det = 1 / self.group.determinant(sampled_group_elements)

        kernels_Rn = kernels_Rn * inv_det[None, :, None, None, None]

        return kernels_H, kernels_Rn


class Sim2GroupKernelGSeparable(GroupKernel):

    def __init__(self, group, kernel_size, in_channels, out_channels, ck_net_num_hidden,
                 ck_net_hidden_size, ck_net_implementation, ck_net_first_omega_0, ck_net_omega_0, sampling_method):
        super(Sim2GroupKernelGSeparable, self).__init__(
            group=group,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            sampling_method=sampling_method
        )

        self.cknet_H = CKNetLinear(
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

        self.cknet_Rn = CKNetLinear(
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

        # spatial grid for scale equivariance depends on the maximum scale of the group
        self.grid_R2 = torch.stack(torch.meshgrid(
            torch.linspace(-self.group.Rplus.max_scale, self.group.Rplus.max_scale, self.kernel_size,
                           dtype=torch.float),
            torch.linspace(-self.group.Rplus.max_scale, self.group.Rplus.max_scale, self.kernel_size,
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

        transformed_grid_H = self.group.normalize(transformed_grid_H)

        transformed_grid_Rn = self.group.left_action_on_Rd(self.group.inverse(sampled_group_elements), grid_Rn)

        return transformed_grid_Rn, transformed_grid_H

    def sample(self, grid_H, sampled_group_elements):
        """ Create a convolutional kernel over the group given an input grid.

        :param num_group_elements: number of group elements to sample from the group
        :param grid_Rn: stack of grids of dimensions [no_grids, 2, kernel_size, kernel_size] containing kernel coordinates
        :param grid_H: stack of grids of dimensions [no_grids, 1, num_group_elements] containing kernel coordinates
        :return:
        """

        num_group_elements = sampled_group_elements.shape[0]

        # obtain kernel grid, localized over lie algebra
        transformed_grid_Rn, transformed_grid_H = self.transform_grid_by_group(self.grid_R2, grid_H, sampled_group_elements)

        # put coordinates in last dimension, flatten into list
        flattened_grid_Rn = transformed_grid_Rn.transpose(1, -1).reshape(-1, 2)

        # flatten into list
        flattened_grid_H = transformed_grid_H.reshape(-1, 2)

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
        kernels_Rn = gF.circular_mask_smooth(kernels_Rn, transformed_grid_Rn)

        # apply smoothed distance mask over the group axis
        kernels_H = gF.group_distance_mask_smooth(
            kernels_H,
            transformed_grid_H.view(num_group_elements, 2, grid_H.shape[0], 1, 1),
            max_dist=torch.log(self.group.Rplus.max_scale) / num_group_elements,
            slope=10.0,
            dim=-1  # only localise over the scale group axis
        )

        # # weight kernels by determinant, this amounts to scalar multiplication along the output group dimension
        inv_det = 1 / self.group.determinant(sampled_group_elements)

        kernels_Rn = kernels_Rn * inv_det[None, :, None, None, None]

        return kernels_H, kernels_Rn

