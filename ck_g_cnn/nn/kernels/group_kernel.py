import torch
import torch.nn as nn

from ck_g_cnn.groups.group import Group


class GroupKernel(nn.Module):

    def __init__(self,
                 group,
                 kernel_size,
                 in_channels,
                 out_channels,
                 sampling_method):
        """ Group convolution kernels are sampled through this module.

        :param group: Group, implements group action on Rn and H subgroups.
        :param kernel_size: Convolution kernel size.
        :param in_channels: number of input channels for convolution kernel.
        :param out_channels: number of output channels for convolution kernel.
        :param sampling_method: Sampling method either 'discretise' or 'uniform'.
        """
        super(GroupKernel, self).__init__()

        self.group = group

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.sampling_method = sampling_method

        # create spatial kernel grid
        if kernel_size == 1:
            self.grid_R2 = torch.tensor([[[0.]], [[0.]]]).to(self.group.identity.device)
        else:
            self.grid_R2 = torch.stack(torch.meshgrid(
                torch.linspace(-1, 1, self.kernel_size, dtype=torch.float),
                torch.linspace(-1, 1, self.kernel_size, dtype=torch.float),
            )).to(self.group.identity.device)

    def sample(self, **kwargs):
        """ Sample convolution kernels for a given number of group elements

        arguments should include:
        :param num_group_elements: the number of group elements to sample over the group dimension
        :param grid_H (in group conv) : grid over the group on which the input function is defined

        should return:
        :return kernels: filter bank extending over all input channels, containing kernels transformed for all
            output group elements.
        :return sampled_group_elements: the relative offsets (group elements) over which the filter bank is defined on
            the subgroup H.
        """
        raise NotImplementedError()

    def transform_grid_by_group(self, **kwargs):
        """ Transform the kernel grid by all groups elements. Grid may be separated over
        spatial and group dimensions before transformation.

        arguments should include:
        :param grid: (optionally multiple) tensors of relative coordinates
        :param elements: tensor of group elements to apply transformations for

        should return:
        :return: list of transformed grids with length num_group_elements
        :return sampled_group_elements: the relative offsets (group elements) over which the filter bank is defined on
            the subgroup H.
        """
        raise NotImplementedError()
