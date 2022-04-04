import torch


def circular_mask(kernels, kernel_grids):
    """ Applies a circular mask to a filter bank, based on the relative distance of a gridpoint the kernel was sampled
        on.

    @param kernels: Kernel bank, [output_channels, output_group, input_channels, (input_group), spatial_1, spatial_2]
    @param kernel_grids: Grid the kernel bank was sampled on [output_group, input_coords, (input_group), spatial_1,
        spatial_2]
    """
    # mask out kernel values at spatial positions with a norm larger than 1
    mask = (torch.norm(kernel_grids[:, :2], dim=1) > 1.0)

    # add output channel dimension
    mask = mask.unsqueeze(0)

    # add input channel (and optionally group input) dimensions to mask
    mask = mask.unsqueeze(2)

    mask = mask.expand_as(kernels)

    # discard kernel values at masked positions
    kernels[mask] = 0
    return kernels


def circular_mask_smooth(kernels, kernel_grids, max_rel_dist=1.0, slope=2.0, from_dim=1, up_to_dim=-0):
    """ Applies a smooth circular mask to a filter bank, based on the relative distance of a gridpoint the kernel was
        sampled on.

    @param kernels: Kernel bank, [output_channels, output_group, input_channels, (input_group), spatial_1, spatial_2]
    @param kernel_grids: Grid the kernel bank was sampled on [output_group, input_coords, (input_group), spatial_1,
        spatial_2]
    """

    smooth_mask = torch.sigmoid(slope * (max_rel_dist - torch.norm(kernel_grids, dim=1)))

    # add output channel dimension
    smooth_mask = smooth_mask.unsqueeze(0)

    # add input channel (and optionally group input) dimensions to mask
    smooth_mask = smooth_mask.unsqueeze(2)

    smooth_mask = smooth_mask.expand_as(kernels)

    return torch.mul(kernels, smooth_mask)


def distance_mask(kernels, kernel_grids, max_rel_dist=1.0):
    """ Applies a distance mask to a filter bank, based on the relative manhattan distance of a gridpoint the kernel was
        sampled on.

    @param kernels: Kernel bank, [output_channels, output_group, input_channels, (input_group), spatial_1, spatial_2]
    @param kernel_grids: Grid the kernel bank was sampled on [output_group, input_coords, (input_group), spatial_1,
        spatial_2]
    """
    #  mask all values with coordinates larger than 1
    mask = torch.logical_or(
        torch.abs(kernel_grids[:, 0]) > max_rel_dist,
        torch.abs(kernel_grids[:, 1]) > max_rel_dist
    )

    # add output channel dim
    mask = mask.unsqueeze(0)

    # add input channel dim
    mask = mask.unsqueeze(2)

    # expand to correct dims
    kernels[mask.expand_as(kernels)] = 0
    return kernels


def distance_mask_smooth(kernels, kernel_grids, max_rel_dist=1.0, slope=5.0):
    """ Applies a smooth circular mask to a filter bank, based on the relative distance of a gridpoint the kernel was
        sampled on.

    @param kernels: Kernel bank, [output_channels, output_group, input_channels, (input_group), spatial_1, spatial_2]
    @param kernel_grids: Grid the kernel bank was sampled on [output_group, input_coords, (input_group), spatial_1,
        spatial_2]
    """

    smooth_mask = torch.amax(torch.abs(kernel_grids)[:, 0:2], dim=1)
    smooth_mask = torch.sigmoid(slope * (max_rel_dist - smooth_mask))

    # add output channel dimension
    smooth_mask = smooth_mask.unsqueeze(0)

    # add input channel (and optionally group input) dimensions to mask
    smooth_mask = smooth_mask.unsqueeze(2)

    smooth_mask = smooth_mask.expand_as(kernels)
    return torch.mul(kernels, smooth_mask)


def group_distance_mask_smooth(kernels, kernel_grids, max_dist=1.0, slope=5.0, dim=-1):
    """ Applies a smooth circular mask to a filter bank, based on the relative distance of a gridpoint the kernel was
        sampled on.

    @param kernels: Kernel bank, [output_channels, output_group, input_channels, (input_group), spatial_1, spatial_2]
    @param kernel_grids: Grid the kernel bank was sampled on [output_group, input_coords, (input_group), spatial_1,
        spatial_2]
    """

    smooth_mask = torch.sigmoid(slope * (max_dist - torch.abs(kernel_grids)[:, dim]))

    # add output channel dimension
    smooth_mask = smooth_mask.unsqueeze(0)

    # add input channel (and optionally group input) dimensions to mask
    smooth_mask = smooth_mask.unsqueeze(2)

    smooth_mask = smooth_mask.expand_as(kernels)

    return torch.mul(kernels, smooth_mask)
