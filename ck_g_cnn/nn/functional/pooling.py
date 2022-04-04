import torch
from torch import nn
import torch.nn.functional as F


def group_max_pooling_RnxH(x, kernel_size=2, stride_Rn=2, stride_H=2):
    """
        Global pooling over both the spatial and groups dimensions.

        x -- Input of dimensions [batch_dim, channels, group_dim, spatial_1, spatial_2]
    """
    pass


def group_max_pooling_Rn(x, kernel_size=2, stride_Rn=2):
    """
        Pooling over only the spatial dimensions.

        x -- Input of dimensions [batch_dim, channels, group_dim, spatial_1, spatial_2]
    """
    # reshape groups into channel dimension
    no_channels = x.shape[1]
    no_group_dims = x.shape[2]

    x = x.reshape(-1, x.shape [1] * x.shape[2], x.shape[3], x.shape[4])
    x = F.max_pool2d(x, kernel_size=kernel_size, stride=stride_Rn)
    return x.reshape(-1, no_channels, no_group_dims, x.shape[-2], x.shape[-1])


def adaptive_group_max_pooling_Rn(x, output_size=1):
    # reshape groups into channel dimension
    no_channels = x.shape[1]
    no_group_dims = x.shape[2]

    x = x.reshape(-1, x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
    x = F.adaptive_max_pool2d(x, output_size=output_size)
    return x.reshape(-1, no_channels, no_group_dims, x.shape[-2], x.shape[-1])


def group_max_pooling_H(x, kernel_size=2, stride_H=2):
    """
    Pooling over only the groups dimension.

    x -- Input of dimensions [batch_dim, channels, group_dim, spatial_1, spatial_2]
    """
    pass

