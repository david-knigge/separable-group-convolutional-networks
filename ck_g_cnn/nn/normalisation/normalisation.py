from torch import nn
import torch.nn.functional as F


class BatchNorm(nn.Module):

    def __init__(self, channels, **kwargs):
        super(BatchNorm, self).__init__()
        self.bn = nn.BatchNorm3d(channels)
        # self.bn.weight.data.fill_(1)
        # self.bn.bias.data.zero_()

    def forward(self, x, grid_H):
        return self.bn(x), grid_H


class InstanceNorm(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, grid_H):
        """ Normalise over group, x, y

        @param x: [batch, channel, group, x, y]
        @param grid_H:
        @return:
        """
        return F.layer_norm(x, x.shape[-3:]), grid_H


class LayerNorm(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, grid_H):
        """ Normalise over channel, group, x, y

        @param x: [batch, channel, group, x, y]
        @param grid_H:
        @return:
        """
        return F.layer_norm(x, x.shape[-4:]), grid_H
