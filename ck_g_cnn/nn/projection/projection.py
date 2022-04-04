import torch
from torch import nn
import math


class ChannelProjection(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((in_channels, out_channels)))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, grid_H):
        """

        @param x: [batch, channels, group, x, y]
        @return:
        """
        x = x.transpose(1, -1)
        return (x @ self.weight).transpose(1, -1), grid_H
