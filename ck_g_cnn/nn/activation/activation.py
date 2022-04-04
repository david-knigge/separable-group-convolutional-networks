from torch import nn
import torch.nn.functional as F


class ReLU(nn.Module):

    def __init__(self, inplace=False):
        """ ReLU module for group convolutional NN.

        :param inplace: Apply ReLU operation inplace.
        """
        super().__init__()
        self.inplace = inplace

    def forward(self, x, grid_H):
        return F.relu(x, inplace=self.inplace), grid_H


class ELU(nn.Module):

    def __init__(self, inplace=False, alpha=1.0):
        """ ELU module for group convolutional NN.

        :param inplace: Apply ReLU operation inplace.
        """
        super().__init__()
        self.inplace = inplace
        self.alpha = alpha

    def forward(self, x, grid_H):
        return F.elu(x, inplace=self.inplace, alpha=self.alpha), grid_H
